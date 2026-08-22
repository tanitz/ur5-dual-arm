"""
Running a two-arm program.

Steps execute on a worker thread so the GUI stays responsive, and every
transition is reported through callbacks rather than touched directly — the
executor knows nothing about Qt.

The awkward part of driving UR controllers over sockets is that `movel`
returns the moment it is sent, not when the arm arrives. There is no
completion event to wait on, so arrival is inferred: the joints have stopped
and the TCP is where it was asked to be. `_wait_until_arrived` is that check,
and it takes a set of arms rather than one — which is what makes `together` a
single line instead of two lines and a barrier.

A line's `link` picks which engine runs it, and they are genuinely different
machines:

    solo/together   movej or movel per arm, and a joint wait
    pair            one world delta to both arms at matched speed, guarded
                    here at 50 Hz because no servo loop is involved
    coupled         the Coordinator, and its guards on every 125 Hz cycle
"""

import threading
import time

import numpy as np

from ..coupling import (
    Coordinator, CouplingError, DRIFT_HARD_MULTIPLE, FIGHT_FORCE, HeldObject,
    limit_uncalibrated_rotation,
)
from ..geometry.kinematics import (
    inv, mat_to_pose, pose_distance, pose_to_mat, xyz_rpy_to_mat,
)
from .steps import (
    CONTROL_KINDS, FOUND_SUFFIX, Program, apply_offset, resolve_target,
    target_kind,
)

IDLE_SPEED = 0.01        # rad/s below which a joint counts as stopped
ARRIVE_LIN = 0.002       # m
ARRIVE_ANG = 0.02        # rad
SETTLE_TIME = 0.15       # s of continuous stillness before believing it
GUARD_PERIOD = 0.02      # s between guard samples on a pair move — 50 Hz
POLL_PERIOD = 0.02       # s between reads while waiting on an input
# A jump can loop, and a loop of nothing but jumps would spin a core without
# ever reaching a step that sleeps. This is not a limit on how long a program
# may run — every arm move resets it — it is what turns `top: jump top` from a
# hung panel into a message.
CONTROL_SPIN_LIMIT = 20000


class ProgramError(RuntimeError):
    pass


class Executor:
    """Runs a Program against a Cell. One at a time."""

    def __init__(self, cell, points, simulate=False, vision=None):
        self.cell = cell
        self.points = points
        self.simulate = simulate
        # The camera, if the cell has one. A program without a FIND never
        # touches it, and a cell without one still runs every other step.
        self.vision = vision
        # The surface the box slides on, if it has been measured: the map from
        # pixels onto it and the places picks were taught against. Set by
        # whoever owns the camera, for the same reason and at the same time.
        # When it holds a map, FIND reads three numbers through it rather than
        # six through `vision.camera_to_world`.
        self.surface = None

        self.object = HeldObject()
        self.coordinator = None
        # what SET_VAR sets and IF reads. Cleared at every start, because a
        # count left over from the last run is a program that behaves
        # differently the second time for reasons nothing on screen shows.
        self.vars = {}
        # what FIND writes: rigid corrections, kept apart from the numbers so
        # `IF count > 3` can never be handed a pose to compare
        self.poses = {}

        self.thread = None
        self._pause = threading.Event()
        self._stop = threading.Event()
        self.running = False
        self.current = -1

        # callbacks: (index, step) / (text) / (ok, message)
        self.on_step = None
        self.on_log = None
        self.on_finished = None

    # -- plumbing ----------------------------------------------------------
    def log(self, text):
        if self.on_log:
            self.on_log(text)
        else:
            self.cell.log(text)

    def start(self, program, only=None, load=None):
        """Run a program, or one line of it.

        `only` is a row index: `▷ To` runs that line by itself, which is how an
        operator drives to a taught position without running what is around it.
        Validation then starts from what is *actually* held rather than from
        nothing, because a single line is being run into a cell that already
        has a state.

        `load` resolves a CALL to another Program by name. The whole call tree
        is expanded here, before the thread starts — a call that cannot be
        loaded is a problem on paper, not a stop half way through a move.
        """
        if self.running:
            raise ProgramError("a program is already running")

        if only is not None:
            if not (0 <= only < len(program.steps)):
                raise ProgramError("no step selected")
            plan = [(program.steps[only], only)]
            check_against = Program(program.name)
            check_against.steps = [program.steps[only]]
        else:
            try:
                plan = program.expand(load or self._no_loader)
            except ValueError as e:
                raise ProgramError(str(e))
            check_against = Program(program.name)
            check_against.steps = [step for step, _i in plan]

        problems, warnings = check_against.check(
            self.points, self.cell.config, holding=self.object.held,
            surfaces=self.taught_on_surface())
        if problems:
            raise ProgramError("; ".join(problems))
        for w in warnings:
            self.log("warning: %s" % w)

        # labels are resolved against the *flattened* plan, so a jump lands
        # where the step actually is rather than where it was written
        labels = {}
        for i, (step, _row) in enumerate(plan):
            if step.kind == "LABEL" and step.enabled:
                labels[(step.get("name") or "").strip()] = i

        self.vars = {}
        self.poses = {}
        self._stop.clear()
        self._pause.clear()
        self.running = True
        self.thread = threading.Thread(
            target=self._run,
            args=(plan, program.loop and only is None, labels),
            daemon=True)
        self.thread.start()

    @staticmethod
    def _no_loader(name):
        raise ValueError("no program library to call '%s' from" % name)

    def pause(self):
        """Hold where the program is, mid-move if that is where it is.

        Each engine has to be held in its own way, because "stop and carry on"
        means something different to each of them:

            solo/together   halt, and re-send the same target on resume. The
            pair            targets are absolute world poses that were already
                            resolved and reach-checked, so the second send
                            finishes the move the first one started rather
                            than working out a new one.
            coupled         freeze the servo plan. Halting here would drop two
                            controllers out from under a box they are both
                            holding, and leave no plan to resume onto.

        Halting is left to whoever is waiting on the move — `pause` only says
        so — because the wait is the one place that knows what was sent and
        can put it back. A step that is merely standing still (DELAY, WAIT_IN,
        the gap between lines) has nothing to halt and just holds.
        """
        self._pause.set()
        if self.coordinator is not None:
            self.coordinator.freeze()
        self.log("paused")

    def resume(self):
        if self.coordinator is not None:
            self.coordinator.thaw()
        self._pause.clear()
        self.log("resumed")

    def _hold_if_paused(self):
        """Block while paused, and return the seconds spent holding.

        Only called where no arm is moving under a command of ours: between
        steps, and inside the two waits that are already standing still.
        Callers with a deadline push it out by what comes back, so a pause
        cannot time out a wait that was nowhere near its limit.

        Raises if Stop is pressed instead of Resume.
        """
        if self._stop.is_set():
            raise ProgramError("stopped")
        if not self._pause.is_set():
            return 0.0
        began = time.monotonic()
        while self._pause.is_set():
            time.sleep(0.05)
            if self._stop.is_set():
                raise ProgramError("stopped")
        return time.monotonic() - began

    def _hold_mid_move(self, resend, guard=None):
        """Stop the arms where they are, hold, then put them back on their way.

        Returns the seconds held, so the arrival deadline can be pushed out —
        an arm that spent two minutes paused has not taken two minutes to get
        anywhere. `guard` keeps running throughout: two arms holding one
        workpiece can still be levering against each other while stopped, and
        that is exactly the fault worth catching before motion starts again.
        """
        self.cell.halt()
        self.log("     held part way — arms stopped")
        began = time.monotonic()
        while self._pause.is_set():
            if self._stop.is_set():
                raise ProgramError("stopped")
            if guard is not None:
                guard()
            time.sleep(0.05)
        if self._stop.is_set():
            raise ProgramError("stopped")
        held = time.monotonic() - began
        resend()
        self.log("     carrying on")
        return held

    def stop(self):
        self._stop.set()
        # a stopped program is not a paused one. Every hold checks the stop
        # flag after the pause flag, so clearing it here releases them into
        # the abort rather than into a resume.
        self._pause.clear()
        if self.coordinator is not None:
            self.coordinator.abort()
        self.cell.halt()

    @property
    def paused(self):
        return self._pause.is_set()

    # -- the loop ----------------------------------------------------------
    def _run(self, plan, loop, labels):
        ok, message = True, "finished"
        try:
            while True:
                index = 0
                spun = 0
                while index < len(plan):
                    step, row = plan[index]
                    self._hold_if_paused()
                    if not step.enabled:
                        index += 1
                        continue
                    # the row this came from, not where it sits in the
                    # flattened plan: a called program has no rows of its own
                    # on the screen
                    self.current = row
                    if self.on_step:
                        self.on_step(row, step)
                    self.log("%3d  %s" % (row + 1, step.describe()))

                    jump = self._execute(step, labels)
                    if jump is None:
                        index += 1
                        spun = 0 if step.kind not in CONTROL_KINDS else spun + 1
                    else:
                        index = jump
                        spun += 1
                    if spun > CONTROL_SPIN_LIMIT:
                        raise ProgramError(
                            "%d jumps without the arms doing anything — this "
                            "program is looping on itself" % spun)
                if not loop:
                    break
        except (ProgramError, CouplingError, OSError, KeyError, ValueError) as e:
            ok, message = False, str(e)
            self.cell.halt()
        finally:
            self._shutdown_coordinator()
            self.running = False
            self.current = -1
            self.log("program %s: %s" % ("finished" if ok else "aborted", message))
            if self.on_finished:
                self.on_finished(ok, message)

    def _execute(self, step, labels=None):
        """Run one step. Returns the index to jump to, or None to carry on."""
        kind = step.kind
        labels = labels or {}
        if kind == "MOVE":
            self._move(step)
        elif kind == "OUT":
            self._set_out(step)
        elif kind == "WAIT_IN":
            self._wait_input(step)
        elif kind == "LABEL":
            pass
        elif kind == "SET_VAR":
            self._set_var(step)
        elif kind == "FIND":
            self._find(step)
        elif kind == "JUMP":
            return self._label_index(step.get("target"), labels)
        elif kind == "IF":
            return self._branch(step, labels)
        elif kind == "BARRIER":
            self._wait_arms_idle(self.cell.connected_ids)
        elif kind == "DELAY":
            self._sleep(float(step.get("seconds", 0)))
        elif kind == "ATTACH":
            self._attach(step)
        elif kind == "DETACH":
            self._detach()
        elif kind == "WHERE":
            self._where()
        else:
            raise ProgramError("cannot execute %r" % kind)
        return None

    # -- resolving a column ------------------------------------------------
    def _arm(self, arm_id):
        arm = self.cell.arms[arm_id]
        if not arm.connected:
            raise ProgramError("arm %s is not connected" % arm_id)
        return arm

    def _correction_for(self, target):
        """The rigid transform a target asks to be carried by, or None.

        A target naming a correction that was never found is refused rather
        than run uncorrected: a pick that quietly falls back to where the box
        used to be is the one failure this whole feature exists to prevent.
        """
        name = (target or {}).get("correct_by")
        if not name:
            return None
        if name not in self.poses:
            raise ProgramError(
                "this line is corrected by %r, and nothing has found it — put "
                "a FIND above it, or branch on %s%s when it finds nothing"
                % (name, name, FOUND_SUFFIX))
        return self.poses[name]

    def _resolve_arm(self, arm_id, target):
        """Where a column sends one arm, in world.

        An offset with no place in it is read against the arm's *live* pose,
        which is why this happens as the line runs rather than when it was
        written.
        """
        arm = self._arm(arm_id)
        return resolve_target(target, self.points,
                              current=arm.tcp_matrix_world(),
                              base=arm.base_matrix(),
                              correction=self._correction_for(target))

    def _check_reachable(self, arm_id, pose_world):
        reachable, why = self.cell.check_reachable(arm_id, pose_world)
        if not reachable:
            raise ProgramError(why)

    def _speed(self, target, default_key="object_lin_speed"):
        return float((target or {}).get("speed")
                     or self.cell.config.limits[default_key])

    # -- MOVE --------------------------------------------------------------
    def _move(self, step):
        link = step.link
        if link == "coupled":
            self._move_coupled(step)
        elif link == "pair":
            self._move_pair(step)
        else:
            self._move_arms(step)

    def _move_arms(self, step):
        """solo and together. The only difference is how many columns are
        filled — both are sent, and then both are waited for."""
        plan = {}
        for arm_id, target in step.arm_slots():
            if target_kind(target) is None:
                continue
            pose = self._resolve_arm(arm_id, target)
            self._check_reachable(arm_id, pose)
            plan[arm_id] = (pose, target)
        if not plan:
            raise ProgramError("this line has no arm to move")

        if self.simulate:
            self._sleep(0.2)
            return

        # every target is resolved and reach-checked before anything is sent,
        # so a bad second column cannot leave the first arm already moving
        def send():
            """Put this line on the wire. Called again to resume a pause.

            The targets are absolute world poses, so the second send finishes
            the move rather than repeating it: a movel carries on down the
            same straight line from wherever it stopped, and a movej lands on
            the same solution because the controller picks the branch nearest
            the configuration it is in — which is one it was already passing
            through.
            """
            for arm_id, (pose, target) in plan.items():
                arm = self.cell.arms[arm_id]
                speed = self._speed(target)
                if target.get("motion", "movej") == "movel":
                    arm.movel_world(pose, vel=speed)
                else:
                    arm.movej_world(pose, vel=speed)

        send()
        self._wait_until_arrived({a: p for a, (p, _t) in plan.items()},
                                 resend=send)

    def _move_pair(self, step):
        """One world delta, given to both arms at matched speed.

        This is the Jog tab's A+B column as a program line. The pairing is in
        the geometry — two arms given the same world translation travel the
        same way at the same speed, so a workpiece between them is carried —
        and not in any attempt to keep two commands in step.

        There is no servo loop here, so the force and separation guards run in
        the arrival wait at 50 Hz instead of at 125. That is enough for a slow
        translation and is why the speed is capped at what a coupled carry
        would use, and why rotation is refused rather than merely discouraged.
        """
        target = step.slot("pair")
        offset = np.asarray(target.get("offset", [0.0] * 6), dtype=float)

        # the validator has said all of this already; this is the last gate
        # before two arms move, and it does not take the validator's word
        if target.get("frame", "world") != "world":
            raise ProgramError("a pair move is world frame only")
        if float(np.max(np.abs(offset[3:]))) > 1e-9:
            raise ProgramError("a pair move cannot rotate — attach and turn it "
                               "coupled")
        if not self.cell.config.translation_calibrated:
            raise ProgramError("a pair move needs the relative base directions "
                               "measured — run tests/check_directions_online.py "
                               "--apply")
        if self.object.held:
            raise ProgramError("something is attached; carry it with a coupled "
                               "move rather than a pair move")

        arm_ids = ("A", "B")
        for a in arm_ids:
            self._arm(a)

        ceiling = float(self.cell.config.limits["object_lin_speed"])
        speed = min(self._speed(target), ceiling)
        if self._speed(target) > ceiling:
            self.log("     pair speed capped at %.0f mm/s" % (ceiling * 1000))

        plan = {}
        for a in arm_ids:
            pose = mat_to_pose(apply_offset(self.cell.arms[a].tcp_matrix_world(),
                                            offset, "world"))
            self._check_reachable(a, pose)
            plan[a] = pose

        if self.simulate:
            self._sleep(0.2)
            return

        guard = self._pair_guard(arm_ids)

        def send():
            for a in arm_ids:
                self.cell.arms[a].movel_world(plan[a], vel=speed)

        send()
        # the guard is built once and kept across a hold on purpose: drift is
        # measured from where the carry began, so a pause in the middle of one
        # must not quietly re-zero how far the arms have come apart
        self._wait_until_arrived(plan, guard=guard, resend=send)

    def _pair_guard(self, arm_ids):
        """What is watched while two uncoupled arms carry one thing.

        The same two questions the coordinated loop asks, and the same answer
        to the first one: drift alone is not a fault, because two controllers
        tracking the same path lag by different amounts and that is harmless
        with nothing between them. It becomes a fault when the wrists are also
        loaded — or when it is so far past the limit that no amount of tracking
        error explains it.
        """
        limit = float(self.cell.config.motion["max_pair_drift"])
        baseline = {a: self.cell.force_vector(a) for a in arm_ids}
        start = self.cell.relative_transform(max_age=None)

        def guard():
            hot = self.cell.force_exceeded(arm_ids, baseline)
            rel = self.cell.relative_transform()
            drift = None
            if rel is not None and start is not None:
                d_t, _d_r = pose_distance(mat_to_pose(start), mat_to_pose(rel))
                drift = d_t
            if drift is not None and drift > limit * DRIFT_HARD_MULTIPLE:
                self.cell.halt()
                raise ProgramError(
                    "the arms have moved %.1f mm relative to each other — far "
                    "past the %.1f mm limit, so this is not tracking error"
                    % (drift * 1000, limit * 1000))
            if hot and drift is not None and drift > limit:
                arm, value, what = hot[0]
                self.cell.halt()
                raise ProgramError(
                    "arm %s is pushing %.0f %s while the pair has drifted "
                    "%.1f mm — they are levering against each other"
                    % (arm, value, what, drift * 1000))
            if hot and float(hot[0][1]) > FIGHT_FORCE * 2:
                arm, value, what = hot[0]
                self.cell.halt()
                raise ProgramError("arm %s is pushing %.0f %s" % (arm, value, what))

        return guard

    def _move_coupled(self, step):
        """The object frame drives both arms, through the Coordinator."""
        if self.coordinator is None or not self.object.held:
            raise ProgramError("nothing is attached")
        target = step.slot("obj")
        kind = target_kind(target)
        lin_speed = target.get("lin_speed")
        ang_speed = target.get("ang_speed")
        frame = target.get("frame", "world")
        pivot = target.get("pivot")

        if kind == "offset":
            # A relative object move is taken apart so each half goes through
            # the entry point that guards it: rotation carries the limit that
            # applies before a touch-off, and translation does not.
            offset = np.asarray(target["offset"], dtype=float)
            rv = offset[3:]
            angle = float(np.linalg.norm(rv))
            if angle > 1e-9:
                self.coordinator.rotate_object(self.object, rv / angle, angle,
                                               ang_speed, frame, pivot)
            if float(np.linalg.norm(offset[:3])) > 1e-9:
                here = self.coordinator.current_pose(self.object)
                moved = apply_offset(pose_to_mat(here),
                                     np.concatenate([offset[:3], np.zeros(3)]),
                                     frame)
                self._carry(mat_to_pose(moved), lin_speed, ang_speed)
            return

        pose = resolve_target(target, self.points)
        self._carry(pose, lin_speed, ang_speed)

    def _carry(self, pose_world, lin_speed, ang_speed):
        """Take the object to an absolute pose, with both arms reach-checked
        first and the turn it implies held to what has been measured."""
        for a in self.object.arm_ids:
            tcp = self.object.tcp_world(a, pose_world)
            self._check_reachable(
                a, np.concatenate([tcp[:3, 3], [0, 0, 0]]))

        # A pose target turns the box as much as a rotation step does, and
        # before a touch-off the error that produces grows with the angle just
        # the same. command_move does not ask, so the question is asked here.
        _d_t, d_r = pose_distance(self.coordinator.current_pose(self.object),
                                  pose_world)
        if len(self.object.arm_ids) > 1 and not self.simulate:
            speed = float(ang_speed or self.cell.config.limits["object_ang_speed"])
            ang_speed, refusal = limit_uncalibrated_rotation(
                d_r, speed, self.cell.config.calibrated)
            if refusal:
                raise ProgramError(refusal)
        self.coordinator.move_object(self.object, pose_world, lin_speed, ang_speed)

    # -- the rest of the vocabulary ----------------------------------------
    # -- I/O ---------------------------------------------------------------
    def _arm_ids_for(self, step):
        arm = step.get("arm")
        return ("A", "B") if arm == "both" else (arm,)

    def _set_out(self, step):
        """Drive a digital output. A gripper is one of these."""
        number = int(step.get("output", 0))
        state = bool(step.get("state", True))
        for arm_id in self._arm_ids_for(step):
            arm = self.cell.arms.get(arm_id)
            if arm is None:
                raise ProgramError("no arm %r to set an output on" % arm_id)
            if arm.connected and not self.simulate:
                arm.motion.set_digital_out(number, state)
        self._sleep(float(step.get("settle", 0.4)))

    def _read_input(self, arm_id, number):
        """One bit of a controller's digital input word."""
        arm = self.cell.arms.get(arm_id)
        if arm is None or not arm.connected:
            raise ProgramError("arm %s is not connected, so its inputs cannot "
                               "be read" % arm_id)
        bits = int(arm.state().get("digital_in_bits", 0))
        return bool(bits >> int(number) & 1)

    def _wait_input(self, step):
        """Hold here until an input reads the way the step asks.

        A timeout of zero waits forever, which is what a program feeding off a
        machine that may be slow actually wants; anything else fails the
        program rather than carrying on as though the signal had arrived,
        because carrying on is how an arm reaches into a fixture that is not
        ready yet.
        """
        arm_id = step.get("arm")
        number = int(step.get("input", 0))
        want = bool(step.get("state", True))
        timeout = float(step.get("timeout", 0) or 0)
        deadline = time.monotonic() + timeout if timeout > 0 else None
        while True:
            held = self._hold_if_paused()
            if deadline is not None:
                deadline += held
            if self.simulate or self._read_input(arm_id, number) == want:
                return
            if deadline is not None and time.monotonic() > deadline:
                raise ProgramError(
                    "arm %s input %d never went %s (waited %.0f s)"
                    % (arm_id, number, "ON" if want else "OFF", timeout))
            time.sleep(POLL_PERIOD)

    # -- variables and branching -------------------------------------------
    def _set_var(self, step):
        name = (step.get("name") or "").strip()
        value = float(step.get("value", 0))
        op = step.get("op", "=")
        current = float(self.vars.get(name, 0.0))
        self.vars[name] = {"=": value,
                           "+=": current + value,
                           "-=": current - value}[op]
        self.log("     %s = %g" % (name, self.vars[name]))

    # -- the camera --------------------------------------------------------
    def _camera_to_world(self):
        vision = self.cell.config.vision
        placed = vision.get("camera_to_world") or {}
        return xyz_rpy_to_mat(placed.get("xyz", [0.0] * 3),
                              placed.get("rpy", [0.0] * 3))

    def taught_on_surface(self):
        """The names a plane file holds, or None when there is no map.

        None is not "none taught". It is "this cell does not read the box off
        a surface", which is what sends a FIND's reference to the point
        library instead, and the two must not be confused: a cell with a map
        and nothing taught on it has to say so rather than quietly look
        somewhere else.
        """
        if self.surface is None or not self.surface.ready:
            return None
        return set(self.surface.references)

    def _find(self, step):
        """Look for the box, and work out how far it has moved.

        What is stored is not where the box is — it is the rigid transform
        from where the box was when the pick was taught to where it is now.
        A step corrected by it is carried the same way the box was, which is
        why the wrist follows the box round instead of spinning on the spot.

        How that transform is measured depends on what this cell has been
        told about itself, and the choice is made here rather than by a
        setting, because both answers cannot be right at once:

            a plane map      three numbers — where the box sits on the surface
                             it slides on, and how far round. Needs no camera
                             placement: the map was fitted through the lens
                             from the box itself.

            camera_to_world  six, from the opening's full solved pose. What a
                             box that can tilt needs, and what a placement
                             this package measures to about 9 mm costs.

        A cell with neither is refused. It used to run the second one against
        an identity transform, which puts every detection in the camera's own
        frame and produces a correction that is confidently wrong — caught, if
        at all, by `max_correction` noticing the size of it.
        """
        into = (step.get("into") or "part").strip()
        reference = (step.get("reference") or "").strip()
        found_var = into + FOUND_SUFFIX
        self.poses.pop(into, None)
        self.vars[found_var] = 0.0

        if self.vision is None:
            raise ProgramError("this cell has no camera to look with")
        if not self.vision.running:
            self.vision.start()

        reading = self.vision.fresh(float(step.get("timeout", 5.0) or 5.0))
        if reading.detection is None:
            self.log("     nothing found: %s"
                     % (reading.error or "nothing standing on the surface"))
            return
        if reading.detection.square:
            self.log("     the box reads as square — its turn is only known "
                     "to a quarter, so a pick that cares which way round it "
                     "is will be wrong one time in two")

        if self.surface is not None and self.surface.ready:
            correction, where = self._on_surface(reference, reading.detection)
            note = "  (corners fit %.1f mm)" % (where.fit_error * 1000)
        elif self.cell.config.vision.get("calibrated"):
            now = reading.detection.matrix_in(self._camera_to_world())
            was = pose_to_mat(self.points.get(reference))
            correction = now @ inv(was)
            note = ""
        else:
            raise ProgramError(
                "this cell cannot turn a detection into a place in itself: it "
                "has no plane map, and the camera has never been placed in "
                "the cell either. Fit one into %s, or calibrate "
                "vision.camera_to_world."
                % self.cell.config.vision.get("plane_file", "the plane file"))

        self._check_correction(correction, into)
        self.poses[into] = correction
        self.vars[found_var] = 1.0
        shift = mat_to_pose(correction)
        self.log("     %s moved %+.1f %+.1f %+.1f mm, turned %+.1f deg%s"
                 % (into, shift[0] * 1000, shift[1] * 1000, shift[2] * 1000,
                    np.degrees(np.linalg.norm(shift[3:])), note))

    def _on_surface(self, reference, detection):
        """The correction from where the box sits on its surface.

        The import is made here rather than at the top of the module for the
        reason `calibrate.solve_hand_eye` makes its own: this file is the
        engine that runs every program, and none of the ones without a FIND in
        them should fail to start on a cell where OpenCV is not installed.
        """
        from ..vision.planar import PlaneMapError
        try:
            return self.surface.correction(
                reference, detection.corners,
                self.cell.config.vision.get("box_size"))
        except PlaneMapError as exc:
            raise ProgramError(str(exc))

    def _check_correction(self, correction, into):
        """A camera may nudge a taught pick. It may not send an arm somewhere
        nobody has ever looked.

        Nothing in this cell checks one arm against the other, and a
        correction is the one number in a program that no operator typed — so
        a misdetection has to be refused here or discovered by an arm.
        """
        vision = self.cell.config.vision
        limit = float(vision.get("max_correction", 0.10))
        limit_deg = float(vision.get("max_correction_deg", 30.0))
        shift = mat_to_pose(correction)
        moved = float(np.linalg.norm(shift[:3]))
        turned = float(np.degrees(np.linalg.norm(shift[3:])))
        if moved > limit or turned > limit_deg:
            raise ProgramError(
                "%s was found %.0f mm and %.0f deg from where it was taught, "
                "past the %.0f mm / %.0f deg a camera may move a pick. Either "
                "the box is not where the program thinks, or that is not the "
                "box." % (into, moved * 1000, turned, limit * 1000, limit_deg))

    def _label_index(self, name, labels):
        name = (name or "").strip()
        if name not in labels:
            raise ProgramError("no label called %r" % name)
        return labels[name]

    def _branch(self, step, labels):
        """Take the jump if the test holds, the other one if it does not."""
        if step.get("source") == "var":
            name = (step.get("name") or "").strip()
            left = float(self.vars.get(name, 0.0))
            right = float(step.get("value", 0))
            compare = step.get("compare", "==")
            held = {"==": left == right, "!=": left != right,
                    "<": left < right, ">": left > right,
                    "<=": left <= right, ">=": left >= right}[compare]
            self.log("     %s is %g, so %s %g is %s"
                     % (name, left, compare, right, held))
        else:
            want = bool(step.get("state", True))
            if self.simulate:
                held = False
            else:
                held = self._read_input(step.get("arm"),
                                        int(step.get("input", 0))) == want
            self.log("     input is %s" % ("as asked" if held else "not"))

        if held:
            return self._label_index(step.get("target"), labels)
        other = (step.get("otherwise") or "").strip()
        return self._label_index(other, labels) if other else None

    def _attach(self, step):
        arm_ids = tuple(a for a in ("A", "B") if self.cell.arms[a].connected)
        if len(arm_ids) < 2 and step.get("origin", "midpoint") == "midpoint":
            raise ProgramError("two connected arms are needed to take hold "
                               "with a midpoint origin")
        self.object = HeldObject(step.get("object", "object"))
        self.object.capture(self.cell, arm_ids, step.get("origin", "midpoint"))
        self.log("     grip span %.1f mm" % ((self.object.span() or 0) * 1000))

        self.coordinator = Coordinator(self.cell, simulate=self.simulate)
        self.coordinator.start(self.object)

    def _detach(self):
        self._shutdown_coordinator()
        self.object.release()

    def _shutdown_coordinator(self):
        if self.coordinator is not None:
            self.coordinator.shutdown()
            self.coordinator = None

    def _where(self):
        """Read the cell out into the log. Moves nothing.

        The panel shows this live, but a program that stopped somewhere
        unexpected is read afterwards, and a dry run has no arms to look at.
        """
        for a in self.cell.connected_ids:
            arm = self.cell.arms[a]
            pose = arm.tcp_pose_world()
            self.log("     %s  %8.1f %8.1f %8.1f mm   %6.1f %6.1f %6.1f deg"
                     % (a, pose[0] * 1000, pose[1] * 1000, pose[2] * 1000,
                        *np.degrees(pose[3:])))
            self.log("        joints  " + "  ".join(
                "%7.1f" % v for v in np.degrees(arm.state()["q_actual"])))
        gap = self.cell.tcp_separation()
        if gap is not None:
            self.log("     gap %.1f mm" % (gap * 1000))
        if self.object.held:
            pose = self.object.pose_world
            self.log("     object '%s'  %8.1f %8.1f %8.1f mm   %6.1f %6.1f %6.1f deg"
                     % (self.object.name, pose[0] * 1000, pose[1] * 1000,
                        pose[2] * 1000, *np.degrees(pose[3:])))

    # -- waiting -----------------------------------------------------------
    def _sleep(self, seconds):
        """Wait out a delay, holding the clock across a pause.

        Counting paused time against the delay would mean Resume skipping
        straight past what is left of it — a 10 s DELAY paused for a minute
        would be over before the operator let go.
        """
        remaining = float(seconds)
        while remaining > 0:
            self._hold_if_paused()
            mark = time.monotonic()
            time.sleep(min(0.02, remaining))
            remaining -= time.monotonic() - mark

    def _wait_until_arrived(self, targets, timeout=60.0, guard=None,
                            resend=None):
        """Return when every arm in `targets` has stopped where it was sent.

        `targets` is {arm id: world pose}. Waiting for the set rather than for
        one arm is what lets a line send both and still know when the line is
        over — the barrier is inside the step instead of being a step of its
        own. `guard` runs on every sample and raises to abort.

        `resend` puts the same command back on the wire, and is what makes
        Pause work in the middle of a move: this is the only place that knows
        the arms are travelling, so it is where they are stopped and started
        again. Without one, a pause here would hold until the arms had already
        arrived — no worse than before, but no better.
        """
        deadline = time.monotonic() + timeout
        still_since = None
        while True:
            if self._stop.is_set():
                raise ProgramError("stopped")
            if self._pause.is_set():
                if resend is None:
                    deadline += self._hold_if_paused()
                else:
                    deadline += self._hold_mid_move(resend, guard)
                # a halted arm was still, and stillness from before the hold
                # says nothing about whether it has arrived since
                still_since = None
            if guard is not None:
                guard()

            settled, worst = True, None
            for arm_id, target in targets.items():
                arm = self.cell.arms[arm_id]
                if not arm.ready():
                    raise ProgramError("arm %s: %s / %s"
                                       % (arm_id, arm.robot_mode(),
                                          arm.safety_mode()))
                st = arm.state()
                moving = float(np.max(np.abs(st["qd_actual"]))) > IDLE_SPEED
                d_t, d_r = pose_distance(arm.tcp_pose_world(), target)
                if moving or d_t >= ARRIVE_LIN or d_r >= ARRIVE_ANG:
                    settled = False
                if worst is None or d_t > worst[1]:
                    worst = (arm_id, d_t, d_r)

            if settled:
                still_since = still_since or time.monotonic()
                if time.monotonic() - still_since >= SETTLE_TIME:
                    return
            else:
                still_since = None

            if time.monotonic() > deadline:
                raise ProgramError(
                    "arm %s did not arrive: still %.1f mm / %.2f deg away"
                    % (worst[0], worst[1] * 1000, np.degrees(worst[2])))
            time.sleep(GUARD_PERIOD)

    def _wait_arms_idle(self, arm_ids, timeout=60.0):
        deadline = time.monotonic() + timeout
        still_since = None
        while True:
            if self._stop.is_set():
                raise ProgramError("stopped")
            moving = False
            for a in arm_ids:
                st = self.cell.arms[a].state()
                if float(np.max(np.abs(st["qd_actual"]))) > IDLE_SPEED:
                    moving = True
            if not moving:
                still_since = still_since or time.monotonic()
                if time.monotonic() - still_since >= SETTLE_TIME:
                    return
            else:
                still_since = None
            if time.monotonic() > deadline:
                raise ProgramError("arms never settled at the barrier")
            time.sleep(0.02)
