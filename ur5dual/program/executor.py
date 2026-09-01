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

from ..axes import shown_xyz
from ..comms import LinkError
from ..coupling import (
    Coordinator, CouplingError, DRIFT_HARD_MULTIPLE, FIGHT_FORCE, HeldObject,
    limit_uncalibrated_rotation,
)
from ..geometry.kinematics import (
    inv, mat_to_pose, pose_distance, pose_to_mat,
    xyz_rpy_to_mat,
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

# The speed dial, as a percentage of what the cell may do — `limits.max_*` for
# a line one arm runs by itself, `limits.object_*` for one that moves a
# workpiece held by both. 100 is the cell flat out.
#
# It comes up at START_SPEED_PCT every launch and is never written to disk, on
# purpose. A dial that remembers is a dial that comes back at whatever the last
# shift left it on, and the failure that matters is the one where somebody
# turned it up to prove a cycle time, closed the panel, and the next person to
# press Run gets that speed on a program they have not seen move.
MIN_SPEED_PCT = 1.0
MAX_SPEED_PCT = 100.0
START_SPEED_PCT = 10.0


def _off_home_rfh(was, now, yaw, tilt):
    """Right/forward/height difference in the frame shown on the camera card."""
    delta = now[:3, 3] - was[:3, 3]
    return ("R%+.1f F%+.1f H%+.1f mm, yaw%+.1f tilt %.1f deg"
            % (delta[0] * 1000, delta[1] * 1000, delta[2] * 1000,
               np.degrees(yaw), np.degrees(tilt)))


def _as_number(value):
    """What a reply is worth to a variable an IF can test.

    A machine answering `12.5` means twelve and a half; one answering `OK`
    means it answered, and 1 is the only honest number for that — variables
    hold numbers here, and inventing a string namespace so that `IF reply ==
    "OK"` could exist would be a second vocabulary for the one case where the
    match on the RECV line has already asked the question.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


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
        # The machines the cell stands next to, if it has any. Owned by
        # whoever owns the camera and for the same reason: a program that
        # hands a machine its start signal must not depend on which panel an
        # operator happened to leave open. A program with no SEND or RECV in
        # it never touches this, and a cell with no links still runs.
        self.links = None
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
        # and, when the hold was taught as a distance from the box rather than
        # as a place in the cell, where that hold is now
        self.stances = {}

        # How fast this run goes, for every line in it. See START_SPEED_PCT.
        self.speed_pct = START_SPEED_PCT

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

        # Say it out loud before anything moves. The dial is one number for
        # the whole run and it is the difference between a crawl and a lunge,
        # so the log has to show what it was set to, not just that Run was
        # pressed.
        self.log("speed %.0f%% — up to %.0f mm/s solo, %.0f mm/s carried"
                 % (self.speed_pct,
                    self._speed("max_lin_speed") * 1000,
                    self._speed("object_lin_speed") * 1000))

        self.vars = {}
        self.poses = {}
        self.stances = {}
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
        elif kind == "SEND":
            self._send(step)
        elif kind == "RECV":
            self._recv(step)
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

    def _stances_for(self, arm_id):
        """The holds this arm has, by the name the FIND that found them used.

        Selecting the arm here rather than inside `resolve_target` keeps that
        function what it is — a name and a place — and puts the one thing that
        knows which arm a column drives in the one place that already does.
        """
        return {name: held[arm_id] for name, held in self.stances.items()
                if arm_id in held}

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
                              correction=self._correction_for(target),
                              stances=self._stances_for(arm_id))

    def _check_reachable(self, arm_id, pose_world):
        reachable, why = self.cell.check_reachable(arm_id, pose_world)
        if not reachable:
            raise ProgramError(why)

    def set_speed_pct(self, pct):
        """Turn the dial, clamped. Safe to call while a program is running:
        the next line sent picks it up, the one already on the wire does not."""
        self.speed_pct = max(MIN_SPEED_PCT, min(MAX_SPEED_PCT, float(pct)))
        return self.speed_pct

    def _speed(self, ceiling_key):
        """What this run may ask for, in m/s or rad/s.

        A percentage of the cell's ceiling, not of the speed stamped on the
        line. That field is still read, written and saved -- a program keeps
        the pace it was taught at -- but the dial is what the arms are given,
        so one number governs a whole run and a program taught at a crawl and
        one taught at speed behave the same way when it is turned down.
        """
        return (self.speed_pct / 100.0
                * float(self.cell.config.limits[ceiling_key]))

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
            speed = self._speed("max_lin_speed")
            for arm_id, (pose, target) in plan.items():
                arm = self.cell.arms[arm_id]
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

        # Both arms are pushing the same workpiece, so 100% here is the
        # coupled-carry ceiling rather than what one arm could do alone.
        speed = self._speed("object_lin_speed")

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
        # Same as a pair line: the object between the grippers sets what
        # 100% means, not what either arm could manage on its own.
        lin_speed = self._speed("object_lin_speed")
        ang_speed = self._speed("object_ang_speed")
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
            ang_speed, refusal = limit_uncalibrated_rotation(
                d_r, ang_speed, self.cell.config.calibrated)
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

    # -- the machines beside the cell --------------------------------------
    def _link_service(self):
        if self.links is None:
            raise ProgramError(
                "this cell has no machines set up — add one on the "
                "Communication tab before a program can talk to it")
        return self.links

    def _send(self, step):
        """Put one named datum on the wire, and carry on.

        Fire and forget on purpose. A machine that answers is a RECV on the
        next line, which is a line the operator can see and time out; folding
        the reply into the send would hide a wait inside a step that reads
        like it does not have one.
        """
        name = (step.get("link") or "").strip()
        item = (step.get("item") or "").strip() or None
        if self.simulate:
            self.log("     (simulated) %s" % step.describe())
            return
        try:
            self._link_service().send(name, item, text=step.get("text"))
        except LinkError as exc:
            raise ProgramError(str(exc))

    def _recv(self, step):
        """Hold here until the machine says what the step is waiting for.

        The same bargain WAIT_IN makes with a digital input, and for the same
        reason: a timeout of zero waits as long as it takes, which is what a
        cell fed by a slow machine wants, and any other timeout fails the
        program rather than carrying on as though the signal had arrived.
        Carrying on is how an arm reaches into a fixture that is not ready.
        """
        name = (step.get("link") or "").strip()
        item = (step.get("item") or "").strip() or None
        into = (step.get("into") or "").strip()
        timeout = float(step.get("timeout", 0) or 0)
        if self.simulate:
            if into:
                self.vars[into] = 1.0
            self.log("     (simulated) %s" % step.describe())
            return

        links = self._link_service()
        deadline = time.monotonic() + timeout if timeout > 0 else None
        while True:
            held = self._hold_if_paused()
            if deadline is not None:
                deadline += held
            try:
                got, value = links.attempt(name, item)
            except LinkError as exc:
                raise ProgramError(str(exc))
            if got:
                if into:
                    self.vars[into] = _as_number(value)
                    self.log("     %s = %g" % (into, self.vars[into]))
                return
            if deadline is not None and time.monotonic() > deadline:
                raise ProgramError(
                    "%s never sent %s (waited %.1f s)"
                    % (name, item or "anything", timeout))
            time.sleep(POLL_PERIOD)

    # -- the camera --------------------------------------------------------
    def _camera_to_world(self):
        vision = self.cell.config.vision
        placed = vision.get("camera_to_world") or {}
        return xyz_rpy_to_mat(placed.get("xyz", [0.0] * 3),
                              placed.get("rpy", [0.0] * 3))

    def taught_on_surface(self):
        """The names a plane file holds, or None when points are the answer.

        None is not "none taught". It is "this cell does not read the box off
        a surface", which is what sends a FIND's reference to the point
        library instead, and the two must not be confused: a cell with a map
        and nothing taught on it has to say so rather than quietly look
        somewhere else.

        The branch is the same one `_find` takes at run time, and has to be:
        an editor that offers names a run would refuse is an editor that lets
        somebody finish a program that cannot start. A cell with no map and no
        camera placement can still offer explicitly taught box-home names:
        those use the installation's small planar axis map instead.
        """
        if self.surface is not None and self.surface.ready:
            return set(self.surface.references)
        if self.cell.config.vision.get("calibrated"):
            return None
        current = self.cell.config.vision.get("box_size")
        names = set()
        for name, record in (
                self.cell.config.vision.get("home_references") or {}).items():
            taught = record.get("box_size") if isinstance(record, dict) else None
            if not taught or not current or (
                    len(taught) == len(current)
                    and max(abs(float(a) - float(b))
                            for a, b in zip(taught, current)) <= 0.001):
                names.add(name)
        return names

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

            box home         the cell's known planar axis mapping carries the
                             saved pre-pick TCP with camera X/Y and box yaw.
                             Robot Z stays at its taught height.

        A cell with none of these is refused. It used to run the second one against
        an identity transform, which puts every detection in the camera's own
        frame and produces a correction that is confidently wrong — caught, if
        at all, by `max_correction` noticing the size of it.
        """
        into = (step.get("into") or "part").strip()
        reference = (step.get("reference") or "").strip()
        found_var = into + FOUND_SUFFIX
        self.poses.pop(into, None)
        self.stances.pop(into, None)
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

        said = None      # a branch may say the whole line itself
        if self.surface is not None and self.surface.ready:
            correction, where = self._on_surface(reference, reading.detection)
            note = "  (corners fit %.1f mm)" % (where.fit_error * 1000)
            # A hold taught as a distance from the box needs no correction of
            # its own — carried to where the box is now, it *is* the answer.
            if self.surface.stance(reference):
                self.stances[into] = self.surface.stance_world(reference, where)
                note += ", held by %s" % "+".join(
                    sorted(self.stances[into]))
        elif self.cell.config.vision.get("calibrated"):
            now = reading.detection.matrix_in(self._camera_to_world())
            was = pose_to_mat(self.points.get(reference))
            correction = now @ inv(was)
            note = ""
        elif reference in (self.cell.config.vision.get("home_references") or {}):
            correction, fixed, seen_off = self._at_fixed_home(
                reference, reading.detection, reading.frame)
            self.stances[into] = fixed
            note = ""
            said = ("%s moved from %s: %s" %
                    (into, reference, seen_off))
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
        if said is None:
            shift = mat_to_pose(correction)
            moved = shown_xyz(shift)
            said = ("%s moved %+.1f %+.1f %+.1f mm, turned %+.1f deg%s"
                    % (into, moved[0] * 1000, moved[1] * 1000,
                       moved[2] * 1000,
                       np.degrees(np.linalg.norm(shift[3:])), note))
        self.log("     " + said)

    def _at_fixed_home(self, reference, detection, frame):
        """Carry a taught TCP by the same R/F/H pose shown under the camera."""
        record = (self.cell.config.vision.get("home_references") or {}).get(
            reference) or {}
        try:
            was = pose_to_mat(record["camera_rfh"])
            tools = {str(arm): pose_to_mat(pose)
                     for arm, pose in record["tools"].items()}
        except (KeyError, TypeError, ValueError) as exc:
            if "camera_rfh" not in record:
                raise ProgramError(
                    "fixed home %r uses old lens coordinates — recreate it "
                    "with Camera > Create Home Box" % reference)
            raise ProgramError(
                "fixed home reference %r is incomplete: %s"
                % (reference, exc))
        taught_size = record.get("box_size")
        current_size = self.cell.config.vision.get("box_size")
        if taught_size and current_size:
            off = max(abs(float(a) - float(b))
                      for a, b in zip(taught_size, current_size))
            if off > 0.001:
                raise ProgramError(
                    "fixed home %r was taught with a %s mm box, but the "
                    "camera is set to %s mm"
                    % (reference,
                       "x".join("%.0f" % (float(v) * 1000)
                                for v in taught_size),
                       "x".join("%.0f" % (float(v) * 1000)
                                for v in current_size)))
        # Imported here so programs without FIND do not depend on OpenCV.
        from ..vision.detect import image_relative_rotation, image_relative_xyz

        vision = self.cell.config.vision
        seen_xyz = image_relative_xyz(
            detection, frame, vision.get("surface"), vision.get("floor"))
        seen_rotation = image_relative_rotation(
            detection, vision.get("surface"), vision.get("floor"))
        if seen_xyz is None or seen_rotation is None:
            raise ProgramError(
                "FIND needs camera R/F/H — run Camera > Measure Surface + Box")
        seen = np.eye(4)
        seen[:3, :3] = seen_rotation
        seen[:3, 3] = seen_xyz
        rfh_delta = seen[:3, 3] - was[:3, 3]
        try:
            axis_map = np.asarray(self.cell.config.vision.get(
                "home_rf_map", [[0.0, -1.0], [1.0, 0.0]]),
                dtype=float).reshape(2, 2)
        except (TypeError, ValueError) as exc:
            raise ProgramError("vision.home_rf_map must be a 2x2 matrix: %s"
                               % exc)
        was_axis = axis_map @ was[:2, 0]
        seen_axis = axis_map @ seen[:2, 0]
        if min(np.linalg.norm(was_axis), np.linalg.norm(seen_axis)) < 1e-6:
            raise ProgramError(
                "box long axis cannot be read through vision.home_rf_map")
        was_yaw = np.arctan2(was_axis[1], was_axis[0])
        seen_yaw = np.arctan2(seen_axis[1], seen_axis[0])
        # A rectangle is the same opening after half a turn.  Take the nearer
        # equivalent so a corner-label swap cannot command a 180 degree move.
        yaw = float((seen_yaw - was_yaw + np.pi / 2) % np.pi - np.pi / 2)
        height_error = abs(float(rfh_delta[2]))
        normal_dot = float(np.clip(
            was[:3, 2] @ seen[:3, 2], -1.0, 1.0))
        tilt = float(np.arccos(normal_dot))
        linear_limit = float(
            self.cell.config.vision.get("home_tolerance", 0.010))
        angular_limit = np.radians(float(
            self.cell.config.vision.get("home_tolerance_deg", 3.0)))
        if height_error > linear_limit or tilt > angular_limit:
            raise ProgramError(
                "box changed height %.1f mm / tilted %.1f deg from home %r — "
                "%s"
                % (height_error * 1000, np.degrees(tilt), reference,
                   _off_home_rfh(was, seen, yaw, tilt)))
        if not tools:
            raise ProgramError(
                "fixed home %r has no saved arm pose" % reference)

        world_delta = np.array([*(axis_map @ rfh_delta[:2]), 0.0],
                               dtype=float)
        anchor = record.get("anchor_world")
        if anchor is None:
            # References saved before anchor_world was introduced remain
            # usable.  One TCP is the box-home point; two TCPs use their
            # midpoint, the same rule used by the teaching dialog now.
            anchor = np.mean([tool[:3, 3] for tool in tools.values()], axis=0)
        try:
            anchor = np.asarray(anchor, dtype=float).reshape(3)
        except ValueError as exc:
            raise ProgramError("fixed home %r has an invalid anchor: %s"
                               % (reference, exc))

        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rotation = np.array([
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ])
        correction = np.eye(4)
        correction[:3, :3] = rotation
        # C carries the taught home anchor to anchor + world_delta, so it is
        # valid both for the saved stance and for targets using correct_by.
        correction[:3, 3] = anchor + world_delta - rotation @ anchor
        carried = {arm: correction @ tool for arm, tool in tools.items()}
        robot = shown_xyz(world_delta)
        return correction, carried, (
            "%s → robot X%+.1f Y%+.1f mm"
            % (_off_home_rfh(was, seen, yaw, tilt),
               robot[0] * 1000, robot[1] * 1000))

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
