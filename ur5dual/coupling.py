"""
Object-centric coordination: the core of two-arm work.

Once both grippers hold one rigid object, the two TCPs are no longer free.
Their poses are fixed relative to the object, and therefore to each other —
a closed kinematic chain. Programming either arm's path directly means
computing that constraint by hand every time.

So this module inverts it. You say where the *object* should go:

    T_world_tcpA = T_world_object · grasp_A
    T_world_tcpB = T_world_object · grasp_B

`grasp_A` and `grasp_B` are captured once, at the moment the grippers close,
and stay constant for as long as the object is held. "Rotate the drum 45
degrees about its own axis" is then one line, and both arms' paths fall out
of it.

Two guards run on every cycle of the coordinated loop, because a closed chain
has no slack: if the arms disagree about where the object is, they lever
against each other and the object or the wrists lose.

  force   either TCP pushing past motion.max_tcp_force
  drift   the measured A-to-B transform wandering from the captured one by
          more than motion.max_pair_drift

Drift alone is not a fault. Servo tracking lags, and two controllers lag by
different amounts, so the arms fall out of step with each other whenever they
move — harmlessly, if nothing is between them. It becomes a fault when the
wrists are also loaded, because then the thing between them is taking the
difference. The guard needs both before it stops anything.
"""

import math
import threading
import time

import numpy as np

from .robot.backends import make_backend
from .geometry.closed_chain import ClosedChainError, ClosedChainSolver
from .geometry.kinematics import (
    interp_pose, inv, mat_to_pose, pose_distance, pose_to_mat,
    rotate_about_own_axis, rotate_about_world_axis, rotvec_to_mat,
)


class CouplingError(RuntimeError):
    pass


# a run of cycles over the drift limit before it counts as a fault: at 125 Hz
# this is a fifth of a second of sustained disagreement, long past anything
# servo lag explains
DRIFT_STRIKES = 25
# above this the wrists are actually loaded, which separates two arms levering
# on a held object from two arms merely tracking badly in free air
FIGHT_FORCE = 40.0
# how often the harmless kind of drift is worth mentioning
DRIFT_WARN_PERIOD = 5.0
# multiple of max_pair_drift past which slack wrists stop being an excuse.
# The forgiving branch below exists for tracking error, which is small and
# bounded; a wrong base transform is neither, and with empty grippers there is
# no force rise to catch it before the two arms meet.
DRIFT_HARD_MULTIPLE = 5.0
# how fast a held jog may change the object's speed, m/s^2. A jog starts and
# stops on a button, and a step change in the target's velocity is a step
# change in every joint's — which the closed chain has to absorb. At 1 m/s^2
# the fastest preset is reached in 50 ms and stopped inside 1.3 mm.
JOG_ACCEL = 1.0
# the same for a rotation jog, rad/s^2. Lower than it looks: a box held 400 mm
# across turns its grippers through 200 mm of arc per radian, so 2 rad/s^2 at
# the box is 0.4 m/s^2 at each wrist — already a third of what the linear jog
# allows, and the wrists are the joints with the least to give.
JOG_ANG_ACCEL = 2.0
# a jog not refreshed this often has lost its button. A GUI thread that stalls,
# or a release event that never arrives, must not leave the arms driving.
JOG_WATCHDOG = 0.5

# An explicit uncalibrated REAL override is for commissioning only.  These are
# deliberately below the smallest normal GUI presets, so a wrong transform is
# visible before it has time to build significant disagreement between arms.
UNCALIBRATED_REAL_LIN_SPEED = 0.001          # m/s
UNCALIBRATED_REAL_ANG_SPEED = math.radians(0.25)  # rad/s
UNCALIBRATED_REAL_DRIFT = 0.001              # m, stop without a force vote
UNCALIBRATED_REAL_DRIFT_STRIKES = 5          # 40 ms at 125 Hz

# Turning the box needs the *distance* between the two bases, which only a
# touch-off measures — hand-taught directions give their relative orientation
# and nothing more. Refusing rotation outright until then was the wrong shape
# for the risk, though: the error a wrong base distance produces grows with
# the angle turned, so a degree is not a hundredth as dangerous as a hundred
# degrees, it is a hundredth as wrong. And the guard that actually catches it
# runs every cycle already — on a 400 mm grip, max_pair_drift of 4 mm is about
# 1.1 degrees of disagreement, so the arms are stopped long before anything is
# strained.
#
# So rotation is allowed before a touch-off, at an angle and a speed small
# enough that the drift guard is certain to win the race. Enough to prove the
# axes point the right way; not enough to carry a job.
UNCALIBRATED_ROTATION = math.radians(5.0)     # rad, per commanded move
UNCALIBRATED_ANG_SPEED = math.radians(2.0)    # rad/s, held jog ceiling


def limit_uncalibrated_rotation(angle, ang_speed, calibrated):
    """How far and how fast the box may be turned given what has been measured.

    Split out of the move itself so it can be checked without two robots and a
    box: this is the one piece of the rotation path that decides whether a
    command is safe, and it would otherwise only ever run on hardware.

    Returns (allowed angular speed, refusal or None).
    """
    if calibrated:
        return ang_speed, None
    if abs(angle) > UNCALIBRATED_ROTATION + 1e-9:
        # a hair of slack on the comparison: the jog panel's largest step
        # preset is exactly this angle, and a button that refuses itself on
        # the last bit of floating point is indistinguishable from a broken one
        return ang_speed, (
            "%.0f degrees is more than can be turned safely before a touch-off: "
            "the distance between the two bases has never been measured, and "
            "the error that produces grows with the angle. %.0f degrees at a "
            "time is allowed for checking the axes; run the touch-off on the "
            "Cell tab to lift the limit"
            % (math.degrees(abs(angle)), math.degrees(UNCALIBRATED_ROTATION)))
    return min(ang_speed, UNCALIBRATED_ANG_SPEED), None


def _smoothstep(s):
    """Ease in and out, so a coordinated move neither starts nor stops with
    a jerk the closed chain would have to absorb."""
    return s * s * (3.0 - 2.0 * s)


class _Jog:
    """One held jog button, as the feed thread sees it.

    Direction and speed belong to whoever is pressing; `v` and the clock
    belong to the feed. Splitting them that way is what lets a finger slide
    from +X to +Y without the object stopping in between — the direction is
    replaced, the speed ramp carries on.

    `kind` says whether the unit vector is a direction to travel along or an
    axis to turn about. Both are world-frame and both ramp the same way, so a
    rotation jog is the translation jog with one line changed — which is the
    point, because an operator sliding from +Z to +RZ should not be able to
    tell that anything different is happening underneath.
    """

    __slots__ = ("unit", "speed", "v", "t", "refreshed", "kind",
                 "origin", "witness")

    def __init__(self, unit, speed, now, kind="lin"):
        self.unit = unit          # unit vector in the world frame
        self.speed = float(speed)  # m/s or rad/s; 0 means ramp down and end
        self.v = 0.0               # what is actually being applied
        self.t = now               # when the feed last advanced this jog
        self.refreshed = now       # when the button last said it was still held
        self.kind = kind          # "lin" to travel, "ang" to turn
        # Where this jog began, commanded and measured, so that letting go can
        # say how far the arms went. Under the uncalibrated cap a held button
        # moves the box a millimetre a second, which is real motion and looks
        # exactly like none at all — and "it does not move" is not a thing that
        # should have to be settled by eye against a 30 mm/s single-arm jog.
        self.origin = None        # commanded object pose at the press
        self.witness = None       # {arm: measured TCP position} at the press

    @property
    def accel(self):
        return JOG_ACCEL if self.kind == "lin" else JOG_ANG_ACCEL


class HeldObject:
    """The workpiece, and how each arm is holding it."""

    def __init__(self, name="object"):
        self.name = name
        self.grasps = {}          # arm id -> 4x4, object frame -> that TCP
        self.pose_world = None    # 6-vector, where the object is now
        self.captured_relative = None   # A's TCP -> B's TCP at capture time
        self.frame_origin = None  # "midpoint", "A" or "B" — how it was planted

    # -- the virtual box, as anything outside the coordinator sees it -------
    # The object has no CAD dimensions and no mass; it is this frame plus the
    # captured grasps. `pose_world` is what the feed thread writes to at servo
    # rate, so a caller that holds on to it is holding a live array — these
    # hand out copies instead, because a readout that can be mutated into the
    # object's actual position is a way to move two arms by accident.
    @property
    def virtual_pose_world(self):
        return None if self.pose_world is None \
            else np.array(self.pose_world, dtype=float)

    def virtual_matrix_world(self):
        return None if self.pose_world is None else pose_to_mat(self.pose_world)

    @property
    def held(self):
        return len(self.grasps) >= 1 and self.pose_world is not None

    @property
    def arm_ids(self):
        return sorted(self.grasps)

    # -- defining the object frame ----------------------------------------
    def capture(self, cell, arm_ids=("A", "B"), origin="midpoint"):
        """Freeze the current grip as the object definition.

        `origin` picks where the object frame is planted:
          midpoint  halfway between the two TCPs, oriented like arm A's TCP.
                    The natural choice for a bottle or a drum held on both
                    sides — the frame sits in the object, not in a gripper.
          A / B     on that arm's TCP. Makes the other arm a follower, which
                    is the leader-follower style expressed in these terms.
        """
        tcp = {}
        for a in arm_ids:
            if not cell.arms[a].connected:
                raise CouplingError("arm %s is not connected" % a)
            tcp[a] = cell.arms[a].tcp_matrix_world()

        if origin in tcp:
            T_w_o = tcp[origin].copy()
        elif origin == "midpoint":
            if len(tcp) < 2:
                raise CouplingError("midpoint origin needs both arms")
            T_w_o = tcp[arm_ids[0]].copy()
            T_w_o[:3, 3] = (tcp[arm_ids[0]][:3, 3] + tcp[arm_ids[1]][:3, 3]) / 2.0
        else:
            raise CouplingError("unknown object origin %r" % origin)

        self.grasps = {a: inv(T_w_o) @ tcp[a] for a in arm_ids}
        self.pose_world = mat_to_pose(T_w_o)
        self.frame_origin = origin
        if len(tcp) == 2:
            self.captured_relative = inv(tcp[arm_ids[0]]) @ tcp[arm_ids[1]]
        return self

    def release(self):
        self.grasps = {}
        self.pose_world = None
        self.captured_relative = None
        self.frame_origin = None

    # -- where the arms must be for a given object pose --------------------
    def tcp_world(self, arm_id, object_pose_world=None):
        T_w_o = pose_to_mat(object_pose_world
                            if object_pose_world is not None else self.pose_world)
        return T_w_o @ self.grasps[arm_id]

    def targets(self, cell, object_pose_world):
        """{arm id: pose in that arm's own base frame} — ready for the wire."""
        out = {}
        for a in self.grasps:
            T_w_tcp = self.tcp_world(a, object_pose_world)
            out[a] = cell.arms[a].world_to_base(mat_to_pose(T_w_tcp))
        return out

    def span(self):
        """Distance between the two grasp points, in metres. Constant while
        the object is held — a change means something slipped."""
        if len(self.grasps) < 2:
            return None
        a, b = self.arm_ids
        return float(np.linalg.norm(self.grasps[a][:3, 3] - self.grasps[b][:3, 3]))

    def to_dict(self):
        return {
            "name": self.name,
            "frame_origin": self.frame_origin,
            "pose_world": None if self.pose_world is None
            else [float(v) for v in self.pose_world],
            "grasps": {a: [float(v) for v in mat_to_pose(T)]
                       for a, T in self.grasps.items()},
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(d.get("name", "object"))
        # None rather than a guess: a box loaded from a file was planted by
        # some earlier session, and claiming it was a midpoint when it may have
        # been arm B would put the frame in the wrong place on the readout
        obj.frame_origin = d.get("frame_origin")
        obj.pose_world = (np.array(d["pose_world"], dtype=float)
                          if d.get("pose_world") else None)
        obj.grasps = {a: pose_to_mat(p) for a, p in (d.get("grasps") or {}).items()}
        return obj


class Coordinator:
    """Drives the held object, both arms from one clock.

    The servo feed is a thread that never stops while the object is held. It
    sends a target every cycle whether or not anything is being commanded —
    when idle, the target is simply where the object already is.

    Two reasons it has to work that way. A caller that runs the loop inline
    blocks whatever thread it is on, and if that is the GUI thread every
    readout freezes for the duration of each move. And a feed that goes quiet
    between moves lets the robot-side script time out, decelerate, and start
    again on the next command — repeatedly, until the stream gives up. A
    continuously fed servo stays hot and the arms hold station between moves
    instead of being dropped and re-caught.

    Anything the loop cannot survive — a guard tripping, a socket breaking —
    stops *both* arms before it reports. One arm still tracking while the
    other has stopped is exactly how two arms tear an object apart.

    Two ways to drive it, and both end up in the same place: `command_move`
    plans a path between two poses, `command_jog` holds a velocity along a
    direction for as long as a button is down. Either way the feed thread
    produces one object pose per cycle and both arms are derived from that
    single pose, so they cannot disagree about where the object is going.

    In `simulate` mode nothing is sent and the whole path is walked
    synchronously, so the geometry can be tested without a robot.
    """

    def __init__(self, cell, simulate=False, drive_robots=True):
        self.cell = cell
        self.simulate = simulate
        # `sim` mode: everything runs — the clock, the solver, the guards, the
        # jog ramp — and only the last step, the write to the controller, is
        # replaced. The joints it would have sent go to the viewer instead, so
        # the model moves and the arms do not.
        self.drive_robots = drive_robots
        self.sim_view = None
        self.backends = {}
        self.trace = []               # simulate mode: (t, {arm: pose})
        self.on_progress = None       # callable(fraction)
        self.object = None
        self.error = None             # what stopped the feed, if anything
        # set when motion.control is `joint`: the closed-chain solver, and the
        # previous cycle's answer, which is what keeps both arms on the branch
        # they were on when the grippers closed
        self.solver = None
        self._joint_seed = None
        self._solve_cache = None      # (pose, result) shared within one cycle

        self._lock = threading.Lock()
        self._dt = 1.0 / float(cell.config.motion["servo_rate_hz"])
        self._plan = None             # (pose_at, duration, t0)
        self._jog = None              # _Jog while a jog button is held
        self._hold = None             # pose held when nothing is running
        self._thread = None
        self._running = False
        self._abort = False
        self.drift = 0.0
        self.force_baseline = None
        self.uncalibrated_real = False
        self._drift_strikes = 0
        self._last_drift_warning = 0.0

    # -- lifecycle ---------------------------------------------------------
    def start(self, obj):
        """Bring up a backend per arm and start feeding the current pose."""
        self.object = obj
        self.uncalibrated_real = False
        self._hold = np.array(obj.pose_world, dtype=float)
        self._plan = None
        self._jog = None
        self.error = None
        self._abort = False
        self._drift_strikes = 0
        # what each arm reads while holding the object but not yet driving it;
        # everything the force guard cares about is measured from here
        self.force_baseline = {a: self.cell.force_vector(a)
                               for a in obj.arm_ids
                               if self.cell.arms[a].connected}
        if self.simulate:
            return self

        # A stopped arm cannot run the program the backend uploads, and the
        # failure it produces — "never connected back" — points at firewalls
        # and firmware rather than at the arm that is simply halted. Check
        # first and say the real thing.
        # Coordinated motion is the one thing that cannot survive a guessed
        # geometry. Each arm is sent its own share of a world-frame path, so a
        # base transform that does not match the bracket sends that arm
        # somewhere else entirely — the arms visibly disagree about which way
        # is up. Single-arm jogging never touches these transforms and stays
        # available; this does not.
        #
        # None of which applies when nothing is being sent. In sim mode an
        # unmeasured cell is not a hazard, it is the thing on screen: the two
        # arms drawn pulling in different directions is exactly what a wrong
        # base transform looks like, and seeing it before the touch-off is
        # more use than being told about it.
        uncalibrated = (self.drive_robots and len(obj.arm_ids) > 1
                        and not self.cell.config.translation_calibrated)
        allow_uncalibrated = bool(
            self.cell.config.motion.get("allow_uncalibrated_real", False))
        if uncalibrated and not allow_uncalibrated:
            # Naming what to do rather than what is missing. The measurement
            # is `ur5dual.geometry.calibration`, driven by tests/check_up_online.py and
            # the touch-off panel; short of running one, the geometry can be
            # entered by hand, and `allow_uncalibrated_real` opens the capped
            # commissioning mode for finding out which of those is needed.
            raise CouplingError(
                "the two bases' relative orientation has never been measured, "
                "so a world-frame move would send the arms in different "
                "directions. Run a touch-off (ur5dual.geometry.calibration), or set "
                "arms.<id>.base in config/cell.yaml from the measured mount "
                "and set translation_calibrated; motion.allow_uncalibrated_real "
                "permits held-button jogging at 1 mm/s in the meantime")
        if uncalibrated:
            self.uncalibrated_real = True
            self.cell.log(
                "WARNING: uncalibrated REAL commissioning override is active "
                "— held jogging only, capped at %.1f mm/s and %.2f deg/s; "
                "stops at %.1f mm pair drift"
                % (UNCALIBRATED_REAL_LIN_SPEED * 1000.0,
                   math.degrees(UNCALIBRATED_REAL_ANG_SPEED),
                   UNCALIBRATED_REAL_DRIFT * 1000.0))

        if self.drive_robots and len(obj.arm_ids) > 1:
            wrong = self.cell.geometry_complaint(obj.arm_ids)
            if wrong:
                raise CouplingError(wrong)

        if self.drive_robots:
            blocked = self.cell.not_ready_reason(obj.arm_ids)
            if blocked:
                raise CouplingError("cannot start the servo loop: %s" % blocked)

        control = str(self.cell.config.motion.get("control", "pose")).lower()
        # Sim mode always solves, whatever the config says. The joints are the
        # product: without them there is nothing to draw, and a viewer fed
        # Cartesian poses would have to run its own inverse kinematics and
        # could then disagree with the cell about which elbow it is showing.
        if not self.drive_robots:
            control = "joint"
        if control == "joint" and not self._start_solver(obj):
            # Joint control asked for and not earned. Falling back rather than
            # refusing, because the error that disqualifies it is precisely the
            # error Cartesian targets do not carry: a pose derived from what
            # the controller itself reported, sent back to that controller to
            # solve, never touches our DH table. Refusing instead left an
            # operator with an attached object, both arms live, and a jog grid
            # that had gone grey with no way forward — which is worse than
            # running the way this cell ran before.
            control = "pose"

        for a in obj.arm_ids:
            # A simulated arm is handed to its own backend as the thing to
            # write to, so the loop closes through it just as it closes
            # through a controller.
            sink = self.cell.arms[a] if (not self.drive_robots
                                         and hasattr(self.cell.arms[a],
                                                     "set_joints")) else None
            backend = make_backend(self.cell.arms[a].cfg.ip,
                                   self.cell.config.motion, control,
                                   self.drive_robots, sink)
            backend.start()
            self.backends[a] = backend
            self.cell.log("arm %s servo backend up (%s, %s targets)"
                          % (a, backend.name, control))

        self._running = True
        self._thread = threading.Thread(target=self._feed, daemon=True,
                                        name="ur5dual-servo")
        self._thread.start()
        return self

    def _start_solver(self, obj):
        """Bring up the closed-chain solver, but only if it can prove itself.

        Solving here means every joint target these arms receive comes out of
        a DH table this side holds rather than the one inside the controller.
        If those two disagree the arms go somewhere other than where the box
        is, and with both grippers closed there is nothing to discover that
        gently. So the first thing the solver does is compute the TCP for the
        joints each arm is *already* sitting at and compare it against the pose
        that arm reports — an exact check, costing one read, that a wrong robot
        generation, an unread tool offset or an unread factory calibration
        cannot pass.

        Returns True if the solver is running. False means it disqualified
        itself and the caller should send Cartesian targets instead; only a
        chain that cannot be built at all raises.
        """
        try:
            solver = ClosedChainSolver(self.cell, obj)
        except ClosedChainError as e:
            raise CouplingError(str(e))

        agreement = solver.verify_against_robots()
        wrong = {a: v for a, v in agreement.items() if not v[2]}
        if wrong and not self.drive_robots:
            # Worth saying, not worth stopping for: a picture drawn from a
            # wrong DH table is a wrong picture, and a wrong picture is how
            # you find out the table is wrong.
            a, (d_p, d_r, _) = sorted(wrong.items())[0]
            self.cell.log(
                "sim: our forward kinematics places arm %s's tool %.0f mm from "
                "where the controller says it is, so the drawing will be out "
                "by about that much — run tests/check_chain_online.py"
                % (a, d_p * 1000.0))
        elif wrong:
            a, (d_p, d_r, _) = sorted(wrong.items())[0]
            self.cell.log(
                "joint control is configured, but our forward kinematics does "
                "not match arm %s — we place its tool %.1f mm and %.1f deg "
                "away from where the controller says it is, which is the size "
                "of that robot's own factory calibration. Falling back to "
                "Cartesian targets, which each controller solves with its own "
                "calibration and which therefore cannot carry that error. "
                "Run tests/check_chain_online.py to see both arms' figures"
                % (a, d_p * 1000.0, np.degrees(d_r)))
            return False

        self.solver = solver
        self._joint_seed = solver.joints_now()
        for arm_id, (margin, sigma) in sorted(solver.branch_report().items()):
            self.cell.log(
                "arm %s locked onto its current IK branch: %.0f deg of joint "
                "travel left, manipulability %.4f%s"
                % (arm_id, np.degrees(margin), sigma,
                   "  — near a singularity, expect poor tracking"
                   if sigma < 0.01 else ""))
        return True

    @property
    def alive(self):
        return self.simulate or (self._running and self._thread is not None
                                 and self._thread.is_alive())

    def shutdown(self):
        self._running = False
        # the feed thread goes first. Clearing the solver while it is still
        # running would drop that cycle into the Cartesian branch and hand a
        # pose to a backend whose robot is expecting joint angles.
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.solver = None
        self._joint_seed = None
        self._solve_cache = None
        for a, backend in self.backends.items():
            try:
                backend.shutdown()
            except Exception as e:
                self.cell.log("arm %s backend shutdown: %s" % (a, e))
        self.backends = {}

    def abort(self):
        self._abort = True
        with self._lock:
            self._jog = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()

    # -- the feed ----------------------------------------------------------
    def _stop_all(self):
        for backend in self.backends.values():
            try:
                backend.stop_motion()
            except Exception:
                pass
        try:
            self.cell.halt()
        except Exception:
            pass

    def _feed(self):
        dt = self._dt
        next_tick = time.monotonic()
        while self._running:
            try:
                now = time.monotonic()
                with self._lock:
                    plan = self._plan
                    jog = self._jog
                    hold = self._hold

                if plan is not None:
                    s = min(1.0, (now - plan[2]) / plan[1])
                    pose = plan[0](s)
                    if s >= 1.0:
                        with self._lock:
                            if self._plan is plan:
                                self._plan = None
                                self._hold = pose
                    if self.on_progress is not None:
                        self.on_progress(s)
                elif jog is not None:
                    pose = self._advance_jog(jog, hold, now)
                else:
                    pose = hold

                # every readout of "where is the object" comes from here, so it
                # tracks the commanded pose rather than jumping at the end of
                # each move — and the next move starts from where this one is
                self.object.pose_world = pose

                self._check(self.object, self.object.arm_ids)
                self._send(pose)

            except Exception as e:
                self.error = e
                with self._lock:
                    self._plan = None
                    self._jog = None
                self._stop_all()
                self.cell.log("servo loop stopped: %s: %s"
                              % (type(e).__name__, e))
                self._running = False
                return

            next_tick += dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()      # fell behind; do not spiral

    def _send(self, pose):
        """One box pose out to both arms, in whichever currency they speak.

        The two branches differ in where the inverse kinematics happens, and
        in nothing else. Cartesian targets leave it to each controller, which
        is fine while the box is light and the arms are far from their limits.
        Joint targets do it here, from the previous cycle's answer, so neither
        arm can quietly change its mind about which elbow it is using.
        """
        if self.solver is None:
            targets = self.object.targets(self.cell, pose)
            for a in self.object.arm_ids:
                self.backends[a].servo_pose(targets[a])
            return

        joints, info = self._solve_cached(pose)
        bad = [(a, m) for a, m in sorted(info.items()) if not m["converged"]]
        if bad:
            a, m = bad[0]
            raise CouplingError(
                "arm %s cannot hold its side of the box there — the nearest it "
                "gets is %.0f mm and %.1f deg away"
                % (a, m["pos_error"] * 1000.0, np.degrees(m["rot_error"])))
        self._joint_seed = joints
        for a in self.object.arm_ids:
            self.backends[a].servo_joints(joints[a])
        if not self.drive_robots:
            # at servo rate rather than at the panel's redraw rate, so a
            # coordinated move is drawn smoothly rather than in steps
            self.cell.publish_sim_view()

    def _solve_cached(self, pose):
        """One solve per pose, however many times the cycle asks about it.

        A jog has to know whether the pose it just produced is reachable
        *before* it commits to it, and the feed then has to solve that same
        pose to send it. Those are the same question, and at 125 Hz asking it
        twice is a millisecond of the eight this loop has.
        """
        pose = np.asarray(pose, dtype=float)
        if self._solve_cache is not None and np.array_equal(self._solve_cache[0],
                                                            pose):
            return self._solve_cache[1]
        result = self.solver.solve(pose, self._joint_seed)
        self._solve_cache = (pose.copy(), result)
        return result

    def _advance_jog(self, jog, pose, now):
        """One cycle of a held jog: ramp the speed, step the pose along it."""
        elapsed = now - jog.t
        jog.t = now
        # A scheduling hiccup must not become a leap. However long this thread
        # was actually away, the target advances by at most a few cycles'
        # worth of travel — the arms have to physically follow it.
        elapsed = max(0.0, min(elapsed, 4.0 * self._dt))

        if now - jog.refreshed > JOG_WATCHDOG:
            jog.speed = 0.0                  # nobody is holding it any more

        step_v = jog.accel * elapsed
        jog.v = (min(jog.speed, jog.v + step_v) if jog.v < jog.speed
                 else max(jog.speed, jog.v - step_v))
        if jog.v <= 0.0:
            if jog.speed <= 0.0:             # ramped down and finished
                finished = False
                with self._lock:
                    if self._jog is jog:
                        self._jog = None
                        finished = True
                if finished:                 # reads sockets; not under the lock
                    self._report_jog(jog, pose)
            return pose

        moved = np.array(pose, dtype=float)
        if jog.kind == "ang":
            # pre-multiply: the axis is the cell's, not the box's, so +RZ is
            # the vertical of the room however the box has been turned. The
            # origin stays put, and for a midpoint grasp that origin is
            # between the grippers — the one pivot that costs both arms the
            # same travel.
            T = pose_to_mat(moved)
            T[:3, :3] = rotvec_to_mat(jog.unit * (jog.v * elapsed)) @ T[:3, :3]
            moved = mat_to_pose(T)
        else:
            moved[:3] += jog.unit * (jog.v * elapsed)

        if not self._jog_reachable(moved):
            # Running out of arm is not a fault — the operator simply cannot
            # go further that way. Ramping the jog down leaves both arms
            # holding station on the last pose that worked, which is a button
            # that stops responding rather than a loop that dies mid-carry.
            jog.speed = 0.0
            return pose

        with self._lock:
            if self._jog is jog and self._plan is None:
                self._hold = moved
        return moved

    def _measure_arms(self):
        """{arm: TCP position in the cell frame} as the robots report it now."""
        out = {}
        for a in (self.object.arm_ids if self.object is not None else ()):
            try:
                out[a] = np.array(self.cell.arms[a].tcp_matrix_world()[:3, 3],
                                  dtype=float)
            except (OSError, ConnectionError, RuntimeError):
                pass
        return out

    def _report_jog(self, jog, pose):
        """What letting go of the button actually achieved, in millimetres.

        Commanded and measured, side by side, because they answer different
        questions and only the pair is worth anything. Commanded says the loop
        did its arithmetic; measured says the arms did something about it. A
        held jog at the uncalibrated cap moves the box a millimetre a second,
        and there is no way to tell that from a dead servo loop by watching.
        """
        if jog.origin is None:
            return
        pose = np.asarray(pose, dtype=float)
        if jog.kind == "ang":
            _, turned = pose_distance(jog.origin, pose)
            commanded = "%.2f deg" % np.degrees(turned)
        else:
            commanded = "%.1f mm" % (
                float(np.linalg.norm(pose[:3] - jog.origin[:3])) * 1000.0)

        moved = []
        for arm_id, was in sorted((jog.witness or {}).items()):
            now = self._measure_arms().get(arm_id)
            if now is not None:
                moved.append("%s %.1f mm"
                             % (arm_id, float(np.linalg.norm(now - was)) * 1000.0))
        self.cell.log(
            "held jog ended: commanded %s%s" % (
                commanded,
                "" if not moved else "; the arms moved " + ", ".join(moved)))

    def _jog_reachable(self, pose):
        """Can both arms actually hold the box there? Only knowable with the
        solver running; without it, the controllers find out for themselves.

        The answer is cached, so the send that follows this cycle costs
        nothing more.
        """
        if self.solver is None or self._joint_seed is None:
            return True
        _, info = self._solve_cached(pose)
        return all(m["converged"] for m in info.values())

    # -- issuing motion ----------------------------------------------------
    def _set_plan(self, pose_at, duration):
        if not self.alive:
            raise CouplingError(
                "the servo loop is not running%s"
                % ("" if self.error is None else " — %s" % self.error))
        self._precheck(pose_at, duration)
        with self._lock:
            self._jog = None          # a planned move outranks a held button
            self._plan = (pose_at, duration, time.monotonic())

    def _precheck(self, pose_at, duration):
        """Walk the whole path in arithmetic before committing the arms to it.

        Only possible with the solver running, and it is most of the reason to
        run one. Without it, a rotation that puts arm B's wrist past its stop
        two thirds of the way through is found out two thirds of the way
        through — box in the air, one arm in a protective stop, the other
        still holding on. A few milliseconds of arithmetic here turns that
        into a refusal with a joint number in it.
        """
        if self.solver is None or self._joint_seed is None:
            return
        plan = self.solver.plan(pose_at, self._joint_seed, duration=duration)
        if plan.ok:
            for arm_id in plan.near_singular:
                self.cell.log(
                    "arm %s passes close to a singularity on this move "
                    "(sigma %.4f) — it will track loosely there"
                    % (arm_id, plan.min_sigma[arm_id]))
            return
        raise CouplingError("this move was not started: %s" % plan.complaint())

    @staticmethod
    def _plan_pose(plan, now):
        pose_at, duration, t0 = plan
        return np.asarray(pose_at(min(1.0, (now - t0) / duration)), dtype=float)

    def current_pose(self, obj=None):
        """Where the object is being commanded *right now*.

        Not the same as `obj.pose_world` half of the time, and the difference
        is the whole point: while a move is in flight the feed is somewhere
        along the path, and the next command has to start from there. Starting
        from the pose the last move began at throws away everything it did —
        press +Z twice quickly and the second press cancels the first.
        """
        with self._lock:
            plan, hold = self._plan, self._hold
        if obj is not None and obj is not self.object:
            return np.asarray(obj.pose_world, dtype=float)   # not what we drive
        if plan is not None:
            return self._plan_pose(plan, time.monotonic())
        if hold is not None:
            return np.array(hold, dtype=float)
        obj = obj if obj is not None else self.object
        return None if obj is None else np.asarray(obj.pose_world, dtype=float)

    def busy(self):
        """True while a planned move is in flight. A held jog is not "busy" —
        it ends when the operator lets go, not on its own, so waiting on one
        would wait forever."""
        with self._lock:
            return self._plan is not None

    def wait_done(self, timeout=120.0):
        deadline = time.monotonic() + timeout
        while self.busy():
            if self.error is not None:
                raise CouplingError(str(self.error))
            if self._abort:
                raise CouplingError("aborted")
            if time.monotonic() > deadline:
                raise CouplingError("coordinated move did not finish in time")
            time.sleep(0.005)
        if self.error is not None:
            raise CouplingError(str(self.error))

    # -- guards ------------------------------------------------------------
    def _check(self, obj, arm_ids):
        if self._abort:
            raise CouplingError("aborted")
        if self.simulate:
            return
        if not self.drive_robots:
            # Every guard below asks a question about robots that are being
            # driven — is either one loaded, are the two levering against each
            # other, has one protective-stopped. In sim the arms are standing
            # still with a workpiece in their grippers, so all three answer
            # about the grip rather than about the motion, and a drift guard
            # measuring the difference between where the arms are and where the
            # simulation has flown off to would trip within a second of the
            # first jog.
            return
        reason = self.cell.not_ready_reason(arm_ids)
        if reason:
            # A protective stop on an arm whose joint is already against its
            # stop is the joint, not the motion. Saying so turns a mystery
            # into a one-move fix.
            if "PROTECTIVE STOP" in reason:
                for arm_id, joint, margin in self.cell.wound_up_joints(arm_ids):
                    reason += ("  —  J%d on arm %s is %.0f deg from its limit; "
                               "unwind it a full turn and the tool still points "
                               "the same way"
                               % (joint + 1, arm_id, np.degrees(margin)))
            raise CouplingError(reason)
        # An arm that is being fed and is not listening. Parking a target
        # always succeeds, so without this the loop keeps solving, keeps
        # passing its guards and keeps reporting itself healthy while nothing
        # on the bench moves — which is exactly how this looked from the panel.
        deaf = [a for a in sorted(self.backends) if self.backends[a].stalled]
        if deaf:
            raise CouplingError(
                "arm %s stopped taking targets — the servo program is no "
                "longer running on that controller. Check the pendant for a "
                "protective stop or a program someone else loaded, then take "
                "hold again" % ", ".join(deaf))
        hot = self.cell.force_exceeded(arm_ids, self.force_baseline)
        if hot:
            raise CouplingError(
                "wrist load rose sharply: "
                + ", ".join("arm %s %+.0f %s" % (a, v, note)
                            for a, v, note in hot))
        if obj.captured_relative is not None and len(arm_ids) == 2:
            # None means one of the feeds is stale; a stale comparison is
            # worse than no comparison, so this cycle simply does not vote
            now = self.cell.relative_transform()
            if now is None:
                return
            drift = float(np.linalg.norm(
                now[:3, 3] - obj.captured_relative[:3, 3]))
            self.drift = drift
            drift_limit = (min(float(self.cell.config.motion["max_pair_drift"]),
                               UNCALIBRATED_REAL_DRIFT)
                           if self.uncalibrated_real else
                           float(self.cell.config.motion["max_pair_drift"]))
            if drift <= drift_limit:
                self._drift_strikes = 0
                return

            # Servo tracking lags the target, and two controllers lag by
            # different amounts, so a single cycle over the limit is normal
            # during a move. Arms genuinely levering against each other stay
            # over it — and push. One cycle is noise; a run of them is not.
            self._drift_strikes += 1
            strikes = (UNCALIBRATED_REAL_DRIFT_STRIKES
                       if self.uncalibrated_real else DRIFT_STRIKES)
            if self._drift_strikes < strikes:
                return

            if self.uncalibrated_real:
                self._drift_strikes = 0
                raise CouplingError(
                    "uncalibrated REAL commissioning stopped: A/B changed "
                    "their captured relationship by %.1f mm (limit %.1f mm) "
                    "— the configured base transforms do not agree"
                    % (drift * 1000.0, drift_limit * 1000.0))

            forces = {a: self.cell.arms[a].tcp_force_magnitude()
                      for a in arm_ids}
            summary = ", ".join("%s %.0f N" % (a, f)
                                for a, f in sorted(forces.items()))
            self._drift_strikes = 0

            hard = DRIFT_HARD_MULTIPLE * drift_limit
            if drift > hard:
                # Past here it stops mattering whether the wrists are loaded.
                # Servo lag does not reach this far, so the two arms are
                # carrying out different motions — which means the geometry
                # relating them is wrong, and they are closing on each other.
                # Empty grippers are not a reason to allow that; they only
                # mean the collision has not happened yet.
                raise CouplingError(
                    "the arms are %.1f mm out of step (%s), far past anything "
                    "servo lag explains — they are being sent in different "
                    "directions, so the transform between the two bases is "
                    "wrong. Re-run the touch-off calibration on the Cell tab"
                    % (drift * 1000, summary))

            if max(forces.values()) <= FIGHT_FORCE:
                # Drift with slack wrists is not the failure this guard exists
                # to catch. Nothing is being levered: either the arms are
                # holding nothing, or whatever they held has already let go.
                # Servo tracking simply is not perfect, and killing the loop
                # for it strands the operator — as it did. Say it, keep going,
                # and do not repeat it more than once every few seconds.
                now = time.monotonic()
                if now - self._last_drift_warning > DRIFT_WARN_PERIOD:
                    self._last_drift_warning = now
                    self.cell.log(
                        "tracking error: the arms are %.1f mm out of step with "
                        "each other, but the wrists are slack (%s) so nothing "
                        "is being strained — continuing"
                        % (drift * 1000, summary))
                return

            raise CouplingError(
                "the arms are %.1f mm out of step and the wrists are loaded "
                "(%s) — they are levering against each other"
                % (drift * 1000, summary))

    # -- motion ------------------------------------------------------------
    # command_* hand the path to the feed thread and return at once, which is
    # what a GUI wants. move_/rotate_ wait for it, which is what a program
    # step wants. Same path either way — only who blocks differs.
    def command_move(self, obj, target_pose_world, lin_speed=None,
                     ang_speed=None):
        if self.uncalibrated_real:
            raise CouplingError(
                "uncalibrated REAL override permits held-button jogging only; "
                "planned and program moves need measured geometry")
        limits = self.cell.config.limits
        lin_speed = float(lin_speed or limits["object_lin_speed"])
        ang_speed = float(ang_speed or limits["object_ang_speed"])

        start = self.current_pose(obj)
        target = np.asarray(target_pose_world, dtype=float)
        d_t, d_r = pose_distance(start, target)
        duration = max(d_t / lin_speed if lin_speed > 0 else 0.0,
                       d_r / ang_speed if ang_speed > 0 else 0.0)
        if duration < 1e-3:
            return 0.0

        if self.simulate:
            self._run_sync(obj, lambda s: interp_pose(start, target,
                                                      _smoothstep(s)), duration)
            self._settle(obj, target)
            return duration

        self._set_plan(lambda s: interp_pose(start, target, _smoothstep(s)),
                       duration)
        return duration

    def command_rotate(self, obj, axis, angle, ang_speed=None,
                       frame="object", pivot=None):
        """Turn the box about one axis.

        `frame` picks whose axis it is. "object" spins it about its own — what
        "turn the drum 45 degrees" means, and what this has always done.
        "world" turns it about the cell's, so RZ is the vertical of the room
        however the box has been tilted since it was picked up, which is what
        an operator means and what a pendant's base-frame buttons do.

        `pivot` is the point it turns around, in world coordinates. None keeps
        the box frame's own origin fixed; for a grasp captured at the midpoint
        that origin sits between the two grippers, which is the one point that
        costs both arms the same travel and neither of them the whole arc.
        """
        if self.uncalibrated_real:
            raise CouplingError(
                "uncalibrated REAL override permits held-button jogging only; "
                "planned and program rotations need calibration")
        if frame not in ("object", "world"):
            raise CouplingError("rotation frame must be object or world, not %r"
                                % frame)
        ang_speed = float(ang_speed or self.cell.config.limits["object_ang_speed"])

        # Rotating moves each TCP along a different arc, and where those arcs
        # run depends on how far apart the arms stand — which the direction
        # teach never measured. Small angles stay inside what the drift guard
        # can catch, so they are allowed and capped rather than refused.
        if len(obj.arm_ids) > 1 and not self.simulate and self.drive_robots:
            ang_speed, refusal = limit_uncalibrated_rotation(
                angle, ang_speed, self.cell.config.calibrated)
            if refusal:
                raise CouplingError(refusal)
        duration = abs(angle) / ang_speed if ang_speed > 0 else 0.0
        start = self.current_pose(obj)
        if duration < 1e-3:
            return 0.0

        if frame == "world":
            path = lambda s: rotate_about_world_axis(
                start, axis, angle * _smoothstep(s), pivot)
        else:
            path = lambda s: rotate_about_own_axis(start, axis,
                                                   angle * _smoothstep(s))
        if self.simulate:
            self._run_sync(obj, path, duration)
            self._settle(obj, path(1.0))
            return duration

        self._set_plan(path, duration)
        return duration

    def command_jog(self, obj, direction_world, speed=None, kind="lin"):
        """Carry the object along a world direction for as long as it is asked.

        `kind="ang"` turns it instead: `direction_world` is then the world axis
        to rotate about and `speed` is rad/s. Everything else is identical,
        including the ramp and the watchdog, because from the operator's side
        RZ is just another button on the same grid.

        This is hold-to-move, so it is called once on the press and again on
        every refresh of the button — the same jog, kept alive, not a new one
        each time. Direction and speed may change between calls without the
        object stopping; only letting go stops it.

        `direction_world` need not be a unit vector: [0, 0, 1] is straight up
        at `speed`, and so is [0, 0, 12]. Both arms are still driven from the
        one object pose the feed produces, so the whole difference between
        this and jogging two arms separately is that here they cannot drift
        apart — the constraint is in the geometry, not in the operator.

        Returns the speed being asked for, in m/s.
        """
        if self.simulate:
            raise CouplingError("a held jog is driven by the live servo feed, "
                                "which simulation does not have")
        if not self.alive:
            raise CouplingError(
                "the servo loop is not running%s"
                % ("" if self.error is None else " — %s" % self.error))
        if obj is not self.object:
            raise CouplingError("the servo loop is holding %r, not %r"
                                % (getattr(self.object, "name", None), obj.name))

        if kind not in ("lin", "ang"):
            raise CouplingError("a jog is lin or ang, not %r" % kind)

        d = np.asarray(direction_world, dtype=float)[:3]
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            self.stop_jog()
            return 0.0
        default = self.cell.config.limits[
            "object_ang_speed" if kind == "ang" else "object_lin_speed"]
        speed = float(speed or default)
        if self.uncalibrated_real:
            speed = min(speed, UNCALIBRATED_REAL_ANG_SPEED
                        if kind == "ang" else UNCALIBRATED_REAL_LIN_SPEED)
        if kind == "ang" and len(obj.arm_ids) > 1 and self.drive_robots:
            # A held button has no total angle to cap, so the ceiling goes on
            # the rate instead: slow enough that the drift guard stops the
            # arms well inside a degree of them disagreeing, and slow enough
            # that whoever is holding the button can see it happen. Angle zero
            # asks only about the speed, which is all a jog has.
            speed, _ = limit_uncalibrated_rotation(
                0.0, speed, self.cell.config.calibrated)
        # measured outside the lock: it reads both arms' feeds, and the feed
        # thread must never wait on a socket to advance a jog already running
        witness = self._measure_arms() if not self.jogging() else None
        now = time.monotonic()
        with self._lock:
            if self._plan is not None:
                # a jog on top of a step takes over from wherever the step got
                # to, rather than snapping back to where it started
                self._hold = self._plan_pose(self._plan, now)
                self._plan = None
            # changing between travelling and turning restarts the ramp rather
            # than carrying the old speed across: they are different units, and
            # 50 mm/s reinterpreted as 50 rad/s is a wrist torn off
            if self._jog is None or self._jog.kind != kind:
                self._jog = _Jog(d / norm, speed, now, kind)
                self._jog.witness = witness
                self._jog.origin = (None if self._hold is None
                                    else np.array(self._hold, dtype=float))
            else:
                self._jog.unit = d / norm
                self._jog.speed = speed
                self._jog.refreshed = now
        return speed

    def stop_jog(self):
        """Let go of the button.

        The jog is asked to ramp down rather than cut, and it ends itself once
        it reaches zero — a target that stops dead is a deceleration the two
        arms have to agree on, and they only agree while the feed is driving
        them both from the same pose.
        """
        with self._lock:
            if self._jog is not None:
                self._jog.speed = 0.0

    def jogging(self):
        with self._lock:
            return self._jog is not None

    def move_object(self, obj, target_pose_world, lin_speed=None, ang_speed=None):
        """Carry the object there, and do not return until it has arrived."""
        duration = self.command_move(obj, target_pose_world, lin_speed, ang_speed)
        if duration and not self.simulate:
            self.wait_done()
        return duration

    def rotate_object(self, obj, axis, angle, ang_speed=None,
                      frame="object", pivot=None):
        """Turn the object, and do not return until it has finished."""
        duration = self.command_rotate(obj, axis, angle, ang_speed, frame, pivot)
        if duration and not self.simulate:
            self.wait_done()
        return duration

    def _settle(self, obj, pose):
        """Where a simulated move leaves the object, for the next one to start from.

        On hardware the feed thread does this: when a plan reaches s = 1 it
        parks the final pose in `_hold`, and `current_pose` — which is what
        every command uses as its starting point — reads it from there.
        Simulation has no feed thread, so `_hold` kept its value from `start()`
        and every step in a program began from wherever the object was when it
        was first picked up. Two steps ran as two independent moves from the
        same origin instead of one after the other, so a dry run of "turn 20
        degrees, turn back 20 degrees" ended 20 degrees from where it began —
        and a dry run is precisely where that is supposed to be caught.
        """
        pose = np.asarray(pose, dtype=float)
        obj.pose_world = pose
        with self._lock:
            self._hold = pose
        return pose

    def _run_sync(self, obj, pose_at, duration):
        """Walk the path inline. Simulation only — on real hardware the
        feed thread owns the sending."""
        arm_ids = obj.arm_ids
        if not self.simulate and not all(a in self.backends for a in arm_ids):
            raise CouplingError("servo backends not started for %s" % arm_ids)

        dt = 1.0 / float(self.cell.config.motion["servo_rate_hz"])
        self._abort = False
        self.trace = []
        t0 = time.monotonic()
        n = 0
        while True:
            t = time.monotonic() - t0 if not self.simulate else n * dt
            s = min(1.0, t / duration)
            pose = pose_at(s)
            targets = obj.targets(self.cell, pose)

            self._check(obj, arm_ids)
            if self.simulate:
                self.trace.append((t, {a: np.array(p) for a, p in targets.items()}))
            else:
                for a in arm_ids:
                    self.backends[a].servo_pose(targets[a])

            if self.on_progress is not None:
                self.on_progress(s)
            if s >= 1.0:
                break

            n += 1
            if not self.simulate:
                sleep = t0 + n * dt - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)

        for a in arm_ids:
            if not self.simulate:
                self.backends[a].stop_motion()
