"""
An arm with no robot behind it.

`SimArm` answers every question `Arm` answers and holds six joint angles
instead of a socket. That is the whole design: the panel, the coordinated
loop, the closed-chain solver and the guards all go on asking the same
questions, and none of them is told which kind of arm answered. A sim mode
built by putting `if simulating` through those callers instead would be a
second control path, tested by nobody, diverging quietly from the one that
drives the robots.

Motion is integrated on read rather than by a thread. A jog sets a joint
velocity and stamps the clock; the next caller to ask where the arm is gets
the angles brought forward to now. There is no timer to start, stop, or leave
running after the window closes, and a caller that asks twice in the same
millisecond gets the same answer twice.

Cartesian jogging goes through the Jacobian, and a held-object move through
the same inverse kinematics the real path uses — so a pose this cannot reach
in simulation is a pose the arm could not have reached either.
"""

import math
import threading
import time

import numpy as np

from ..geometry.kinematics import inv, mat_to_pose, pose_to_mat, rotvec_to_mat
from ..geometry.ur_kinematics import fk_tcp, ik, jacobian

GRAVITY = 9.80665

# Where each simulated arm stands before anything has told it otherwise.
# Values are kept in degrees here so each J1..J6 starting angle is easy to
# configure; SimArm converts them to radians for the kinematics below.
HOME_DEG = {
    "A": (-39, -55, 121, -176, -113, 47),
    "B": (-141, -125, -121, -4, 113, 133),
}
HOME_Q = {arm: tuple(math.radians(d) for d in q)
          for arm, q in HOME_DEG.items()}

# And the way out of it. Solved rather than guessed: each arm's inverse
# kinematics for a point 150 mm either side of the centre line, in front of the
# mast at working height, checked against the joint limits and for a
# manipulability well clear of a singularity (both land near 0.23).
#
# The result is two tools 300 mm apart with a box-sized gap between them, which
# is what makes ATTACH and then a coordinated move do something visible. This
# is also the default simulated starting posture configured above.
#
# Whole degrees, and B is A's mirror image rather than a second IK solution of
# its own. The two mountings are mirror images through the y=0 plane, and for a
# UR that reflection is exactly q -> -q + (180, 180, 0, 180, 0, 180) degrees, so
# the pair matches limb for limb, both tools point at each other across the
# centre line, and B's manipulability cannot drift from A's. Solving the two
# arms separately did not: it picked different elbow and wrist branches, and
# left B's tool facing away from A. Rounding to whole degrees costs 2 mm.
#
# These assume `mount.rotate_deg_A/B` are zero. J1 turns about the very axis a
# pad is spun about, so rotating a pad by r degrees moves that arm's tool
# unless J1 here is shifted by -r to cancel it. Change a rotate number and this
# joint has to follow it, or Ready pose throws the arms apart.
READY_DEG = {
    "A": (-39, -55, 121, -176, -113, 47),
    "B": (-141, -125, -121, -4, 113, 133),
}
READY_Q = {arm: tuple(math.radians(d) for d in q)
           for arm, q in READY_DEG.items()}

# A velocity command that has not been renewed within this is dropped, exactly
# as `speedl`'s own `t=` argument does on a real controller. Without it a jog
# whose release never arrived would keep integrating forever, and the arm on
# screen would wander off while nobody was touching anything.
VELOCITY_WATCHDOG = 0.3

# Joint speed a simulated arm will admit to. Only used to fill in `qd_actual`
# for the monitors; nothing is being driven, so nothing is limited by it.
MAX_JOINT_SPEED = math.pi


class SimStream:
    """Stands in for the real-time feed, which is never stale here."""

    def __init__(self):
        self.reconnects = 0
        self.on_reconnect = None

    @property
    def age(self):
        return 0.0

    def close(self):
        pass


class SimMotion:
    """The URScript vocabulary, carried out in arithmetic.

    Every method matches `ArmMotion`'s signature and does the same thing to a
    simulated arm that the script would have done to a real one — which is why
    the jog panel needs no knowledge that it is driving a simulation.
    """

    def __init__(self, arm):
        self.arm = arm

    # -- jogging -----------------------------------------------------------
    def step(self, frame, axis, sign, preset):
        from . import motion as M
        if frame == "joint":
            delta = np.zeros(6)
            delta[axis] = sign * M.STEP_ANG[preset]
            self.arm.nudge_joints(delta)
            return
        twist = np.zeros(6)
        twist[axis] = sign * (M.STEP_LIN[preset] if axis < 3 else M.STEP_ANG[preset])
        self.arm.nudge_cartesian(twist, frame)

    def step_period(self, frame, preset):
        from . import motion as M
        if frame == "joint":
            return max(0.10, M.STEP_ANG[preset] / M.STEP_JVEL + 0.03)
        return max(0.10, M.STEP_LIN[preset] / M.STEP_VEL + 0.03)

    def speed(self, frame, cmd, preset):
        from . import motion as M
        cmd = np.asarray(cmd, dtype=float)
        if frame == "joint":
            self.arm.set_joint_velocity(cmd * M.CONT_ANG[preset])
            return
        twist = cmd.copy()
        twist[:3] *= M.CONT_LIN[preset]
        twist[3:] *= M.CONT_ANG[preset]
        self.arm.set_cartesian_velocity(twist, frame)

    def halt(self, frame="base"):
        self.arm.set_joint_velocity(np.zeros(6))

    def halt_all(self):
        self.arm.set_joint_velocity(np.zeros(6))

    # -- moves -------------------------------------------------------------
    def movej_pose(self, pose, vel=None, acc=None):
        self.arm.go_to_pose(pose)

    def movel_pose(self, pose, vel=None, acc=None):
        self.arm.go_to_pose(pose)

    # -- everything else ---------------------------------------------------
    def set_digital_out(self, n, on):
        self.arm.digital_out[int(n)] = bool(on)

    def freedrive_on(self):
        # Nothing to release: there is no weight on a simulated arm and nobody
        # can push it. Recorded so the panel's checks see a consistent state
        # rather than an arm that refuses to admit it was asked.
        self.arm.freedrive = True

    def freedrive_off(self):
        self.arm.freedrive = False


class SimArm:
    """One arm, simulated. Same surface as `Arm`, no sockets."""

    def __init__(self, arm_cfg, q=None):
        self.cfg = arm_cfg
        self.id = arm_cfg.id
        self.polyscope = "simulated"
        self.remote_control = None
        self.stream_events = []
        self.stream = SimStream()
        self.sender = None
        self.motion = SimMotion(self)
        # A simulated arm has no pendant to read a tool offset from. Zero is
        # the honest answer — TCP and flange coincide — rather than borrowing
        # a number from a robot that is not here.
        self.tcp_offset = np.zeros(6)
        self.digital_out = {}
        self.freedrive = False

        self._lock = threading.RLock()
        self._q = np.array(HOME_Q[self.id] if q is None else q, dtype=float)
        self._qd = np.zeros(6)
        now = time.monotonic()
        self._advance_stamp = now
        self._command_stamp = now
        self._target_stamp = now
        self._velocity_mode = False

    # -- lifecycle ---------------------------------------------------------
    @property
    def connected(self):
        return True

    def connect(self, timeout=5.0):
        return self

    def disconnect(self):
        self.set_joint_velocity(np.zeros(6))

    def dashboard(self, timeout=5.0):
        raise RuntimeError("arm %s is simulated — there is no controller to "
                           "power on, release brakes on, or unlock" % self.id)

    # -- the state everything else reads -----------------------------------
    def _advance_locked(self):
        """Integrate only an explicit velocity jog, never a joint target.

        A servoJ target also has a finite-difference velocity for the monitor,
        but extrapolating that velocity between targets moves past the target
        and makes the next servo cycle pull back.  Position targets therefore
        update ``_qd`` for display without entering velocity mode.
        """
        now = time.monotonic()
        elapsed = now - self._advance_stamp
        self._advance_stamp = now
        if not self._velocity_mode:
            if now - self._target_stamp > VELOCITY_WATCHDOG:
                self._qd = np.zeros(6)
            return
        if not np.any(self._qd):
            self._velocity_mode = False
            return
        if now - self._command_stamp > VELOCITY_WATCHDOG:
            # the command was never renewed; treat it as released rather than
            # letting GUI state reads renew a command nobody is still holding
            self._qd = np.zeros(6)
            self._velocity_mode = False
            return
        self._q = self._q + self._qd * max(0.0, elapsed)

    def joints(self):
        with self._lock:
            self._advance_locked()
            return self._q.copy()

    def state(self):
        with self._lock:
            self._advance_locked()
            q, qd = self._q.copy(), self._qd.copy()
        pose = mat_to_pose(fk_tcp(q, pose_to_mat(self.tcp_offset)))
        return {
            "q_actual": q,
            "qd_actual": qd,
            "q_target": q,
            "tcp_pose": pose,
            "tcp_speed": np.zeros(6),
            "tcp_force": np.zeros(6),
            "robot_mode": 7.0,           # RUNNING
            "safety_mode": 1.0,          # NORMAL
            "digital_in_bits": 0,
            "motor_temp": np.zeros(6),
            "tool_accel": self._tool_accel(q),
            "time": time.monotonic(),
            "packet_size": 0,
        }

    def _tool_accel(self, q):
        """What a flange accelerometer would read: gravity, and only gravity."""
        R_base_tcp = fk_tcp(q, pose_to_mat(self.tcp_offset))[:3, :3]
        up_base = self.base_matrix()[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        return R_base_tcp.T @ up_base * GRAVITY

    def robot_mode(self):
        return "RUNNING"

    def safety_mode(self):
        return "NORMAL"

    def ready(self):
        return True

    def tcp_force_magnitude(self):
        return 0.0

    # -- frames, identical to Arm ------------------------------------------
    def base_matrix(self):
        return self.cfg.base_matrix()

    def tcp_pose(self):
        return mat_to_pose(fk_tcp(self.joints(), pose_to_mat(self.tcp_offset)))

    def tcp_matrix_world(self):
        return self.base_matrix() @ fk_tcp(self.joints(),
                                           pose_to_mat(self.tcp_offset))

    def tcp_pose_world(self):
        return mat_to_pose(self.tcp_matrix_world())

    def world_to_base(self, pose_world):
        return mat_to_pose(inv(self.base_matrix()) @ pose_to_mat(pose_world))

    def base_to_world(self, pose_base):
        return mat_to_pose(self.base_matrix() @ pose_to_mat(pose_base))

    def up_in_base(self):
        return self.base_matrix()[:3, :3].T @ np.array([0.0, 0.0, 1.0])

    # -- being driven ------------------------------------------------------
    def set_joints(self, q):
        """Put the arm exactly here. What a servo backend does every cycle."""
        q = np.asarray(q, dtype=float)
        with self._lock:
            self._advance_locked()
            now = time.monotonic()
            dt = max(1e-3, now - self._target_stamp)
            self._qd = np.clip((q - self._q) / dt, -MAX_JOINT_SPEED,
                               MAX_JOINT_SPEED)
            self._q = q.copy()
            self._target_stamp = now
            self._advance_stamp = now
            self._velocity_mode = False

    def set_joint_velocity(self, qd):
        with self._lock:
            self._advance_locked()
            self._qd = np.asarray(qd, dtype=float).copy()
            now = time.monotonic()
            self._command_stamp = now
            self._advance_stamp = now
            self._velocity_mode = bool(np.any(self._qd))

    def set_cartesian_velocity(self, twist, frame="base"):
        """Resolve a tool twist into joint rates through the Jacobian."""
        with self._lock:
            self._advance_locked()
            q = self._q.copy()
        twist = np.asarray(twist, dtype=float).copy()
        if frame == "tool":
            R = fk_tcp(q, pose_to_mat(self.tcp_offset))[:3, :3]
            twist = np.concatenate([R @ twist[:3], R @ twist[3:]])
        J = jacobian(q, pose_to_mat(self.tcp_offset))
        # damped, so a jog towards a singularity slows down instead of
        # demanding a joint rate no arm could make
        qd = J.T @ np.linalg.solve(J @ J.T + (0.05 ** 2) * np.eye(6), twist)
        self.set_joint_velocity(np.clip(qd, -MAX_JOINT_SPEED, MAX_JOINT_SPEED))

    def nudge_joints(self, delta):
        with self._lock:
            self._advance_locked()
            self._q = self._q + np.asarray(delta, dtype=float)
            self._qd = np.zeros(6)
            self._velocity_mode = False

    def nudge_cartesian(self, twist, frame="base"):
        """One step-per-press increment, in the frame the panel names."""
        q = self.joints()
        tool = pose_to_mat(self.tcp_offset)
        T = fk_tcp(q, tool)
        twist = np.asarray(twist, dtype=float)
        target = T.copy()
        if frame == "tool":
            target = T @ pose_to_mat(twist)
        else:
            target[:3, 3] = T[:3, 3] + twist[:3]
            if np.any(twist[3:]):
                target[:3, :3] = rotvec_to_mat(twist[3:]) @ T[:3, :3]
        self.go_to_matrix(target)

    def go_to_pose(self, pose_base):
        self.go_to_matrix(pose_to_mat(pose_base))

    def go_to_matrix(self, T_base_tcp):
        q, info = ik(T_base_tcp, self.joints(), pose_to_mat(self.tcp_offset))
        if info["converged"]:
            self.set_joints(q)
            self.set_joint_velocity(np.zeros(6))
        return info["converged"]

    def halt(self):
        self.set_joint_velocity(np.zeros(6))
