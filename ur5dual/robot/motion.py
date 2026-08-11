"""
URScript for one arm: jogging, absolute moves, I/O, freedrive.

Everything here is stateless request-response over the secondary interface —
good for teaching, jogging and point-to-point moves. The coordinated 125 Hz
work lives in backends.py instead, because that needs a controller that stays
resident on the robot.

Poses in this module are in the arm's own base frame, the frame its
controller speaks. Converting to and from the cell's world frame is Arm's
job, not this module's.
"""

import math

import numpy as np

FRAMES = ("base", "tool", "joint")

AXIS_NAMES = ("X", "Y", "Z", "RX", "RY", "RZ")
JOINT_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")

# increment / speed presets, index 0..3
STEP_LIN = (0.001, 0.005, 0.010, 0.050)                  # m per step
STEP_ANG = tuple(math.radians(a) for a in (0.5, 1.0, 5.0, 10.0))
CONT_LIN = (0.010, 0.030, 0.060, 0.120)                  # m/s
CONT_ANG = (0.05, 0.15, 0.30, 0.60)                      # rad/s

STEP_VEL = 0.05      # m/s   executing a Cartesian step
STEP_ACC = 0.5       # m/s^2
STEP_JVEL = 0.35     # rad/s executing a joint step
STEP_JACC = 1.0      # rad/s^2

WATCHDOG = 0.4       # t= on speedl/speedj: motion dies if not refreshed
REFRESH = 0.08       # how often a held jog button refreshes the command

SAFETY_NAMES = {
    1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE STOP", 4: "RECOVERY",
    5: "SAFEGUARD STOP", 6: "SYSTEM E-STOP", 7: "ROBOT E-STOP",
    8: "VIOLATION", 9: "FAULT",
}
ROBOT_MODE_NAMES = {
    0: "DISCONNECTED", 1: "CONFIRM SAFETY", 2: "BOOTING", 3: "POWER OFF",
    4: "POWER ON", 5: "IDLE", 6: "BACKDRIVE", 7: "RUNNING",
}

RUNNABLE_SAFETY = ("NORMAL", "REDUCED")


def _vec(values):
    return ",".join("%.6f" % v for v in values)


class ArmMotion:
    """URScript generator bound to one arm's script socket."""

    def __init__(self, sender, stream=None):
        self.sender = sender
        self.stream = stream          # only needed for tool-frame velocities

    # -- jogging: one increment -------------------------------------------
    def step(self, frame, axis, sign, preset):
        if frame == "joint":
            self.sender.send(
                "def jog_j():\n"
                "  q = get_actual_joint_positions()\n"
                "  q[%d] = q[%d] + %.6f\n"
                "  movej(q, a=%f, v=%f)\n"
                "end\n"
                % (axis, axis, sign * STEP_ANG[preset], STEP_JACC, STEP_JVEL))
            return

        d = np.zeros(6)
        d[axis] = sign * (STEP_LIN[preset] if axis < 3 else STEP_ANG[preset])
        pose = "p[%s]" % _vec(d)

        if frame == "tool":
            # pose_trans applies the delta in the TCP's own frame
            self.sender.send("movel(pose_trans(get_actual_tcp_pose(), %s), "
                             "a=%f, v=%f)" % (pose, STEP_ACC, STEP_VEL))
        elif axis < 3:
            # pose_add adds the position and leaves the rotation untouched
            self.sender.send("movel(pose_add(get_actual_tcp_pose(), %s), "
                             "a=%f, v=%f)" % (pose, STEP_ACC, STEP_VEL))
        else:
            # rotate about a base axis through the TCP point: keep the current
            # position, pre-multiply the current rotation
            self.sender.send(
                "def jog_rot():\n"
                "  p_c = get_actual_tcp_pose()\n"
                "  p_r = pose_trans(%s, p[0,0,0,p_c[3],p_c[4],p_c[5]])\n"
                "  movel(p[p_c[0],p_c[1],p_c[2],p_r[3],p_r[4],p_r[5]], "
                "a=%f, v=%f)\n"
                "end\n" % (pose, STEP_ACC, STEP_VEL))

    def step_period(self, frame, preset):
        """Spacing between queued steps, so at most ~one is outstanding."""
        if frame == "joint":
            return max(0.10, STEP_ANG[preset] / STEP_JVEL + 0.03)
        return max(0.10, STEP_LIN[preset] / STEP_VEL + 0.03)

    # -- jogging: continuous ----------------------------------------------
    def speed(self, frame, cmd, preset):
        """`cmd` is a 6-vector of -1/0/+1 direction flags."""
        from ..geometry.kinematics import rotvec_to_mat

        cmd = np.asarray(cmd, dtype=float)
        if frame == "joint":
            self.sender.send("speedj([%s], a=%f, t=%f)"
                             % (_vec(cmd * CONT_ANG[preset]), STEP_JACC, WATCHDOG))
            return

        v = cmd.copy()
        v[:3] *= CONT_LIN[preset]
        v[3:] *= CONT_ANG[preset]
        if frame == "tool":
            R = rotvec_to_mat(self.stream.latest()["tcp_pose"][3:])
            v = np.concatenate([R @ v[:3], R @ v[3:]])
        self.sender.send("speedl([%s], a=%f, t=%f)" % (_vec(v), STEP_ACC, WATCHDOG))

    def halt(self, frame="base"):
        if frame == "joint":
            self.sender.send("stopj(%f)" % STEP_JACC)
        else:
            self.sender.send("stopl(%f)" % STEP_ACC)

    def halt_all(self):
        """Both stop forms, for the panic button — whichever applies wins."""
        self.sender.send("stopl(%f)" % STEP_ACC)
        self.sender.send("stopj(%f)" % STEP_JACC)

    # -- absolute moves ----------------------------------------------------
    def movej_q(self, q, vel=STEP_JVEL, acc=STEP_JACC):
        self.sender.send("movej([%s], a=%f, v=%f)" % (_vec(q), acc, vel))

    def movej_pose(self, pose, vel=STEP_VEL, acc=STEP_ACC):
        """Joint-interpolated move to a Cartesian pose — no straight-line
        guarantee, but it will not stall in a singularity on the way."""
        self.sender.send("movej(p[%s], a=%f, v=%f)" % (_vec(pose), acc, vel))

    def movel_pose(self, pose, vel=STEP_VEL, acc=STEP_ACC):
        self.sender.send("movel(p[%s], a=%f, v=%f)" % (_vec(pose), acc, vel))

    # -- tool ---------------------------------------------------------------
    def set_digital_out(self, n, on):
        self.sender.send("set_digital_out(%d, %s)" % (n, "True" if on else "False"))

    def freedrive_on(self):
        # freedrive only lives as long as a program runs, so give it one
        self.sender.send("def fd():\n  freedrive_mode()\n"
                         "  while True:\n    sync()\n  end\nend\n")

    def freedrive_off(self):
        self.sender.send("end_freedrive_mode()")
