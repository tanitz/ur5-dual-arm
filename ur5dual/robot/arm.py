"""
One arm: its sockets, its motion vocabulary, and the transform that says
where it stands in the cell.

The whole reason this class exists is that a UR controller only ever talks
about its *own* base frame. Two arms bolted to the same column each report
poses in their own frame, and those two frames say nothing about each other
until you put the mounting geometry between them. `tcp_pose_world()` and
`world_to_base()` are that bridge, and they are the only place the conversion
happens.
"""

import numpy as np

from . import motion as M
from ..geometry import ur_kinematics as UK
from ..geometry.kinematics import inv, mat_to_pose, pose_to_mat
from .transport import Dashboard, ScriptSender, StateStream, read_geometry


GRAVITY = 9.80665
# rad/s below which an arm counts as standing still, so its accelerometer is
# reading gravity and not its own motion
STILL_SPEED = 0.01


class ArmError(RuntimeError):
    pass


class Arm:
    def __init__(self, arm_cfg):
        self.cfg = arm_cfg
        self.id = arm_cfg.id
        self.stream = None
        self.sender = None
        self.motion = None
        self.polyscope = ""
        self.tcp_offset = None
        self.remote_control = None
        # The DH table that describes *this* arm, settled at connect. Starts as
        # the published one because that is all there is until a controller
        # answers.
        self.dh = UK.UR5_DH
        self.dh_source = "the published UR5 table"
        self.dh_error = None
        # picked up and logged by whoever owns this arm; a feed that had to be
        # remade is something the operator must be told about
        self.stream_events = []

    # -- lifecycle ---------------------------------------------------------
    @property
    def connected(self):
        return self.stream is not None

    def connect(self, timeout=5.0):
        self.stream = StateStream(self.cfg.ip, timeout=timeout)
        self.stream.on_reconnect = self._stream_recovered
        self.sender = ScriptSender(self.cfg.ip, timeout=timeout)
        self.motion = M.ArmMotion(self.sender, self.stream)
        with Dashboard(self.cfg.ip, timeout=timeout) as db:
            self.polyscope = db.polyscope_version()
            self.remote_control = db.is_in_remote_control()
        _, self.tcp_offset, calibration = read_geometry(self.cfg.ip,
                                                        timeout=timeout)
        self._settle_kinematics(calibration)
        return self

    def _settle_kinematics(self, calibration):
        """Pick the DH table that matches this arm, and record how well.

        Two arms off the same drawing are not the same machine: each is
        measured at the factory and its corrections kept in its own controller.
        Millimetres, which a single arm working from its own reported poses
        never notices — and which every joint target computed on this side
        carries straight into a workpiece held by two arms. Left to the
        published table, one of these two arms sat 3 mm out and the closed-chain
        solver was right to refuse to drive it.
        """
        state = self.stream.latest()
        tcp = None if self.tcp_offset is None else pose_to_mat(self.tcp_offset)
        self.dh, self.dh_source, self.dh_error = UK.choose_dh(
            np.array(state["q_actual"], dtype=float), state["tcp_pose"],
            tcp, calibration)
        return self.dh_error

    # Reconnects that stop being bad luck and start being a second client.
    # One or two over a session is a network hiccup; a steady trickle is
    # another program holding port 30003, which older controllers cannot serve
    # twice — they stall this one mid-packet and never resume.
    RECONNECT_SUSPICION = 3

    def _stream_recovered(self, ip, reason):
        self.stream_events.append("arm %s feed restarted — %s" % (self.id, reason))
        n = self.stream.reconnects if self.stream is not None else 0
        if n == self.RECONNECT_SUSPICION:
            self.stream_events.append(
                "arm %s has now dropped its feed %d times — something else is "
                "probably reading port 30003 on this robot. `ss -tnp | grep "
                "%s:30003` on the Jetson will name it; the RViz bridge "
                "(ur5dual-rviz) is the usual one, and this controller is too "
                "old to serve two clients at once. A feed that stalls without "
                "erroring is worse than one that breaks: every reader keeps "
                "getting the last sample, so the coordinated loop would carry "
                "the object using a pose that has moved on"
                % (self.id, n, ip))

    def disconnect(self):
        for obj in (self.stream, self.sender):
            if obj is not None:
                obj.close()
        self.stream = self.sender = self.motion = None

    def dashboard(self, timeout=5.0):
        return Dashboard(self.cfg.ip, timeout=timeout)

    # -- state -------------------------------------------------------------
    def state(self):
        if not self.connected:
            raise ArmError("arm %s is not connected" % self.id)
        return self.stream.latest()

    def robot_mode(self):
        return M.ROBOT_MODE_NAMES.get(int(self.state()["robot_mode"]), "?")

    def safety_mode(self):
        return M.SAFETY_NAMES.get(int(self.state()["safety_mode"]), "?")

    def ready(self):
        """Powered, brakes off, and not sitting in any kind of stop."""
        if not self.connected:
            return False
        st = self.state()
        return (M.ROBOT_MODE_NAMES.get(int(st["robot_mode"])) == "RUNNING"
                and M.SAFETY_NAMES.get(int(st["safety_mode"])) in M.RUNNABLE_SAFETY)

    def tcp_force_magnitude(self):
        return float(np.linalg.norm(self.state()["tcp_force"][:3]))

    # -- frames ------------------------------------------------------------
    def base_matrix(self):
        """world -> this arm's base."""
        return self.cfg.base_matrix()

    def tcp_pose(self):
        """TCP in the arm's own base frame — what the controller reports."""
        return np.array(self.state()["tcp_pose"], dtype=float)

    def tcp_matrix_world(self):
        return self.base_matrix() @ pose_to_mat(self.tcp_pose())

    def tcp_pose_world(self):
        return mat_to_pose(self.tcp_matrix_world())

    def world_to_base(self, pose_world):
        """A pose expressed in the cell frame -> the frame this arm speaks."""
        return mat_to_pose(inv(self.base_matrix()) @ pose_to_mat(pose_world))

    def up_in_base(self):
        """Which way is up, in this arm's own base frame — measured, not typed.

        The flange accelerometer feels gravity and nothing else while the arm
        is still, so it points along the vertical wherever the arm happens to
        be standing. Rotating that reading out of the tool frame and into the
        base frame gives the one thing the mounting numbers only claim.

        Returns None when the arm is moving (the reading is then acceleration
        plus gravity and means nothing) or when the firmware is too old to
        report the field.
        """
        st = self.state()
        accel = st.get("tool_accel")
        if accel is None:
            return None
        if float(np.max(np.abs(st["qd_actual"]))) > STILL_SPEED:
            return None

        a = np.array(accel, dtype=float)
        norm = float(np.linalg.norm(a))
        # at rest this is 1 g; anything else is a moving arm or a wrong field
        if not 0.85 * GRAVITY < norm < 1.15 * GRAVITY:
            return None

        # the accelerometer sits on the flange, and tcp_pose is the TCP — the
        # two differ by whatever tool offset is configured on the pendant
        R_base_tcp = pose_to_mat(self.tcp_pose())[:3, :3]
        R_flange_tcp = (np.eye(3) if self.tcp_offset is None
                        else pose_to_mat(self.tcp_offset)[:3, :3])
        # an accelerometer at rest reads the force holding it up, so the
        # vector points along +up, not along gravity
        return R_base_tcp @ R_flange_tcp.T @ (a / norm)

    def base_to_world(self, pose_base):
        return mat_to_pose(self.base_matrix() @ pose_to_mat(pose_base))

    # -- commands, in world coordinates ------------------------------------
    def movel_world(self, pose_world, vel=M.STEP_VEL, acc=M.STEP_ACC):
        self.motion.movel_pose(self.world_to_base(pose_world), vel=vel, acc=acc)

    def movej_world(self, pose_world, vel=M.STEP_VEL, acc=M.STEP_ACC):
        self.motion.movej_pose(self.world_to_base(pose_world), vel=vel, acc=acc)

    def halt(self):
        if self.motion is not None:
            self.motion.halt_all()
