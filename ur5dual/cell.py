"""
The cell: both arms, the geometry that relates them, and the safety checks
that only make sense once there are two.

A single arm cannot collide with itself in any way its own controller does
not already guard against. Two arms sharing a workspace can, and nothing in
the socket interface knows that — so the guards live here: a keep-out box
around the column, a force ceiling, and the drift check that catches the two
arms starting to fight each other while holding one object.
"""

import math

import numpy as np

from .robot import motion as M
from .robot.arm import Arm
from .robot.sim_arm import READY_Q, SimArm
from .sim_view import SimViewSender
from .config import ARM_IDS, CellConfig
from .geometry.kinematics import inv, mat_to_pose, pose_to_mat


# every UR joint runs to +/-360 degrees
JOINT_LIMIT = 2.0 * math.pi


class CellError(RuntimeError):
    pass


class Cell:
    def __init__(self, config=None, simulated=False):
        self.config = config or CellConfig.load()
        self.simulated = bool(simulated)
        self.sim_view = None
        self.arms = {}
        self.listeners = []               # callables taking one log string
        self._build_arms()

    def _build_arms(self, seed=None):
        maker = SimArm if self.simulated else Arm
        self.arms = {a: maker(self.config.arms[a]) for a in ARM_IDS}
        if self.simulated and seed:
            for a, q in seed.items():
                if a in self.arms and q is not None:
                    self.arms[a].set_joints(q)
        self._make_sim_view()

    def _make_sim_view(self):
        if getattr(self, "sim_view", None) is not None:
            self.sim_view.close()
        # REAL uses the same loopback channel as SIM.  The GUI is already the
        # authoritative reader of port 30003; RViz must not open a second
        # connection to an older controller and starve this safety feed.
        self.sim_view = SimViewSender()

    def publish_sim_view(self, lease=0.0):
        """Send the simulated cell's joint angles to whoever is drawing it.

        Lives here rather than in the coordinated loop, which is where it was
        first put and where it was wrong: the loop only runs once an object has
        been taken hold of, so jogging a single arm — or simply looking at the
        cell before attaching anything — published nothing at all. RViz then
        fell back to the robots' own feeds, found no robots, and drew every
        joint at zero while the panel's own readouts moved. The arms are
        simulated the whole time the cell is; so is the picture of them.
        """
        joints = {}
        for a in ARM_IDS:
            arm = self.arms.get(a)
            if arm is None:
                continue
            try:
                joints[a] = arm.state()["q_actual"]
            except (OSError, ConnectionError, RuntimeError):
                continue
        # Sending an empty REAL announcement is intentional: it tells RViz to
        # release its fallback robot feeds before this cell connects.
        return self.sim_view.send(joints,
                                  mode="sim" if self.simulated else "real",
                                  lease=lease)

    def sim_ready_pose(self):
        """Put the simulated arms where they can actually be driven.

        Restores the configured ready posture: the two tools are a box-width
        apart and both wrists are well conditioned.

        Does nothing to real arms. Nothing here should ever be able to command
        a robot to jump to a stored pose.
        """
        if not self.simulated:
            return False
        for a in ARM_IDS:
            arm = self.arms.get(a)
            if arm is not None and a in READY_Q:
                arm.set_joints(np.array(READY_Q[a], dtype=float))
                arm.set_joint_velocity(np.zeros(6))
        self.publish_sim_view()
        gap = self.tcp_separation()
        self.log("simulated arms moved to the ready posture — tools %s apart"
                 % ("%.0f mm" % (gap * 1000) if gap else "an unknown distance"))
        return True

    def set_simulated(self, simulated):
        """Swap the whole cell between real arms and simulated ones.

        Everything above this line — the panel, the coordinated loop, the
        solver, the guards — keeps asking a `cell.arms[id]` the same questions.
        Which class answers is settled here and nowhere else, so there is no
        second control path to keep in step with the first.

        Going to simulation carries the real arms' current joint angles across
        when they are known, so the model starts where the cell actually is
        rather than jumping to a home pose the robots are nowhere near.
        """
        simulated = bool(simulated)
        if simulated == self.simulated:
            return False
        seed = None
        if simulated:
            seed = {}
            for a in ARM_IDS:
                arm = self.arms.get(a)
                try:
                    if arm is not None and arm.connected:
                        seed[a] = np.array(arm.state()["q_actual"], dtype=float)
                except (OSError, ConnectionError, RuntimeError):
                    pass
        self.disconnect()
        self.simulated = simulated
        self._build_arms(seed)
        self.log("cell is now %s" % ("simulated — no robot is being contacted"
                                     if simulated else
                                     "live; press Connect for each arm"))
        return True

    # -- plumbing ----------------------------------------------------------
    def log(self, text):
        for fn in self.listeners:
            fn(text)

    def __getitem__(self, arm_id):
        return self.arms[arm_id]

    @property
    def enabled_ids(self):
        return [a for a in ARM_IDS if self.config.arms[a].enabled]

    @property
    def connected_ids(self):
        return [a for a in ARM_IDS if self.arms[a].connected]

    def connect(self, arm_ids=None):
        """Connect the arms asked for, reporting failures rather than
        raising: a cell with one arm reachable is still worth driving."""
        results = {}
        for a in (arm_ids or self.enabled_ids):
            try:
                self.arms[a].connect()
                results[a] = True
                self.log("arm %s connected — %s" % (a, self.arms[a].polyscope))
                # Whether joint control is available for this arm, said at the
                # one moment it can be found out. Left unsaid, the answer only
                # showed up later as the coordinated loop declining to use it.
                arm = self.arms[a]
                if arm.dh_error is not None:
                    self.log("arm %s kinematics: %s, matching its controller "
                             "to %.2f mm"
                             % (a, arm.dh_source, arm.dh_error * 1000))
                if self.arms[a].remote_control is False:
                    self.log("arm %s is in LOCAL control — commands will be "
                             "ignored until you switch to Remote Control" % a)
            except OSError as e:
                results[a] = False
                self.log("arm %s connect failed: %s" % (a, e))
        return results

    def disconnect(self):
        for a in ARM_IDS:
            self.arms[a].disconnect()

    def reload_config(self, config):
        """Swap in edited geometry without dropping the sockets."""
        self.config = config
        for a in ARM_IDS:
            self.arms[a].cfg = config.arms[a]

    # -- state -------------------------------------------------------------
    def ready(self, arm_ids=None):
        ids = arm_ids or self.connected_ids
        return bool(ids) and all(self.arms[a].ready() for a in ids)

    def not_ready_reason(self, arm_ids=None):
        for a in (arm_ids or self.connected_ids):
            arm = self.arms[a]
            if not arm.connected:
                return "arm %s not connected" % a
            if not arm.ready():
                return "arm %s: %s / %s" % (a, arm.robot_mode(), arm.safety_mode())
        return None

    def halt(self):
        for a in self.connected_ids:
            try:
                self.arms[a].halt()
            except OSError as e:
                self.log("arm %s stop failed: %s" % (a, e))

    # -- two-arm geometry --------------------------------------------------
    def tcp_separation(self):
        """Distance between the two TCPs, in metres, in the world frame."""
        if len(self.connected_ids) < 2:
            return None
        pa = self.arms["A"].tcp_matrix_world()[:3, 3]
        pb = self.arms["B"].tcp_matrix_world()[:3, 3]
        return float(np.linalg.norm(pb - pa))

    def feeds_fresh(self, max_age=0.05):
        """True when every connected arm's real-time feed is current.

        Anything that compares the two arms against each other is only as
        good as the older of the two samples. One feed lagging by 200 ms
        while an arm moves at 50 mm/s puts 10 mm of pure fiction into the
        comparison — enough to trip a guard that is watching for 4.
        """
        for a in self.connected_ids:
            if self.arms[a].stream.age > max_age:
                return False
        return True

    def relative_transform(self, max_age=0.05):
        """A's TCP -> B's TCP, or None if either feed is too stale to trust.

        Constant while both hold one rigid object, which is what makes it
        worth watching — and why a stale reading must not be mistaken for a
        real change.
        """
        if len(self.connected_ids) < 2:
            return None
        if max_age is not None and not self.feeds_fresh(max_age):
            return None
        return inv(self.arms["A"].tcp_matrix_world()) @ self.arms["B"].tcp_matrix_world()

    def geometry_complaint(self, arm_ids=None):
        """What is plainly wrong with the configured geometry, or None.

        Costs nothing and is worth doing before anything moves. Calibration
        trims a geometry that is roughly right; it cannot rescue one that
        describes a different cell. An arm whose base transform has it
        reaching back towards the mast is in the second category, and the
        symptom is unmistakable once two arms are holding one object: each
        works out its own share of the same world path, the shares disagree,
        and the arms drive into each other — squeezing whatever is between
        them, or colliding outright when the grippers are empty.

        The test is only meaningful for arms standing off a mast, which is
        every layout this cell has ever had. It looks at where the base
        actually sits rather than assuming a side, so it survives a yawed or
        hand-measured cell.
        """
        if not bool(self.config.mount.get("enforce_outward_guard", True)):
            return None
        for a in (arm_ids or self.enabled_ids):
            outward = self.config.base_axis_outward(a)
            if outward < 0:
                return ("arm %s's configured base has it reaching back towards "
                        "the mast (outward %+.2f). That is not a small error to "
                        "be calibrated out — it is the wrong way round, and a "
                        "world-frame move will send the two arms into each "
                        "other. Correct arms.%s.base — or the mount numbers it "
                        "is derived from — in config/cell.yaml"
                        % (a, outward, a))
        return None

    # -- guards ------------------------------------------------------------
    def joint_headroom(self, arm_id):
        """(joint index, radians of travel left) for the tightest joint.

        A UR joint runs to +/-360 degrees, and a wrist that has wound its way
        out to one end has no travel left in that direction even though the
        tool is pointing somewhere perfectly ordinary — J6 at -360 and J6 at 0
        hold the tool identically. Ask for one more degree the wrong way and
        the controller answers with a protective stop, which from the outside
        looks like the robot objecting to the motion rather than to the number.

        So it gets measured. Unwinding the joint by a full turn costs nothing
        and gives the whole range back.
        """
        q = np.array(self.arms[arm_id].state()["q_actual"])
        margin = JOINT_LIMIT - np.abs(q)
        i = int(np.argmin(margin))
        return i, float(margin[i])

    def wound_up_joints(self, arm_ids=None, warn_at=math.radians(15)):
        """[(arm, joint index, margin)] for joints close to their stops."""
        out = []
        for a in (arm_ids or self.connected_ids):
            i, margin = self.joint_headroom(a)
            if margin < warn_at:
                out.append((a, i, margin))
        return out

    def force_vector(self, arm_id):
        return np.array(self.arms[arm_id].state()["tcp_force"][:3], dtype=float)

    def force_exceeded(self, arm_ids=None, baseline=None):
        """Which arms are pushing too hard, and by how much.

        With a `baseline` — the force each arm reported when it took hold —
        the test is on the *rise* since then. That is the number that means
        something: the standing offset from an unconfigured payload cancels,
        and what is left is what the arm has started pushing against.

        Without one it falls back to the absolute ceiling, which only catches
        a hard collision.
        """
        hot = []
        rise_limit = float(self.config.motion.get("max_force_rise", 40.0))
        abs_limit = float(self.config.motion["max_tcp_force"])
        for a in (arm_ids or self.connected_ids):
            f = self.force_vector(a)
            if baseline is not None and a in baseline:
                rise = float(np.linalg.norm(f - baseline[a]))
                if rise > rise_limit:
                    hot.append((a, rise, "N more than when it took hold"))
                    continue
            if float(np.linalg.norm(f)) > abs_limit:
                hot.append((a, float(np.linalg.norm(f)), "N absolute"))
        return hot

    def check_reachable(self, arm_id, pose_world, max_reach=0.85, min_reach=0.20):
        """Cheap reach test in the arm's own base frame.

        A UR5 reaches 850 mm and cannot fold closer than roughly 200 mm to
        its own axis. This rejects the targets that are obviously impossible
        before any motion is commanded; it is not a substitute for the
        controller's own IK, which is the real authority.
        """
        p = self.arms[arm_id].world_to_base(pose_world)[:3]
        r = float(np.linalg.norm(p))
        if r > max_reach:
            return False, "%.0f mm from arm %s base — beyond its %.0f mm reach" % (
                r * 1000, arm_id, max_reach * 1000)
        if r < min_reach:
            return False, "%.0f mm from arm %s base — inside its dead zone" % (
                r * 1000, arm_id)
        return True, ""
