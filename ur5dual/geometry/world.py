"""
Where the two arms are in the cell frame, and what to send to move them there.

A UR controller only ever speaks about its own base. Two arms bolted to one
column each report poses in their own frame, and those frames say nothing
about each other until the mounting geometry is put between them. `arm.py`
crosses that bridge for an arm whose socket is open, and `closed_chain.py`
crosses it for two arms already holding a box. Neither answers the question
this module exists for: given six joint angles — read a moment ago, saved in
config/points.json, or typed into a script — where is that tool in the cell,
and what has to be sent to put it somewhere else in the cell?

Both directions, because a move needs both:

    joints  ──fk──▶  arm base  ──base──▶  world          `pose`
    world   ──base_inv──▶  arm base  ──ik──▶  joints     `joints_for`
    world   ──base_inv──▶  arm base                      `to_base`, for the wire

Both arms, because two arms standing in the same posture are only *visibly*
the same in the world frame. Their joint angles differ by however the two
brackets are turned, and each base frame calls the same place in the room
something else — so "are they in the same pose" and "how far apart are the
tools" are questions that can only be asked here.

Which point on the arm is meant matters more here than anywhere else:

    flange   what UR calls tool0, the face at the end of joint 6
    TCP      the flange times the tool offset set on the pendant

Mixing them up is a quiet error of exactly one tool length, so every method
says which one it returns. Built without tool offsets, the TCP *is* the
flange — which is also what an arm with no tool configured reports.

Nothing here opens a socket or reads a clock. `from_cell` takes a cell so it
can copy each arm's measured tool offset and factory DH table, and reads
nothing from it afterwards; `from_config` needs only cell.yaml, so a world
Cartesian can be worked out for a cell that is switched off.
"""

import numpy as np

from .closed_chain import ArmChain
from .kinematics import (
    inv, mat_to_pose, pose_to_mat, rotate_about_world_axis,
)
from .ur_kinematics import fk, nearest_turn


class WorldFrameError(RuntimeError):
    pass


def _as_matrix(target):
    """A 4x4 as given, or the 4x4 a 6-vector pose stands for."""
    T = np.asarray(target, dtype=float)
    return T if T.shape == (4, 4) else pose_to_mat(T)


class WorldCartesian:
    """The cell's fixed geometry, and the world-frame arithmetic on top of it.

    Built once and asked many times. The base transforms, the tool offsets and
    the DH tables are constants of the cell, and re-reading them per call is
    how a tool offset comes to change halfway through a move.
    """

    def __init__(self, chains):
        self.chains = dict(chains)
        if not self.chains:
            raise WorldFrameError("a world frame with no arms in it says "
                                  "nothing about anything")

    # -- building ----------------------------------------------------------
    @classmethod
    def from_config(cls, config, arm_ids=None, tools=None, dh=None):
        """From config/cell.yaml alone — no robot, nothing to connect to.

        `tools` is {arm: 6-vector tool offset} and `dh` is {arm: DH table},
        both optional. cell.yaml holds neither on purpose: the pendant is the
        authority on the tool, and each controller is the authority on its own
        factory calibration. Left out, this reports the flange from the
        published UR5 table, which describes these two arms to about 3 mm.
        """
        ids = sorted(config.arms) if arm_ids is None else list(arm_ids)
        tools = tools or {}
        dh = dh or {}
        return cls({a: ArmChain(a, config.arms[a].base_matrix(),
                                tools.get(a), None, dh.get(a))
                    for a in ids})

    @classmethod
    def from_cell(cls, cell, arm_ids=None):
        """From a live cell, taking each arm's own tool offset and DH table.

        Worth preferring wherever there is a cell to hand. An arm that has
        been connected has read the tool actually set on its pendant and the
        corrections measured for that particular machine at the factory, and
        both are millimetres away from the drawing — the same millimetres the
        closed-chain solver refuses to drive without.
        """
        ids = sorted(cell.arms) if arm_ids is None else list(arm_ids)
        return cls({a: ArmChain.from_arm(cell.arms[a]) for a in ids})

    @property
    def arm_ids(self):
        return sorted(self.chains)

    def chain(self, arm_id):
        try:
            return self.chains[arm_id]
        except KeyError:
            raise WorldFrameError("no arm %r in this cell — it holds %s"
                                  % (arm_id, ", ".join(self.arm_ids))) from None

    def has_tool(self, arm_id):
        """False when this arm's TCP is its flange, tool offset unknown."""
        return self.chain(arm_id).tcp is not None

    # -- joints -> the cell frame ------------------------------------------
    def flange_matrix(self, arm_id, q):
        """Joint angles -> 4x4 *flange* pose in the world frame.

        The face a gauge can be laid across, which is what makes it the thing
        worth comparing against a measurement (see tools/flange_log.py).
        """
        chain = self.chain(arm_id)
        return chain.base @ fk(np.asarray(q, dtype=float), chain.dh)

    def flange_pose(self, arm_id, q):
        """Joint angles -> [x y z rx ry rz] flange pose in the world frame."""
        return mat_to_pose(self.flange_matrix(arm_id, q))

    def tcp_matrix(self, arm_id, q):
        """Joint angles -> 4x4 *TCP* pose in the world frame."""
        return self.chain(arm_id).tcp_world(np.asarray(q, dtype=float))

    def pose(self, arm_id, q):
        """Joint angles -> [x y z rx ry rz] TCP pose in the world frame.

        The everyday one: what the panel shows, what a taught point holds, and
        what `PointLibrary` stores — so a pose worked out here can be compared
        against one taught by hand without converting anything.
        """
        return mat_to_pose(self.tcp_matrix(arm_id, q))

    def poses(self, joints):
        """{arm: 6-vector world TCP pose} for {arm: six joint angles}."""
        return {a: self.pose(a, q) for a, q in joints.items()}

    # -- the cell frame <-> the frame a controller speaks ------------------
    def to_base(self, arm_id, pose_world):
        """A world pose -> the same pose in that arm's own base frame.

        The last step before the wire. `movel_pose`, `servo_pose` and every
        target a controller is ever handed are in its own frame; the cell
        frame exists on this side of the socket and nowhere else.
        """
        return mat_to_pose(self.chain(arm_id).base_inv @ _as_matrix(pose_world))

    def to_world(self, arm_id, pose_base):
        """A pose as the controller reports it -> the same pose in the cell."""
        return mat_to_pose(self.chain(arm_id).base @ _as_matrix(pose_base))

    def direction_to_base(self, arm_id, direction_world):
        """A world *direction* -> the same direction in that arm's base frame.

        A vector, not a point, so only the rotation applies — which is exactly
        the difference between jogging along the room's X and translating a
        pose. Linear and angular parts of a twist about the tool transform by
        this same rotation, there being no translation cross-term.
        """
        return self.chain(arm_id).base[:3, :3].T @ np.asarray(direction_world,
                                                              dtype=float)[:3]

    # -- the cell frame -> joints ------------------------------------------
    def joints_for(self, arm_id, target_world, seed, tidy=True, **kw):
        """Where this arm's joints must be to put its TCP there. (q, info).

        `seed` is not a hint, it is the choice: eight postures put the flange
        in the same place, and the solver walks downhill from the seed, so it
        returns the one nearest to where the arm already is. Passing the arm's
        current angles is therefore both the fastest answer and the only one
        it can reach without travelling through the others.

        `tidy` slides each joint to the turn nearest the seed — two answers a
        full revolution apart are equally correct and one of them winds the
        wrist the long way round.

        `info["converged"]` is the only thing a caller must check. An
        unconverged q is wherever the solver gave up, not a posture to send.
        """
        seed = np.asarray(seed, dtype=float)
        q, info = self.chain(arm_id).solve(_as_matrix(target_world), seed, **kw)
        if tidy:
            q = nearest_turn(q, seed)
        return q, info

    def solve(self, targets, seed, tidy=True, **kw):
        """({arm: q}, {arm: info}) for one world target per arm."""
        joints, info = {}, {}
        for a, target in targets.items():
            joints[a], info[a] = self.joints_for(a, target, seed[a], tidy, **kw)
        return joints, info

    def solve_strict(self, targets, seed, **kw):
        """`solve`, but a target an arm cannot reach raises rather than reports.

        What a move wants: there is nothing to do with a partial answer, and
        sending one arm to a pose the other could not match is how a workpiece
        gets pulled out of a gripper.
        """
        joints, info = self.solve(targets, seed, **kw)
        bad = [(a, m) for a, m in sorted(info.items()) if not m["converged"]]
        if bad:
            a, m = bad[0]
            raise WorldFrameError(
                "arm %s cannot put its tool there — the nearest it gets is "
                "%.1f mm and %.1f deg away"
                % (a, m["pos_error"] * 1000.0, np.degrees(m["rot_error"])))
        return joints

    # -- moving, in the cell frame -----------------------------------------
    def moved(self, joints, delta_world):
        """Both tools carried by the same world offset. {arm: 4x4}.

        The whole point of doing it here rather than per arm: one vector in
        the cell frame becomes a different vector in each base frame, and two
        arms each given "+100 mm in X" in their own frames go two different
        ways. Given the same *world* offset they travel parallel, so whatever
        is between the grippers keeps its grip and its orientation.
        """
        d = np.asarray(delta_world, dtype=float)[:3]
        out = {}
        for a, q in joints.items():
            T = self.tcp_matrix(a, q)
            T[:3, 3] = T[:3, 3] + d
            out[a] = T
        return out

    def turned(self, joints, axis, angle, pivot=None):
        """Both tools turned about an axis of the cell. {arm: 4x4}.

        A rigid turn of the pair: each tool swings around the pivot and its
        orientation turns with it, so the two tools keep their relationship
        and anything held between them turns as one piece.

        `axis` is "x"/"y"/"z", an index, or a world vector. `pivot` defaults
        to the point midway between the two tools — the one place a turn costs
        both arms the same travel; pass a point in world coordinates to swing
        the pair about somewhere else in the room instead.
        """
        poses = self.poses(joints)
        if pivot is None:
            pivot = np.mean([p[:3] for p in poses.values()], axis=0)
        return {a: pose_to_mat(rotate_about_world_axis(p, axis, angle, pivot))
                for a, p in poses.items()}

    def targets_in_base(self, targets):
        """{arm: 4x4 or pose in world} -> {arm: 6-vector in that arm's frame}.

        Ready for `arm.motion.movel_pose` or a servo backend, for the times
        the controllers are left to solve their own IK.
        """
        return {a: self.to_base(a, T) for a, T in targets.items()}

    # -- the pair ----------------------------------------------------------
    def separation(self, joints, arm_ids=None):
        """Distance between two tools, in metres, in the cell frame."""
        ids = self.arm_ids if arm_ids is None else list(arm_ids)
        if len(ids) < 2:
            return None
        a, b = ids[:2]
        return float(np.linalg.norm(self.tcp_matrix(a, joints[a])[:3, 3]
                                    - self.tcp_matrix(b, joints[b])[:3, 3]))

    def relative(self, joints, arm_ids=None):
        """The first arm's TCP -> the second arm's TCP, as a 4x4.

        Constant while both hold one rigid workpiece, which is what makes it
        the thing to watch: a change in it is the two arms starting to
        disagree about where that workpiece is.
        """
        ids = self.arm_ids if arm_ids is None else list(arm_ids)
        if len(ids) < 2:
            return None
        a, b = ids[:2]
        return inv(self.tcp_matrix(a, joints[a])) @ self.tcp_matrix(b, joints[b])

    def midpoint(self, joints, arm_ids=None):
        """The frame halfway between the two tools, as a 6-vector in world.

        Position midway between them, orientation taken from the first arm —
        the same convention `HeldObject.capture(origin="midpoint")` plants an
        object frame with, so a pose worked out here can be handed straight to
        a coordinated move once the grippers have closed on something.
        """
        ids = self.arm_ids if arm_ids is None else list(arm_ids)
        if len(ids) < 2:
            return None
        a, b = ids[:2]
        T = self.tcp_matrix(a, joints[a])
        T[:3, 3] = (T[:3, 3] + self.tcp_matrix(b, joints[b])[:3, 3]) / 2.0
        return mat_to_pose(T)


def world_poses(config, joints, tools=None, dh=None):
    """The one-liner: {arm: world TCP pose} from cell.yaml and joint angles.

    For a script that wants the answer rather than an object to keep. Anything
    that asks more than once should build a `WorldCartesian` and hold it.
    """
    return WorldCartesian.from_config(config, sorted(joints), tools,
                                      dh).poses(joints)
