"""
The two arms and the box as one closed kinematic chain.

`coupling.py` already turns one object pose into one Cartesian target per arm.
That is the right idea and this module does not replace it — it replaces what
happens next. Today those Cartesian targets go over the wire and each
controller solves for joints on its own, in its own way, with no knowledge
that another arm is holding the other side of the same box. Two things go
wrong there, and only when the box is heavy enough to matter:

  branch flips   eight joint solutions put the flange in the same place, and
                 the controller may hand you a different one from the last
                 cycle. On one arm that is a startling but survivable lurch.
                 On two arms holding one box it is the box.

  no lookahead   a rotation that runs arm B's wrist into its limit at 40
                 degrees is discovered at 40 degrees, with the box in the air.

Solving here fixes both. Joints come out of `ur_kinematics.ik` seeded with the
previous cycle's answer, which cannot leave the branch it started on, and the
whole path can be walked in a few milliseconds of arithmetic before a single
target is sent.

What it does *not* do is buy redundancy. Two 6-DOF arms rigidly holding one
box is twelve joints against twelve constraint equations — a square system,
one solution per branch, nothing spare to steer with. Arms that do have room
to spare in this situation have seven joints each, which is why the papers
about null-space optimisation are all written on iiwas and Frankas. Here the
only choice available is which of the eight branches each arm sits on, and
that choice is made once, when the grippers close.
"""

import numpy as np

from .kinematics import inv, mat_to_pose, pose_to_mat
from .ur_kinematics import (
    JOINT_LIMITS, SIGMA_MIN_WARN, UR5_DH, fk_tcp, ik, limit_margin,
    manipulability,
    nearest_turn, within_limits,
)


class ClosedChainError(RuntimeError):
    pass


# A UR5 joint will do 180 deg/s, but a servo loop tracking a target is not the
# same as a planned move: the arm is always a little behind, and the further
# behind it is the worse the two arms agree. Plans are checked against a
# fraction of the rating, so the loop is never asked to work at the edge.
JOINT_SPEED_MAX = np.pi          # rad/s, the arm's own rating
JOINT_SPEED_PLAN = 0.35 * np.pi  # rad/s, what a coordinated plan may ask for

# how many points along a path get checked. At the default object speeds a
# carry lasts a second or two, so this samples every few tens of milliseconds
# — fine enough to catch a wrist passing through a singularity, coarse enough
# to stay well under a millisecond of arithmetic.
PLAN_SAMPLES = 64


class ArmChain:
    """One arm's fixed geometry, gathered once so the solver need not ask again.

    The two transforms here are the entire bridge between a pose in the cell
    and six joint angles: where this arm's base stands in the world, and what
    tool is bolted to its flange. Both are read at capture time rather than
    per cycle, because both are constants — and because a tool offset that
    changed mid-carry would mean somebody edited the pendant while two arms
    were holding a box.
    """

    def __init__(self, arm_id, base_matrix, tcp_offset=None, limits=None,
                 dh=None):
        self.id = arm_id
        self.base = np.asarray(base_matrix, dtype=float)
        self.base_inv = inv(self.base)
        self.tcp = None if tcp_offset is None else pose_to_mat(tcp_offset)
        self.limits = JOINT_LIMITS if limits is None else np.asarray(limits, float)
        # Per arm, not per project: the two robots in this cell were measured
        # separately at the factory and disagree with the published table, and
        # with each other, by millimetres.
        self.dh = UR5_DH if dh is None else dh

    @classmethod
    def from_arm(cls, arm, limits=None):
        return cls(arm.id, arm.base_matrix(), arm.tcp_offset, limits,
                   getattr(arm, "dh", None))

    def tcp_world(self, q):
        """Joint angles -> TCP pose as a 4x4 in the cell frame."""
        return self.base @ fk_tcp(q, self.tcp, self.dh)

    def solve(self, T_world_tcp, q_seed, **kw):
        """Where this arm's joints must be to put its TCP there."""
        return ik(self.base_inv @ T_world_tcp, q_seed, self.tcp, dh=self.dh,
                  **kw)


class ChainPlan:
    """The result of walking a path before any of it is sent.

    Every field is something an operator can be told in a sentence, because
    the point of planning offline is to replace "the arms stopped and I do not
    know why" with "arm B's wrist runs out 40 degrees into this rotation".
    """

    def __init__(self, joints, samples):
        self.joints = joints          # [ {arm: q}, ... ] one per sample
        self.samples = samples        # [ s, ... ] the path parameter
        self.unreachable = []         # (sample index, arm, pos_error, rot_error)
        self.limit_hits = []          # (sample index, arm, joint, angle, lo, hi)
        self.min_sigma = {}           # arm -> worst manipulability on the path
        self.max_joint_speed = {}     # arm -> rad/s the path demands
        self.duration = None

    @property
    def ok(self):
        return not self.unreachable and not self.limit_hits and not self.too_fast

    @property
    def too_fast(self):
        return [a for a, v in self.max_joint_speed.items() if v > JOINT_SPEED_PLAN]

    @property
    def near_singular(self):
        return [a for a, v in self.min_sigma.items() if v < SIGMA_MIN_WARN]

    def complaint(self):
        """One paragraph saying what is wrong, or None if nothing is."""
        parts = []
        if self.unreachable:
            i, a, ep, er = self.unreachable[0]
            where = "%.0f%% of the way along this path" % (100.0 * self.samples[i])
            if ep * 1000.0 < 1.0 and np.degrees(er) < 1.0:
                # The solver got arbitrarily close without ever arriving, which
                # is what the edge of the workspace looks like from the inside:
                # the arm is not short by a distance, it is out of geometry.
                # Quoting the residual here reads as "0 mm short", which sounds
                # like nothing is wrong.
                parts.append(
                    "arm %s runs out of reach at %s — it can approach that "
                    "pose but not hold it, which is the edge of its workspace "
                    "rather than a near miss" % (a, where))
            else:
                parts.append(
                    "arm %s cannot reach %s — the nearest it gets is %.1f mm "
                    "and %.1f deg away, so the box would have to leave one "
                    "gripper to get there"
                    % (a, where, ep * 1000.0, np.degrees(er)))
        if self.limit_hits:
            i, a, j, angle, lo, hi = self.limit_hits[0]
            parts.append(
                "arm %s runs J%d out to %.0f deg at %.0f%% of the way along, "
                "past its %.0f deg limit — unwind that joint a full turn "
                "before starting and the tool still points the same way"
                % (a, j + 1, np.degrees(angle), 100.0 * self.samples[i],
                   np.degrees(hi if angle > 0 else lo)))
        for a in self.too_fast:
            parts.append(
                "arm %s would have to turn a joint at %.0f deg/s to keep up, "
                "over the %.0f deg/s a coordinated move allows — slow the "
                "move down or shorten it"
                % (a, np.degrees(self.max_joint_speed[a]),
                   np.degrees(JOINT_SPEED_PLAN)))
        for a in self.near_singular:
            parts.append(
                "arm %s passes within %.4f of a singularity on this path, "
                "where small box motions need large joint motions — it will "
                "track badly there even if it does not fault"
                % (a, self.min_sigma[a]))
        return "  ".join(parts) if parts else None


class ClosedChainSolver:
    """Object pose in, both arms' joint angles out.

    Held open deliberately: `solve` is what the servo loop calls every cycle,
    and `plan` is what a program step calls once before committing. They run
    the same arithmetic, so a plan that passes is a promise about what the
    loop will do rather than a separate opinion about it.
    """

    def __init__(self, cell, obj, limits=None):
        self.cell = cell
        self.object = obj
        if not obj.grasps:
            raise ClosedChainError(
                "nothing is held — the chain is only closed once both grippers "
                "are on the box and ATTACH has captured the grasps")
        self.chains = {}
        for arm_id in obj.arm_ids:
            arm = cell.arms[arm_id]
            if not arm.connected:
                raise ClosedChainError("arm %s is not connected" % arm_id)
            self.chains[arm_id] = ArmChain.from_arm(arm, limits)

    @property
    def arm_ids(self):
        return sorted(self.chains)

    # -- reading the robots (state only; nothing here commands motion) ------
    def joints_now(self):
        """{arm: q} as the controllers report it this instant."""
        return {a: np.array(self.cell.arms[a].state()["q_actual"], dtype=float)
                for a in self.arm_ids}

    def verify_against_robots(self, tol=0.002):
        """Does our forward kinematics agree with what the arms report?

        The single most useful thing in this module, and the first thing to
        run. Everything downstream — every joint target, every plan, every
        promise that the two arms agree about where the box is — rests on DH
        parameters, a tool offset, and a base transform all being right. This
        compares our own FK against the TCP each controller reports for the
        joints it is sitting at, which catches a wrong tool length, an e-series
        DH table, or a base transform that was never measured, without moving
        anything.

        Returns {arm: (position error m, rotation error rad, ok)}.
        """
        out = {}
        for a in self.arm_ids:
            arm = self.cell.arms[a]
            state = arm.state()
            q = np.array(state["q_actual"], dtype=float)
            mine = fk_tcp(q, self.chains[a].tcp, self.chains[a].dh)
            theirs = pose_to_mat(np.array(state["tcp_pose"], dtype=float))
            d_p = float(np.linalg.norm(mine[:3, 3] - theirs[:3, 3]))
            d_r = float(np.linalg.norm(
                mat_to_pose(mine @ inv(theirs))[3:]))
            out[a] = (d_p, d_r, d_p < tol and d_r < tol * 10)
        return out

    # -- the solve ---------------------------------------------------------
    def targets_world(self, object_pose_world):
        """{arm: 4x4 TCP pose in the cell frame} for a given box pose."""
        T_w_o = pose_to_mat(object_pose_world)
        return {a: T_w_o @ self.object.grasps[a] for a in self.arm_ids}

    def solve(self, object_pose_world, seed, tidy=True, **kw):
        """({arm: q}, {arm: info}) — both arms' joints for one box pose.

        `seed` is the previous answer, and it is doing more work than it
        appears to. It picks the branch, it keeps the wrist from unwinding,
        and it makes the solve cheap: one servo cycle of box motion is well
        under a millimetre, so the seed is already almost the answer.

        `tidy` slides each joint to the turn nearest the seed. Two answers a
        full revolution apart put the box in the same place and are equally
        correct, but one of them makes the arm take the long way round.
        """
        targets = self.targets_world(object_pose_world)
        joints, info = {}, {}
        for a in self.arm_ids:
            q, meta = self.chains[a].solve(targets[a], seed[a], **kw)
            if tidy:
                q = nearest_turn(q, seed[a])
            joints[a], info[a] = q, meta
        return joints, info

    def solve_strict(self, object_pose_world, seed, **kw):
        """`solve`, but a pose one arm cannot reach is an error, not a report.

        What the servo loop wants: it has no way to act on a partial answer,
        and sending one arm to a pose the other could not match is precisely
        how a box gets pulled out of a gripper.
        """
        joints, info = self.solve(object_pose_world, seed, **kw)
        bad = [(a, m) for a, m in info.items() if not m["converged"]]
        if bad:
            a, m = bad[0]
            raise ClosedChainError(
                "arm %s cannot hold its side of the box there — the nearest it "
                "gets is %.0f mm and %.1f deg away"
                % (a, m["pos_error"] * 1000.0, np.degrees(m["rot_error"])))
        return joints

    # -- walking a path before committing to it ----------------------------
    def plan(self, pose_at, seed, duration=None, samples=PLAN_SAMPLES):
        """Walk `pose_at(s)` for s in [0, 1] and report what it would cost.

        `pose_at` is the same callable the coordinator feeds to its servo
        thread, so this checks the actual path rather than a straight line
        between its ends — which matters for a rotation, where the ends can
        both be comfortable and the middle not.

        Pass `duration` to have joint speeds checked too; without it the plan
        still reports reach, limits and singularities, which do not depend on
        how fast the path is walked.
        """
        s_values = np.linspace(0.0, 1.0, int(samples))
        plan = ChainPlan([], list(s_values))
        plan.duration = duration
        q_seed = {a: np.array(seed[a], dtype=float) for a in self.arm_ids}
        min_sigma = {a: np.inf for a in self.arm_ids}
        max_speed = {a: 0.0 for a in self.arm_ids}
        previous = None

        for i, s in enumerate(s_values):
            pose = np.asarray(pose_at(float(s)), dtype=float)
            joints, info = self.solve(pose, q_seed)
            reached = {a: info[a]["converged"] for a in self.arm_ids}

            for a in self.arm_ids:
                meta = info[a]
                min_sigma[a] = min(min_sigma[a], meta["sigma_min"])
                if not reached[a]:
                    plan.unreachable.append(
                        (i, a, meta["pos_error"], meta["rot_error"]))
                    # the joints that came back are wherever the solver gave up,
                    # not a posture the arm would ever be in — reporting them
                    # as limit violations would bury the real fault in noise
                    continue
                for j, angle, lo, hi in within_limits(joints[a],
                                                      self.chains[a].limits):
                    plan.limit_hits.append((i, a, j, angle, lo, hi))

            if previous is not None and duration:
                dt = duration * (s_values[i] - s_values[i - 1])
                if dt > 0:
                    for a in self.arm_ids:
                        if not reached[a]:
                            continue
                        step = float(np.max(np.abs(joints[a] - previous[a])))
                        max_speed[a] = max(max_speed[a], step / dt)

            plan.joints.append(joints)
            # A failed solve is not a place to start the next one from. Keeping
            # the last posture that did work means a path with an unreachable
            # patch in the middle is still checked properly on the far side,
            # instead of every later sample failing from a ruined seed.
            q_seed = {a: (joints[a] if reached[a] else q_seed[a])
                      for a in self.arm_ids}
            previous = q_seed

        plan.min_sigma = {a: float(v) for a, v in min_sigma.items()}
        plan.max_joint_speed = {a: float(v) for a, v in max_speed.items()}
        return plan

    # -- choosing where to stand before the box is picked up ---------------
    def branch_report(self, seed=None):
        """How much room each arm has left, on the branch it is currently on.

        The one decision this geometry actually leaves open. Both arms are on
        a branch already — whichever one they were driven onto before the
        grippers closed — and nothing in a carry can change it, because
        changing branch means passing through a singularity with a box in
        both hands. So the time to look at this number is before ATTACH, and
        the way to act on it is to jog the arm to a better posture and grip
        again.

        Returns {arm: (margin in rad to the nearest joint limit, sigma_min)}.
        """
        seed = self.joints_now() if seed is None else seed
        return {a: (limit_margin(seed[a], self.chains[a].limits),
                    manipulability(seed[a], self.chains[a].tcp,
                                   self.chains[a].dh))
                for a in self.arm_ids}

    def internal_twist(self, joints):
        """How far the two arms disagree about where the box is, in this model.

        Zero by construction when both solves converged, because both were
        derived from the same box pose — which is the whole reason to solve
        here rather than let two controllers do it separately. It is not zero
        when a solve fell short, and then this is the residual the box would
        have to absorb, so it is worth measuring rather than assuming.
        """
        if len(self.arm_ids) < 2:
            return 0.0, 0.0
        a, b = self.arm_ids
        actual = inv(self.chains[a].tcp_world(joints[a])) @ \
            self.chains[b].tcp_world(joints[b])
        residual = mat_to_pose(inv(self.object.grasps[a]) @ self.object.grasps[b]
                               @ inv(actual))
        return (float(np.linalg.norm(residual[:3])),
                float(np.linalg.norm(residual[3:])))
