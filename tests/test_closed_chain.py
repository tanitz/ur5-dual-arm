"""The two arms and the box as one chain — arithmetic only, no robot.

The cell here is a stub: two base transforms straight out of config/cell.yaml
and a pair of joint angles, with no sockets behind them. That is enough,
because everything this module claims is geometry. The grasps are built from
forward kinematics, so the chain starts out exactly closed and any error that
appears afterwards was introduced by the solver rather than by the fixture.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.geometry import kinematics as K
from ur5dual.geometry import ur_kinematics as UK
from ur5dual.geometry.closed_chain import ClosedChainSolver, JOINT_SPEED_PLAN
from ur5dual.config import CellConfig
from ur5dual.coupling import (
    UNCALIBRATED_ANG_SPEED, UNCALIBRATED_ROTATION, Coordinator, HeldObject,
    limit_uncalibrated_rotation,
)

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


# ── a cell with no sockets behind it ──────────────────────────────────────
TOOL = [0.0, 0.0, 0.12, 0.0, 0.0, 0.0]      # 120 mm gripper, no rotation


class StubArm:
    def __init__(self, arm_id, xyz, rpy, q):
        self.id = arm_id
        self.connected = True
        self.tcp_offset = np.array(TOOL, dtype=float)
        self._base = K.xyz_rpy_to_mat(xyz, rpy)
        self.q = np.array(q, dtype=float)

    def base_matrix(self):
        return self._base

    def tcp_matrix_world(self):
        return self._base @ UK.fk_tcp(self.q, K.pose_to_mat(self.tcp_offset))

    def state(self):
        return {"q_actual": self.q.copy(),
                "tcp_pose": K.mat_to_pose(
                    UK.fk_tcp(self.q, K.pose_to_mat(self.tcp_offset)))}

    def world_to_base(self, pose_world):
        return K.mat_to_pose(K.inv(self._base) @ K.pose_to_mat(pose_world))


class StubCell:
    def __init__(self, arms, config=None):
        self.arms = arms
        self.config = config
        self.messages = []

    def log(self, message):
        self.messages.append(message)

    def force_vector(self, arm_id):
        return np.zeros(6)


# the mounting geometry this cell actually has: two arms on one column,
# 500 mm apart, tilted 120 degrees outward
Q_A = np.array([0.0, -1.2, 1.4, -1.2, -1.5, 0.0])
Q_B = np.array([0.0, -1.2, 1.4, -1.2, -1.5, 0.0])
cell = StubCell({
    "A": StubArm("A", [0.0, 0.25, 1.2], [-2.0943951, 0.0, 0.0], Q_A),
    "B": StubArm("B", [0.0, -0.25, 1.2], [2.0943951, 0.0, 0.0], Q_B),
})

obj = HeldObject("box").capture(cell, ("A", "B"), origin="midpoint")
seed = {"A": Q_A.copy(), "B": Q_B.copy()}
solver = ClosedChainSolver(cell, obj)

print("the chain as captured")
check("both grasps were captured", sorted(obj.grasps) == ["A", "B"])
print("       grip span %.0f mm" % (obj.span() * 1000))
print("       box frame %s mm" % np.round(np.asarray(obj.pose_world[:3]) * 1000, 1))

joints, info = solver.solve(obj.pose_world, seed)
check("solving at the captured pose converges for both arms",
      all(m["converged"] for m in info.values()))
check("and returns the joints the arms were already at",
      max(float(np.max(np.abs(joints[a] - seed[a]))) for a in "AB") < 1e-6,
      "worst %.2e rad" % max(float(np.max(np.abs(joints[a] - seed[a]))) for a in "AB"))

d_p, d_r = solver.internal_twist(joints)
check("the arms agree about where the box is", d_p < 1e-9 and d_r < 1e-9,
      "%.2e m, %.2e rad" % (d_p, d_r))

check("our FK agrees with the (stubbed) controller pose",
      all(ok for _, _, ok in solver.verify_against_robots().values()))


print("\ncarrying the box in the world frame")
for name, pose in (
        ("+20 mm along world X", np.concatenate([obj.pose_world[:3] + [0.02, 0, 0],
                                                 obj.pose_world[3:]])),
        ("-15 mm along world Z", np.concatenate([obj.pose_world[:3] - [0, 0, 0.015],
                                                 obj.pose_world[3:]])),
        ("10 deg about world Z", K.rotate_about_world_axis(obj.pose_world, "z",
                                                           np.radians(10))),
        ("8 deg about world X", K.rotate_about_world_axis(obj.pose_world, "x",
                                                          np.radians(8))),
        ("6 deg about world Y", K.rotate_about_world_axis(obj.pose_world, "y",
                                                          np.radians(6)))):
    q, meta = solver.solve(pose, seed)
    converged = all(m["converged"] for m in meta.values())
    twist_p, twist_r = solver.internal_twist(q)
    moved = max(float(np.max(np.abs(q[a] - seed[a]))) for a in "AB")
    # each TCP lands within the solver's own 1e-6 m tolerance, so the relative
    # transform between them can be out by twice that and no more — anything
    # larger would be the two arms genuinely disagreeing
    check(name, converged and twist_p < 1e-5 and twist_r < 1e-5 and moved > 1e-4,
          "joints moved %.1f deg, box strain %.1e m" % (np.degrees(moved), twist_p))


print("\nworld axes are not the box's own axes")
tilted = K.rotate_about_own_axis(obj.pose_world, "x", np.radians(40))
own = K.rotate_about_own_axis(tilted, "z", np.radians(30))
world = K.rotate_about_world_axis(tilted, "z", np.radians(30))
check("on a tilted box the two rotations disagree",
      np.linalg.norm(np.asarray(own[3:]) - np.asarray(world[3:])) > 0.05,
      "%.3f rad apart" % np.linalg.norm(np.asarray(own[3:]) - np.asarray(world[3:])))

upright = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
check("on an unrotated box they are the same thing",
      np.allclose(K.rotate_about_own_axis(upright, "z", 0.4),
                  K.rotate_about_world_axis(upright, "z", 0.4), atol=1e-12))
check("a full turn about a world axis comes back to where it started",
      np.allclose(K.pose_to_mat(K.rotate_about_world_axis(tilted, "y", 2 * np.pi)),
                  K.pose_to_mat(tilted), atol=1e-9))


print("\nthe pivot")
spun = K.rotate_about_world_axis(obj.pose_world, "z", np.radians(25))
check("with no pivot the box spins in place",
      np.allclose(spun[:3], obj.pose_world[:3], atol=1e-12))

pivot = np.array(obj.pose_world[:3]) + np.array([0.3, 0.0, 0.0])
swung = K.rotate_about_world_axis(obj.pose_world, "z", np.radians(25), pivot=pivot)
check("with a pivot the box swings around it instead",
      np.linalg.norm(np.asarray(swung[:3]) - np.asarray(obj.pose_world[:3])) > 0.1)
check("and keeps its distance from that pivot",
      abs(np.linalg.norm(np.asarray(swung[:3]) - pivot)
          - np.linalg.norm(np.asarray(obj.pose_world[:3]) - pivot)) < 1e-12)
check("a world-Z rotation about a pivot does not change height",
      abs(swung[2] - obj.pose_world[2]) < 1e-12)


print("\nwalking a path before committing to it")
start = np.array(obj.pose_world, dtype=float)
lift = start.copy()
lift[2] += 0.05
plan = solver.plan(lambda s: K.interp_pose(start, lift, s), seed, duration=2.0)
check("a 50 mm lift plans clean", plan.ok, plan.complaint() or "")
print("       min sigma  A %.4f  B %.4f" % (plan.min_sigma["A"], plan.min_sigma["B"]))
print("       peak joint speed  A %.1f  B %.1f deg/s"
      % (np.degrees(plan.max_joint_speed["A"]),
         np.degrees(plan.max_joint_speed["B"])))
check("the plan covers the whole path", len(plan.joints) == len(plan.samples))
check("consecutive samples do not jump branches",
      max(float(np.max(np.abs(plan.joints[i + 1][a] - plan.joints[i][a])))
          for i in range(len(plan.joints) - 1) for a in "AB") < 0.1)

far = start.copy()
far[0] += 3.0
bad = solver.plan(lambda s: K.interp_pose(start, far, s), seed)
check("a path that leaves the workspace is refused", not bad.ok)
check("and says which arm gives out first", "arm" in (bad.complaint() or ""))
print("       %s" % (bad.complaint() or "").split("  ")[0])

rushed = solver.plan(lambda s: K.interp_pose(start, lift, s), seed, duration=0.05)
check("the same lift in 50 ms is refused for joint speed",
      bool(rushed.too_fast),
      "peak %.0f deg/s, limit %.0f"
      % (np.degrees(max(rushed.max_joint_speed.values())),
         np.degrees(JOINT_SPEED_PLAN)))


print("\nhow far the box may be turned before a touch-off")
free, refusal = limit_uncalibrated_rotation(np.radians(90), 0.30, True)
check("a measured cell turns as far and as fast as asked",
      refusal is None and free == 0.30)

speed, refusal = limit_uncalibrated_rotation(np.radians(5), 0.30, False)
check("an unmeasured cell allows a small turn, slowly",
      refusal is None and speed == UNCALIBRATED_ANG_SPEED,
      "%.1f deg/s" % np.degrees(speed))

_, refusal = limit_uncalibrated_rotation(np.radians(90), 0.30, False)
check("and refuses a large one", refusal is not None)
check("saying what would lift the limit", "touch-off" in (refusal or ""))

# the jog panel's largest step preset is exactly the cap, and a button that
# refuses itself on the last bit of floating point reads as broken
_, refusal = limit_uncalibrated_rotation(UNCALIBRATED_ROTATION, 0.30, False)
check("the cap itself is allowed, not refused by a rounding error",
      refusal is None)

check("a jog asks only about speed and is never refused",
      limit_uncalibrated_rotation(0.0, 0.30, False)[1] is None)


print("\na dry run composes its steps, rather than restarting each one")
# The regression: on hardware the feed thread parks each finished move's pose
# for the next one to start from, and simulation has no feed thread. Every
# step therefore began from wherever the object was picked up, so a dry run of
# "turn 20 degrees, turn back" ended 20 degrees out — in the one mode whose
# entire purpose is catching that before it reaches the arms.
sim_cell = StubCell(cell.arms, CellConfig.load())
sim_obj = HeldObject("box").capture(sim_cell, ("A", "B"), origin="midpoint")
sim = Coordinator(sim_cell, simulate=True)
sim.start(sim_obj)
origin = np.array(sim_obj.pose_world, dtype=float)

sim.rotate_object(sim_obj, "z", np.radians(20), frame="world")
turned = np.array(sim_obj.pose_world, dtype=float)
check("the first turn moves the box",
      np.degrees(K.pose_distance(origin, turned)[1]) > 19.0)

sim.rotate_object(sim_obj, "z", np.radians(-20), frame="world")
_, back = K.pose_distance(origin, np.array(sim_obj.pose_world, dtype=float))
check("turning back returns it exactly", np.degrees(back) < 1e-6,
      "%.2e deg out" % np.degrees(back))

for axis in ("x", "y", "z"):
    for frame in ("world", "object"):
        before = np.array(sim_obj.pose_world, dtype=float)
        sim.rotate_object(sim_obj, axis, np.radians(12), frame=frame)
        sim.rotate_object(sim_obj, axis, np.radians(-12), frame=frame)
        _, out = K.pose_distance(before,
                                 np.array(sim_obj.pose_world, dtype=float))
        if np.degrees(out) > 1e-6:
            check("%s %s round trip" % (frame, axis), False,
                  "%.3f deg out" % np.degrees(out))
            break
    else:
        continue
    break
else:
    check("every axis round-trips in both frames (6 pairs)", True)

lifted = np.array(sim_obj.pose_world, dtype=float)
lifted[2] += 0.05
sim.move_object(sim_obj, lifted)
sim.move_object(sim_obj, np.array(sim_obj.pose_world, dtype=float) - [0, 0, 0.05, 0, 0, 0])
_, _ = K.pose_distance(lifted, np.array(sim_obj.pose_world, dtype=float))
check("a carry then its reverse lands back too",
      abs(float(sim_obj.pose_world[2]) - float(origin[2])) < 1e-9,
      "%.2e m out" % abs(float(sim_obj.pose_world[2]) - float(origin[2])))


print("\nposture before the grippers close")
report = solver.branch_report(seed)
check("every arm reports room left and a manipulability",
      all(len(v) == 2 for v in report.values()))
for arm_id, (margin, sigma) in sorted(report.items()):
    print("       arm %s  %.0f deg from a stop, sigma %.4f"
          % (arm_id, np.degrees(margin), sigma))

print("\n%d failed" % fail)
sys.exit(1 if fail else 0)
