"""The world-frame Cartesian layer — arithmetic only, no robot.

Everything `world.py` claims is geometry, so all of it can be checked against
the real config/cell.yaml with the cell switched off. The posture used is the
one the cell actually stands in for two-arm work: `sim_arm.READY_Q`, where the
two arms are mirror images of each other and the tools point at each other
across the centre line. That is the case the module exists for — the two sets
of joint angles look nothing alike, and only the world frame says the arms are
holding the same place.

What is checked is the pair of things a move depends on: that a world pose and
six joint angles say the same thing in both directions, and that a rigid move
of the pair comes out rigid.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.config import ARM_IDS, CellConfig
from ur5dual.geometry import kinematics as K
from ur5dual.geometry import ur_kinematics as UK
from ur5dual.geometry.world import WorldCartesian, WorldFrameError, world_poses
from ur5dual.robot.sim_arm import READY_Q, SimArm

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


TOOL = [0.0, 0.0, 0.12, 0.0, 0.0, 0.0]      # 120 mm gripper, no rotation

Q = {a: np.array(READY_Q[a], dtype=float) for a in ARM_IDS}

config = CellConfig.load()
world = WorldCartesian.from_config(config, tools={a: TOOL for a in ARM_IDS})

print("both arms in the ready posture, read out of config/cell.yaml")
for a in ARM_IDS:
    p = world.pose(a, Q[a])
    print("       arm %s  J %s deg" % (a, " ".join("%7.1f" % v
                                                   for v in np.degrees(Q[a]))))
    print("            x %7.1f  y %7.1f  z %7.1f mm   rx ry rz %6.1f %6.1f "
          "%6.1f deg" % (p[0] * 1000, p[1] * 1000, p[2] * 1000,
                         *np.degrees(p[3:])))
print("       tools %.1f mm apart, midpoint at x %.1f y %.1f z %.1f mm"
      % (world.separation(Q) * 1000, *(world.midpoint(Q)[:3] * 1000)))


print("\nframes")
for a in ARM_IDS:
    mine = world.tcp_matrix(a, Q[a])
    theirs = config.arms[a].base_matrix() @ UK.fk_tcp(Q[a], K.pose_to_mat(TOOL))
    check("arm %s world TCP is its base transform times FK, and nothing else" % a,
          float(np.max(np.abs(mine - theirs))) < 1e-12)

for a in ARM_IDS:
    tool_length = float(np.linalg.norm(
        world.tcp_matrix(a, Q[a])[:3, 3] - world.flange_matrix(a, Q[a])[:3, 3]))
    check("arm %s flange and TCP differ by exactly the tool" % a,
          abs(tool_length - TOOL[2]) < 1e-12,
          "%.4f mm out" % ((tool_length - TOOL[2]) * 1000))

no_tool = WorldCartesian.from_config(config)
check("with no tool offset the TCP is the flange",
      float(np.max(np.abs(no_tool.tcp_matrix("A", Q["A"])
                          - no_tool.flange_matrix("A", Q["A"])))) < 1e-12)
check("and it says which it is reporting",
      not no_tool.has_tool("A") and world.has_tool("A"))

for a in ARM_IDS:
    p = world.pose(a, Q[a])
    d_t, d_r = K.pose_distance(p, world.to_world(a, world.to_base(a, p)))
    check("arm %s world -> base -> world round trip" % a,
          d_t < 1e-12 and d_r < 1e-9, "%.2e m" % d_t)

# A direction is not a point: only the rotation may touch it, and the answer
# is the world axis written in base coordinates rather than the other way up.
up_base = world.direction_to_base("A", [0.0, 0.0, 1.0])
check("a world direction keeps its length in a base frame",
      abs(float(np.linalg.norm(up_base)) - 1.0) < 1e-12)
check("and it is the base rotation, transposed, that took it there",
      float(np.max(np.abs(up_base
                          - config.arms["A"].base_matrix()[2, :3]))) < 1e-12)


print("\njoints out of a world pose")
for a in ARM_IDS:
    q, info = world.joints_for(a, world.pose(a, Q[a]), Q[a])
    check("arm %s solves the pose it is already standing in" % a,
          info["converged"], "%.4f mm short" % (info["pos_error"] * 1000))
    check("arm %s comes back to the angles it started from" % a,
          float(np.max(np.abs(q - Q[a]))) < 1e-6,
          "%.2e rad" % float(np.max(np.abs(q - Q[a]))))


print("\ncarrying the pair 50 mm along the room's X")
# One world vector, two different vectors in the two base frames. Two arms
# each given "+50 mm in X" in their own frame would go two different ways.
targets = world.moved(Q, [0.05, 0.0, 0.0])
joints = world.solve_strict(targets, Q)
for a in ARM_IDS:
    travelled = float(np.linalg.norm(world.tcp_matrix(a, joints[a])[:3, 3]
                                     - world.tcp_matrix(a, Q[a])[:3, 3]))
    check("arm %s travels the 50 mm it was given" % a,
          abs(travelled - 0.05) < 1e-5, "%.3f mm" % (travelled * 1000))
gap = abs(world.separation(joints) - world.separation(Q))
check("the two tools stay as far apart as they were", gap < 1e-5,
      "%.4f mm" % (gap * 1000))
check("and keep the same relationship, which is what holding one thing means",
      float(np.max(np.abs(world.relative(joints) - world.relative(Q)))) < 1e-5)

# The same move as a Cartesian target per arm, for the times each controller
# is left to solve its own IK. Both routes have to describe one motion.
in_base = world.targets_in_base(targets)
for a in ARM_IDS:
    d_t, d_r = K.pose_distance(in_base[a],
                               K.mat_to_pose(K.inv(config.arms[a].base_matrix())
                                             @ targets[a]))
    check("arm %s Cartesian target says the same as the joint answer" % a,
          d_t < 1e-9 and d_r < 1e-9)

far = world.moved(Q, [3.0, 0.0, 0.0])
_, info_far = world.solve(far, Q)
check("a pose three metres away is reported unreachable, not solved",
      not all(m["converged"] for m in info_far.values()))
try:
    world.solve_strict(far, Q)
    check("and solve_strict refuses it rather than sending it", False)
except WorldFrameError as e:
    check("and solve_strict refuses it rather than sending it",
          "cannot put its tool there" in str(e))


print("\nturning the pair")
angle = np.radians(20.0)
joints = world.solve_strict(world.turned(Q, "z", angle), Q)
gap = abs(world.separation(joints) - world.separation(Q))
check("a turn about the room's Z keeps the tools as far apart", gap < 1e-5,
      "%.4f mm" % (gap * 1000))
moved_mid = float(np.linalg.norm(world.midpoint(joints)[:3]
                                 - world.midpoint(Q)[:3]))
check("and turns about the midpoint, which stays where it was",
      moved_mid < 1e-5, "%.4f mm" % (moved_mid * 1000))
_, turned_by = K.pose_distance(world.pose("A", Q["A"]),
                               world.pose("A", joints["A"]))
check("each tool turns by the angle that was asked for",
      abs(turned_by - angle) < 1e-5,
      "%.4f deg out" % np.degrees(turned_by - angle))

pivot = world.tcp_matrix("A", Q["A"])[:3, 3]
about_a = world.turned(Q, "z", angle, pivot=pivot)
stayed = float(np.linalg.norm(about_a["A"][:3, 3] - pivot))
check("a pivot on one tool leaves that tool exactly where it is",
      stayed < 1e-12, "%.4e mm" % (stayed * 1000))


print("\ntwo arms in one posture, seen from the two frames")
# The pair is a mirror image limb for limb, so the joint angles differ by up
# to 180 degrees while the tools face each other across a few centimetres.
# Nothing but the world frame can say that.
spread = float(np.max(np.abs(np.degrees(Q["A"] - Q["B"]))))
check("the two arms' joint angles are nothing alike", spread > 90.0,
      "%.0f deg apart at the widest joint" % spread)
check("while their tools are a workpiece apart", world.separation(Q) < 0.40,
      "%.1f mm" % (world.separation(Q) * 1000))
check("the A -> B transform is the same one either way round",
      float(np.max(np.abs(K.inv(world.relative(Q))
                          - world.relative(Q, arm_ids=["B", "A"])))) < 1e-12)

quick = world_poses(config, Q, tools={a: TOOL for a in ARM_IDS})
check("the one-liner agrees with the object it is built on",
      all(float(np.max(np.abs(quick[a] - world.pose(a, Q[a])))) < 1e-12
          for a in ARM_IDS))

# The other way of building one: from arms rather than from the file, so each
# arm brings its own tool offset and its own factory DH table. Simulated arms
# here, which have both attributes and no socket behind either.
sim_arms = {a: SimArm(config.arms[a]) for a in ARM_IDS}
for a in ARM_IDS:
    sim_arms[a].set_joints(Q[a])
from_arms = WorldCartesian.from_cell(type("StubCell", (), {"arms": sim_arms})())
check("from_cell places a tool where that arm's own pose_world does",
      all(K.pose_distance(from_arms.pose(a, Q[a]),
                          sim_arms[a].base_to_world(sim_arms[a].tcp_pose()))[0]
          < 1e-12 for a in ARM_IDS))

try:
    world.pose("C", Q["A"])
    check("an arm that is not in this cell is named, not KeyErrored", False)
except WorldFrameError as e:
    check("an arm that is not in this cell is named, not KeyErrored",
          "no arm 'C'" in str(e))

print("\n%d failed" % fail)
sys.exit(1 if fail else 0)
