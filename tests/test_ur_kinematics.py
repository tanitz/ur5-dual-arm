"""UR5 forward kinematics, Jacobian and IK — arithmetic only, no robot.

Nothing here talks to a controller. What it cannot prove is that the DH table
matches the arms in this cell; only `check_chain_online.py` can do that, by
comparing this FK against what the robots report. These checks prove the
maths is self-consistent, which has to hold first.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.geometry import kinematics as K
from ur5dual.geometry import ur_kinematics as UK

rng = np.random.default_rng(11)
fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


def random_q(n=1):
    """Joint sets away from the obvious singularities, so a Jacobian test is
    measuring the formula rather than the conditioning."""
    out = []
    while len(out) < n:
        q = rng.uniform(-np.pi, np.pi, size=6)
        if UK.manipulability(q) > 0.02:
            out.append(q)
    return out


TOOL = K.pose_to_mat([0.0, 0.0, 0.15, 0.0, 0.0, 0.3])   # a plausible gripper


print("forward kinematics")
home = np.zeros(6)
T = UK.fk(home)
# with every joint at zero the UR5 stretches out along its own x, and the
# flange sits d1 up, a2+a3 out, and d4-d6 across — straight off the DH table
expected = np.array([UK.UR5_DH["a"][1] + UK.UR5_DH["a"][2],
                     -UK.UR5_DH["d"][3] - UK.UR5_DH["d"][5],
                     UK.UR5_DH["d"][0] - UK.UR5_DH["d"][4]])
check("zero pose matches the DH table by hand",
      np.allclose(T[:3, 3], expected, atol=1e-9),
      "%s vs %s" % (np.round(T[:3, 3], 6), np.round(expected, 6)))
check("zero pose is a proper rotation",
      abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-12)

for q in random_q(50):
    R = UK.fk(q)[:3, :3]
    if abs(np.linalg.det(R) - 1.0) > 1e-9 or not np.allclose(R @ R.T, np.eye(3), atol=1e-9):
        check("random pose stays orthonormal", False, str(q))
        break
else:
    check("random pose stays orthonormal (50x)", True)

check("tool offset moves the TCP by exactly the tool",
      np.allclose(UK.fk_tcp(home, TOOL), UK.fk(home) @ TOOL, atol=1e-12))


print("\njacobian against numerical differentiation")
h = 1e-6
worst = 0.0
for q in random_q(25):
    J = UK.jacobian(q, TOOL)
    J_num = np.zeros((6, 6))
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = h
        T_plus = UK.fk_tcp(q + dq, TOOL)
        T_minus = UK.fk_tcp(q - dq, TOOL)
        J_num[:3, i] = (T_plus[:3, 3] - T_minus[:3, 3]) / (2 * h)
        J_num[3:, i] = K.mat_to_rotvec(T_plus[:3, :3] @ T_minus[:3, :3].T) / (2 * h)
    worst = max(worst, float(np.max(np.abs(J - J_num))))
check("analytic == numerical (25 random poses)", worst < 1e-5,
      "worst column error %.2e" % worst)


print("\ninverse kinematics")
solved = 0
for q in random_q(60):
    T_target = UK.fk_tcp(q, TOOL)
    # a seed a few degrees away, the way a servo loop always has one
    seed = q + rng.normal(scale=0.05, size=6)
    q_out, info = UK.ik(T_target, seed, TOOL)
    if not info["converged"]:
        check("nearby seed converges", False, "from %s" % np.round(seed, 3))
        break
    if np.max(np.abs(UK.fk_tcp(q_out, TOOL) - T_target)) > 1e-5:
        check("solution reproduces the target pose", False)
        break
    solved += 1
else:
    check("nearby seed converges and reproduces the pose (60x)", solved == 60)

# The claim that earns this module its place: a seed on one branch never comes
# back on another. Perturb hard enough to be a real disturbance, not hard
# enough to be a different posture, and the answer must stay put.
stayed = 0
for q in random_q(40):
    T_target = UK.fk_tcp(q, TOOL)
    seed = q + rng.normal(scale=0.15, size=6)
    q_out, info = UK.ik(T_target, seed, TOOL)
    if info["converged"] and np.max(np.abs(q_out - q)) < 0.5:
        stayed += 1
check("a seeded solve stays on the seed's branch (40x)", stayed >= 38,
      "%d/40" % stayed)

far = UK.fk_tcp(np.zeros(6), TOOL).copy()
far[:3, 3] += np.array([5.0, 0.0, 0.0])          # a metre-scale reach away
_, info = UK.ik(far, np.zeros(6), TOOL, max_iter=60)
check("an unreachable pose reports failure rather than a wrong answer",
      not info["converged"], "residual %.0f mm" % (info["pos_error"] * 1000))


print("\njoint bookkeeping")
q = np.array([0.1, -1.0, 1.2, -0.4, 1.5, 6.0])
check("nearest_turn removes a needless full revolution",
      np.allclose(UK.nearest_turn(q, np.zeros(6))[5], 6.0 - 2 * np.pi, atol=1e-12))
check("nearest_turn leaves an already-near joint alone",
      np.allclose(UK.nearest_turn(q, q), q, atol=1e-12))
check("nearest_turn does not change where the tool is",
      np.allclose(UK.fk(UK.nearest_turn(q, np.zeros(6))), UK.fk(q), atol=1e-9))

over = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 7.0])
check("within_limits catches a joint past its stop",
      [j for j, *_ in UK.within_limits(over)] == [5])
check("within_limits passes a legal pose", UK.within_limits(q) == [])
check("limit_margin shrinks as a joint approaches its stop",
      UK.limit_margin(np.array([0.0] * 6)) > UK.limit_margin(over * 0.85))


print("\nsingularity measure")
# wrist singularity: joint 5 at zero lines up joints 4 and 6
check("q5 = 0 is detected as near-singular",
      UK.manipulability(np.array([0.0, -1.0, 1.0, -0.5, 0.0, 0.0])) < UK.SIGMA_MIN_WARN)
check("a normal posture is not",
      UK.manipulability(np.array([0.0, -1.2, 1.4, -1.0, -1.5, 0.0])) > UK.SIGMA_MIN_WARN)

print("\nthe table that describes the machine, not the drawing")
# Every UR is measured at the factory and its corrections kept in its own
# controller. Those corrections are millimetres, which is nothing for one arm
# working from its own reported poses and everything for two arms whose joint
# targets are computed here — it is what made the closed-chain solver refuse
# arm B, correctly, when it was handed the published table.
deltas = {"theta": [1e-3, -2e-4, 5e-4, 0.0, 3e-4, 0.0],
          "a": [1e-4, -3e-4, 2e-4, 0.0, 0.0, 0.0],
          "d": [-2e-4, 0.0, 0.0, 1e-4, 5e-5, 0.0],
          "alpha": [2e-4, -1e-4, 0.0, 0.0, 0.0, 0.0]}
real = UK.with_corrections(deltas)
check("corrections are added to the published table, not swapped in",
      np.allclose(real["a"], UK.UR5_DH["a"] + np.array(deltas["a"])))
check("the published table has no joint-angle offsets of its own",
      np.allclose(UK.UR5_DH["theta"], 0.0))

q = np.array([0.3, -1.1, 1.3, -0.9, -1.4, 0.6])
check("a corrected table moves the tool by about the size of the corrections",
      1e-4 < np.linalg.norm(UK.fk(q, real)[:3, 3] - UK.fk(q)[:3, 3]) < 0.01,
      "%.2f mm" % (np.linalg.norm(UK.fk(q, real)[:3, 3]
                                  - UK.fk(q)[:3, 3]) * 1000))
check("a theta offset turns that joint and nothing else",
      np.allclose(UK.fk(q, UK.with_corrections(
          {"theta": [0.0, 0.0, 0.0, 0.0, 0.0, 0.2]})),
          UK.fk(q + np.array([0, 0, 0, 0, 0, 0.2])), atol=1e-12))

# The referee is the robot: whichever table reproduces the pose it reports for
# the joints it is sitting at wins. So which convention a firmware uses for its
# calibration is measured rather than assumed.
reported = K.mat_to_pose(UK.fk_tcp(q, None, real))
picked, name, err = UK.choose_dh(q, reported, None, deltas)
check("a firmware that did send corrections would still be read right",
      err < 1e-9 and "as corrections" in name, "%.4f mm  %s" % (err * 1000, name))
picked, name, err = UK.choose_dh(q, K.mat_to_pose(UK.fk_tcp(q)), None, deltas)
check("given one built to the published table, the published one is picked",
      err < 1e-9 and "published" in name, "%.4f mm  %s" % (err * 1000, name))
check("with no calibration to read it falls back to the published table",
      UK.choose_dh(q, reported, None, None)[1] == "the published UR5 table")
d_p, d_r = UK.tcp_disagreement(q, reported, None, UK.UR5_DH)
check("and the disagreement it is choosing on is the one the panel reports",
      d_p > 1e-4 and d_r >= 0.0, "%.2f mm  %.2f deg" % (d_p * 1000,
                                                        np.degrees(d_r)))

print("\narm B's real calibration, as its controller sends it")
# Copied off 192.168.250.30 by tests/check_primary_calibration.py, with the
# tool offset and joint angles it was sitting at, and the TCP its controller
# reported for them. Nothing here is invented: this is the case the published
# table gets wrong by 3.18 mm, kept so it cannot quietly come back.
ARM_B = {
    "theta": [5.93575604e-07, 5.74269825e-02, -5.17280307e-02,
              -5.70541078e-03, 2.60735664e-05, 4.05833587e-05],
    "a": [2.12820159e-04, -4.24333853e-01, -3.92225924e-01,
          4.91181201e-05, 1.30167257e-04, 0.0],
    "d": [0.08931901, -54.62308586, 54.47576641, 0.25776813,
          0.09484562, 0.08251619],
    "alpha": [1.57056496, -4.46709056e-04, -1.50992708e-02,
              1.57061304, -1.57057276, 0.0],
}
ARM_B_Q = np.radians([-23.22, -107.36, -108.03, -45.77, 62.90, 7.55])
ARM_B_TOOL = K.pose_to_mat([0, 0, 0.2, 0, 0, 0])
ARM_B_TCP = np.array([0.4369, -0.4456, 0.0311])     # what the controller says

real = UK.as_dh(ARM_B)
here = UK.fk_tcp(ARM_B_Q, ARM_B_TOOL, real)[:3, 3]
published = UK.fk_tcp(ARM_B_Q, ARM_B_TOOL)[:3, 3]
check("the calibration table reproduces the arm the controller describes",
      np.linalg.norm(here - ARM_B_TCP) < 3e-4,
      "%.2f mm" % (np.linalg.norm(here - ARM_B_TCP) * 1000))
check("the published table is out by the 3.18 mm that was actually measured",
      abs(np.linalg.norm(published - ARM_B_TCP) - 0.00318) < 3e-4,
      "%.2f mm" % (np.linalg.norm(published - ARM_B_TCP) * 1000))

reported = K.mat_to_pose(UK.fk_tcp(ARM_B_Q, ARM_B_TOOL, real))
picked, name, err = UK.choose_dh(ARM_B_Q, reported, ARM_B_TOOL,
                                 dict(ARM_B, status=1))
check("choose_dh reads it as a finished table, not as corrections",
      err < 1e-9 and name == "this robot's own calibration",
      "%.4f mm  %s" % (err * 1000, name))

# Arm A: status 0, and what it sends back is the published table verbatim. It
# then wins on an exact tie, and calling that "this robot's own calibration"
# would dress up the arm nobody has measured as the well-measured one.
echoed = dict(UK.as_dh(UK.UR5_DH), status=0)
_, name, err = UK.choose_dh(q, K.mat_to_pose(UK.fk_tcp(q)), None, echoed)
check("a controller holding no calibration is not credited with one",
      err < 1e-12 and "all this controller holds" in name, name)

# d2 and d3 come back as tens of metres: a DH chain cannot express a small
# change in the angle between two parallel joints, so the fit escapes into
# offsets that cancel. Everything downstream is derived from the frames as
# given and stays exact — which is the claim being pinned here, because a
# plausibility check on these numbers would throw the right table away.
origins = [float(np.linalg.norm(T[:3, 3])) for T in UK.link_frames(ARM_B_Q, real)]
check("joint 2's frame really does sit tens of metres from the base",
      origins[2] > 50.0, "%.1f m" % origins[2])
check("and the tool still lands inside a UR5's reach",
      origins[-1] < 1.0, "%.2f m" % origins[-1])
check("the Jacobian is unharmed by it",
      abs(UK.manipulability(ARM_B_Q, ARM_B_TOOL, real)
          - UK.manipulability(ARM_B_Q, ARM_B_TOOL)) < 0.01,
      "sigma %.4f vs %.4f" % (UK.manipulability(ARM_B_Q, ARM_B_TOOL, real),
                              UK.manipulability(ARM_B_Q, ARM_B_TOOL)))
solved, info = UK.ik(UK.fk_tcp(ARM_B_Q, ARM_B_TOOL, real),
                     ARM_B_Q + np.radians([1, -1, 1, -1, 1, -1]),
                     ARM_B_TOOL, real)
check("and IK converges on it from a degree away, back to the same branch",
      info["converged"] and np.max(np.abs(solved - ARM_B_Q)) < 1e-6,
      "%.1e m residual" % info["pos_error"])

print("\n%d failed" % fail)
sys.exit(1 if fail else 0)
