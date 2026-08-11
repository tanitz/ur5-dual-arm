"""Does the Python forward kinematics match the arms in this cell?

Everything the closed-chain solver produces rests on three things being right:
the UR5 DH table, the tool offset set on each pendant, and the base transform
in cell.yaml. The first two can be checked exactly, right now, without moving
anything — each controller reports both the joint angles it is sitting at and
the TCP pose those angles produce, so our own FK has a ground truth to be
wrong against.

Run this before letting the solver drive anything. A DH table for the wrong
robot generation shows up here as a couple of centimetres and nowhere else
until two arms are holding a box.

Read-only. Nothing is commanded, no program is uploaded, no arm moves.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ur5dual.geometry import kinematics as K
from ur5dual.geometry import ur_kinematics as UK
from ur5dual.cell import Cell
from ur5dual.config import CellConfig

cell = Cell(CellConfig.load())
cell.connect(("A", "B"))
arms = [a for a in ("A", "B") if cell.arms[a].connected]

print("what each arm is sitting at")
for a in arms:
    st = cell.arms[a].state()
    q = np.array(st["q_actual"], dtype=float)
    print("  arm %s  J1-J6  %s deg" % (a, np.round(np.degrees(q), 1)))
    print("         tool offset  %s"
          % ("none configured" if cell.arms[a].tcp_offset is None
             else np.round(cell.arms[a].tcp_offset, 4)))

print("\nwhich DH table each arm needs")
# The published table describes the design. Each of these two arms was measured
# at the factory and its corrections stored in its own controller, and the
# difference is millimetres — which is what a closed-chain solver refuses over.
for a in arms:
    arm = cell.arms[a]
    published, _ = UK.tcp_disagreement(
        np.array(arm.state()["q_actual"], dtype=float),
        arm.state()["tcp_pose"],
        None if arm.tcp_offset is None else K.pose_to_mat(arm.tcp_offset),
        UK.UR5_DH)
    print("  arm %s  using %s" % (a, arm.dh_source))
    print("         published table would be %.2f mm out; this one is %.2f mm"
          % (published * 1000, (arm.dh_error or 0.0) * 1000))

print("\nour forward kinematics against the controller's own TCP report")
worst = 0.0
for a in arms:
    st = cell.arms[a].state()
    q = np.array(st["q_actual"], dtype=float)
    tool = (None if cell.arms[a].tcp_offset is None
            else K.pose_to_mat(cell.arms[a].tcp_offset))

    mine = UK.fk_tcp(q, tool, cell.arms[a].dh)
    theirs = K.pose_to_mat(np.array(st["tcp_pose"], dtype=float))
    d_p = float(np.linalg.norm(mine[:3, 3] - theirs[:3, 3]))
    d_r = float(np.linalg.norm(K.mat_to_rotvec(mine[:3, :3] @ theirs[:3, :3].T)))
    worst = max(worst, d_p)

    print("  arm %s  we say  %s m" % (a, np.round(mine[:3, 3], 4)))
    print("         it says %s m" % np.round(theirs[:3, 3], 4))
    print("         apart by %.2f mm and %.3f deg   %s"
          % (d_p * 1000, np.degrees(d_r),
             "ok" if d_p < 0.002 else "<-- the DH table or the tool offset is wrong"))

print("\nwhat that means")
if worst < 0.002:
    print("  under 2 mm: the DH tables and the tool offsets are right, so joint")
    print("  targets computed here will land where they are meant to.")
elif worst < 0.02:
    print("  a few millimetres. Either the controller did not send its")
    print("  calibration — the line above says which table each arm ended up")
    print("  with — or the tool offset is wrong: check Installation -> TCP on")
    print("  each pendant. The panel falls back to Cartesian targets meanwhile,")
    print("  which each controller solves with its own calibration; that works,")
    print("  but the path is no longer checked before it starts.")
else:
    print("  centimetres: this is the wrong DH table for these robots. UR5 CB3")
    print("  and UR5e have different link lengths — check ur_type in cell.yaml")
    print("  against the label on the controller.")

print("\nposture, and how much room it leaves")
print("  (this is the branch each arm would be locked onto by an ATTACH now —")
print("   a carry cannot change branch, so a bad posture is fixed before, not after)")
for a in arms:
    q = np.array(cell.arms[a].state()["q_actual"], dtype=float)
    tool = (None if cell.arms[a].tcp_offset is None
            else K.pose_to_mat(cell.arms[a].tcp_offset))
    margin = UK.limit_margin(q)
    sigma = UK.manipulability(q, tool)
    tight = int(np.argmin(np.minimum(q - UK.JOINT_LIMITS[:, 0],
                                     UK.JOINT_LIMITS[:, 1] - q)))
    print("  arm %s  nearest stop is J%d at %.0f deg away   sigma %.4f  %s"
          % (a, tight + 1, np.degrees(margin), sigma,
             "ok" if sigma > UK.SIGMA_MIN_WARN else "<-- near a singularity"))

if len(arms) == 2:
    print("\nwhere the two TCPs are, in the cell frame")
    p = {}
    for a in arms:
        p[a] = cell.arms[a].tcp_matrix_world()[:3, 3]
        print("  arm %s  %s m" % (a, np.round(p[a], 4)))
    print("  %.0f mm apart — this is the grip span an ATTACH would freeze"
          % (np.linalg.norm(p["A"] - p["B"]) * 1000))

cell.disconnect()

print("""
This proves our kinematics matches each arm on its own. It says nothing about
the transform *between* the two bases, which no single-arm measurement can
reach — that still comes from the touch-off on the Cell tab, and until it does
the two arms will disagree about where the box is by however wrong cell.yaml
is, no matter how good the FK is.
""")
