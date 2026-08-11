"""Is each arm's configured base orientation the one it actually has?

Nothing computed from cell.yaml can answer that: every world-frame number in
this project is produced by the very transform under suspicion, so a wrong
transform stays self-consistent and every check passes.

An independent reference is needed, and there is one on the wire already. The
TCP force these arms report while holding nothing is dominated by the weight
of a tool their payload setting does not know about — a constant pull that
points *down* in the real world, expressed in each arm's own base frame. Two
arms therefore give two independent measurements of where down is, and the
configuration has to agree with both.

Read-only. Nothing is commanded.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ur5dual.cell import Cell
from ur5dual.config import CellConfig

cell = Cell(CellConfig.load())
cell.connect(("A", "B"))

print("what the configuration claims")
down_world = np.array([0.0, 0.0, -1.0])
claimed, measured = {}, {}
for a in ("A", "B"):
    R = cell.arms[a].base_matrix()[:3, :3]
    claimed[a] = R.T @ down_world          # world down, in this arm's base frame
    print("  arm %s  down should be  %s  in its own base frame"
          % (a, np.round(claimed[a], 3)))

print("\nwhat the arms report (force offset while holding nothing)")
for a in ("A", "B"):
    f = np.zeros(3)
    for _ in range(50):
        f += cell.force_vector(a)
    f /= 50.0
    n = np.linalg.norm(f)
    measured[a] = f / n if n > 1e-6 else f
    print("  arm %s  |F| %5.1f N  pulls towards %s" % (a, n, np.round(measured[a], 3)))

print("\nagreement")
worst = 0.0
for a in ("A", "B"):
    cos = float(np.clip(claimed[a] @ measured[a], -1, 1))
    angle = np.degrees(np.arccos(cos))
    worst = max(worst, angle)
    print("  arm %s  configured 'down' and measured pull differ by %6.1f deg  %s"
          % (a, angle, "ok" if angle < 25 else "<-- the base orientation is wrong"))

print("\nwhat this means for a world +Z object jog")
up_world = np.array([0.0, 0.0, 1.0])
for a in ("A", "B"):
    R = cell.arms[a].base_matrix()[:3, :3]
    commanded = R.T @ up_world            # the direction the arm is told to go
    real_up = -measured[a]                # the direction that is actually up
    cos = float(np.clip(commanded @ real_up, -1, 1))
    print("  arm %s  told to move %s, which is %5.1f deg from true up"
                    % (a, np.round(commanded, 3), np.degrees(np.arccos(cos))))

print("\nand relative to each other")
# the angle between the two arms' real 'up' directions, mapped into the world
# through the configured transforms: if the config were right this would be 0
ups = {}
for a in ("A", "B"):
    ups[a] = cell.arms[a].base_matrix()[:3, :3] @ (-measured[a])
cos = float(np.clip(ups["A"] @ ups["B"], -1, 1))
print("  the two arms disagree about which way is up by %.1f deg"
      % np.degrees(np.arccos(cos)))
print("  (0 deg would mean both base transforms are consistent with gravity;")
print("   this is the number that decides whether +Z sends them the same way)")

cell.disconnect()

print("""
Caveat: the force offset is not pure gravity. Joint friction and error in the
arm's own mass model ride along with it and both change with pose, so treat
these angles as an indicator, not a measurement. A disagreement of a few
degrees means little; a hundred means the transforms are wrong.

The numbers that fix it come from two places, neither of which this can reach:
  the pendant   Installation -> General -> Mounting gives each arm's tilt and
                rotate against gravity, which is what makes world +Z mean up
  the wizard    the Cell tab's touch-off gives arm A -> arm B exactly, which
                is what keeps two arms from fighting over one object
""")
