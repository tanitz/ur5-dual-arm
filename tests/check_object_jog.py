"""
A held jog on the real arms: +Z for a second and a half, then back down.

THE ARMS MOVE. Travel is about 30 mm each way, under servo control from one
clock — clear the workspace and keep the E-stop within reach.

What it is actually checking is the gap between the two TCPs. Both grippers
are on one rigid thing, so that number is fixed by physics; every millimetre
it changes while the object is being jogged is a millimetre the two arms are
levering against each other.

    python3 tests/check_object_jog.py
"""

import sys, time
sys.path.insert(0, "/home/jetson/UR5")
import numpy as np
from ur5dual.cell import Cell
from ur5dual.config import CellConfig
from ur5dual.coupling import Coordinator, HeldObject

SPEED = 0.020          # m/s
HOLD = 1.5             # s with the button down

cell = Cell(CellConfig.load()); cell.listeners.append(lambda t: print("   .", t))
cell.connect(("A", "B"))
obj = HeldObject("probe").capture(cell, ("A", "B"), "midpoint")
coord = Coordinator(cell); coord.start(obj)

home = np.array(obj.pose_world)
start_gap = cell.tcp_separation()
print("holding '%s', grippers %.1f mm apart" % (obj.name, start_gap * 1000))


def tcps():
    return {a: cell.arms[a].tcp_matrix_world()[:3, 3].copy() for a in ("A", "B")}


def jog(direction, label):
    """Hold a direction for HOLD seconds, refreshing as the GUI does."""
    worst = 0.0
    was = tcps()                       # where the arms are, not where we asked
    t0 = time.time()
    granted = coord.command_jog(obj, direction, SPEED)
    while time.time() - t0 < HOLD:
        coord.command_jog(obj, direction, SPEED)      # what a held button does
        time.sleep(0.08)                              # M.REFRESH
        worst = max(worst, abs(cell.tcp_separation() - start_gap))
        if coord.error is not None:
            raise SystemExit("servo loop died: %s" % coord.error)
    coord.stop_jog()
    while coord.jogging():                            # it ramps down on its own
        time.sleep(0.02)
    time.sleep(0.3)
    moved = np.array(obj.pose_world) - home
    now = tcps()
    # The commanded figure alone cannot tell a loop that is driving the arms
    # from one that is talking to itself, and an uncalibrated cell caps a held
    # jog at 1 mm/s — which is real motion and looks like none.
    print("%s: asked for %.1f mm/s, granted %.1f mm/s" % (label, SPEED * 1000,
                                                          granted * 1000))
    print("     commanded  %+6.1f %+6.1f %+6.1f mm from where it started"
          % (moved[0] * 1000, moved[1] * 1000, moved[2] * 1000))
    print("     the arms actually moved  %s"
          % "   ".join("%s %.1f mm" % (a, np.linalg.norm(now[a] - was[a]) * 1000)
                       for a in ("A", "B")))
    print("     worst gap change %.2f mm" % (worst * 1000))
    return worst


up = jog([0, 0, 1], "held +Z")
down = jog([0, 0, -1], "held -Z")

back = np.array(obj.pose_world) - home
print("back where it started to %.1f mm" % (np.linalg.norm(back) * 1000))
print("worst arm-to-arm disagreement over the whole run: %.2f mm"
      % (max(up, down) * 1000))
print("feed still alive:", coord.alive, " error:", coord.error)
coord.shutdown(); cell.disconnect()
