"""
One real coordinated move: 5 mm up and straight back down, both arms.

THE ARMS MOVE. Total travel is 5 mm, but they move together under servo
control — clear the workspace and keep the E-stop within reach.

    python3 tests/check_object_move.py
"""

import sys, time
sys.path.insert(0, "/home/jetson/UR5")
import numpy as np
from ur5dual.cell import Cell
from ur5dual.config import CellConfig
from ur5dual.coupling import Coordinator, HeldObject

cell = Cell(CellConfig.load()); cell.listeners.append(lambda t: print("   .", t))
cell.connect(("A", "B"))
obj = HeldObject("probe").capture(cell, ("A", "B"), "midpoint")
coord = Coordinator(cell); coord.start(obj)

print("feed alive:", coord.alive)
time.sleep(1.0)
print("after 1 s idle — alive:", coord.alive, " error:", coord.error)

start_gap = cell.tcp_separation()
home = np.array(obj.pose_world)
target = home.copy(); target[2] += 0.005          # 5 mm up

t0 = time.time()
coord.command_move(obj, target)
print("command returned in %.1f ms (must not block)" % ((time.time()-t0)*1000))
coord.wait_done()
up_gap = cell.tcp_separation()
print("at target: gap %.1f mm (was %.1f)" % (up_gap*1000, start_gap*1000))

coord.move_object(obj, home)
time.sleep(0.4)
back_gap = cell.tcp_separation()
print("back home: gap %.1f mm" % (back_gap*1000))
print("feed still alive:", coord.alive, " error:", coord.error)
print("gap change over the whole run: %.2f mm" % (abs(back_gap-start_gap)*1000))
coord.shutdown(); cell.disconnect()
