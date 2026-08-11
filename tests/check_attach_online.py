"""
The ATTACH path, exercised against both real arms without moving them.

This is everything the Object tab's ATTACH button does — capture the grip,
bring up a servo backend on each arm, then run the coordinated loop — except
that the "move" is to the object's *current* pose. Both arms are therefore
commanded to stay exactly where they already are, at the full 125 Hz, with
the force and drift guards live.

That makes it the honest rehearsal: if the servo loop cannot hold station, it
certainly cannot carry a bottle, and finding out costs nothing here.

    python3 tests/check_attach_online.py

THE ARMS ARE UNDER SERVO CONTROL while this runs. They are commanded to hold
position, but keep the E-stop within reach.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.cell import Cell
from ur5dual.config import ARM_IDS, CellConfig
from ur5dual.coupling import Coordinator, CouplingError, HeldObject

HOLD_TOLERANCE = 0.003          # m; station-keeping should be far tighter
fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=3.0,
                    help="how long to hold station")
args = parser.parse_args()

config = CellConfig.load()
cell = Cell(config)
cell.listeners.append(lambda text: print("       . %s" % text))

print("connecting")
cell.connect(ARM_IDS)
online = cell.connected_ids
check("both arms online", len(online) == 2, ", ".join(online) or "none")
if len(online) < 2:
    sys.exit(1)
check("both arms ready", cell.ready(), cell.not_ready_reason() or "RUNNING / NORMAL")

print("capturing the grip (pure geometry, nothing sent)")
obj = HeldObject("rehearsal")
try:
    obj.capture(cell, ("A", "B"), origin="midpoint")
except CouplingError as e:
    check("capture", False, str(e))
    sys.exit(1)
check("object frame captured", obj.held)
check("grip span is a real distance", obj.span() and obj.span() > 0,
      "%.1f mm" % (obj.span() * 1000))
for arm_id in ("A", "B"):
    regenerated = obj.tcp_world(arm_id)
    err = float(np.linalg.norm(regenerated - cell.arms[arm_id].tcp_matrix_world()))
    # not 1e-9: the feed is live, so a fresh sample can arrive between the
    # capture and this line and the arm will have twitched a micron
    check("arm %s grasp reproduces its TCP" % arm_id, err < 5e-4, "%.2e m" % err)

print("starting the servo backends (%s)" % config.motion["backend"])
coordinator = Coordinator(cell)
try:
    coordinator.start(obj)
except Exception as e:
    check("backends up", False, "%s: %s" % (type(e).__name__, str(e)[:100]))
    cell.disconnect()
    sys.exit(1)
check("backends up on both arms", len(coordinator.backends) == 2)

print("letting the continuous feed hold station for %.1f s" % args.seconds)
start_tcp = {a: cell.arms[a].tcp_matrix_world()[:3, 3].copy() for a in obj.arm_ids}
worst = {a: 0.0 for a in obj.arm_ids}
# No move is commanded: the feed thread sends the object's current pose every
# cycle on its own. Surviving an idle stretch is the whole point — the old
# design went quiet between moves and the robot-side script eventually
# dropped the connection, which is what "broken pipe" was.
deadline = time.time() + args.seconds
while time.time() < deadline:
    for a in obj.arm_ids:
        worst[a] = max(worst[a], float(np.linalg.norm(
            cell.arms[a].tcp_matrix_world()[:3, 3] - start_tcp[a])))
    time.sleep(0.05)

check("feed survived the idle stretch", coordinator.alive,
      "" if coordinator.error is None else str(coordinator.error))
check("no error was recorded", coordinator.error is None,
      str(coordinator.error) if coordinator.error else "")
for a in obj.arm_ids:
    check("arm %s held station" % a, worst[a] < HOLD_TOLERANCE,
          "worst drift %.2f mm" % (worst[a] * 1000))

separation_now = cell.tcp_separation()
check("grip separation unchanged",
      abs(separation_now - obj.span()) < HOLD_TOLERANCE,
      "%.1f mm vs %.1f mm captured" % (separation_now * 1000, obj.span() * 1000))

coordinator.shutdown()
cell.halt()
cell.disconnect()
print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
