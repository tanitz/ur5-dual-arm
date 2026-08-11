"""
Which way is up, according to the robots rather than to cell.yaml.

NOTHING MOVES. Both arms are read standing still; the flange accelerometers
feel gravity and that is the whole measurement. Safe to run at any time, with
or without something in the grippers.

It answers the two questions a wrong-direction world jog raises, and they have
different fixes:

  pair error    do the two arms agree with each other about which way is up?
                They can only disagree if the transform between them is wrong,
                which is what the touch-off wizard and the direction teach
                measure. Big number here = re-run one of those.

  world error   does the pair as a whole hang the way cell.yaml says it does?
                Nothing in this project has ever measured that — arm A's
                orientation is typed in as `mount.tilt_deg`. Big number here =
                every world jog leaves at that angle, both arms in perfect
                agreement, and nothing feels wrong to the robots at all.

    python3 tests/check_up_online.py            # report only
    python3 tests/check_up_online.py --apply    # and write the correction
"""

import sys
sys.path.insert(0, "/home/jetson/UR5")
import numpy as np
from ur5dual.geometry import calibration as C
from ur5dual.cell import Cell
from ur5dual.config import CellConfig

np.set_printoptions(precision=3, suppress=True)
apply = "--apply" in sys.argv

cell = Cell(CellConfig.load()); cell.listeners.append(lambda t: print("   .", t))
cell.connect(("A", "B"))
if len(cell.connected_ids) < 2:
    print("both arms need to be connected; got %s" % cell.connected_ids)

try:
    r = C.level_report(cell)
except C.CalibrationError as e:
    raise SystemExit("cannot measure: %s" % e)

print()
print("measured, per arm")
for a in r["arms"]:
    print("  arm %s  raw accelerometer %s  (should be ~9.81 long)"
          % (a, np.array(cell.arms[a].state()["tool_accel"])))
    print("         up in its own base %s" % r["up_in_base"][a])
    print("         base tilt   measured %6.1f deg   cell.yaml says %6.1f deg"
          % (r["tilt_deg"][a], r["config_tilt_deg"][a]))
print()
print("  Cross-check the measured tilt against the pendant: Installation ->")
print("  General -> Mounting shows the same angle, and the robot's own gravity")
print("  compensation already depends on it. If the measured figure comes out")
print("  as 180 minus the pendant's, this firmware reports the accelerometer")
print("  the other way up and the sign in arm.up_in_base needs flipping —")
print("  say so and nothing else in the reading changes.")

print()
if r["pair_error_deg"] is None:
    print("pair error   — (needs both arms)")
else:
    print("pair error   %.2f deg   arm A and arm B agreeing about up"
          % r["pair_error_deg"])
print("world error  %.2f deg   configured world +Z away from real up"
      % r["world_error_deg"])
print("             real up sits at %s in the world frame as configured"
      % r["measured_up_world"])

for w in r["warnings"]:
    print()
    print("  !! " + w)

print()
print("so a +Z jog today lifts the object along a line %.1f deg off vertical, "
      % r["world_error_deg"])
print("and +X and +Y are tipped out of horizontal by the same amount.")

if not apply:
    print()
    print("nothing was changed. re-run with --apply to write the correction")
    cell.disconnect()
    raise SystemExit(0)

cfg = cell.config
before = {a: cfg.arms[a].rpy.copy() for a in ("A", "B")}
C.apply_level(cfg, r)
cfg.save()
print()
print("applied — both arms turned by the same %.2f deg, so the transform "
      "between them is untouched" % r["world_error_deg"])
for a in ("A", "B"):
    print("  arm %s rpy %s -> %s deg"
          % (a, np.degrees(before[a]), np.degrees(cfg.arms[a].rpy)))
print("saved to %s" % cfg.path)
cell.disconnect()
