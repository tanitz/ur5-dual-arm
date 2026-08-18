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

import math
import sys
sys.path.insert(0, "/home/jetson/UR5")
import numpy as np
from ur5dual.geometry import calibration as C
from ur5dual.cell import Cell
from ur5dual.config import CellConfig

np.set_printoptions(precision=3, suppress=True)
tipped = "--apply-tipped" in sys.argv
apply = tipped or "--apply" in sys.argv

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
print("the same reading, if instead the mast is plumb and the brackets are not")
for a in r["arms"]:
    tilt, spin = r["bracket_fit"][a]
    print("  arm %s  mount.tilt_deg_%s %6.2f   mount.rotate_deg_%s %+6.2f"
          % (a, a, tilt, a, spin))
print("  Gravity fixes two of the three angles each base has, and those two")
print("  numbers spend them on this arm's own bracket instead of on the world.")
print("  It fits the measurement exactly as well as the correction below does.")
print("  Note what each keeps: one shared rotation leaves the A-to-B transform")
print("  the touch-off measured untouched, per-arm bracket numbers rewrite it.")

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

# Applying is a claim about the building, so print the claim in millimetres a
# plumb line can check. Applied to a cell whose mast is in fact upright, the
# correction levels the world onto whatever the brackets and the sensor are
# doing, and every world jog afterwards carries that error instead.
lean = -np.array(r["measured_up_world"], dtype=float)[:2]
if np.linalg.norm(lean) > 1e-9:
    lean = lean / np.linalg.norm(lean) * math.sin(
        math.radians(r["world_error_deg"])) * float(cell.config.mount["column_height"])
    print()
    print("--apply-tipped would claim the mast is out of plumb by %.2f deg,"
          % r["world_error_deg"])
    print("its top standing %s mm from the plumb line through its foot in"
          % np.round(lean * 1000.0, 0))
    print("world XY, and would move both flanges there. Put a level on the")
    print("mast before choosing that one. On a mast that is plumb, --apply")
    print("turns the brackets instead and leaves the flanges alone.")

if not apply:
    print()
    print("nothing was changed. re-run with --apply (plumb mast, turn the "
          "brackets)")
    print("or --apply-tipped (the mast itself is out of plumb, turn the cell)")
    cell.disconnect()
    raise SystemExit(0)

cfg = cell.config
before = {a: cfg.arms[a].base_matrix() for a in ("A", "B")}
a_to_b_before = cfg.a_to_b()
C.apply_level(cfg, r, tipped=tipped)
cfg.save()
print()
print("applied — both arms turned by the same %.2f deg, about %s"
      % (r["world_error_deg"],
         "the cell's foot: the flanges moved with them"
         if tipped else "their own flanges, which have not moved"))
for a in ("A", "B"):
    print("  arm %s rpy %s -> %s deg"
          % (a, np.degrees(C.mat_to_rpy(before[a][:3, :3])),
             np.degrees(cfg.arms[a].rpy)))
    if tipped:
        print("         xyz %s -> %s m" % (before[a][:3, 3], cfg.arms[a].xyz))

# The pair is what two-arm work is built on, so say what happened to it either
# way rather than leaving it to be discovered by a workpiece being pulled out
# of a gripper.
drift = np.linalg.norm(cfg.a_to_b()[:3, 3] - a_to_b_before[:3, 3]) * 1000.0
print()
if drift < 1e-6:
    print("the A-to-B transform is untouched, translation included")
else:
    print("arm B now sits %.1f mm from where arm A had it — turning the "
          "brackets in place" % drift)
    print("moves the pair unless the twist is exactly about the crossbar. "
          "Re-run the touch-off")
    print("if this cell is calibrated for two-arm work (it says calibrated=%s)."
          % cfg.calibrated)
print("saved to %s" % cfg.path)
cell.disconnect()
