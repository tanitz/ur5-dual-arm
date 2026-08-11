"""
The mounting preset, and the sign error it is easy to make.

The cell is a mast with a horizontal crossbar and one arm hanging off each
end. Both arms must reach *away* from the mast. Getting the roll sign
backwards tips them inward instead — and every symmetric check (the angle
between the two base axes, the separation, the heights) still passes, because
inward-inward is just as symmetric as outward-outward. Only a test that knows
which way is out can catch it.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.config import CellConfig

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


cfg = CellConfig()
cfg.mount.update({"column_height": 1.20, "spacing": 0.50, "tilt_deg": 120.0,
                  "yaw_deg": 0.0, "style": "pedestal"})
cfg.apply_mount_preset()

print("crossbar placement")
check("arm A on the +Y end", cfg.arms["A"].xyz[1] > 0)
check("arm B on the -Y end", cfg.arms["B"].xyz[1] < 0)
check("spacing is the flange-to-flange distance",
      abs(cfg.base_separation() - 0.50) < 1e-9, "%.4f m" % cfg.base_separation())
check("both at the crossbar height",
      abs(cfg.arms["A"].xyz[2] - 1.20) < 1e-9
      and abs(cfg.arms["B"].xyz[2] - 1.20) < 1e-9)

print("the arms reach outward, not into the mast")
for arm_id in ("A", "B"):
    out = cfg.base_axis_outward(arm_id)
    check("arm %s base Z points away from the column" % arm_id, out > 0,
          "outward component %+.3f" % out)
check("outward component matches sin(tilt)",
      abs(cfg.base_axis_outward("A") - math.sin(math.radians(120))) < 1e-7,
      "%+.6f vs %+.6f" % (cfg.base_axis_outward("A"),
                          math.sin(math.radians(120))))

print("tilt behaves at the ends of its range")
for tilt, expect_z, label in ((0.0, +1.0, "flange up, arm stands off a table"),
                              (90.0, 0.0, "flange vertical, arm reaches sideways"),
                              (180.0, -1.0, "arm hangs upside down")):
    cfg.mount["tilt_deg"] = tilt
    cfg.apply_mount_preset()
    z_up = cfg.arms["A"].base_matrix()[2, 2]
    check("tilt %3.0f deg: %s" % (tilt, label), abs(z_up - expect_z) < 1e-7,
          "base Z vertical component %+.3f" % z_up)

print("the pair yaws about the mast as one")
cfg.mount.update({"tilt_deg": 120.0, "yaw_deg": 90.0})
cfg.apply_mount_preset()
check("yaw 90 deg swings the +Y end of the bar onto -X",
      abs(cfg.arms["A"].xyz[0] + 0.25) < 1e-6
      and abs(cfg.arms["A"].xyz[1]) < 1e-6,
      str(np.round(cfg.arms["A"].xyz, 4)))
check("still reaching outward after a yaw",
      cfg.base_axis_outward("A") > 0.8,
      "%+.3f" % cfg.base_axis_outward("A"))
check("separation unchanged by yaw", abs(cfg.base_separation() - 0.50) < 1e-6)

print("a measured cell is not overwritten by the preset")
cfg.set_custom_mount()
before = cfg.arms["B"].base_matrix().copy()
cfg.mount["spacing"] = 1.5
check("apply_mount_preset is a no-op in custom style",
      cfg.apply_mount_preset() is False
      and np.allclose(cfg.arms["B"].base_matrix(), before))

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
