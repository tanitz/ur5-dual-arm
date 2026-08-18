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
from ur5dual.ros.launch_support import xacro_mappings

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

# Measuring the cell says nothing about whether it still has a mast. These
# were once the same flag, so calibrating made the frame vanish from RViz
# with nothing in the log to say why.
print("the drawn frame survives calibration")
mapping = xacro_mappings(cfg)
check("custom style still draws the frame", mapping["show_frame"] == "true",
      "style=%s show_frame=%s" % (cfg.mount["style"], mapping["show_frame"]))
cfg.mount["show_frame"] = False
check("show_frame false turns it off",
      xacro_mappings(cfg)["show_frame"] == "false")
cfg.mount["show_frame"] = True
check("a config predating the flag still draws it",
      xacro_mappings(CellConfig({"mount": {"style": "custom"}}))["show_frame"]
      == "true")

# The mast is welded to the flanges. It used to be drawn from `world`, which
# is the same picture right up until a calibration turns the bases — then the
# structure stood bolt upright with the arms leaning off the ends of it, and
# the one view an operator judges a calibration by was the one thing that
# could not show the calibration.
print("the drawn structure hangs off the flanges")
cfg = CellConfig()
cfg.mount.update({"column_height": 1.20, "spacing": 0.50, "tilt_deg": 135.0,
                  "yaw_deg": 0.0, "style": "pedestal"})
cfg.apply_mount_preset()


def rot_y(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def bar_ends(config):
    """Where the drawn crossbar's two ends land, in world coordinates."""
    F = config.mount_frame_matrix()
    half = config.base_separation() / 2.0
    return F[:3, 3] + F[:3, 1] * half, F[:3, 3] - F[:3, 1] * half


def frame_checks(config, label):
    F = config.mount_frame_matrix()
    end_a, end_b = bar_ends(config)
    check("%s: bar ends sit on both flanges" % label,
          np.allclose(end_a, config.arms["A"].xyz, atol=1e-9)
          and np.allclose(end_b, config.arms["B"].xyz, atol=1e-9),
          "A off by %.3g mm" % (np.linalg.norm(end_a - config.arms["A"].xyz) * 1e3))
    check("%s: mast square to the bar" % label,
          abs(float(F[:3, 2] @ F[:3, 1])) < 1e-9)
    check("%s: mast drawn plumb" % label,
          np.allclose(F[:3, 2], [0.0, 0.0, 1.0], atol=1e-9),
          str(np.round(F[:3, 2], 4)))
    return F


F = frame_checks(cfg, "level cell")
check("level cell: crossbar in the middle, at column height",
      np.allclose(F[:3, 3], [0.0, 0.0, 1.20], atol=1e-9), str(np.round(F[:3, 3], 4)))

# A twisted crossbar, an arm bolted round on its flange, a pad machined off
# angle: the brackets are wrong and the column is not. Reading the mast back
# out of the pads would draw this as a leaning cell — the picture that had
# this cell chasing a lean it does not have.
before = {a: cfg.arms[a].xyz.copy() for a in ("A", "B")}
for arm_id in ("A", "B"):
    T = cfg.arms[arm_id].base_matrix()
    T[:3, :3] = rot_y(-6.0) @ T[:3, :3]
    cfg.arms[arm_id].set_base_matrix(T)
cfg.set_custom_mount()

F = frame_checks(cfg, "brackets 6 deg out")
check("brackets 6 deg out: the column has not moved",
      all(np.allclose(cfg.arms[a].xyz, before[a], atol=1e-9) for a in ("A", "B")))
# a 6 deg twist of the bar swings a pad by less than 6 deg: only the part of
# its face square to the bar's axis moves
t = math.radians(cfg.tilt_of("A"))
square = np.array([0.0, math.sin(t), math.cos(t)])
expect = math.degrees(math.acos(float(square @ (rot_y(-6.0) @ square))))
off = math.degrees(math.acos(min(1.0, float(F[:3, :3] @ square
                                            @ cfg.arms["A"].base_matrix()[:3, 2]))))
check("brackets 6 deg out: the error shows on the pads instead",
      abs(off - expect) < 1e-4,   # set_base_matrix rounds the rpy it stores
      "pad A sits %.3f deg off square to the bar" % off)

# a cell whose column really is out of plumb: apply_level(tipped=True) turns
# the flanges about the foot as well, and the bar tips off level with them
for arm_id in ("A", "B"):
    T = np.eye(4)
    T[:3, :3] = rot_y(-6.0)
    cfg.arms[arm_id].set_base_matrix(T @ cfg.arms[arm_id].base_matrix())
check("a cell turned about its foot tips the bar, not just the pads",
      abs(cfg.mount_frame_matrix()[2, 3] - 1.20) > 1e-4
      or abs(cfg.mount_frame_matrix()[0, 3]) > 1e-4,
      "crossbar middle %s" % np.round(cfg.mount_frame_matrix()[:3, 3], 4))

mapping = xacro_mappings(cfg)
check("the frame reaches the description",
      "frame_xyz" in mapping and "frame_rpy" in mapping, str(sorted(mapping)))
check("the drawn bar is as long as the flanges are apart",
      abs(float(mapping["bar_length"]) - cfg.base_separation()) < 1e-6,
      "%s m drawn, %.6f m measured, mount.spacing says %.3f"
      % (mapping["bar_length"], cfg.base_separation(), cfg.mount["spacing"]))

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
