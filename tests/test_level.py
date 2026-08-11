"""
Levelling the world frame against gravity, checked without a robot.

A fake arm reports the "up" a real accelerometer would report if the cell were
hung the way the test says it is hung. The config is then given something
different on purpose, which is the whole situation this code exists for: the
arms know where up is, cell.yaml only claims to.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.geometry import calibration as C
from ur5dual.geometry import kinematics as K
from ur5dual.cell import Cell
from ur5dual.config import CellConfig

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


class FakeArm:
    """An arm bolted on at `R_true`, reading gravity honestly."""

    def __init__(self, arm_cfg, R_true):
        self.cfg = arm_cfg
        self.id = arm_cfg.id
        self.connected = True
        self.R_true = np.asarray(R_true, dtype=float)

    def base_matrix(self):
        return self.cfg.base_matrix()          # what the config *claims*

    def up_in_base(self):
        return self.R_true.T @ np.array([0.0, 0.0, 1.0])


def make_cell(R_true_a, R_true_b):
    cfg = CellConfig()
    cfg.apply_mount_preset()
    cfg.arms["B"].enabled = True
    cell = Cell(cfg)
    cell.arms["A"] = FakeArm(cfg.arms["A"], R_true_a)
    cell.arms["B"] = FakeArm(cfg.arms["B"], R_true_b)
    return cell


def rot(axis, deg):
    k = np.zeros(3)
    k["xyz".index(axis)] = 1.0
    return K.rotvec_to_mat(k * math.radians(deg))


print("a cell hung exactly as configured")
cfg0 = CellConfig()
cfg0.apply_mount_preset()
R_a = cfg0.arms["A"].base_matrix()[:3, :3]
R_b = cfg0.arms["B"].base_matrix()[:3, :3]
cell = make_cell(R_a, R_b)
r = C.level_report(cell)
# 1e-4 deg, not 1e-9: these come out of an acos near 1, where the argument's
# last bit is worth about 1e-6 of a degree. Tighter than that is testing the
# arithmetic, not the geometry.
check("the arms agree about up", r["pair_error_deg"] < 1e-4,
      "%.2e deg" % r["pair_error_deg"])
check("world +Z is already up", r["world_error_deg"] < 1e-4,
      "%.2e deg" % r["world_error_deg"])
check("nothing to warn about", not r["warnings"], str(r["warnings"]))
check("measured tilt matches the mount number",
      all(abs(r["tilt_deg"][a] - r["config_tilt_deg"][a]) < 1e-6 for a in "AB"),
      str({a: round(r["tilt_deg"][a], 2) for a in "AB"}))

print("the pair is really hung 12 deg off what the mount numbers say")
# both brackets out by the same amount — the arms still agree with each other
tip = rot("x", 12.0)
cell = make_cell(tip.T @ R_a, tip.T @ R_b)
r = C.level_report(cell)
check("the arms still agree with each other", r["pair_error_deg"] < 1e-4,
      "%.2e deg" % r["pair_error_deg"])
check("and that is the point: nothing feels wrong to them",
      not any("disagree" in w for w in r["warnings"]))
check("but world +Z is 12 deg off real up",
      abs(r["world_error_deg"] - 12.0) < 1e-6, "%.3f deg" % r["world_error_deg"])

a_to_b_before = cell.config.a_to_b()
C.apply_level(cell.config, r)
after = C.level_report(cell)
check("levelling puts world +Z back on the vertical",
      after["world_error_deg"] < 1e-9, "%.2e deg" % after["world_error_deg"])
check("and the A-to-B transform came through untouched",
      np.allclose(cell.config.a_to_b(), a_to_b_before, atol=1e-12))
check("the mount is marked measured afterwards",
      cell.config.mount["style"] == "custom")

print("one bracket wrong: the arms themselves disagree")
cell = make_cell(R_a, rot("x", 20.0).T @ R_b)
r = C.level_report(cell)
check("the disagreement is measured", abs(r["pair_error_deg"] - 20.0) < 0.5,
      "%.2f deg" % r["pair_error_deg"])
check("and blamed on the transform between the arms",
      any("disagree" in w for w in r["warnings"]))
check("levelling is not offered as the fix for that",
      any("touch-off" in w or "teach" in w for w in r["warnings"]))

print("the blind spot, stated out loud")
# A relative error that is a pure rotation about the vertical moves neither
# arm's sense of up. This check exists so nobody reads a clean pair error as
# proof the two arms are related correctly — it clears two angles, not three.
cell = make_cell(R_a, rot("z", 25.0).T @ R_b)
r = C.level_report(cell)
check("a 25 deg yaw error between the arms is invisible to gravity",
      r["pair_error_deg"] < 1e-4, "%.2e deg" % r["pair_error_deg"])
check("so is the levelling it would need",
      r["world_error_deg"] < 1e-4, "%.2e deg" % r["world_error_deg"])

print("an arm that cannot answer")


class MuteArm(FakeArm):
    def up_in_base(self):
        return None


cell = make_cell(R_a, R_b)
cell.arms["A"] = MuteArm(cell.config.arms["A"], R_a)
cell.arms["B"] = MuteArm(cell.config.arms["B"], R_b)
try:
    C.level_report(cell)
    check("a cell that cannot measure says so", False, "it returned a report")
except C.CalibrationError as e:
    check("a cell that cannot measure says so", "gravity" in str(e))

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
