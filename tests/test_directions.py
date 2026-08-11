"""
Teaching the bases' relative orientation by hand.

Both grippers hold one workpiece and someone pushes it around. Each arm sees
the same physical motion in its own base frame, and that pair of views is the
equation. Synthetic motions from a known rotation are fed in here, so the
answer is knowable and the solver has to recover it — including from the
degenerate pushes that look fine and determine nothing.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.geometry import kinematics as K
from ur5dual.geometry.calibration import CalibrationError, DirectionCalibration

rng = np.random.default_rng(23)
fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


# the truth: arm B's base is yawed 180 deg from A's and tipped over
R_true = K.rpy_to_mat([math.radians(35), math.radians(-12), math.pi])


def teach(directions, noise=0.0, scale=0.08):
    """Feed displacements along `directions`, seen from both base frames."""
    cal = DirectionCalibration()
    for d in directions:
        d_a = np.asarray(d, dtype=float)
        d_a = d_a / np.linalg.norm(d_a) * scale
        d_b = R_true @ d_a
        if noise:
            d_a = d_a + rng.normal(scale=noise, size=3)
            d_b = d_b + rng.normal(scale=noise, size=3)
        cal.pairs.append((d_a, d_b))
    return cal


print("clean pushes along three axes")
cal = teach([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
R, rep = cal.solve()
check("recovers the planted rotation", np.allclose(R, R_true, atol=1e-9),
      "rms %.4f mm" % rep["rms_mm"])
check("residual is zero", rep["rms_mm"] < 1e-6)
check("no warnings", not rep["warnings"], str(rep["warnings"]))

print("realistic hand pushes with 0.5 mm of slop")
errs = []
for _ in range(30):
    cal = teach(rng.normal(size=(8, 3)), noise=0.0005)
    R, rep = cal.solve()
    errs.append(math.degrees(np.linalg.norm(
        K.mat_to_rotvec(R_true.T @ R))))
check("median orientation error under 1 deg", float(np.median(errs)) < 1.0,
      "median %.2f deg, 95th %.2f deg"
      % (float(np.median(errs)), float(np.percentile(errs, 95))))

print("what a wrong rotation would cost")
# the point of the whole exercise: a world +Z push must reach both arms
worst = 0.0
for _ in range(30):
    cal = teach(rng.normal(size=(8, 3)), noise=0.0005)
    R, _ = cal.solve()
    up = np.array([0.0, 0.0, 1.0])
    sent_to_b = R @ up                    # where arm B is actually sent
    should_be = R_true @ up
    worst = max(worst, math.degrees(math.acos(
        float(np.clip(sent_to_b @ should_be, -1, 1)))))
check("arm B is sent within 1 deg of the right direction", worst < 1.0,
      "worst %.2f deg" % worst)

print("refuses what it cannot determine")
try:
    teach([(1, 0, 0), (0, 1, 0)]).solve()
    check("two motions rejected", False)
except CalibrationError as e:
    check("two motions rejected", True, str(e)[:44])

# pushing back and forth along one line: any number of samples, still no
# information about the rotation around that line
cal = teach([(1, 0, 0), (-1, 0, 0), (1, 0, 0), (-1, 0, 0), (2, 0, 0)])
R, rep = cal.solve()
check("one-directional pushing is flagged",
      any("all three directions" in w for w in rep["warnings"]),
      "spread %.3f" % rep["spread"])
check("and it does not recover the rotation (which is why the flag matters)",
      not np.allclose(R, R_true, atol=1e-3))

# everything in a plane is the same trap, one dimension up
cal = teach([(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, -1, 0), (2, 1, 0)])
check("planar pushing is flagged",
      any("all three directions" in w for w in cal.solve()[1]["warnings"]))

print("a slipping grip shows up in the residual")
cal = teach(rng.normal(size=(6, 3)))
cal.pairs[3] = (cal.pairs[3][0], cal.pairs[3][1] + np.array([0.006, 0, 0]))
report = cal.solve()[1]
check("slip raises a warning",
      any("let go of the workpiece" in w for w in report["warnings"]),
      "worst motion %.1f mm out, rest agree to %.1f mm"
      % (report["max_mm"], report["rms_mm"]))

# and the reverse: a sound grip must not be accused of slipping
false_alarms = 0
for _ in range(100):
    if any("let go" in w
           for w in teach(rng.normal(size=(6, 3)), noise=0.0005).solve()[1]["warnings"]):
        false_alarms += 1
check("a sound grip is not accused", false_alarms <= 2,
      "%d false alarms in 100" % false_alarms)

print("short wobbles are ignored while sampling")
cal = DirectionCalibration()


class FakeArm:
    def __init__(self, p):
        self.p = np.array(p, dtype=float)

    def tcp_pose(self):
        return np.concatenate([self.p, np.zeros(3)])


class FakeCell:
    def __init__(self):
        self.arms = {"A": FakeArm([0, 0, 0]), "B": FakeArm([0, 0, 0])}


cell = FakeCell()
cal.start(cell)
cell.arms["A"].p = np.array([0.002, 0, 0])     # 2 mm: noise
cell.arms["B"].p = R_true @ cell.arms["A"].p
ok, why = cal.commit(cell)
check("a 2 mm wobble is not recorded", not ok, why)
check("and it says what to do", "at least 10 mm" in why)

cell.arms["A"].p = np.zeros(3)
cell.arms["B"].p = np.zeros(3)
cal.start(cell)
cell.arms["A"].p = np.array([0.05, 0, 0])      # one arm pushed, one not
ok, why = cal.commit(cell)
check("one arm moving alone is rejected", not ok)
check("and it names the real cause", "same rigid object" in why, why[:60])

cell.arms["A"].p = np.zeros(3)
cell.arms["B"].p = np.zeros(3)
cal.start(cell)
cell.arms["A"].p = np.array([0.05, 0, 0])      # a genuine linked push
cell.arms["B"].p = R_true @ cell.arms["A"].p
ok, why = cal.commit(cell)
check("a 50 mm linked push is recorded", ok, why)

# three presses must accumulate, not overwrite one another
before = len(cal.pairs)
for step in ((0, 0.05, 0), (0, 0, 0.05)):
    cell.arms["A"].p = np.zeros(3)
    cell.arms["B"].p = np.zeros(3)
    cal.start(cell)
    cell.arms["A"].p = np.array(step, dtype=float)
    cell.arms["B"].p = R_true @ cell.arms["A"].p
    cal.commit(cell)
check("each press adds a push rather than replacing it",
      len(cal.pairs) == before + 2, "%d pushes" % len(cal.pairs))

cell.arms["A"].p = np.zeros(3)
cell.arms["B"].p = np.zeros(3)
cal.start(cell)
cell.arms["A"].p = np.array([0.05, 0, 0])      # tips travel unequal distances
cell.arms["B"].p = R_true @ np.array([0.02, 0, 0])
ok, why = cal.commit(cell)
check("mismatched travel is rejected", not ok)
check("and it says the link is not rigid", "slipped or flexed" in why)

print("applying it moves only arm B's orientation")
from ur5dual.config import CellConfig

cfg = CellConfig()
cfg.apply_mount_preset()
a_before = cfg.arms["A"].base_matrix().copy()
b_pos_before = cfg.arms["B"].xyz.copy()
cal = teach([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
cal.apply_to_config(cfg)
R_ba = cfg.arms["B"].base_matrix()[:3, :3].T @ cfg.arms["A"].base_matrix()[:3, :3]
check("the configured relative rotation is now the measured one",
      np.allclose(R_ba, R_true, atol=1e-6))
check("arm A untouched", np.allclose(cfg.arms["A"].base_matrix(), a_before))
check("arm B's position untouched — motion never measured it",
      np.allclose(cfg.arms["B"].xyz, b_pos_before))
check("straight-line moves unlocked", cfg.translation_calibrated)
check("turning the object still locked", not cfg.calibrated)

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
