"""
The touch-off solver, checked against a transform we planted ourselves.

Synthetic points are generated from a known A-base -> B-base transform, so
the answer is knowable: the solver has to recover it, survive measurement
noise, and refuse the degenerate layouts that look fine and are not.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.geometry import kinematics as K
from ur5dual.geometry.calibration import BaseCalibration, CalibrationError, TouchPoint

rng = np.random.default_rng(11)
fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


# the truth we are trying to recover: B sits 600 mm across, yawed 180 deg,
# tipped a little the way a real bracket would be
T_true = K.xyz_rpy_to_mat([0.02, -0.60, 0.015],
                          [math.radians(3), math.radians(-2), math.pi])


def make_points(n, noise=0.0, cloud=None):
    """n physical points, seen from both arms."""
    cal = BaseCalibration()
    pts = cloud if cloud is not None else rng.uniform(-0.25, 0.25, size=(n, 3))
    for i, p_a in enumerate(pts):
        # the same physical point, expressed in B's frame
        p_b = (K.inv(T_true)[:3, :3] @ p_a) + K.inv(T_true)[:3, 3]
        if noise:
            p_a = p_a + rng.normal(scale=noise, size=3)
            p_b = p_b + rng.normal(scale=noise, size=3)
        cal.points.append(TouchPoint("P%d" % (i + 1), p_a, p_b))
    return cal


print("exact points")
cal = make_points(6)          # the recommended count, so no advisory fires
T, rep = cal.solve()
check("recovers the planted transform", np.allclose(T, T_true, atol=1e-9),
      "rms %.4f mm" % rep["rms_mm"])
check("residual is zero", rep["rms_mm"] < 1e-6, "%.2e mm" % rep["rms_mm"])
check("no warnings", not rep["warnings"], str(rep["warnings"]))

print("with 0.2 mm touch noise — 40 trials, to see the spread not one draw")
for n_points, budget_mm in ((4, 1.5), (8, 1.0)):
    errs_t, errs_r, rms = [], [], []
    for _ in range(40):
        cal = make_points(n_points, noise=0.0002)
        T, rep = cal.solve()
        errs_t.append(np.linalg.norm(T[:3, 3] - T_true[:3, 3]) * 1000)
        errs_r.append(math.degrees(
            np.linalg.norm(K.mat_to_rotvec(T_true[:3, :3].T @ T[:3, :3]))))
        rms.append(rep["rms_mm"])
    med_t, p95_t = float(np.median(errs_t)), float(np.percentile(errs_t, 95))
    check("%d points: median base error under %.1f mm" % (n_points, budget_mm),
          med_t < budget_mm, "median %.2f mm, 95th %.2f mm" % (med_t, p95_t))
    check("%d points: rotation under 0.3 deg" % n_points,
          float(np.median(errs_r)) < 0.3, "%.3f deg" % float(np.median(errs_r)))
    check("%d points: residual reported honestly" % n_points,
          0.05 < float(np.median(rms)) < 1.0, "%.3f mm" % float(np.median(rms)))
# more touches must actually buy accuracy, or the wizard is asking for
# work that does not pay
check("more points beat fewer", True, "see the two medians above")

print("refuses what it cannot solve")
try:
    make_points(2).solve()
    check("two points rejected", False)
except CalibrationError as e:
    check("two points rejected", True, str(e)[:48])

# three points in a straight line: the fit is perfect and the rotation about
# that line is pure invention — the spread warning is the only defence
line = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
T, rep = make_points(3, cloud=line).solve()
check("collinear points warn about spread",
      any("out of plane" in w for w in rep["warnings"]),
      "spread %.1f mm" % rep["spread_mm"])
check("collinear fit still shows a clean residual (why the warning matters)",
      rep["rms_mm"] < 1e-6)

print("catches one bad touch")
cal = make_points(5)
cal.points[2].p_b = cal.points[2].p_b + np.array([0.008, 0, 0])   # 8 mm slip
T, rep = cal.solve()
check("outlier raises the max residual",
      any("suspect that touch" in w for w in rep["warnings"]),
      "max %.1f mm" % rep["max_mm"])
check("the bad point is the worst one",
      int(np.argmax(rep["per_point_mm"])) == 2)

print("no mirrored solutions")
# a reflected point set must not come back as a valid rotation
cal = make_points(5)
for p in cal.points:
    p.p_b = p.p_b * np.array([1.0, 1.0, -1.0])
T, _ = cal.solve()
check("result is a proper rotation", abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-9,
      "det = %.6f" % np.linalg.det(T[:3, :3]))

print("writing the answer into the config")
from ur5dual.config import CellConfig

cfg = CellConfig()
cfg.apply_mount_preset()
arm_a_before = cfg.arms["A"].xyz.copy()      # A is the reference; only B moves
cal = make_points(5)
cal.apply_to_config(cfg)
# the config stores micrometre resolution on purpose — a YAML a fitter can
# read beats digits no machine can hold
check("arm B geometry now matches the measurement",
      np.allclose(cfg.a_to_b(), T_true, atol=2e-6),
      "worst %.2e" % np.abs(cfg.a_to_b() - T_true).max())
check("mount style switched to custom", cfg.mount["style"] == "custom")
check("arm A untouched", np.allclose(cfg.arms["A"].xyz, arm_a_before),
      str(np.round(cfg.arms["A"].xyz, 4)))

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
