"""Pairing the two flanges: what one gauge reading fixes, and what several do.

Checked against a transform planted here, so the answer is knowable. The
interesting cases are the negative ones — a single reading must *not* claim
the three numbers it cannot see, and a set of readings all taken with the
flange axis pointing the same way must be refused rather than fitted.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.config import CellConfig
from ur5dual.geometry import kinematics as K
from ur5dual.geometry.calibration import (
    CalibrationError, FlangePairCalibration, PAIR_TRUST_MM, facing_flanges,
    flange_pair_transform, refine_from_flange_pair,
)
from ur5dual.geometry.ur_kinematics import UR5_DH, fk

rng = np.random.default_rng(7)
fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


# the truth: B half a metre across from A, turned to face it, on a bracket a
# couple of degrees out of square the way a real one is
T_TRUE = K.xyz_rpy_to_mat([0.010, -0.510, 0.004],
                          [math.radians(2.0), math.radians(-1.5), math.pi])


def pairing(gap=None, spin=None, offset=0.0):
    """One posture with the two flanges facing each other. (F_a, F_b, gap)."""
    F_a = fk(rng.uniform(-2.0, 2.0, 6), UR5_DH)
    gap = rng.uniform(0.01, 0.20) if gap is None else gap
    spin = rng.uniform(-math.pi, math.pi) if spin is None else spin
    G = facing_flanges(gap, spin, rng.normal(0.0, offset, 2))
    return F_a, K.inv(T_TRUE) @ F_a @ G, gap


print("what the geometry of a facing pair is")
G = facing_flanges(0.043)
check("the far flange's Z opposes the near one's",
      np.allclose(G[:3, 2], [0.0, 0.0, -1.0]))
check("its origin sits at the measured gap along that axis",
      abs(G[2, 3] - 0.043) < 1e-15, "%.4f mm" % (G[2, 3] * 1000))
check("and it is a rotation, not a reflection",
      abs(np.linalg.det(G[:3, :3]) - 1.0) < 1e-12)
spun = facing_flanges(0.043, spin=math.radians(37.0))
check("spinning one flange changes neither the gap nor the parallelism",
      abs(spun[2, 3] - 0.043) < 1e-15 and np.allclose(spun[:3, 2], G[:3, 2]))
check("which is exactly why a gauge cannot measure it",
      float(np.max(np.abs(spun[:3, :3] - G[:3, :3]))) > 0.5)

F_a, F_b, gap = pairing(spin=0.0)
T = flange_pair_transform(F_a, F_b, gap)
check("one pairing with the spin and offset known gives the transform exactly",
      np.allclose(T, T_TRUE, atol=1e-12),
      "worst %.2e" % np.abs(T - T_TRUE).max())


print("\ncorrecting a configured geometry with one reading")
# A cell that is 12 mm and 2 degrees out — a bracket measured off a drawing.
SEED = K.xyz_rpy_to_mat([0.0, -0.498, 0.0],
                        [math.radians(0.5), 0.0, math.pi - math.radians(2.0)])
F_a, F_b, gap = pairing(gap=0.043, spin=math.radians(20.0), offset=0.0)
T = refine_from_flange_pair(SEED, F_a, F_b, gap)
G = K.inv(F_a) @ T @ F_b
check("the faces come out parallel", np.allclose(G[:3, 2], [0, 0, -1], atol=1e-12),
      "axis %s" % np.round(G[:3, 2], 9))
check("the gap comes out as measured", abs(G[2, 3] - gap) < 1e-12,
      "%.4f mm" % (G[2, 3] * 1000))
before = K.mat_to_pose(K.inv(T_TRUE) @ SEED)
after = K.mat_to_pose(K.inv(T_TRUE) @ T)
check("and the transform is closer to the truth than it was",
      np.linalg.norm(after[3:]) < np.linalg.norm(before[3:]),
      "%.2f deg -> %.2f deg" % (np.degrees(np.linalg.norm(before[3:])),
                                np.degrees(np.linalg.norm(after[3:]))))
# The three numbers a gauge is blind to have to survive untouched: this is
# what stops a single reading from being mistaken for a calibration.
G_seed = K.inv(F_a) @ SEED @ F_b
check("it does not invent the spin about the common axis",
      abs(math.atan2(G[1, 0], G[0, 0]) - math.atan2(G_seed[1, 0], G_seed[0, 0]))
      < math.radians(2.5),
      "%.2f deg of drift" % math.degrees(abs(math.atan2(G[1, 0], G[0, 0])
                                             - math.atan2(G_seed[1, 0], G_seed[0, 0]))))
concentric = refine_from_flange_pair(SEED, F_a, F_b, gap, concentric=True)
G_c = K.inv(F_a) @ concentric @ F_b
check("concentric=True is the extra assumption, and only when asked for",
      np.allclose(G_c[:2, 3], 0.0) and float(np.linalg.norm(G[:2, 3])) > 1e-4,
      "sideways %.1f mm otherwise" % (np.linalg.norm(G[:2, 3]) * 1000))

F_a, F_b, gap = pairing(spin=0.0)
check("a geometry that already explains the reading is left alone",
      np.allclose(refine_from_flange_pair(T_TRUE, F_a, F_b, gap), T_TRUE,
                  atol=1e-9))


print("\nfitting several readings")
for model, offset in (("separation", 0.0), ("facing", 0.02)):
    cal = FlangePairCalibration(model)
    for _ in range(10):
        F_a, F_b, gap = pairing(offset=offset)
        cal.add(F_a, F_b, gap + rng.normal(0.0, 0.0002))     # 0.2 mm gauge slop
    T, report = cal.solve(SEED)
    err = K.mat_to_pose(K.inv(T_TRUE) @ T)
    check("%s recovers the planted transform from 10 readings" % model,
          np.linalg.norm(err[:3]) < 0.002 and np.linalg.norm(err[3:]) < 0.01,
          "%.2f mm, %.3f deg, rms %.2f mm"
          % (np.linalg.norm(err[:3]) * 1000, np.degrees(np.linalg.norm(err[3:])),
             report["rms_mm"]))
    check("  and says nothing is wrong with them", not report["warnings"],
          "; ".join(report["warnings"]))

# All the readings taken with the flange axis pointing one way. The fit has
# nothing to say about the two directions square to it, and saying so is the
# whole job — a transform that fits every reading and is wrong sideways is
# precisely what would send two arms into each other.
flat = FlangePairCalibration("separation")
F_a0, _, _ = pairing()
for _ in range(8):
    gap = rng.uniform(0.02, 0.05)
    F_b = K.inv(T_TRUE) @ F_a0 @ facing_flanges(gap, rng.uniform(-3, 3))
    flat.add(F_a0, F_b, gap)
check("readings along one line score no spread", flat.spread(SEED) < 0.05,
      "%.3f" % flat.spread(SEED))
_, report = flat.solve(SEED)
check("and the report says the geometry square to them is still a guess",
      any("much the same direction" in w for w in report["warnings"]))

back_to_back = FlangePairCalibration("facing")
for _ in range(4):
    F_a, F_b, gap = pairing()
    back_to_back.add(F_a, F_b, gap)
F_a, F_b, gap = pairing()
back_to_back.add(F_a, K.inv(T_TRUE) @ F_a @ K.xyz_rpy_to_mat([0, 0, gap], [0, 0, 0]),
                 gap, name="turned round")
_, report = back_to_back.solve(SEED)
check("a pair standing back to back is named rather than fitted",
      "turned round" in report["flipped"], str(report["flipped"]))

thin = FlangePairCalibration("separation")
thin.add(*pairing()[:2], 0.043)
try:
    thin.solve(SEED)
    check("one reading is refused as a fit, whatever else it is good for", False)
except CalibrationError as e:
    check("one reading is refused as a fit, whatever else it is good for",
          "need at least" in str(e))


print("\nwriting it into the config")
cfg = CellConfig()
cfg.apply_mount_preset()
a_before = cfg.arms["A"].base_matrix().copy()
cal = FlangePairCalibration("separation")
# a cell whose bracket is 5 mm and half a degree off the preset it was built
# from, which is the size of error this measurement exists to find
T_TRUE = cfg.a_to_b() @ K.xyz_rpy_to_mat([0.004, -0.003, 0.002],
                                         [math.radians(0.4), 0.0, 0.0])
for _ in range(12):
    F_a, F_b, gap = pairing()
    cal.add(F_a, F_b, gap)
T, report = cal.apply_to_config(cfg)
check("arm B now matches the fitted geometry",
      np.allclose(cfg.a_to_b(), T, atol=2e-6),
      "worst %.2e" % np.abs(cfg.a_to_b() - T).max())
check("arm A is untouched — it is the reference",
      np.allclose(cfg.arms["A"].base_matrix(), a_before))
check("the mount is marked hand-measured", cfg.mount["style"] == "custom")
check("a straight-line carry is now allowed", cfg.translation_calibrated)
check("and turning a held object only if the readings agree to %.0f mm"
      % PAIR_TRUST_MM,
      cfg.calibrated == (report["max_mm"] <= PAIR_TRUST_MM),
      "worst %.2f mm" % report["max_mm"])

cfg2 = CellConfig()
cfg2.apply_mount_preset()
try:
    flat.apply_to_config(cfg2)
    check("an under-determined fit refuses to be written", False)
except CalibrationError as e:
    check("an under-determined fit refuses to be written",
          "much the same direction" in str(e))
check("and leaves the config as it found it", not cfg2.translation_calibrated)

print("\nFAILURES: %d" % fail)
sys.exit(1 if fail else 0)
