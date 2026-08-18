"""
Measuring the whole cell from a workpiece both arms are holding.

`test_directions.py` covers the translation-only teach, which throws the
orientations away and can only answer half the question. This covers the
AX=ZB solve that keeps them, and the two things that buys: turning the
workpiece is allowed, and the base *positions* come out as well as their
orientations.

Everything here is synthetic, generated from a planted (Z, X) pair, so the
answer is knowable and the solver has to find it — including from the badly
posed sessions that look productive and determine nothing.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.geometry import kinematics as K
from ur5dual.geometry import calibration as C
from ur5dual.geometry.calibration import (CalibrationError,
                                          HeldObjectCalibration)

rng = np.random.default_rng(11)
fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


# the truth: arm B's base half a metre away, yawed most of a half turn and
# tipped over, which is roughly how this cell is really built
Z_TRUE = K.xyz_rpy_to_mat([0.02, 0.58, -0.03],
                          [math.radians(88.0), math.radians(2.0),
                           math.radians(-4.0)])
# and the grip: the two gripper frames 340 mm apart across the workpiece,
# facing each other and rolled a little
X_TRUE = K.xyz_rpy_to_mat([0.34, 0.02, -0.01],
                          [math.radians(178.0), 0.0, math.radians(25.0)])


def placements(turns, slides=None, pos_noise=0.0, ang_noise_deg=0.0):
    """Build a session: `turns` are rotvecs applied to the workpiece."""
    cal = HeldObjectCalibration()
    slides = slides if slides is not None else [(0, 0, 0)] * len(turns)
    for rv, slide in zip(turns, slides):
        M = np.eye(4)
        M[:3, :3] = K.rotvec_to_mat(np.asarray(rv, dtype=float))
        M[:3, 3] = np.array([0.45, 0.05, 0.35]) + np.asarray(slide, float)
        N = K.inv(Z_TRUE) @ M @ X_TRUE
        if pos_noise or ang_noise_deg:
            for T in (M, N):
                T[:3, 3] += rng.normal(scale=pos_noise, size=3)
                T[:3, :3] = K.rotvec_to_mat(
                    rng.normal(scale=math.radians(ang_noise_deg), size=3)
                ) @ T[:3, :3]
        cal.add(M, N)
    return cal


def err_mm_deg(Z):
    """How far a solved transform is from the planted one."""
    E = K.inv(Z_TRUE) @ Z
    return (float(np.linalg.norm(E[:3, 3])) * 1000,
            math.degrees(float(np.linalg.norm(K.mat_to_rotvec(E[:3, :3])))))


# turns about three axes, which is what an operator waggling the workpiece
# about actually produces
GOOD_TURNS = [(0, 0, 0), (0.7, 0, 0), (0, 0.7, 0), (0, 0, 0.7),
              (0.5, 0.5, 0), (-0.6, 0, 0.4), (0, -0.5, 0.5), (0.3, -0.4, -0.3)]
GOOD_SLIDES = rng.normal(scale=0.06, size=(8, 3))

print("clean placements, turned about several axes")
cal = placements(GOOD_TURNS, GOOD_SLIDES)
Z, rep = cal.solve()
dp, da = err_mm_deg(Z)
check("recovers the planted base transform", dp < 1e-6 and da < 1e-6,
      "%.6f mm, %.6f deg" % (dp, da))
check("recovers the grip too — the tape-measure check",
      abs(rep["grip_mm"] - np.linalg.norm(X_TRUE[:3, 3]) * 1000) < 1e-6,
      "grip %.1f mm" % rep["grip_mm"])
check("residual is zero", rep["rms_mm"] < 1e-6)
check("no warnings", not rep["warnings"], str(rep["warnings"]))

print("the whole point: the base POSITION, which no push can measure")
check("position solved, not just orientation", dp < 1e-6,
      "translation teach cannot produce this number at all")

print("realistic hand-guiding, 0.5 mm and 0.1 deg of pose slop")
dps, das = [], []
for _ in range(40):
    Z, rep = placements(GOOD_TURNS, GOOD_SLIDES,
                        pos_noise=0.0005, ang_noise_deg=0.1).solve()
    dp, da = err_mm_deg(Z)
    dps.append(dp)
    das.append(da)
check("median base position within 2.5 mm", float(np.median(dps)) < 2.5,
      "median %.2f mm, 95th %.2f mm"
      % (float(np.median(dps)), float(np.percentile(dps, 95))))
check("median base orientation within 0.5 deg", float(np.median(das)) < 0.5,
      "median %.3f deg, 95th %.3f deg"
      % (float(np.median(das)), float(np.percentile(das, 95))))

print("coarser slop, 2 mm and 0.5 deg — a badly set TCP or a soft grip")
dps = [err_mm_deg(placements(GOOD_TURNS, GOOD_SLIDES, pos_noise=0.002,
                             ang_noise_deg=0.5).solve()[0])[0]
       for _ in range(40)]
check("degrades gracefully rather than diverging", float(np.median(dps)) < 12.0,
      "median %.1f mm" % float(np.median(dps)))

print("what the old solver refused, this one accepts")
# a session where the workpiece turned during every move: the exact thing
# `DirectionCalibration` rejects as a slipped grip
cal = placements([(0, 0, 0), (0.04, 0.01, 0.02), (0.08, -0.03, 0.05),
                  (0.6, 0.2, -0.1), (-0.3, 0.5, 0.2), (0.1, -0.6, 0.4)],
                 rng.normal(scale=0.08, size=(6, 3)))
Z, rep = cal.solve()
dp, da = err_mm_deg(Z)
check("a session with a turn in every move still solves exactly",
      dp < 1e-6 and da < 1e-6, "%.6f mm, %.6f deg" % (dp, da))

print("a rough seed is a candidate, never an assumption")
cal = placements(GOOD_TURNS, GOOD_SLIDES)
wrong = K.xyz_rpy_to_mat([0.3, 0.9, 0.2],
                         [math.radians(60), 0.0, math.radians(40)])
Z, rep = cal.solve(seed=wrong)
check("a badly wrong seed loses to the data", err_mm_deg(Z)[0] < 1e-6,
      "%.6f mm from truth despite a seed 400 mm out" % err_mm_deg(Z)[0])


print("refuses what it cannot determine")
try:
    placements([(0, 0, 0), (0.5, 0, 0)]).solve()
    check("two placements rejected", False)
except CalibrationError as e:
    check("two placements rejected", True, str(e)[:46])

# every turn about the same axis: any number of placements, and the distance
# between the bases square to that axis is still unknown
cal = placements([(0, 0, 0), (0, 0, 0.4), (0, 0, 0.8), (0, 0, 1.2),
                  (0, 0, 1.6), (0, 0, 2.0)], rng.normal(scale=0.06, size=(6, 3)))
Z, rep = cal.solve()
check("turning about one axis only is flagged as undetermined",
      not rep["determined"]
      and any("do not determine the geometry" in w for w in rep["warnings"]),
      "amplification %s" % rep["amplification"])
check("and it is the base POSITION that is lost",
      err_mm_deg(Z)[0] > 5.0, "%.1f mm out" % err_mm_deg(Z)[0])
check("while the orientation still comes out exactly — that part IS determined",
      err_mm_deg(Z)[1] < 1e-6, "%.6f deg" % err_mm_deg(Z)[1])
check("and the fit is not left stranded at a bad residual",
      rep["rms_mm"] < 1e-6, "rms %.6f mm" % rep["rms_mm"])

print("a slipped grip shows up in the residual and gets named")
cal = placements(GOOD_TURNS, GOOD_SLIDES)
cal.samples[4].N[:3, 3] += np.array([0.006, 0.0, 0.0])
rep = cal.solve()[1]
check("slip raises a warning",
      any("shifted its bite" in w for w in rep["warnings"]),
      "worst %.1f mm, rms %.1f mm" % (rep["max_mm"], rep["rms_mm"]))
check("and it names the placement that did it",
      "P5" in " ".join(rep["warnings"]),
      [w for w in rep["warnings"] if "shifted" in w][0][:40])

false_alarms = 0
for _ in range(100):
    if any("shifted its bite" in w
           for w in placements(GOOD_TURNS, GOOD_SLIDES, pos_noise=0.0005,
                               ang_noise_deg=0.1).solve()[1]["warnings"]):
        false_alarms += 1
check("a sound grip is rarely accused", false_alarms <= 5,
      "%d false alarms in 100" % false_alarms)

print("small turns are flagged as noise amplifiers, not silently trusted")
cal = placements([(0, 0, 0), (0.15, 0, 0), (0, 0.15, 0), (0.1, 0.1, 0),
                  (0, 0, 0.15), (0.1, 0, 0.1)], rng.normal(scale=0.06, size=(6, 3)))
rep = cal.solve()[1]
check("a session of 10 deg turns warns about the base positions",
      any("do not pin the base positions down" in w for w in rep["warnings"]),
      "widest turn %.0f deg, amplification %.1f"
      % (cal.largest_turn_deg(), rep["amplification"]))
check("and refuses to be trusted with the base positions",
      not rep["determined"],
      "amplification %.1f against a bar of %.0f"
      % (rep["amplification"], C.MAX_POSITION_AMPLIFICATION))

# the failure the offline script run turned up: with real readings in them,
# placements that never turned the workpiece are not exactly singular, only
# hopelessly ill-conditioned, and the rank test alone lets them through
cal = placements([(0.2, 0.1, 0)] * 8, rng.normal(scale=0.06, size=(8, 3)),
                 pos_noise=0.0005, ang_noise_deg=0.1)
Z, rep = cal.solve()
check("a NOISY session with no turn is caught too, not just a clean one",
      not rep["determined"],
      "amplification %.0f" % rep["amplification"])
check("and it is not silently applied", not rep.get("marked_calibrated", False))


print("sampling refuses a placement that repeats one already taken")


class FakeArm:
    def __init__(self):
        self.pose = np.zeros(6)

    @property
    def connected(self):
        return True

    def tcp_pose(self):
        return self.pose.copy()


class FakeCell:
    def __init__(self):
        self.arms = {"A": FakeArm(), "B": FakeArm()}

    def place(self, M):
        self.arms["A"].pose = K.mat_to_pose(M)
        self.arms["B"].pose = K.mat_to_pose(K.inv(Z_TRUE) @ M @ X_TRUE)


cell = FakeCell()
cal = HeldObjectCalibration()


def pose_at(rv, xyz):
    M = np.eye(4)
    M[:3, :3] = K.rotvec_to_mat(rv)
    M[:3, 3] = xyz
    return M


cell.place(pose_at([0, 0, 0], [0.45, 0.05, 0.35]))
ok, why = cal.commit(cell)
check("the first placement is always recorded", ok, why)

cell.place(pose_at([0.01, 0, 0], [0.453, 0.05, 0.35]))
ok, why = cal.commit(cell)
check("3 mm and half a degree later is refused", not ok, why[:52])
check("and it says what to do", "move or turn the workpiece further" in why)

cell.place(pose_at([0.6, 0, 0], [0.45, 0.05, 0.35]))
ok, why = cal.commit(cell)
check("turning it on the spot IS a new placement", ok, why[:46])

cell.place(pose_at([0.6, 0, 0], [0.55, 0.09, 0.30]))
ok, why = cal.commit(cell)
check("and so is sliding it without turning", ok)
check("each press adds a placement rather than replacing one",
      len(cal.samples) == 3, "%d placements" % len(cal.samples))

print("the fallback for a workpiece that cannot be turned")
cal = placements([(0.2, 0.1, 0)] * 5,
                 [(0, 0, 0), (0.09, 0, 0), (0.09, 0.08, 0),
                  (0.09, 0.08, 0.07), (0.02, 0.03, 0.05)])
rep = cal.solve()[1]
check("a session with no turn at all is flagged",
      any("do not determine the geometry" in w for w in rep["warnings"]))
direction = cal.as_direction_calibration()
check("but the placements still yield translation pairs",
      len(direction.pairs) == 4, "%d pairs" % len(direction.pairs))
R_ba, drep = direction.solve()
R_ba_true = Z_TRUE[:3, :3].T
check("and the fallback recovers the relative orientation",
      np.allclose(R_ba, R_ba_true, atol=1e-9),
      "rms %.4f mm" % drep["rms_mm"])

print("applying it writes arm B's position AND orientation")
from ur5dual.config import CellConfig

cfg = CellConfig()
cfg.apply_mount_preset()
a_before = cfg.arms["A"].base_matrix().copy()
cal = placements(GOOD_TURNS, GOOD_SLIDES)
Z, rep = cal.apply_to_config(cfg)
check("the configured A-to-B transform is now the measured one",
      np.allclose(cfg.a_to_b(), Z_TRUE, atol=1e-6))
check("arm A untouched", np.allclose(cfg.arms["A"].base_matrix(), a_before))
check("straight-line moves unlocked", cfg.translation_calibrated)
check("turning a held object unlocked too — the touch-off's job, done",
      cfg.calibrated and rep["marked_calibrated"],
      "worst %.3f mm" % rep["max_mm"])

cfg2 = CellConfig()
cfg2.apply_mount_preset()
cal = placements([(0, 0, 0), (0, 0, 0.5), (0, 0, 1.0), (0, 0, 1.5)])
try:
    cal.apply_to_config(cfg2)
    check("a one-axis session refuses to be applied", False)
except CalibrationError as e:
    check("a one-axis session refuses to be applied", True, str(e)[:44])
check("and cell.yaml was left alone", not cfg2.calibrated)

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
