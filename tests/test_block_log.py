"""
The block log: what gets written, what it refuses, and that it fits back.

No robot and no sockets. A block session is eight careful seatings that take
an hour to collect, and until now nothing kept them — so what is checked here
is that the file is worth having. Two of these are refusals rather than sums,
and they are the reason it is: a gauge reading and a block placement are both
one distance and two postures, and a file holding some of each is wrong about
half of them whichever way it is later read. Nothing in the numbers would
show it.

    python3 tests/test_block_log.py
"""

import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.config import ARM_IDS, CellConfig
from ur5dual.geometry import kinematics as K
from ur5dual.geometry import ur_kinematics as UK
from ur5dual.geometry.calibration import FlangePairCalibration, facing_flanges
from ur5dual.tools import flange_fit as FF
from ur5dual.tools import flange_log as FL

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(11)
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
# and B is not the arm on the drawing. Its controller holds corrections worth
# a couple of millimetres, which is the size of what a block session measures
DH = {"A": UK.UR5_DH,
      "B": UK.with_corrections({"d": np.array([2e-4, 0, 0, -3e-4, 4e-4, 6e-4]),
                                "a": np.array([0, 3e-4, -2e-4, 0, 0, 0])})}
DH_SOURCE = {"A": FL.PUBLISHED_DH, "B": "this robot's own calibration"}
GAP_MM = 84.88


def placements(count):
    """`count` postures with the two flanges seated on the same block.

    Arm A goes wherever it likes — which is what a block re-clamped into a new
    attitude amounts to — and arm B is asked for the pose that puts its face
    flat on the other side, through the transform being planted.
    """
    out = []
    while len(out) < count:
        q_a = rng.uniform(-2.0, 2.0, 6)
        F_a = UK.fk(q_a, DH["A"])
        G = facing_flanges(GAP_MM / 1000.0, spin=rng.uniform(-math.pi, math.pi))
        q_b, info = UK.ik(K.inv(T_TRUE) @ F_a @ G,
                          rng.uniform(-2.0, 2.0, 6), dh=DH["B"])
        if info["converged"]:
            out.append({"A": q_a, "B": q_b})
    return out


def write_log(path, poses, model=FL.FACING, dh=DH, dh_source=DH_SOURCE):
    data = FL.load_log(path)
    data["meta"] = FL.log_meta("check_block_online", dh_source, dh, model=model)
    for q in poses:
        data["samples"].append(FL.build_sample(GAP_MM, q, dh))
    FL.save_log(path, data)
    return data


tmp = tempfile.mkdtemp(prefix="block-log-")
LOG = os.path.join(tmp, "block_log.json")
poses = placements(8)


print("what a session writes down")
write_log(LOG, poses)
data = FL.load_log(LOG)
check("every placement is in the file", len(data["samples"]) == 8,
      "%d" % len(data["samples"]))
check("as joint angles, in degrees, for both arms",
      all(len(s[a]["joints_deg"]) == 6 for s in data["samples"] for a in ARM_IDS))
check("the file says what its distances mean",
      data["meta"]["model"] == "facing", data["meta"].get("model", "nothing"))
check("and which kinematics the poses came from",
      data["meta"]["kinematics"] == DH_SOURCE)
check("arm B's own table travels with them, to the last digit",
      np.allclose(data["meta"]["dh"]["B"]["d"], DH["B"]["d"], atol=0, rtol=0),
      "%s" % np.round(np.array(data["meta"]["dh"]["B"]["d"]) - DH["B"]["d"], 12))
first = data["samples"][0]
check("the angles survive the round trip to a ten-thousandth of a degree",
      float(np.max(np.abs(np.radians(first["A"]["joints_deg"]) - poses[0]["A"])))
      < math.radians(1e-4) + 1e-12)


print("\nfitting the file back, with no robot switched on")
SEED = K.xyz_rpy_to_mat([0.004, -0.498, 0.0],
                        [math.radians(0.5), 0.0, math.pi - math.radians(2.0)])
cal = FlangePairCalibration.from_log(data, model=data["meta"]["model"],
                                     dh=FF.dh_from_meta(data["meta"]))
T, report = cal.solve(SEED)
moved = K.mat_to_pose(K.inv(T_TRUE) @ T)
check("the planted transform comes back",
      float(np.linalg.norm(moved[:3])) < 5e-5,
      "%.3f mm, %.4f deg" % (np.linalg.norm(moved[:3]) * 1000,
                             np.degrees(np.linalg.norm(moved[3:]))))
check("the placements agree with it", report["max_mm"] < 0.01,
      "worst %.4f mm" % report["max_mm"])
check("and they were posed well enough to say so",
      report["spread"] >= 0.25 and not report["warnings"],
      "spread %.2f%s" % (report["spread"],
                         "; " + "; ".join(report["warnings"])
                         if report["warnings"] else ""))

# the whole reason the tables are in the file
published = FlangePairCalibration.from_log(
    data, model=data["meta"]["model"], dh={a: UK.UR5_DH for a in ARM_IDS})
T_pub, report_pub = published.solve(SEED)
off = float(np.linalg.norm(K.mat_to_pose(K.inv(T_TRUE) @ T_pub)[:3])) * 1000
check("fitting the same samples against the published table is millimetres out",
      off > 0.5, "%.2f mm from the truth" % off)


print("\nwhat the file refuses")
check("an empty file takes either kind",
      FL.model_clash({"meta": {}, "samples": []},
                     FL.log_meta("x", DH_SOURCE, DH, model=FL.FACING)) is None)
check("a block session will not go in among gauge readings",
      "cannot share one" in (FL.model_clash(
          {"meta": {"model": "separation"}, "samples": [1]},
          FL.log_meta("x", DH_SOURCE, DH, model=FL.FACING)) or ""))
check("nor gauge readings in among block placements",
      FL.model_clash(data, FL.log_meta("x", DH_SOURCE, DH,
                                       model=FL.SEPARATION)) is not None)
check("a log written before the field existed is read as gauge readings",
      FL.model_clash({"meta": {"kinematics": {}}, "samples": [1]},
                     FL.log_meta("x", DH_SOURCE, DH,
                                 model=FL.SEPARATION)) is None)
check("more of the same kind is what a file is for",
      FL.model_clash(data, FL.log_meta("x", DH_SOURCE, DH,
                                       model=FL.FACING)) is None)
check("and two DH tables still cannot share a file",
      FL.kinematics_clash(data, FL.log_meta(
          "x", {a: FL.PUBLISHED_DH for a in ARM_IDS},
          {a: UK.UR5_DH for a in ARM_IDS}, model=FL.FACING)) is not None)


print("\nthe offline fit reads the file's own meaning")
cfg = CellConfig()
cfg.apply_mount_preset()
cfg.arms["B"].set_base_matrix(cfg.arms["A"].base_matrix() @ SEED)
cfg_path = os.path.join(tmp, "cell.yaml")
cfg.save(cfg_path)
out = subprocess.run(
    [sys.executable, "-m", "ur5dual.tools.flange_fit",
     "--file", LOG, "--config", cfg_path, "--fit"],
    cwd=REPO, capture_output=True, text=True)
check("the fit runs", out.returncode == 0, out.stderr.strip()[-200:])
check("and reads the samples as facing without being told to",
      "read as facing" in out.stdout and "with the facing model" in out.stdout)
check("nothing is written without --apply", "nothing written" in out.stdout)
check("cell.yaml is untouched by a report",
      CellConfig.load(cfg_path).a_to_b().tolist() == cfg.a_to_b().tolist())

for name in os.listdir(tmp):
    os.remove(os.path.join(tmp, name))
os.rmdir(tmp)

print("\nFAILURES: %d" % fail)
sys.exit(1 if fail else 0)
