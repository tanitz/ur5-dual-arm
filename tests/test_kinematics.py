"""Round-trip and identity checks for ur5dual.geometry.kinematics."""

import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.geometry import kinematics as K

rng = np.random.default_rng(7)
fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


print("rotvec <-> matrix")
for _ in range(200):
    rv = rng.normal(size=3)
    rv = rv / np.linalg.norm(rv) * rng.uniform(0, math.pi - 1e-3)
    back = K.mat_to_rotvec(K.rotvec_to_mat(rv))
    if np.linalg.norm(back - rv) > 1e-9:
        check("random rotvec round trip", False, str(rv))
        break
else:
    check("random rotvec round trip (200x)", True)

# the two cases the Rodrigues formula degenerates on
check("theta = 0", np.linalg.norm(K.mat_to_rotvec(K.rotvec_to_mat([0, 0, 0]))) < 1e-12)
for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
             np.array([0, 0, 1.0]), np.array([1.0, 1.0, 0]) / math.sqrt(2)):
    rv = axis * math.pi
    R = K.rotvec_to_mat(rv)
    back = K.mat_to_rotvec(R)
    # at exactly pi, +k and -k describe the same rotation
    same = min(np.linalg.norm(back - rv), np.linalg.norm(back + rv)) < 1e-6
    check("theta = pi about %s" % axis, same, "%s -> %s" % (rv, back))

print("rpy <-> matrix")
for _ in range(200):
    rpy = np.array([rng.uniform(-math.pi, math.pi),
                    rng.uniform(-math.pi / 2 + 0.02, math.pi / 2 - 0.02),
                    rng.uniform(-math.pi, math.pi)])
    back = K.mat_to_rpy(K.rpy_to_mat(rpy))
    if np.linalg.norm(back - rpy) > 1e-9:
        check("random rpy round trip", False, "%s -> %s" % (rpy, back))
        break
else:
    check("random rpy round trip (200x)", True)

print("pose <-> matrix, inverse")
for _ in range(200):
    pose = np.concatenate([rng.normal(size=3), rng.normal(size=3) * 0.8])
    T = K.pose_to_mat(pose)
    if np.linalg.norm(K.mat_to_pose(T) - pose) > 1e-9:
        check("pose round trip", False)
        break
    if np.linalg.norm(K.inv(T) @ T - np.eye(4)) > 1e-9:
        check("inv(T) @ T = I", False)
        break
else:
    check("pose round trip + inverse (200x)", True)

print("interpolation")
a = np.array([0.0, 0, 0, 0, 0, 0])
b = np.array([1.0, 2.0, 3.0, 0, 0, math.pi / 2])
check("s=0 gives a", np.linalg.norm(K.interp_pose(a, b, 0.0) - a) < 1e-9)
check("s=1 gives b", np.linalg.norm(K.interp_pose(a, b, 1.0) - b) < 1e-9)
mid = K.interp_pose(a, b, 0.5)
check("s=0.5 halves the translation", np.allclose(mid[:3], [0.5, 1.0, 1.5]))
check("s=0.5 halves the rotation", abs(np.linalg.norm(mid[3:]) - math.pi / 4) < 1e-9)

print("rotate about own axis")
# a frame at (1,0,0) yawed 90 deg, spun about its own z, must not translate
pose = np.array([1.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2])
out = K.rotate_about_own_axis(pose, "z", math.radians(45))
check("origin stays put", np.allclose(out[:3], pose[:3], atol=1e-12))
check("rotation accumulates",
      abs(np.linalg.norm(out[3:]) - math.radians(135)) < 1e-9)
# spinning about its own x is NOT the same as about world x, when yawed
out_x = K.rotate_about_own_axis(pose, "x", math.radians(30))
R_expect = K.rotvec_to_mat(pose[3:]) @ K.rotvec_to_mat([math.radians(30), 0, 0])
check("own-axis, not world-axis",
      np.allclose(K.rotvec_to_mat(out_x[3:]), R_expect, atol=1e-12))

print("pose_distance")
d_t, d_r = K.pose_distance([0, 0, 0, 0, 0, 0], [0.03, 0.04, 0, 0, 0, math.radians(10)])
check("translation = 50 mm", abs(d_t - 0.05) < 1e-12, "%.6f" % d_t)
check("rotation = 10 deg", abs(d_r - math.radians(10)) < 1e-12)

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
