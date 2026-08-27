"""Marker pose and eye-to-hand transform, without a physical camera."""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.geometry.kinematics import inv, pose_to_mat  # noqa: E402
from ur5dual.vision.calibrate import solve_hand_eye        # noqa: E402
from ur5dual.vision.camera import Frame, Intrinsics        # noqa: E402
from ur5dual.vision.markers import find_marker             # noqa: E402


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name
          + (("  " + detail) if detail else ""))
    if not condition:
        raise AssertionError(name)


print("a printed marker is read in the same dictionary the popup names")
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
marker = cv2.aruco.drawMarker(dictionary, 42, 200)
image = np.full((480, 640), 255, dtype=np.uint8)
image[140:340, 220:420] = marker
colour = np.dstack([image, image, image])
intrinsics = Intrinsics(640, 480, 600, 600)
found = find_marker(Frame(np.ones((480, 640)), intrinsics, colour), 42, 0.05)
check("ID 42 is found", found.id == 42)
check("its centre is the middle of the rendered target",
      np.linalg.norm(found.centre - [320, 240]) < 1.0, str(found.centre))
check("its translation is finite and in front of the lens",
      np.all(np.isfinite(found.matrix)) and found.matrix[2, 3] > 0)


print("\neye-to-hand returns world<-camera for a marker on the flange")
camera_to_world = pose_to_mat([0.30, -0.20, 0.80, 0.20, -0.10, 0.30])
flange_to_marker = pose_to_mat([0.02, 0.01, 0.05, 0.10, 0.20, -0.10])
poses = [
    [-0.50, 0.10, 0.80, 0.00, 0.00, 0.00],
    [-0.45, 0.17, 0.83, 0.20, 0.00, 0.00],
    [-0.40, 0.05, 0.86, 0.00, 0.25, 0.00],
    [-0.55, 0.20, 0.80, 0.00, 0.00, 0.30],
    [-0.48, 0.12, 0.90, -0.20, 0.15, 0.10],
    [-0.38, 0.18, 0.85, 0.10, -0.20, 0.25],
    [-0.52, 0.03, 0.88, 0.25, 0.10, -0.20],
    [-0.43, 0.09, 0.81, -0.15, -0.20, -0.10],
]
flanges = [pose_to_mat(pose) for pose in poses]
readings = [inv(camera_to_world) @ flange @ flange_to_marker
            for flange in flanges]
solved, marker_on_flange, spread = solve_hand_eye(flanges, readings)
check("camera_to_world is recovered", np.allclose(solved, camera_to_world,
                                                   atol=1e-7))
check("the marker mounting offset is recovered",
      np.allclose(marker_on_flange, flange_to_marker, atol=1e-7))
check("consistent samples have zero spread", spread < 1e-8, str(spread))
