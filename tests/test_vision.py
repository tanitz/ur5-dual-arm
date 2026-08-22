"""Checks for the known-size open-box detector used by the Camera tab."""

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.vision.camera import Frame, Intrinsics, RealSenseCamera, make_camera
from ur5dual.vision.detect import (
    DetectionError, OpenBoxDetector, OpeningTracker, detect_opening_quad,
    solve_opening_pnp,
)
from ur5dual.vision.service import VisionService


fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name +
          (("  " + detail) if detail else ""))
    if not ok:
        fail += 1


K = Intrinsics(640, 480, 600.0, 600.0, 320.0, 240.0)
MATRIX = np.array([[K.fx, 0, K.cx], [0, K.fy, K.cy], [0, 0, 1.0]])
SIZE = (0.60, 0.40, 0.20)
OBJECT = np.array([[-.30, -.20, 0], [.30, -.20, 0],
                   [.30, .20, 0], [-.30, .20, 0]], float)


def scene(rvec=(0.70, 0.05, 0.02), tvec=(0.03, 0.08, 1.20)):
    rvec = np.asarray(rvec, float).reshape(3, 1)
    tvec = np.asarray(tvec, float).reshape(3, 1)
    corners, _ = cv2.projectPoints(OBJECT, rvec, tvec, MATRIX, None)
    image = np.full((480, 640, 3), 40, np.uint8)
    cv2.fillConvexPoly(image, np.rint(corners).astype(np.int32), (170, 170, 170))
    return Frame(np.full((480, 640), tvec[2, 0]), K, color=image), tvec.ravel()


print("four edges become one metric pose")
frame, truth = scene()
corners, edges = detect_opening_quad(frame.color, (80, 100, 560, 380))
pose, error = solve_opening_pnp(corners, frame.intrinsics, *SIZE[:2])
check("the edge image and four ordered corners are returned",
      edges.shape == frame.depth.shape and corners.shape == (4, 2))
check("solvePnP recovers translation from the configured opening size",
      np.linalg.norm(pose[:3, 3] - truth) < 0.005,
      str(np.round(pose[:3, 3] - truth, 4)))
check("the projected rectangle agrees with its measured corners",
      error < 1.0, "%.2f px" % error)

try:
    detect_opening_quad(np.zeros((480, 640, 3), np.uint8))
    missing = False
except DetectionError:
    missing = True
check("an empty picture is not invented into a box", missing)


print("temporal filtering")
tracker = OpeningTracker(alpha=0.2, max_error=4, max_jump=35,
                         confirm_frames=3, hold_frames=5)
check("one frame is not enough to lock", tracker.update(corners, error)[0] is None)
tracker.update(corners + 1, error)
locked, state = tracker.update(corners, error)
check("three agreeing frames lock", locked is not None and state == "LOCKED")
held, state = tracker.update(corners + 100, error)
check("one jumping answer holds the old corners",
      state.startswith("HOLD") and np.max(np.abs(held - locked)) < 1e-9)
held, state = tracker.update(corners, 20.0)
check("a poor reprojection is rejected without moving the pose",
      state.startswith("REJECT") and np.max(np.abs(held - locked)) < 1e-9)


print("the object handed to Camera, Teach and FIND")
detector = OpenBoxDetector(box_size=SIZE, roi=(80, 100, 560, 380),
                           confirm_frames=1)
notes = {}
found = detector.find(frame, notes)
check("one accepted opening becomes a detection", found is not None)
check("its frame origin is the centre of the opening",
      np.linalg.norm(found.matrix()[:3, 3] - truth) < 0.005)
check("it carries the eight wireframe corners drawn by Camera",
      found.landmarks_3d().shape == (8, 3))
check("depth remains an independent diagnostic",
      abs(found.depth_center - truth[2]) < 1e-9)


print("the service keeps the tracker and writes diagnostics")
work = tempfile.mkdtemp(prefix="openbox-vision-")
service = VisionService({
    "source": "sim", "box_size": SIZE, "roi": (80, 100, 560, 380),
    "sim_plane_z": 1.20, "confirm_frames": 3,
    # asked for by name: the CSV is off unless someone is measuring, and
    # here someone is
    "log_enabled": True, "log_dir": work,
}).start()
try:
    reading = service.fresh(3.0)
    log_path = service.csv_log.path
finally:
    service.stop()
check("a fresh reading waits through LOCKING for a confirmed pose",
      reading.detection is not None, reading.why_not())
check("the session CSV is written when it is asked for",
      log_path.exists() and len(log_path.read_text().splitlines()) >= 3,
      str(log_path))
check("asking for a RealSense still constructs the real camera adapter",
      isinstance(make_camera({"source": "realsense"}), RealSenseCamera))

print("\nFAILURES: %d" % fail)
sys.exit(1 if fail else 0)
