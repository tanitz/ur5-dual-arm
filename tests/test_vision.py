"""Checks for the known-size open-box detector used by the Camera tab."""

import csv
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.vision.camera import Frame, Intrinsics, RealSenseCamera, make_camera
from ur5dual.vision.charuco import BoardPlane
from ur5dual.vision.detect import (
    DetectionError, OpenBoxDetector, OpeningTracker, detect_opening_quad,
    height_check, solve_opening_pnp, surface_plane,
)
from ur5dual.vision.service import VisionService
import ur5dual.vision.detect as detect_module


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
upper = found.display_axis_origin()
upper_pixel = np.array([
    upper[0] * K.fx / upper[2] + K.cx,
    upper[1] * K.fy / upper[2] + K.cy,
])
edge = (int(np.argmin([
    (found.corners[i, 1] + found.corners[(i + 1) % 4, 1]) * 0.5
    for i in range(4)
])) + 2) % 4
expected_upper = (found.corners[edge]
                  + found.corners[(edge + 1) % 4]) * 0.5
check("the display-axis anchor is on the rim opposite the upper edge",
      np.linalg.norm(upper_pixel - expected_upper) < 2.0,
      "%s vs %s" % (np.round(upper_pixel, 1),
                      np.round(expected_upper, 1)))
check("depth remains an independent diagnostic",
      abs(found.depth_center - truth[2]) < 1e-9)


print("a small fixed box chooses the best depth-valid outline")
small_object = np.array([[-.10, -.05, 0], [.10, -.05, 0],
                         [.10, .05, 0], [-.10, .05, 0]], float)
small_corners, _ = cv2.projectPoints(
    small_object, np.array([0.5, 0.02, 0.01]),
    np.array([0.03, -0.04, 0.70]), MATRIX, None)
small_corners = small_corners.reshape(4, 2)
bad_corners = small_corners.copy()
bad_corners[2] += [40.0, -30.0]
small_detector = OpenBoxDetector(box_size=(0.20, 0.10, 0.10),
                                 confirm_frames=1)
_bright, _rim, _choose = (detect_module.find_bright_quads,
                          detect_module.find_rim_quad,
                          detect_module.choose_size)
try:
    detect_module.find_bright_quads = lambda *_a, **_k: [bad_corners,
                                                          small_corners]
    detect_module.find_rim_quad = lambda *_a, **_k: (None, None)
    detect_module.choose_size = lambda *_a, **_k: ((0.20, 0.10), 0.0, None)
    chosen, _measured, _off = small_detector._find_fixed_small(frame)
finally:
    detect_module.find_bright_quads = _bright
    detect_module.find_rim_quad = _rim
    detect_module.choose_size = _choose
_good_error = solve_opening_pnp(small_corners, K, 0.20, 0.10)[1]
_bad_error = solve_opening_pnp(bad_corners, K, 0.20, 0.10)[1]
check("it does not return the first candidate merely because depth agrees",
      np.allclose(chosen, small_corners)
      and _good_error < 1.0 and _bad_error > 4.0,
      "chose %.1f px over %.1f px" % (_good_error, _bad_error))


print("a box sliding along a table changes camera Z and not its height")
# The rig this cell actually has: a lens looking down and along at a table,
# 62 degrees off square-on, which is why camera Z and height parted company
# in the first place. The surface normal leans back towards the lens, and a
# rim standing on it is turned by the same amount — 118 degrees of roll,
# which is what the Camera tab reads on the real one.
TILT = np.radians(62.0)
NORMAL = np.array([0.0, -np.sin(TILT), -np.cos(TILT)])
# Along the surface, directly away from the lens. Sliding here is the one
# motion that must move camera Z and leave a height reading alone.
DOWNHILL = np.array([0.0, -np.cos(TILT), np.sin(TILT)])
WALL = SIZE[2]
RIM_AT_REST = np.array([0.0, 0.05, 1.30])
TABLE = BoardPlane(NORMAL, float(NORMAL @ RIM_AT_REST) - WALL)


def slid(metres):
    """The rim centre, `metres` further along the table from the lens."""
    return tuple(RIM_AT_REST + metres * DOWNHILL)


_rvec = (np.pi - TILT, 0.02, 0.01)
_frames = [scene(rvec=_rvec, tvec=slid(d))[0] for d in (0.0, 0.30)]
_seen = [OpenBoxDetector(box_size=SIZE, surface=TABLE.to_dict(),
                         confirm_frames=1).find(f) for f in _frames]
check("both positions are found at all",
      all(d is not None for d in _seen))
_dz = abs(_seen[1].centre[2] - _seen[0].centre[2])
_dh = abs(_seen[1].surface_height - _seen[0].surface_height)
check("camera Z moves by most of the slide, as the geometry says it must",
      _dz > 0.20, "%.0f mm of camera Z for 300 mm of slide" % (_dz * 1000))
check("height above the measured surface stays put across the same slide",
      _dh < 0.005, "%.1f mm" % (_dh * 1000))
check("that height is the box's own wall, not a distance from the lens",
      abs(_seen[0].surface_height - WALL) < 0.005,
      "%.1f mm against a %.0f mm wall"
      % (_seen[0].surface_height * 1000, WALL * 1000))
check("the readout line names the height and its error against the wall",
      "above surface" in height_check(_seen[0]), height_check(_seen[0]))

# Without a surface the pose is unchanged and only the height line goes.
_bare = OpenBoxDetector(box_size=SIZE, confirm_frames=1).find(_frames[0])
check("a cell that has not measured its surface still detects normally",
      _bare is not None
      and np.allclose(_bare.centre, _seen[0].centre, atol=1e-9)
      and _bare.surface_height is None)
check("and says which procedure would give it the missing number",
      "Measure Surface + Box" in height_check(_bare), height_check(_bare))
check("a surface that will not parse is not fatal to the detector",
      surface_plane({"normal": "nonsense"}) is None
      and surface_plane(None) is None)
check("a plane survives the trip through the config file's dict form",
      np.allclose(surface_plane(TABLE.to_dict()).normal, TABLE.normal))


print("the service keeps the tracker and writes diagnostics")
work = tempfile.mkdtemp(prefix="openbox-vision-")
service = VisionService({
    "source": "sim", "box_size": SIZE, "roi": (80, 100, 560, 380),
    "sim_plane_z": 1.20, "confirm_frames": 3,
    # the sim's own table, so the height column has something to hold
    "surface": {"normal": [0.0, 0.0, -1.0], "offset": -1.20},
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
_rows = [r for r in csv.DictReader(
             line for line in log_path.read_text().splitlines()
             if not line.startswith("#"))
         if r["found"] == "1"]
check("and carries the height beside camera Z, which is what a slide test "
      "is read from",
      bool(_rows) and all(r["surface_height_m"] for r in _rows),
      "%d found rows, height %s" % (len(_rows),
                                    _rows[0]["surface_height_m"] if _rows else "-"))
check("asking for a RealSense still constructs the real camera adapter",
      isinstance(make_camera({"source": "realsense"}), RealSenseCamera))

print("\nFAILURES: %d" % fail)
sys.exit(1 if fail else 0)
