"""Checks for the three-number placement a fixed camera over one plane gives.

The camera here is the one this cell has: the intrinsics a RealSense reported
in `captures/box_20260821_083610.npz`, at the distance and the angle the same
capture was taken from. Every position is projected through it, so what these
checks measure is the arithmetic and not a lens.
"""

import math
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.vision.camera import Intrinsics
from ur5dual.vision.planar import (
    PlaneFile, PlaneMap, PlaneMapError, Placement, box_on_plane,
    fit_rectangle, planar_correction, rim_corners, spread_of, wrap_half,
)
from ur5dual.vision.rim import order_corners


fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name +
          (("  " + detail) if detail else ""))
    if not ok:
        fail += 1


# ── the cell, as measured ─────────────────────────────────────────────────
K = Intrinsics(640, 480, 607.51171875, 607.4620971679688,
               327.8101806640625, 248.51937866210938)
SIZE = (0.60, 0.40, 0.20)
TILT_DEG = 62.0                       # of the lens from vertical, as mounted
RANGE_M = 1.176


def camera_at(tilt_deg=TILT_DEG, distance=RANGE_M):
    """World-to-camera for a lens looking down at the plane's origin."""
    angle = math.radians(tilt_deg)
    position = np.array([0.0, -distance * math.sin(angle),
                         distance * math.cos(angle)])
    forward = -position / np.linalg.norm(position)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    rotation = np.column_stack([right, np.cross(forward, right), forward])
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = rotation
    camera_to_world[:3, 3] = position
    return np.linalg.inv(camera_to_world)


def pixels_of(x, y, yaw, lift=0.0, world_to_camera=None):
    """Where a box at (x, y, yaw) puts its four rim corners in the picture."""
    world_to_camera = camera_at() if world_to_camera is None else world_to_camera
    corners = rim_corners(x, y, yaw, SIZE)
    points = np.column_stack([corners, np.full(len(corners), lift)])
    camera = (world_to_camera[:3, :3] @ points.T).T + world_to_camera[:3, 3]
    return np.column_stack([K.fx * camera[:, 0] / camera[:, 2] + K.cx,
                            K.fy * camera[:, 1] / camera[:, 2] + K.cy])


PLACES = [(-0.25, -0.20, 0.0), (0.25, -0.20, math.radians(20)),
          (0.25, 0.20, math.radians(-20)), (-0.25, 0.20, math.radians(10))]


def fitted_map(world_to_camera=None):
    pixels, plane = [], []
    for x, y, yaw in PLACES:
        pixels.append(pixels_of(x, y, yaw, world_to_camera=world_to_camera))
        plane.append(rim_corners(x, y, yaw, SIZE))
    return PlaneMap.from_samples(np.vstack(pixels), np.vstack(plane),
                                 height=SIZE[2])


# ── the map ───────────────────────────────────────────────────────────────
print("a plane map is the exact model of this geometry, not a fit to it")
plane_map = fitted_map()
check("four placements give sixteen correspondences",
      plane_map.samples == 16, plane_map.description)
check("an exact geometry leaves nothing unexplained",
      plane_map.worst < 1e-6, "%.2e m worst" % plane_map.worst)

errors = []
for x, y, yaw in [(0.0, 0.0, 0.0), (0.30, 0.15, 0.0),
                  (0.20, -0.30, math.radians(25)),
                  (-0.30, 0.10, math.radians(-25))]:
    found = box_on_plane(pixels_of(x, y, yaw), plane_map, SIZE)
    errors.append((abs(found.x - x), abs(found.y - y),
                   abs(wrap_half(found.yaw - yaw))))
errors = np.array(errors)
check("x and y come back at places it was never fitted at",
      errors[:, :2].max() < 1e-6, "%.2e m" % errors[:, :2].max())
check("so does the turn, across the plus or minus 25 deg this cell allows",
      errors[:, 2].max() < 1e-6, "%.2e rad" % errors[:, 2].max())

far = box_on_plane(pixels_of(0.90, 0.80, math.radians(15)), plane_map, SIZE)
check("and far outside the patch the samples covered",
      abs(far.x - 0.90) < 1e-6 and abs(far.y - 0.80) < 1e-6,
      "%.2e m" % max(abs(far.x - 0.90), abs(far.y - 0.80)))

check("a map survives being written down and read back",
      np.allclose(PlaneMap.from_dict(plane_map.to_dict()).matrix,
                  plane_map.matrix))

# ── what it refuses ───────────────────────────────────────────────────────
print("\nwhat it refuses to answer")
try:
    PlaneMap.from_samples(np.zeros((3, 2)), np.zeros((3, 2)))
    check("three correspondences are refused", False)
except PlaneMapError as e:
    check("three correspondences are refused", True, str(e)[:44])

line = np.array([[t, 0.0] for t in (-0.30, -0.10, 0.10, 0.30, 0.45)])
try:
    PlaneMap.from_samples(np.arange(10, dtype=float).reshape(5, 2), line)
    check("places along a line are refused, not fitted perfectly", False)
except PlaneMapError as e:
    check("places along a line are refused, not fitted perfectly", True,
          str(e)[:52])
check("spread_of is what saw it",
      spread_of(line)[1] / spread_of(line)[0] < 0.10,
      "%.3f" % (spread_of(line)[1] / spread_of(line)[0]))

one = PlaneMap.from_samples(pixels_of(0.1, 0.05, math.radians(8)),
                            rim_corners(0.1, 0.05, math.radians(8), SIZE))
check("but one box placement is already two-dimensional, and is accepted",
      one.samples == 4, one.description)
check("solved from four, it can say nothing about its own error",
      one.worst < 1e-6, "%.2e m, which it would report for any four" % one.worst)

# ── the rectangle, and the labels ─────────────────────────────────────────
print("\na known rectangle is its own guard against the image's corner labels")
truth_yaw = math.radians(18.0)
mapped = rim_corners(0.1, 0.2, truth_yaw, SIZE)
centre, yaw, error, swapped = fit_rectangle(mapped, *SIZE[:2])
check("it recovers the centre and the turn it was built from",
      np.allclose(centre, [0.1, 0.2]) and abs(wrap_half(yaw - truth_yaw)) < 1e-12,
      "%.4f deg" % math.degrees(wrap_half(yaw - truth_yaw)))
check("and says the rectangle sat on the corners",
      error < 1e-12, "%.2e m" % error)
check("the front edge was the long one", not swapped)

rolled = np.roll(mapped, 1, axis=0)
_c, rolled_yaw, rolled_error, rolled_swapped = fit_rectangle(rolled, *SIZE[:2])
check("labels one out are detected rather than believed", rolled_swapped)
check("and are corrected, not merely flagged: the turn comes back true",
      abs(wrap_half(rolled_yaw - truth_yaw)) < 1e-12,
      "%.4f deg against %.4f" % (math.degrees(rolled_yaw),
                                 math.degrees(truth_yaw)))
check("the wrong labelling is the one that cannot fit 600x400",
      rolled_error < 1e-12, "%.2e m" % rolled_error)

half = fit_rectangle(np.roll(mapped, 2, axis=0), *SIZE[:2])
check("a half turn is not detectable, and is reported as the same placement",
      abs(wrap_half(half[1] - truth_yaw)) < 1e-12 and not half[3])

print("\nthe image's corner labels do roll inside this cell's own travel")
worst_yaw, rolled_at = 0.0, []
for degrees in (-25, -18, -12, 0, 12, 18, 25):
    yaw = math.radians(degrees)
    ordered = order_corners(pixels_of(0.0, 0.0, yaw).copy())
    found = box_on_plane(ordered, plane_map, SIZE)
    worst_yaw = max(worst_yaw, abs(wrap_half(found.yaw - yaw)))
    if found.side_swapped:
        rolled_at.append(degrees)
check("they roll well inside plus or minus 25 deg",
      len(rolled_at) > 0, "rolled at %s deg" % rolled_at)
check("and turning one way is not the same as turning the other",
      all(d < 0 for d in rolled_at) and max(rolled_at) < 0,
      "only negative angles rolled: %s" % rolled_at)
check("and the placement is right at every one of them anyway",
      worst_yaw < 1e-6, "%.2e rad worst" % worst_yaw)

# ── the correction ────────────────────────────────────────────────────────
print("\nthe correction is the transform a taught pick is carried by")
was = box_on_plane(pixels_of(0.0, 0.0, 0.0), plane_map, SIZE)
now = box_on_plane(pixels_of(0.12, -0.08, math.radians(21)), plane_map, SIZE)
correction = planar_correction(now, was)
check("it carries where the box was onto where the box is",
      np.allclose(correction @ was.matrix(), now.matrix(), atol=1e-9))
check("it leaves height alone",
      abs(correction[2, 3]) < 1e-12 and np.allclose(correction[2, :3], [0, 0, 1]))

# a pick taught 200 mm from the middle of the box, and where it must end up
taught_pick = np.array([0.20, 0.0, 0.0, 1.0])
carried = correction @ taught_pick
turn = math.radians(21)
expected = np.array([0.12 + 0.20 * math.cos(turn),
                     -0.08 + 0.20 * math.sin(turn), 0.0])
check("a pick 200 mm off the middle swings round with the box, not on itself",
      np.allclose(carried[:3], expected, atol=1e-9),
      "%.3f mm" % (1000 * np.linalg.norm(carried[:3] - expected)))

flipped = Placement(now.x, now.y, now.yaw + math.pi)
check("a box read a half turn round corrects by the small angle",
      np.allclose(planar_correction(flipped, was), correction, atol=1e-12))
check("wrap_half puts every difference in the nearer half",
      abs(wrap_half(math.radians(185)) - math.radians(5)) < 1e-12 and
      abs(wrap_half(math.radians(-100)) - math.radians(80)) < 1e-12)

# ── what it costs ─────────────────────────────────────────────────────────
print("\nwhat noise costs, and what leaving the plane costs")
rng = np.random.default_rng(11)
spread_x, spread_y, spread_yaw = [], [], []
for _ in range(400):
    noisy = pixels_of(0.30, 0.20, math.radians(15)) + rng.normal(0, 1.0, (4, 2))
    found = box_on_plane(noisy, plane_map, SIZE)
    spread_x.append(found.x)
    spread_y.append(found.y)
    spread_yaw.append(found.yaw)
sx, sy = np.std(spread_x) * 1000, np.std(spread_y) * 1000
syaw = math.degrees(np.std(spread_yaw))
check("one pixel of corner noise stays inside two millimetres across the view",
      sx < 2.0, "%.2f mm" % sx)
check("and inside five along it, which is the foreshortened axis",
      sy < 5.0, "%.2f mm sideways vs %.2f mm along" % (sx, sy))
check("the turn holds under the same noise", syaw < 0.6, "%.3f deg" % syaw)

lifted = box_on_plane(pixels_of(0.0, 0.0, 0.0, lift=0.010), plane_map, SIZE)
slip = math.hypot(lifted.x, lifted.y) / 0.010
check("a box off its plane slips by the tangent of the mounting angle",
      abs(slip - math.tan(math.radians(TILT_DEG))) < 0.10,
      "%.2f mm per mm, tan %.0f deg = %.2f"
      % (slip, TILT_DEG, math.tan(math.radians(TILT_DEG))))

overhead = camera_at(tilt_deg=35.0)
flat_map = fitted_map(overhead)
lifted = box_on_plane(pixels_of(0.0, 0.0, 0.0, lift=0.010,
                                world_to_camera=overhead), flat_map, SIZE)
check("which is why a nearer-overhead lens is worth more than any arithmetic",
      math.hypot(lifted.x, lifted.y) / 0.010 < slip / 2,
      "%.2f mm per mm at 35 deg against %.2f at %.0f"
      % (math.hypot(lifted.x, lifted.y) / 0.010, slip, TILT_DEG))

# ── the file the surface is written to ────────────────────────────────────
print("\nthe map and the places picks were taught against live together")
work = tempfile.mkdtemp(prefix="openbox-plane-")
path = os.path.join(work, "plane.json")

check("a file that is not there yet loads as an empty one",
      not PlaneFile.load(path).ready, PlaneFile.load(path).description)
try:
    PlaneFile.load(path).plane_map()
    check("but using what is not there is refused", False)
except PlaneMapError as e:
    check("but using what is not there is refused", True, str(e)[:46])

store = PlaneFile(plane_map, box_size=SIZE, path=path)
taught_at = box_on_plane(pixels_of(0.05, -0.10, math.radians(6)),
                         plane_map, SIZE)
store.teach("box_home", taught_at)
store.save()
check("it is written where it was asked to be", os.path.exists(path))

reloaded = PlaneFile.load(path)
check("and comes back with the map it was given",
      reloaded.ready and np.allclose(reloaded.map.matrix, plane_map.matrix),
      reloaded.description)
check("and with the box size, so a caller need not repeat it",
      tuple(reloaded.box_size) == SIZE)
back = reloaded.reference("box_home")
check("and with the place the pick was taught against, to the micron",
      abs(back.x - taught_at.x) < 1e-6 and abs(back.y - taught_at.y) < 1e-6
      and abs(back.yaw - taught_at.yaw) < 1e-9,
      back.describe())

try:
    reloaded.reference("nowhere")
    check("a reference nothing was taught against is refused", False)
except PlaneMapError as e:
    check("a reference nothing was taught against is refused", True,
          "and it says what is there" if "box_home" in str(e) else str(e)[:40])

moved_to = (0.05 + 0.14, -0.10 - 0.06, math.radians(6 + 19))
correction, found = reloaded.correction("box_home", pixels_of(*moved_to))
check("the file turns four image corners into the transform FIND stores",
      np.allclose(correction @ taught_at.matrix(), found.matrix(), atol=1e-6),
      found.describe())
shift = correction[:2, 3] + correction[:2, :2] @ taught_at.xy - taught_at.xy
check("which moves the box the way the box moved",
      np.allclose(shift, [0.14, -0.06], atol=1e-6),
      "%.1f %.1f mm" % (shift[0] * 1000, shift[1] * 1000))
check("and turns it the way the box turned",
      abs(math.degrees(math.atan2(correction[1, 0], correction[0, 0])) - 19)
      < 1e-4)

open(path, "w", encoding="utf-8").write("{ not json")
try:
    PlaneFile.load(path)
    check("a file that is there but unreadable is not silently empty", False)
except PlaneMapError as e:
    check("a file that is there but unreadable is not silently empty", True,
          str(e)[:40])

print("\nFAILURES: %d" % fail)
sys.exit(1 if fail else 0)
