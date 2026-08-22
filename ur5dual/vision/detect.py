"""Open-box pose detection used by the Camera tab, FIND and the CLI tool.

Find the four corners of a known rectangular opening, solve its metric pose
from calibrated intrinsics, reject geometrically bad answers, and stabilise
the remaining corners across frames. Camera coordinates follow RealSense:
+X right, +Y down, +Z forward. The returned frame is centred on the opening.

Corner-finding itself lives in `rim.py`, and so does the depth measurement
this module carries alongside every pose: how far the solved pose puts the
near rim, against how far the sensor says it is. That number is reported and
never used to reject, because it can only be earned where depth is real. A
synthetic frame with one flat depth everywhere would fail it while being
exactly right, and a detector that refused to answer there would be a
detector no test could hold still.
"""

import math

import cv2
import numpy as np

from .rim import (STANDARD_SIZES, choose_size, depth_disagreement,
                  find_bright_quads, find_rim_quad)


DEFAULT_BOX_SIZE = (0.60, 0.40, 0.20)

# How close to the search window's edge a corner may sit before it is more
# likely resting *on* that edge than on the box. Measured on this cell: an ROI
# six pixels too narrow moved the pose 34 mm and left the reprojection error
# at 1.6 px, which is to say invisible. Only depth caught it.
EDGE_MARGIN = 3.0

# What the depth cross-check may differ by and still be called agreement.
AGREE_MM = 30.0


class DetectionError(RuntimeError):
    pass


def roi_bounds(shape, roi=None):
    """The window to search in, as pixels of this frame.

    No ROI, which is now the default, means the whole picture. A window used
    to be mandatory, on the reasoning that it keeps a cluttered workshop out
    of the search — but the corner finder ranks candidates by hull area and by
    how much of the contour lies along the rectangle it reduces to, and does
    that job better than a box drawn once in a config file ever did.

    What the window did instead was fail silently. A box that strays outside
    it is not reported missing; it is reported at the window's own edge, with
    a reprojection error as good as any correct answer. Measured on this cell:
    six pixels of clipping moved the pose 34 mm at a reprojection error of
    1.6 px. Only the depth cross-check saw it.
    """
    height, width = shape[:2]
    if roi is None:
        return 0, 0, width - 1, height - 1
    if len(roi) != 4:
        raise DetectionError("ROI needs x1 y1 x2 y2")
    x1, y1, x2, y2 = map(int, roi)
    x1, x2 = sorted(np.clip([x1, x2], 0, width - 1).astype(int))
    y1, y2 = sorted(np.clip([y1, y2], 0, height - 1).astype(int))
    if x2 - x1 < 100 or y2 - y1 < 80:
        raise DetectionError("ROI is too small")
    return x1, y1, x2, y2


def detect_opening_quad(image, roi=None):
    """Return front-left, front-right, back-right, back-left image corners.

    A thin wrapper now: the work is `rim.find_rim_quad`, and what is left here
    is the ROI contract and the failure this package raises. The Hough
    line-grouping that used to live here has gone. It assumed the rim would
    arrive as four long straight segments in known angle bands, which held
    until a box had work standing proud of its rim — and then it held for no
    frame at all, measured 0 detections in 100 on this cell's own crate.
    """
    if image is None or np.asarray(image).ndim != 3:
        raise DetectionError("the camera did not provide a colour image")
    bounds = roi_bounds(image.shape, roi)
    corners, edges = find_rim_quad(image, bounds)
    if corners is None:
        raise DetectionError("no four-sided opening in ROI")
    return corners, edges


def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx, 0.0, intrinsics.cx],
                     [0.0, intrinsics.fy, intrinsics.cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def solve_opening_pnp(corners, intrinsics, length=0.60, width=0.40):
    object_points = np.array([
        [-length / 2, -width / 2, 0.0],
        [ length / 2, -width / 2, 0.0],
        [ length / 2,  width / 2, 0.0],
        [-length / 2,  width / 2, 0.0]], dtype=np.float64)
    matrix = camera_matrix(intrinsics)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, np.asarray(corners, np.float64), matrix, None,
        # ITERATIVE remains stable when the opening is nearly square-on.
        # IPPE is excellent at a tilt but becomes ambiguous in the simulated
        # (and calibration) view and can report >5 px error for exact corners.
        flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise DetectionError("solvePnP could not estimate the opening pose")
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, matrix, None)
    error = float(np.sqrt(np.mean(np.sum(
        (projected.reshape(-1, 2) - corners) ** 2, axis=1))))
    transform = np.eye(4)
    transform[:3, :3] = cv2.Rodrigues(rvec)[0]
    transform[:3, 3] = tvec.reshape(3)
    return transform, error


class OpeningTracker:
    def __init__(self, alpha=0.20, max_error=4.0, max_jump=35.0,
                 confirm_frames=4, hold_frames=15):
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.max_error = float(max_error)
        self.max_jump = float(max_jump)
        self.confirm_frames = max(1, int(confirm_frames))
        self.hold_frames = max(0, int(hold_frames))
        self.reset()

    def reset(self):
        self.corners = None
        self.candidate = None
        self.candidate_count = 0
        self.misses = 0

    @staticmethod
    def _jump(a, b):
        return float(np.max(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1)))

    def _confirm(self, corners):
        corners = np.asarray(corners, np.float64)
        if (self.candidate is not None and
                self._jump(corners, self.candidate) <= self.max_jump):
            self.candidate = 0.5 * self.candidate + 0.5 * corners
            self.candidate_count += 1
        else:
            self.candidate = corners.copy()
            self.candidate_count = 1
        return self.candidate_count >= self.confirm_frames

    def hold(self, reason):
        self.misses += 1
        if self.corners is not None and self.misses <= self.hold_frames:
            return self.corners.copy(), reason
        return None, reason

    def update(self, corners, error):
        if error > self.max_error:
            return self.hold("REJECT %.1f px" % error)
        corners = np.asarray(corners, np.float64)
        if self.corners is None:
            if not self._confirm(corners):
                return None, "LOCKING %d/%d" % (
                    self.candidate_count, self.confirm_frames)
            self.corners = self.candidate.copy()
            self.candidate = None
            self.candidate_count = 0
            self.misses = 0
            return self.corners.copy(), "LOCKED"
        jump = self._jump(corners, self.corners)
        if jump <= self.max_jump:
            self.corners = ((1.0 - self.alpha) * self.corners +
                            self.alpha * corners)
            self.candidate = None
            self.candidate_count = 0
            self.misses = 0
            return self.corners.copy(), "TRACKING"
        if self._confirm(corners):
            self.corners = self.candidate.copy()
            self.candidate = None
            self.candidate_count = 0
            self.misses = 0
            return self.corners.copy(), "RELOCKED"
        return self.hold("HOLD jump %.0f px" % jump)


def size_check(found):
    """One line on whether measured depth backs up the configured size.

    The pose came out of four corners and two numbers typed by a person, so it
    reads as confident whatever those numbers were. This is the only line that
    can contradict it — and when it does, it names the likelier of the two
    causes rather than always blaming the size: a box whose corners are
    sitting on the edge of the search window is a box that has been cropped,
    not a box of the wrong size.
    """
    if found is None:
        return ""
    if found.depth_disagree is None:
        return "size check: no depth at the near rim"
    off = found.depth_disagree * 1000
    if abs(off) <= AGREE_MM:
        return "size check %+.0f mm — agrees with depth" % off
    if found.clipped:
        return ("size check %+.0f mm — the opening runs off the edge of the "
                "view; move it fully into frame" % off)
    return "size check %+.0f mm — check the opening size" % off


class Detection:
    """Solved opening pose with the interface used by FIND and CameraPanel."""

    def __init__(self, transform, size, corners, reprojection_error=0.0,
                 state="TRACKING", depth_center=0.0, depth_disagree=None,
                 clipped=False):
        self._matrix = np.asarray(transform, dtype=float).copy()
        self.size = tuple(float(v) for v in size[:2])
        self.height = float(size[2])
        self.corners = np.asarray(corners, dtype=float).copy()
        self.reprojection_error = float(reprojection_error)
        self.state = str(state)
        self.depth_center = float(depth_center)
        # metres between where this pose puts the near rim and where depth
        # measured it; None when depth could not answer there. The size that
        # was configured is only as good as this number.
        self.depth_disagree = (None if depth_disagree is None
                               else float(depth_disagree))
        # whether a corner is sitting on the edge of the search window, which
        # is the other thing a large disagreement can mean
        self.clipped = bool(clipped)
        self.pixels = int(abs(cv2.contourArea(self.corners.astype(np.float32))))
        self.centre = self._matrix[:3, 3].copy()
        self.yaw = float(math.atan2(self._matrix[1, 0], self._matrix[0, 0]))

    @property
    def square(self):
        long_side, short_side = max(self.size), min(self.size)
        return long_side > 0 and (long_side - short_side) / long_side < 0.10

    @property
    def rotation(self):
        return self._matrix[:3, :3].copy()

    @property
    def translation(self):
        return self.centre + self.rotation @ np.array(
            [0.0, 0.0, -self.height / 2])

    @property
    def scale(self):
        return np.array([*self.size, self.height], dtype=float)

    def axes(self):
        return tuple(self.rotation[:, i] for i in range(3))

    def landmarks_3d(self):
        length, width = self.size
        top = np.array([[-length / 2, -width / 2, 0.0],
                        [ length / 2, -width / 2, 0.0],
                        [ length / 2,  width / 2, 0.0],
                        [-length / 2,  width / 2, 0.0]])
        bottom = top.copy()
        bottom[:, 2] = -self.height
        return np.vstack([top, bottom]) @ self.rotation.T + self.centre

    def matrix(self):
        return self._matrix.copy()

    def matrix_in(self, camera_to_world):
        return np.asarray(camera_to_world, dtype=float) @ self._matrix

    def describe(self):
        return ("open box %.0fx%.0f mm at %.0f %.0f %.0f mm, "
                "reprojection %.1f px, %s%s" %
                (self.size[0] * 1000, self.size[1] * 1000,
                 self.centre[0] * 1000, self.centre[1] * 1000,
                 self.centre[2] * 1000, self.reprojection_error, self.state,
                 "" if self.depth_disagree is None
                 else ", depth agrees to %+.0f mm"
                      % (self.depth_disagree * 1000)))


def _touches_edge(corners, roi, margin=EDGE_MARGIN):
    """Is any corner resting on the boundary of the window it was found in?"""
    corners = np.asarray(corners, dtype=float)
    x1, y1, x2, y2 = roi
    return bool(np.any(corners[:, 0] <= x1 + margin) or
                np.any(corners[:, 0] >= x2 - margin) or
                np.any(corners[:, 1] <= y1 + margin) or
                np.any(corners[:, 1] >= y2 - margin))


def _median_depth(depth, corners, radius=2):
    if depth is None:
        return 0.0
    u, v = map(int, np.rint(np.mean(corners, axis=0)))
    height, width = depth.shape
    patch = depth[max(0, v - radius):min(height, v + radius + 1),
                  max(0, u - radius):min(width, u + radius + 1)]
    valid = patch[patch > 0]
    return float(np.median(valid)) if len(valid) else 0.0


class OpenBoxDetector:
    """Stateful detector; one instance belongs to one camera stream."""

    def __init__(self, box_size=DEFAULT_BOX_SIZE, roi=None,
                 smoothing=0.20, max_reprojection=4.0,
                 max_corner_jump=35.0, confirm_frames=4, hold_frames=15,
                 auto_size=False, box_sizes=None):
        if len(box_size) == 2:
            box_size = (*box_size, DEFAULT_BOX_SIZE[2])
        self.box_size = tuple(float(v) for v in box_size)
        self.roi = tuple(int(v) for v in roi) if roi is not None else None
        self.auto_size = bool(auto_size)
        self.set_box_sizes(box_sizes or ())
        self.tracker = OpeningTracker(
            smoothing, max_reprojection, max_corner_jump,
            confirm_frames, hold_frames)

    def reset(self):
        self.tracker.reset()

    def set_box_sizes(self, sizes):
        """Remember optional L/W/H profiles used after auto size selection."""
        profiles = []
        for entry in sizes:
            try:
                values = tuple(float(v) for v in entry)
            except (TypeError, ValueError):
                continue
            if len(values) >= 3 and min(values[:3]) > 0:
                profiles.append(values[:3])
        self.box_sizes = profiles

    def _height_for(self, size):
        """Height saved for this L/W pair, or the currently configured one."""
        wanted = np.asarray(size, dtype=float)
        best = None
        for profile in self.box_sizes:
            direct = np.abs(wanted - np.asarray(profile[:2]))
            swapped = np.abs(wanted - np.asarray(profile[1::-1]))
            error = direct if np.linalg.norm(direct) <= np.linalg.norm(swapped) \
                else swapped
            if best is None or np.linalg.norm(error) < best[0]:
                best = (float(np.linalg.norm(error)), float(np.max(error)),
                        profile[2])
        # The detector's standard 195 mm face and an operator's nominal
        # 200 mm entry are the same stock in this cell.  Ten millimetres keeps
        # that practical rounding without borrowing a height from another SKU.
        if best is not None and best[1] <= 0.010:
            return best[2]
        return self.box_size[2]

    def find(self, frame, notes=None):
        notes = {} if notes is None else notes
        notes["roi"] = roi_bounds(frame.depth.shape, self.roi)
        notes["box_size"] = self.box_size
        raw_corners = None
        raw_error = None
        raw_pose = None
        try:
            raw_corners, measured_size, measured_off = self._find_corners(frame)
            if measured_size is not None:
                if not np.allclose(measured_size, self.box_size[:2],
                                   atol=1e-6):
                    # Never combine corners held from the previous object
                    # with a newly measured size; that would produce a
                    # plausible-looking pose at the wrong scale while the
                    # tracker is waiting to relock.
                    self.tracker.reset()
                self.box_size = (measured_size[0], measured_size[1],
                                 self._height_for(measured_size))
                notes.update(measured_size=measured_size,
                             measured_size_off=measured_off)
            raw_pose, raw_error = solve_opening_pnp(
                raw_corners, frame.intrinsics, *self.box_size[:2])
            corners, state = self.tracker.update(raw_corners, raw_error)
        except (DetectionError, cv2.error, ValueError) as exc:
            corners, state = self.tracker.hold("HOLD no opening")
            notes["reason"] = str(exc)
        notes.update(state=state, raw_corners=raw_corners,
                     raw_reprojection=raw_error, raw_pose=raw_pose)
        if corners is None:
            notes.setdefault("reason", state)
            return None
        transform, _ = solve_opening_pnp(
            corners, frame.intrinsics, *self.box_size[:2])
        depth = _median_depth(frame.depth, corners)
        # The independent half of the answer: everything above came out of the
        # colour image and an assumed size, and this is the only step that can
        # contradict it. Reported, not enforced — see the module docstring.
        disagree, edge = depth_disagreement(frame, corners, transform,
                                            self.box_size[1])
        clipped = _touches_edge(corners, notes["roi"])
        notes.update(corners=corners, depth_center=depth,
                     depth_disagree=disagree, clipped=clipped,
                     near_edge_mm=None if edge is None
                     else edge["length"] * 1000.0)
        return Detection(transform, self.box_size, corners,
                         raw_error or 0.0, state, depth, disagree, clipped)

    def _find_corners(self, frame):
        """Find the normal rim, or a depth-confirmed small bright carton."""
        if not self.auto_size:
            # Fixed size controls the metric pose, but it must also select the
            # matching RGB search strategy.  A 195x100 face is below the
            # large-rim detector's normal 5000 px cutoff, so merely changing
            # solvePnP's numbers cannot make that face appear.
            if max(self.box_size[:2]) <= 0.30:
                return self._find_fixed_small(frame)
            corners, _ = detect_opening_quad(frame.color, self.roi)
            return corners, None, None

        bounds = roi_bounds(frame.color.shape, self.roi)
        # Bright candidates go first deliberately.  A crate can remain the
        # largest quadrilateral while a small carton sits on it; depth must
        # confirm a standard size before that smaller face is accepted.
        candidates = find_bright_quads(frame.color, bounds)
        rim_error = None
        try:
            corners, _ = detect_opening_quad(frame.color, self.roi)
            candidates.append(corners)
        except DetectionError as exc:
            # A small box may be the only rectangle in view and fall below
            # the normal rim finder's area threshold.
            rim_error = exc
        sizes = [(length / 1000.0, width / 1000.0)
                 for length, width in STANDARD_SIZES]
        for candidate in candidates:
            size, off, _ = choose_size(candidate, frame, solve_opening_pnp,
                                       candidates=sizes,
                                       tolerance=AGREE_MM / 1000.0)
            if size is not None:
                return candidate, size, off
        if rim_error is not None:
            raise rim_error
        return corners, None, None

    def _find_fixed_small(self, frame):
        """A small configured face, accepted only when depth fits its size."""
        bounds = roi_bounds(frame.color.shape, self.roi)
        candidates = find_bright_quads(frame.color, bounds)

        # A small box need not be white.  Run the ordinary edge/hull strategy
        # at the small-object area threshold as a second route, rather than
        # allowing the much larger crate underneath to win by area.
        rim, _ = find_rim_quad(frame.color, bounds, min_area=1500)
        if rim is not None:
            candidates.append(rim)

        fixed = [tuple(self.box_size[:2])]
        closest = None
        for candidate in candidates:
            size, off, _ = choose_size(candidate, frame, solve_opening_pnp,
                                       candidates=fixed,
                                       tolerance=AGREE_MM / 1000.0)
            if off is not None and (closest is None or abs(off) < abs(closest)):
                closest = off
            if size is not None:
                return candidate, None, off
        if closest is None:
            raise DetectionError("no small box has usable depth at its near edge")
        raise DetectionError("no small box agrees with the fixed size "
                             "(closest %+.0f mm)" % (closest * 1000.0))


def find_box(frame, notes=None, detector=None, **settings):
    """Compatibility entry point; services should keep a detector instance."""
    if detector is None:
        settings.setdefault("confirm_frames", 1)
        detector = OpenBoxDetector(**settings)
    return detector.find(frame, notes=notes)
