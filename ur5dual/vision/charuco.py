"""
The table, measured rather than typed.

Everything this cell believes about how far away anything is comes from four
corners and two numbers somebody entered on the Camera tab: `solve_opening_pnp`
puts the box at whatever depth makes an opening of the configured size fit the
rim it found. That answer is exact when the size is right and *proportionally*
wrong when it is not — and proportionally wrong in every direction at once,
because a pinhole's lateral scale is its depth. A crate typed 5% too wide
reads 5% too far away and moves 5% too far sideways, and nothing on screen
contradicts either. It is the one error that looks like no error.

A printed board contradicts both. Its squares are 25 mm because they were
drawn 25 mm, so the pose that fits them is metric without anyone being asked
the size of anything, and the plane it defines is the plane the boxes stand
on. Once that plane is known, the depth of any pixel on the table stops being
an inference and becomes arithmetic — a ray met with a plane — which is what
the rest of this module is.

Three things follow from that, and they are the reason to print the sheet:

  the true depth of any pixel on the surface, with no box size involved

  the true opening of a box, by meeting four rim rays with the plane the rim
  stands on. The size stops being typed. That is the fix for a cell whose
  directions are right and whose distances are not, because both come from
  the same wrong number.

  whether the depth stream agrees, checked against 35 corners whose distance
  is known exactly rather than against a flat wall somebody trusts.

The board is an instrument, not a calibration. Nothing here is written to a
config file, nothing here moves an arm, and the board's own frame is left
where OpenCV puts it — its origin at the first inner corner, +Z out of the
printed face. Tying that frame to the cell's world is `camera_to_world`'s job
and is deliberately not attempted from one sheet of paper lying on a table.
"""

import cv2
import numpy as np

from .markers import camera_matrix


DICTIONARY_ID = cv2.aruco.DICT_4X4_50

# The sheet in docs/calibration: charuco_6x8_25mm_18mm_4x4_50_a4.pdf, whose
# raster is identical to what CharucoBoard_create draws for these numbers.
SQUARES_X, SQUARES_Y = 6, 8
SQUARE_LENGTH = 0.025
MARKER_LENGTH = 0.018

# The millimetres the sheet's check line was drawn at. `printed_board` reads a
# measurement of it as the scale a printer actually applied.
SCALE_LINE = 100.0

# Of the 35 inner corners. A pose from a handful of corners in one part of the
# picture is a pose whose tilt is guessed, and tilt is the whole point here:
# every depth this module returns is the plane's, and a plane pivoted about
# its middle is exactly right there and wrong everywhere a box might be.
MIN_CORNERS = 12

# What the corners may miss their own projection by before the pose is refused.
# Well above the ~0.2 px a flat sheet in focus achieves, and well below what a
# curled print or a wrong square size produces.
MAX_REPROJECTION = 2.0


class BoardError(RuntimeError):
    pass


def make_board(squares_x=SQUARES_X, squares_y=SQUARES_Y,
               square=SQUARE_LENGTH, marker=MARKER_LENGTH):
    """The printed sheet, as OpenCV's description of it."""
    return cv2.aruco.CharucoBoard_create(
        int(squares_x), int(squares_y), float(square), float(marker),
        cv2.aruco.Dictionary_get(DICTIONARY_ID))


def printed_board(check_line=SCALE_LINE):
    """The board as it actually came off a printer, not as it was drawn.

    Every sheet carries a line drawn 100 mm long. A printer that shrank the
    page to fit its own margins — the usual one, and usually to about 95% —
    prints that line short by exactly the factor it shrank everything else by,
    the squares included. Told what the line really measures, this returns a
    board whose squares are the size they physically are, and every distance
    downstream is right again.

    A rescaled sheet is not a damaged one. Detection reads patterns and cares
    nothing for size; only the pose carries metres, and it carries them from
    these numbers. What matters is that the number is *measured* rather than
    assumed, which is the same claim this whole module makes about a box.

    Two things are worth knowing before relying on it rather than reprinting.

    The first is that nothing downstream can catch a wrong answer here. A
    homography through a planar target fits perfectly at any scale — a board
    declared 5% large simply sits 5% further away, with the same corners, the
    same reprojection, and no residual anywhere that changes. This is the very
    error the board was printed to expose, moved one step back; the only
    independent witness left is `depth_agreement`.

    The second is that a ruler across one 25 mm square measures scale to a few
    percent at best. The 100 mm line exists because it is four times longer,
    and reading the whole 150 mm board width is better still.
    """
    scale = float(check_line) / SCALE_LINE
    if not 0.5 <= scale <= 2.0:
        raise BoardError(
            "a %.1f mm line where 100 mm was drawn is not a scaled print, it "
            "is a mismeasurement or the wrong sheet" % float(check_line))
    return make_board(square=SQUARE_LENGTH * scale,
                      marker=MARKER_LENGTH * scale)


def _grey(color):
    image = np.asarray(color)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


class BoardPlane:
    """A plane in the camera's frame, as `normal · X = offset` in metres.

    The normal is kept pointing back at the lens, so `lifted` raising a plane
    by a box's height raises it *towards* the camera whichever way round
    OpenCV happened to resolve the board's own axes. A sign convention that
    depends on how a sheet of paper was laid down is a sign convention that
    will be wrong one day with nothing to show for it.
    """

    def __init__(self, normal, offset):
        normal = np.asarray(normal, dtype=float).reshape(3)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            raise BoardError("a plane needs a direction to face")
        self.normal = normal / length
        self.offset = float(offset) / length
        if self.offset > 0:
            # `offset` is the signed distance from the lens along the normal.
            # Positive means the normal points from the camera to the plane;
            # this class promises the other one.
            self.normal = -self.normal
            self.offset = -self.offset

    @property
    def distance(self):
        """How far the lens is off the plane, along the normal, in metres."""
        return abs(self.offset)

    def lifted(self, height):
        """The plane a surface `height` metres above this one lies in."""
        return BoardPlane(self.normal, self.offset + float(height))

    def rays(self, intrinsics, pixels):
        """Unit-Z directions through each pixel, in the camera's frame."""
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        return np.column_stack([
            (pixels[:, 0] - intrinsics.cx) / intrinsics.fx,
            (pixels[:, 1] - intrinsics.cy) / intrinsics.fy,
            np.ones(len(pixels))])

    def depth_at(self, intrinsics, pixels):
        """The metres of Z at which each pixel's ray meets this plane.

        This is the number the rest of the cell has been inferring from a
        typed box size. Here nothing is assumed about what is being looked
        at — only where the surface is.
        """
        rays = self.rays(intrinsics, pixels)
        along = rays @ self.normal
        if np.any(np.abs(along) < 1e-9):
            raise BoardError(
                "a ray runs along the surface, so nothing on it has a depth — "
                "the board is being seen almost edge-on")
        depths = self.offset / along
        if np.any(depths <= 0):
            raise BoardError(
                "the surface meets that ray behind the lens; the board pose "
                "and the pixels are not of the same picture")
        return depths

    def points_at(self, intrinsics, pixels):
        """Where each pixel lands on this plane, as (N, 3) metres."""
        rays = self.rays(intrinsics, pixels)
        return rays * self.depth_at(intrinsics, pixels)[:, None]

    def height_of(self, point):
        """How far a camera-frame point stands off this surface, in metres.

        Positive is towards the lens, which is up off a table under it. This
        is the number a cell wants whenever it asks "did the box change
        height", and it is not the same question as "did camera Z change".
        On a rig looking down at 61 degrees, sliding a box 100 mm along the
        table moves camera Z by 88 mm while this stays put — which is the
        whole reason a plane is worth measuring rather than assuming the lens
        axis points at the floor.
        """
        return float(self.normal @ np.asarray(point, dtype=float).reshape(3)
                     - self.offset)

    def to_dict(self):
        """The two numbers, plus one a person reading the file can check."""
        return {"normal": [float(v) for v in self.normal],
                "offset": float(self.offset),
                "tilt_deg": float(np.degrees(self.tilt))}

    @classmethod
    def from_dict(cls, d):
        if not d:
            return None
        return cls(d["normal"], d["offset"])

    def describe(self):
        return ("surface %.1f mm from the lens, tilted %.1f deg from square on"
                % (self.distance * 1000, np.degrees(self.tilt)))

    @property
    def tilt(self):
        """Angle between the surface's normal and the lens axis, in radians.

        `planar.py` prices this: an error in a box's height costs its tangent
        in lateral miss, so a cell that reads distances through a plane wants
        this small and a cell that reads full 6-DOF poses wants it large.
        """
        return float(np.arccos(np.clip(abs(self.normal[2]), -1.0, 1.0)))


class BoardPose:
    """Where the printed board is, and therefore where the table is."""

    def __init__(self, matrix, corners, ids, board_xy, reprojection,
                 sheet=None):
        self.matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
        self.corners = np.asarray(corners, dtype=float).reshape(-1, 2)
        self.ids = np.asarray(ids, dtype=int).reshape(-1)
        self.board_xy = np.asarray(board_xy, dtype=float).reshape(-1, 2)
        self.reprojection = float(reprojection)
        # The printed sheet's own extent, recorded because a caller measuring
        # something else against this plane may need to know how much of the
        # picture the paper covers.
        self.sheet = (None if sheet is None
                      else (float(sheet[0]), float(sheet[1])))

    def height_of(self, point):
        """How far a camera-frame point stands off the board, in metres."""
        return self.plane.height_of(point)

    @property
    def plane(self):
        """The board's own plane, in the camera's frame."""
        normal = self.matrix[:3, 2]
        return BoardPlane(normal, float(normal @ self.matrix[:3, 3]))

    def describe(self):
        return ("board from %d corners, fits to %.2f px; %s"
                % (len(self.corners), self.reprojection,
                   self.plane.describe()))


def find_board(frame, board=None):
    """The printed board's pose as camera<-board, in metres.

    Refused rather than returned when too few corners were seen or when the
    ones that were do not sit where the pose puts them: a board pose is used
    here to say how far away things are, and a pose that is merely the best
    available is a wrong distance with no symptom.
    """
    if frame is None or frame.color is None:
        raise BoardError("camera did not provide a colour image")
    board = board or make_board()
    dictionary = cv2.aruco.Dictionary_get(DICTIONARY_ID)
    image = _grey(frame.color)
    corners, ids, _rejected = cv2.aruco.detectMarkers(
        image, dictionary, parameters=cv2.aruco.DetectorParameters_create())
    if ids is None or not len(ids):
        raise BoardError("no ArUco markers of the board's dictionary are visible")
    found, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, image, board)
    if not found or charuco_ids is None or found < MIN_CORNERS:
        raise BoardError(
            "only %d of the board's corners were found; %d are the fewest "
            "that fix its tilt" % (int(found or 0), MIN_CORNERS))

    matrix = camera_matrix(frame.intrinsics)
    distortion = np.zeros(5, dtype=float)
    ok, rotation_vec, translation = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, matrix, distortion, None, None)
    if not ok:
        raise BoardError(
            "the board's corners were found but no pose fits them; they are "
            "probably all along one row")

    board_points = np.asarray(board.chessboardCorners, dtype=float)[
        charuco_ids.reshape(-1)]
    projected, _jacobian = cv2.projectPoints(
        board_points, rotation_vec, translation, matrix, distortion)
    seen = np.asarray(charuco_corners, dtype=float).reshape(-1, 2)
    reprojection = float(np.sqrt(np.mean(np.sum(
        (projected.reshape(-1, 2) - seen) ** 2, axis=1))))
    if reprojection > MAX_REPROJECTION:
        raise BoardError(
            "the board's corners miss their own projection by %.1f px, over "
            "the %.1f px limit — the print is curled, was scaled by the "
            "printer, or is not the %.0f mm board this expects"
            % (reprojection, MAX_REPROJECTION, SQUARE_LENGTH * 1000))

    rotation, _ = cv2.Rodrigues(rotation_vec)
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(translation, dtype=float).ravel()
    squares_x, squares_y = board.getChessboardSize()
    square = board.getSquareLength()
    return BoardPose(pose, seen, charuco_ids.reshape(-1),
                     board_points[:, :2], reprojection,
                     sheet=(squares_x * square, squares_y * square))


class Opening:
    """A box's rim, measured off the surface it stands on."""

    def __init__(self, points, sides, diagonals):
        self.points = np.asarray(points, dtype=float).reshape(4, 3)
        self.sides = np.asarray(sides, dtype=float).reshape(4)
        self.diagonals = np.asarray(diagonals, dtype=float).reshape(2)

    @property
    def length(self):
        """The longer pair of opposite sides, averaged, in metres."""
        return float(max(np.mean(self.sides[0::2]), np.mean(self.sides[1::2])))

    @property
    def width(self):
        return float(min(np.mean(self.sides[0::2]), np.mean(self.sides[1::2])))

    @property
    def squareness(self):
        """How far from a rectangle, as the difference of the diagonals."""
        return float(abs(self.diagonals[0] - self.diagonals[1]))

    def against(self, size):
        """What a configured size is out by, as a factor on each side.

        Two numbers rather than one, and deliberately not averaged into one.
        A size typed with the wrong *shape* as well as the wrong scale cannot
        be corrected by multiplying at all, and an average hides exactly that:
        it produces a factor that works at the distance it was measured and
        drifts everywhere else, which is the fudge this module exists to
        replace. When the two agree, the configured size is the right shape
        and their common value is what every distance the detector reports is
        out by — the pose it solves scales with the size it was given. When
        they do not, the answer is not a factor but `length` and `width`.
        """
        length, width = float(size[0]), float(size[1])
        if min(length, width) <= 0:
            raise BoardError("a box has no zero-length side")
        return (self.length / length, self.width / width)

    def describe(self):
        return ("opening %.1f x %.1f mm, sides %s mm, out of square %.1f mm"
                % (self.length * 1000, self.width * 1000,
                   " ".join("%.1f" % (side * 1000) for side in self.sides),
                   self.squareness * 1000))


def measure_opening(plane, intrinsics, corners, height=0.0):
    """The real size of a rim, from where its corners meet the surface.

    `corners` are the four rim pixels in the order `rim.order_corners` gives
    them, and `height` is how far the rim stands above the board — the box's
    height when the board is on the table beside it, and zero when the board
    is lying in the opening itself. Nothing is assumed about the box: the
    plane fixes the depth, and the depth fixes the size.
    """
    corners = np.asarray(corners, dtype=float).reshape(4, 2)
    points = plane.lifted(height).points_at(intrinsics, corners)
    sides = [float(np.linalg.norm(points[(i + 1) % 4] - points[i]))
             for i in range(4)]
    diagonals = [float(np.linalg.norm(points[2] - points[0])),
                 float(np.linalg.norm(points[3] - points[1]))]
    return Opening(points, sides, diagonals)


def depth_agreement(frame, pose):
    """What the depth stream says at corners whose distance is known exactly.

    Returns the median bias and the spread, both in metres, over the corners
    where depth had anything to say. A bias is a depth stream that needs its
    scale looked at; a spread is noise, and tells you how much of the box
    height check's tolerance is the sensor rather than the box.
    """
    if frame.depth is None:
        raise BoardError("this frame carries no depth")
    depth = np.asarray(frame.depth, dtype=float)
    truth = pose.plane.depth_at(frame.intrinsics, pose.corners)
    rows = np.rint(pose.corners[:, 1]).astype(int)
    columns = np.rint(pose.corners[:, 0]).astype(int)
    inside = ((rows >= 0) & (rows < depth.shape[0])
              & (columns >= 0) & (columns < depth.shape[1]))
    measured = np.zeros(len(truth))
    measured[inside] = depth[rows[inside], columns[inside]]
    said = measured > 0
    if not np.any(said):
        raise BoardError("the depth stream has nothing at any board corner")
    error = measured[said] - truth[said]
    return (float(np.median(error)), float(np.std(error)), int(said.sum()))
