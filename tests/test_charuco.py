"""The printed board as a ruler: true depth, and a box size nobody typed."""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.vision.camera import Frame, Intrinsics            # noqa: E402
from ur5dual.vision.charuco import (                           # noqa: E402
    BoardError, BoardPlane, MARKER_LENGTH, SQUARES_X, SQUARES_Y,
    SQUARE_LENGTH, depth_agreement, find_board, make_board, measure_opening,
    printed_board,
)
from ur5dual.vision.detect import solve_opening_pnp            # noqa: E402


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name
          + (("  " + detail) if detail else ""))
    if not condition:
        raise AssertionError(name)


SHEET = (SQUARES_X * SQUARE_LENGTH, SQUARES_Y * SQUARE_LENGTH)
INTRINSICS = Intrinsics(640, 480, 600.0, 600.0)
BOARD = make_board()


def camera_from_board(tilt_deg, centre, sheet=None):
    """A board lying face-up on a table, seen from a lens above and in front.

    Rx(180) turns the sheet's +Z to face the camera, which is how a board on a
    table is seen; the tilt is then how far this rig is from looking straight
    down at it.
    """
    flip = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    angle = np.radians(tilt_deg)
    tilt = np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(angle), -np.sin(angle)],
                     [0.0, np.sin(angle), np.cos(angle)]])
    matrix = np.eye(4)
    matrix[:3, :3] = tilt @ flip
    sheet = SHEET if sheet is None else sheet
    matrix[:3, 3] = np.asarray(centre, float) - matrix[:3, :3] @ [
        sheet[0] / 2.0, sheet[1] / 2.0, 0.0]
    return matrix


def picture_of(pose, sheet=None):
    """The board rendered as this camera would see it at `pose`.

    The sheet is drawn flat and warped through the homography its own four
    corners give, which is exact for a plane and needs no renderer.
    """
    sheet = SHEET if sheet is None else sheet
    scale = 2000.0                      # pixels per metre on the flat sheet
    flat = BOARD.draw((int(SHEET[0] * scale), int(SHEET[1] * scale)))
    rotation = cv2.Rodrigues(pose[:3, :3])[0]
    sheet_corners = np.array([[0.0, 0.0, 0.0], [sheet[0], 0.0, 0.0],
                              [sheet[0], sheet[1], 0.0], [0.0, sheet[1], 0.0]])
    seen, _ = cv2.projectPoints(
        sheet_corners, rotation, pose[:3, 3],
        np.array([[INTRINSICS.fx, 0, INTRINSICS.cx],
                  [0, INTRINSICS.fy, INTRINSICS.cy], [0, 0, 1.0]]),
        np.zeros(5))
    # `board.draw` puts the board's origin at the bottom-left of the sheet it
    # draws — its +Y runs up the image, not down it. Pairing the corners the
    # other way renders the back of the paper, and a mirrored ArUco code is
    # not a code.
    flat_corners = np.float32([[0, flat.shape[0]],
                               [flat.shape[1], flat.shape[0]],
                               [flat.shape[1], 0], [0, 0]])
    warp = cv2.getPerspectiveTransform(flat_corners,
                                       seen.reshape(-1, 2).astype(np.float32))
    canvas = cv2.warpPerspective(
        flat, warp, (INTRINSICS.width, INTRINSICS.height),
        borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return np.dstack([canvas] * 3)


def pixels_of(points):
    """Where camera-frame points land in the picture."""
    points = np.asarray(points, float).reshape(-1, 3)
    return np.column_stack([
        points[:, 0] * INTRINSICS.fx / points[:, 2] + INTRINSICS.cx,
        points[:, 1] * INTRINSICS.fy / points[:, 2] + INTRINSICS.cy])


print("a board on the table is read as the plane the boxes stand on")
# 0.70 m is where this cell's lens sits: `home_references.box_home` recorded
# the box at 714 mm.
TILT, CENTRE = 30.0, [0.0, 0.02, 0.70]
pose = camera_from_board(TILT, CENTRE)
frame = Frame(np.zeros((480, 640)), INTRINSICS, picture_of(pose))
read = find_board(frame)
check("the board's place comes back from the picture",
      np.linalg.norm(read.matrix[:3, 3] - pose[:3, 3]) < 0.001,
      "%.2f mm out" % (np.linalg.norm(read.matrix[:3, 3] - pose[:3, 3]) * 1000))
turned = np.degrees(np.linalg.norm(cv2.Rodrigues(
    read.matrix[:3, :3].T @ pose[:3, :3])[0]))
check("and so does the way it is lying", turned < 0.5, "%.2f deg out" % turned)
check("its corners fit their own projection", read.reprojection < 0.5,
      "%d corners, %.2f px" % (len(read.corners), read.reprojection))

plane = read.plane
check("the surface's tilt is the tilt it was rendered at",
      abs(np.degrees(plane.tilt) - TILT) < 0.5,
      "%.2f deg" % np.degrees(plane.tilt))
check("its normal faces the lens rather than away from it",
      plane.normal[2] < 0, str(np.round(plane.normal, 4)))


print("\nthe depth of a pixel on that surface is arithmetic, not inference")
board_points = np.column_stack([
    np.asarray(BOARD.chessboardCorners, float)[:, :2],
    np.zeros(35)])
truth = (pose[:3, :3] @ board_points.T).T + pose[:3, 3]
depths = plane.depth_at(INTRINSICS, pixels_of(truth))
check("every corner's depth is the depth it was drawn at",
      np.max(np.abs(depths - truth[:, 2])) < 1e-3,
      "worst %.2f mm" % (np.max(np.abs(depths - truth[:, 2])) * 1000))
points = plane.points_at(INTRINSICS, pixels_of(truth))
check("and the whole point comes back, not only its Z",
      np.max(np.linalg.norm(points - truth, axis=1)) < 1e-3)


print("\nand a box standing on it is measured rather than typed")
TRUE_SIZE, HEIGHT = (0.186, 0.094), 0.100
# Off the table is towards the lens, which is the way `BoardPlane` keeps its
# normal pointing and the way `lifted` moves a surface.
up = plane.normal
on_surface = (pose[:3, :3] @ np.array([
    [SHEET[0] / 2 - TRUE_SIZE[0] / 2, SHEET[1] / 2 - TRUE_SIZE[1] / 2, 0.0],
    [SHEET[0] / 2 + TRUE_SIZE[0] / 2, SHEET[1] / 2 - TRUE_SIZE[1] / 2, 0.0],
    [SHEET[0] / 2 + TRUE_SIZE[0] / 2, SHEET[1] / 2 + TRUE_SIZE[1] / 2, 0.0],
    [SHEET[0] / 2 - TRUE_SIZE[0] / 2, SHEET[1] / 2 + TRUE_SIZE[1] / 2, 0.0],
]).T).T + pose[:3, 3]
rim = on_surface + HEIGHT * up
rim_pixels = pixels_of(rim)

opening = measure_opening(plane, INTRINSICS, rim_pixels, HEIGHT)
check("the rim's real size comes off the surface it stands on",
      abs(opening.length - TRUE_SIZE[0]) < 5e-4
      and abs(opening.width - TRUE_SIZE[1]) < 5e-4, opening.describe())
check("a rectangle reads as square", opening.squareness < 5e-4)
check("measuring at the wrong height gets the wrong size, as it must",
      abs(measure_opening(plane, INTRINSICS, rim_pixels, 0.0).length
          - TRUE_SIZE[0]) > 0.01,
      "%.1f mm at the table instead of the rim"
      % (measure_opening(plane, INTRINSICS, rim_pixels, 0.0).length * 1000))


print("\nwhich is the factor every distance the detector reports is out by")
TYPED = (0.200, 0.100)
typed_pose, _typed_error = solve_opening_pnp(rim_pixels, INTRINSICS, *TYPED)
true_pose, _true_error = solve_opening_pnp(rim_pixels, INTRINSICS, *TRUE_SIZE)
along, across = opening.against(TYPED)
check("a size typed 7% large puts the box 7% far away",
      abs(typed_pose[2, 3] / true_pose[2, 3]
          - np.mean([TYPED[0] / TRUE_SIZE[0], TYPED[1] / TRUE_SIZE[1]])) < 0.01,
      "%.0f mm typed vs %.0f mm true"
      % (typed_pose[2, 3] * 1000, true_pose[2, 3] * 1000))
check("the board names how far out each side is",
      abs(along - TRUE_SIZE[0] / TYPED[0]) < 0.002
      and abs(across - TRUE_SIZE[1] / TYPED[1]) < 0.002,
      "x%.4f along, x%.4f across" % (along, across))
check("they disagree, because this size is the wrong shape as well as big",
      abs(along - across) > 0.005,
      "which is why the answer is %.1f x %.1f mm and not one factor"
      % (opening.length * 1000, opening.width * 1000))
check("the rim's true depth is the board's, lifted by the box",
      abs(true_pose[2, 3] - plane.lifted(HEIGHT).depth_at(
          INTRINSICS, [pixels_of(rim.mean(axis=0)[None, :])[0]])[0]) < 2e-3,
      "%.1f mm" % (true_pose[2, 3] * 1000))


print("\nand what the depth stream says at 35 places whose distance is known")
depth_image = np.zeros((480, 640))
rows = np.rint(read.corners[:, 1]).astype(int)
columns = np.rint(read.corners[:, 0]).astype(int)
depth_image[rows, columns] = (plane.depth_at(INTRINSICS, read.corners)
                              + 0.004)          # a sensor reading 4 mm long
bias, spread, said = depth_agreement(
    Frame(depth_image, INTRINSICS, frame.color), read)
check("a biased depth stream is named as biased",
      abs(bias - 0.004) < 5e-4 and said == len(read.corners),
      "%+.1f mm over %d corners, spread %.1f mm"
      % (bias * 1000, said, spread * 1000))
try:
    depth_agreement(Frame(np.zeros((480, 640)), INTRINSICS, frame.color), read)
    check("and a stream with nothing to say is refused, not averaged", False)
except BoardError as exc:
    check("and a stream with nothing to say is refused, not averaged",
          True, str(exc))


print("\nhow far away an A4 board can still be believed")
reach = {}
for metres in (0.35, 0.70, 1.10):
    try:
        far = find_board(Frame(
            np.zeros((480, 640)), INTRINSICS,
            picture_of(camera_from_board(TILT, [0.0, 0.02, metres]))))
        reach[metres] = np.linalg.norm(
            far.matrix[:3, 3] - camera_from_board(
                TILT, [0.0, 0.02, metres])[:3, 3])
    except BoardError:
        reach[metres] = None
check("close up it is a fraction of a millimetre",
      reach[0.35] is not None and reach[0.35] < 0.0005,
      "%.2f mm at 350 mm" % (reach[0.35] * 1000))
check("at this cell's working distance it is still under a millimetre",
      reach[0.70] is not None and reach[0.70] < 0.001,
      "%.2f mm at 700 mm" % (reach[0.70] * 1000))
check("and past a metre the 18 mm codes run out of pixels, as printed",
      reach[1.10] is None or reach[1.10] > 0.002,
      "no answer" if reach[1.10] is None
      else "%.1f mm at 1100 mm" % (reach[1.10] * 1000))


print("\na sheet a printer shrank, told what size it really came out")
SHRUNK = 95.0                                    # the check line measured 95 mm
small = (SHEET[0] * SHRUNK / 100.0, SHEET[1] * SHRUNK / 100.0)
shrunk_pose = camera_from_board(TILT, [0.0, 0.02, 0.70], small)
shrunk = Frame(np.zeros((480, 640)), INTRINSICS,
               picture_of(shrunk_pose, small))
believed = find_board(shrunk)                    # taken at its drawn 25 mm
told = find_board(shrunk, printed_board(SHRUNK))  # told what it measures
check("a 95% print read as 25 mm sits 5% too far away",
      abs(believed.plane.distance / told.plane.distance - 100.0 / SHRUNK)
      < 0.005,
      "%.1f mm believed vs %.1f mm true"
      % (believed.plane.distance * 1000, told.plane.distance * 1000))
check("and told the truth it lands where it actually is",
      abs(np.linalg.norm(told.matrix[:3, 3] - shrunk_pose[:3, 3])) < 0.001,
      "%.2f mm out"
      % (np.linalg.norm(told.matrix[:3, 3] - shrunk_pose[:3, 3]) * 1000))
check("the wrong size leaves no trace in the fit, which is why it is asked for",
      abs(believed.reprojection - told.reprojection) < 0.01,
      "%.2f px either way" % told.reprojection)
try:
    printed_board(40.0)
    check("a line measured half its length is refused as a mismeasurement",
          False)
except BoardError as exc:
    check("a line measured half its length is refused as a mismeasurement",
          True, str(exc))


print("\nand a board it cannot trust is refused rather than returned")
try:
    find_board(Frame(np.zeros((480, 640)), INTRINSICS,
                     np.full((480, 640, 3), 255, np.uint8)))
    check("an empty picture is refused", False)
except BoardError as exc:
    check("an empty picture is refused", True, str(exc))

try:
    BoardPlane([0.0, 0.0, 0.0], 1.0)
    check("a plane with no direction is refused", False)
except BoardError as exc:
    check("a plane with no direction is refused", True, str(exc))

print("\nFAILURES: 0")
