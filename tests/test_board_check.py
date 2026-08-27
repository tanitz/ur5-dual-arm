"""The board tool's two staged captures: what `a` writes, and what it refuses."""

import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.config import CellConfig                         # noqa: E402
from ur5dual.tools import board_check                         # noqa: E402
from ur5dual.tools.board_check import (                       # noqa: E402
    Staged, _apply_live, best_rim, live, rank_rims, rim_candidates,
)
from ur5dual.vision.camera import Frame, SimCamera            # noqa: E402
from ur5dual.vision.charuco import (                          # noqa: E402
    BoardPlane, find_board, printed_board,
)
from ur5dual.vision.detect import OpenBoxDetector             # noqa: E402


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name
          + (("  " + detail) if detail else ""))
    if not condition:
        raise AssertionError(name)


WORK = tempfile.mkdtemp(prefix="board-check-")
CONFIG = os.path.join(WORK, "cell.yaml")
REAL, TYPED = (0.160, 0.080), (0.200, 0.100)
PLANE_Z = 0.70
# The sim's box really is this tall. `RIM` is what cell.yaml claims, and the
# two differ on purpose: the height is one more number that would otherwise be
# typed, and a plane put 20 mm wrong is a size measured wrong.
SIM_HEIGHT, RIM = 0.08, 0.10


def a_config():
    fresh = CellConfig.load("config/cell.yaml")
    fresh.vision["box_size"] = [TYPED[0], TYPED[1], RIM]
    # An uncalibrated cell, whatever this one happens to be. The repo's own
    # config is the base here, and once somebody has actually run the board
    # against this cell it carries a real surface — which silently turned
    # "step 2 wrote the plane" into a check that passed before it ran.
    fresh.vision["surface"] = None
    fresh.save_vision(CONFIG)
    return CellConfig.load(CONFIG)


def a_box_picture():
    """The sim's box on its table, with no board anywhere — step one's view."""
    lens = SimCamera(plane_z=PLANE_Z, box_size=REAL,
                     box_height=SIM_HEIGHT).open()
    frame = lens.read()
    lens.close()
    return frame


class FlatBoard:
    """A board pose standing in for a sheet lying on the sim's own table.

    The sim camera cannot render a ChArUco sheet, and `find_board` is already
    measured against a rendered one in test_charuco. What is under test here
    is the pairing: a surface from one picture, a rim from another.
    """

    def __init__(self, z=PLANE_Z):
        self.matrix = np.diag([1.0, -1.0, -1.0, 1.0])
        self.matrix[:3, 3] = [0.0, 0.0, z]
        self.corners = np.zeros((35, 2))
        self.sheet = (0.15, 0.20)

    @property
    def plane(self):
        from ur5dual.vision.charuco import BoardPlane
        return BoardPlane([0.0, 0.0, -1.0], -self.matrix[2, 3])

    def height_of(self, point):
        plane = self.plane
        return float(plane.normal @ np.asarray(point, float).reshape(3)
                     - plane.offset)


print("step one keeps a rim without knowing any size")
picture = a_box_picture()
notes = {}
found = OpenBoxDetector(box_size=(*TYPED, RIM),
                        confirm_frames=1).find(picture, notes)
check("the detector answers even though the configured size is 40 mm wrong",
      found is not None, str(notes.get("reason") or notes.get("state")))
check("and says so through depth rather than by refusing",
      found.depth_disagree is not None and abs(found.depth_disagree) > 0.030,
      "depth disagrees by %+.0f mm" % (found.depth_disagree * 1000))
rim = best_rim(picture, rim_candidates(picture))
check("the same rim is there for this tool to keep, size or no size",
      rim is not None)
check("and it is four corners", np.asarray(rim).shape == (4, 2))


# The bug this ranking exists for: a carton standing on a crate. The crate lid
# is a bigger rectangle than the carton's face and step one kept it every
# time, while the Camera tab locked the carton — measured on this cell, 91764
# px of crate against 9714 px of carton, 186 px apart and nothing saying so.
def _stacked(near_mm, area_px):
    """A quad of a given size whose near rim reads a given depth."""
    half = np.sqrt(area_px) / 2.0
    return np.array([[320 - half, 240 + half], [320 + half, 240 + half],
                     [320 + half, 240 - half], [320 - half, 240 - half]])


_crate, _carton = _stacked(499, 91764), _stacked(452, 9714)
_depth = np.full((480, 640), 0.499)
cv2.fillConvexPoly(_depth, np.rint(_carton).astype(np.int32), 0.452)
_stack = Frame(_depth, picture.intrinsics, picture.color)
_order = rank_rims(_stack, [_crate, _carton])
check("the carton standing on the crate is offered before the crate itself, "
      "though the crate's lid is nine times its area",
      np.allclose(_order[0], _carton),
      "%.0f px first" % abs(cv2.contourArea(_order[0].astype(np.float32))))
check("and the crate is still offered, because a bare crate is a box too",
      len(_order) == 2 and np.allclose(_order[1], _crate))
check("a rim depth cannot reach sorts to the back rather than winning",
      np.allclose(rank_rims(Frame(np.zeros((480, 640)), picture.intrinsics,
                                  picture.color), [_crate, _carton])[0],
                  _crate))


print("\nand step two turns it into millimetres")
staged = Staged()
check("with nothing captured it says which key to press",
      staged.opening()[2] == "capture the box first — press 1")
staged.take_box(picture, rim)
check("after the box it asks for the board where the box stood",
      "press 2" in staged.opening()[2], staged.opening()[2])
staged.take_board(FlatBoard())
opening, height, how = staged.opening()
check("the pair measures the opening",
      opening is not None and abs(opening.length - REAL[0]) < 0.010
      and abs(opening.width - REAL[1]) < 0.010,
      "" if opening is None else "%.1f x %.1f mm against %.0f x %.0f"
      % (opening.length * 1000, opening.width * 1000,
         REAL[0] * 1000, REAL[1] * 1000))
check("and the rim's height is measured, not taken from the config",
      how == "measured" and abs(height - SIM_HEIGHT) < 0.015,
      "%.0f mm measured where cell.yaml says %.0f mm"
      % (height * 1000, RIM * 1000))
wrong = Staged(asked_height=RIM)
wrong.take_box(picture, rim)
wrong.take_board(FlatBoard())
wrong.height = lambda: (RIM, "asked for")
# A height error is a pure scale error on the size: measuring at 600 mm what
# stands at 620 mm shrinks everything by that ratio, tilt or no tilt.
check("had the config's height been believed, the size would be millimetres out",
      abs(wrong.opening()[0].length - REAL[0]) > 0.005,
      "%.1f mm at the configured height against %.0f mm real"
      % (wrong.opening()[0].length * 1000, REAL[0] * 1000))


print("\nwhat `a` writes")
config = a_config()
said = _apply_live(config, staged, CONFIG, log=lambda text: None)
kept = CellConfig.load(CONFIG).vision
written = kept["box_size"]
check("all three numbers reach cell.yaml, the height included",
      abs(written[0] - opening.length) < 1e-3
      and abs(written[1] - opening.width) < 1e-3
      and abs(written[2] - height) < 1e-3, str(written))
check("and it says so in words an operator can read", "wrote" in said, said)
check("the surface goes with it, which is the half a program needs",
      kept["surface"] is not None
      and abs(kept["surface"]["offset"] + PLANE_Z) < 1e-6,
      str(kept.get("surface")))
check("and it comes back as the plane it was",
      abs(BoardPlane.from_dict(kept["surface"]).height_of([0, 0, PLANE_Z]))
      < 1e-9)
check("a box slid across that plane keeps its height, where camera Z does not",
      abs(BoardPlane.from_dict(kept["surface"]).height_of(
          [0.1, 0.0, PLANE_Z])) < 1e-9)

print("\nand the surface alone is worth writing")
config = a_config()
board_only = Staged()
board_only.take_board(FlatBoard())
said = _apply_live(config, board_only, CONFIG, log=lambda text: None)
kept = CellConfig.load(CONFIG).vision
check("step 2 on its own saves the plane", kept["surface"] is not None, said)
check("and leaves the box size alone, having measured no box",
      kept["box_size"] == [TYPED[0], TYPED[1], RIM], str(kept["box_size"]))

config = a_config()
half = Staged()
half.take_box(picture, rim)
said = _apply_live(config, half, CONFIG, log=lambda text: None)
check("a box with no surface behind it writes nothing",
      CellConfig.load(CONFIG).vision["box_size"] == [TYPED[0], TYPED[1], RIM]
      and CellConfig.load(CONFIG).vision["surface"] is None)
check("and says what is still missing", "press 2" in said, said)

config = a_config()
said = _apply_live(config, Staged(), CONFIG, log=lambda text: None)
check("neither does an empty one", "press 1" in said, said)

print("\na rim height depth cannot see")
blind = Staged()
blind.take_box(Frame(np.zeros((480, 640)), picture.intrinsics, picture.color),
               rim)
blind.take_board(FlatBoard())
check("is asked for rather than assumed",
      blind.opening()[0] is None and "--height" in blind.opening()[2],
      blind.opening()[2])
asked = Staged(asked_height=RIM)
asked.take_box(Frame(np.zeros((480, 640)), picture.intrinsics, picture.color),
               rim)
asked.take_board(FlatBoard())
check("and `--height` then supplies it, saying so",
      asked.opening()[0] is not None and asked.opening()[2] == "asked for")


print("\nthe window itself, with keys typed at it")
os.environ.setdefault("DISPLAY", ":0")           # the window is stubbed out
WINDOW_CALLS = ("namedWindow", "imshow", "resizeWindow", "waitKey",
                "getWindowProperty", "destroyWindow", "imwrite")


class Spy(SimCamera):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.closed = False

    def close(self):
        self.closed = True


def press(keys, visible=-1.0):
    camera = Spy(plane_z=PLANE_Z, box_size=REAL, box_height=SIM_HEIGHT)
    real = {name: getattr(cv2, name) for name in WINDOW_CALLS}
    was = board_check.open_camera
    script = [ord(k) for k in keys]
    deadline = time.monotonic() + 15.0

    def wait_key(delay):
        time.sleep(0.005)
        if time.monotonic() > deadline:
            return ord("q")
        return script.pop(0) if script else ord("q")

    cv2.namedWindow = lambda *a, **k: None
    cv2.imshow = lambda *a, **k: None
    cv2.resizeWindow = lambda *a, **k: None
    cv2.waitKey = wait_key
    cv2.getWindowProperty = lambda *a, **k: visible
    cv2.destroyWindow = lambda *a, **k: None
    cv2.imwrite = lambda *a, **k: True
    board_check.open_camera = lambda vision, log=print: camera
    try:
        return live(a_config(), {"source": "sim"}, printed_board(), None,
                    (*TYPED, RIM), CONFIG, log=lambda text: None), camera
    finally:
        for name, fn in real.items():
            setattr(cv2, name, fn)
        board_check.open_camera = was


written, camera = press("q")
check("q ends it and lets the lens go", camera.closed)
check("a run that captured nothing wrote nothing", written == [])

written, camera = press("1q")
check("1 keeps the rim without the board being anywhere", camera.closed)

written, camera = press("2q")
check("2 with no board in view does not crash", camera.closed)
check("and nothing was written", CellConfig.load(CONFIG).vision["box_size"]
      == [TYPED[0], TYPED[1], RIM])

written, camera = press("1raq")
check("r sends it back to step 1, so a writes nothing after it",
      CellConfig.load(CONFIG).vision["box_size"] == [TYPED[0], TYPED[1], RIM])

written, camera = press("sq")
check("s writes a picture", len(written) == 1, str(written))

shutil.rmtree(WORK, ignore_errors=True)
print("\nFAILURES: 0")
