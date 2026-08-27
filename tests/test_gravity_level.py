"""Which way is down, and what the floor under a crate is."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.vision.charuco import BoardPlane                  # noqa: E402
from ur5dual.vision.detect import (                            # noqa: E402
    Detection, OpenBoxDetector, height_check,
)
from ur5dual.vision.imu import (                               # noqa: E402
    G, ImuError, down_from_samples, level_error,
)
from ur5dual.tools.level_check import floor_plane              # noqa: E402


fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + (("  " + detail) if detail else ""))
    if not ok:
        fail += 1


# This cell's own geometry: a lens looking down and along at 61.5 degrees, so
# true vertical is 61.5 degrees off the lens axis and nowhere near camera Z.
TILT = np.radians(61.5)
UP = np.array([0.0, -np.sin(TILT), -np.cos(TILT)])
DOWN = -UP


print("gravity out of an accelerometer at rest")
# At rest the sensor reads the push holding it up, not the pull on it. Get
# that sign wrong and every height in the cell is upside down while every
# magnitude stays right.
_rest = np.tile(-DOWN * G, (30, 1)) + np.random.default_rng(0).normal(0, 0.002, (30, 3))
_down, _spread = down_from_samples(_rest)
check("down comes back as the direction gravity pulls, not the reaction to it",
      float(_down @ DOWN) > 0.9999, "%.5f" % float(_down @ DOWN))
check("and it is a unit vector", abs(np.linalg.norm(_down) - 1) < 1e-9)
check("the spread says the rig was still", _spread < 0.001, "%.5f of g" % _spread)

try:
    down_from_samples(np.tile(-DOWN * G, (30, 1))
                      + np.random.default_rng(1).normal(0, 2.0, (30, 3)))
    check("a camera that was moving is refused", False)
except ImuError as exc:
    check("a camera that was moving is refused, not averaged into an answer",
          "moving" in str(exc), str(exc)[:52])
try:
    down_from_samples(np.tile([0.0, 0.0, 2.0], (30, 1)))
    check("a stream that is not the accelerometer is refused", False)
except ImuError as exc:
    check("and a reading that is not gravity at all is refused too",
          "not gravity" in str(exc), str(exc)[:52])


print("\nhow far a measured surface is out of level")
_level = BoardPlane(UP, float(UP @ np.array([0.0, 0.0, 1.30])))
check("a surface square to gravity reads level",
      np.degrees(level_error(_level.normal, DOWN)) < 1e-6)
_tipped = BoardPlane(
    np.array([np.sin(np.radians(2.0)), 0, 0]) + UP * np.cos(np.radians(2.0)),
    float(UP @ np.array([0.0, 0.0, 1.30])))
check("and one lying 2 deg out reads 2 deg out",
      abs(np.degrees(level_error(_tipped.normal, DOWN)) - 2.0) < 0.05,
      "%.2f deg" % np.degrees(level_error(_tipped.normal, DOWN)))
check("which side of the plane it was measured from cannot change that",
      abs(level_error(_tipped.normal, DOWN)
          - level_error(-np.asarray(_tipped.normal), DOWN)) < 1e-12)


print("\nthe floor under it")
DROP = 0.152                      # tape from the floor to the crate lid
_floor = floor_plane(_level, DOWN, DROP)
check("the floor is square to gravity, whatever the crate was",
      np.degrees(level_error(_floor.normal, DOWN)) < 1e-6)
_on_surface = _level.offset * _level.normal
check("and it sits the taped distance below the surface",
      abs(_floor.height_of(_on_surface) - DROP) < 1e-9,
      "%.1f mm" % (_floor.height_of(_on_surface) * 1000))
# The whole reason gravity is read rather than assumed: a crate 2 deg out
# would otherwise tilt the room's idea of height along with it.
_from_tipped = floor_plane(_tipped, DOWN, DROP)
check("a crate out of level does not tilt the floor it stands on",
      np.degrees(level_error(_from_tipped.normal, DOWN)) < 1e-6)


print("\nwhat the Camera tab then reads")
_rim = np.eye(4)
_rim[:3, 3] = np.array([0.0, 0.0, 1.30]) + 0.0967 * UP
_seen = Detection(_rim, (0.1924, 0.0756, 0.0967), np.zeros((4, 2)),
                  surface_height=_level.height_of(_rim[:3, 3]),
                  floor_height=_floor.height_of(_rim[:3, 3]))
check("the rim's height above the floor is the box plus the crate",
      abs(_seen.floor_height - (DROP + 0.0967)) < 1e-6,
      "%.1f mm" % (_seen.floor_height * 1000))
check("and the line names the floor while keeping how proud the box stands",
      "above floor" in height_check(_seen) and "proud" in height_check(_seen),
      height_check(_seen))
check("without a floor datum it still answers against the surface",
      "above surface" in height_check(Detection(
          _rim, (0.1924, 0.0756, 0.0967), np.zeros((4, 2)),
          surface_height=0.0967)))
check("a detector handed a floor that will not parse still detects",
      OpenBoxDetector(floor={"normal": "nonsense"}).floor is None)

print("\nFAILURES: %d" % fail)
sys.exit(1 if fail else 0)
