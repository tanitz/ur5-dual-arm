#!/usr/bin/env python3
"""
Is the surface level, and how high is the floor under it.

    scripts/ur5dual-level                  # what the IMU says about the surface
    scripts/ur5dual-level --floor 152      # ...and the floor 152 mm below it

The Camera tab's `Measure Surface + Box` workflow measures the surface a box
stands on, and every height built
on it is measured along that surface's own normal. That answers "how tall is
the box" and says nothing about whether the surface is level — a crate lid on
a table 2 degrees out is a perfectly good plane, and a height above it is a
perfectly good number that is not a height above the floor.

The D435i's accelerometer settles it without a picture. At rest the only
force on it is gravity, so the direction it reads is the vertical the room
was built to, and the angle between that and the measured surface is how far
out of level the surface is. Nothing here needs the board, or a box, or an
arm — only a camera that has been left alone for a second.

`--floor` supplies the one thing gravity cannot: where zero is. Measure once,
with a tape, from the floor to the surface the boxes stand on, and this
writes `vision.floor` — the same plane the surface is, moved down by that
much and squared to true vertical rather than to the surface. The Camera tab
then reads heights above the floor instead of above the crate.

Only one process may hold the camera. Stop the Camera tab's live view first.
"""

import argparse
from datetime import datetime

import numpy as np

from ..config import CellConfig, DEFAULT_PATH
from ..vision.charuco import BoardPlane
from ..vision.imu import ImuError, level_error, read_gravity


# What a surface may be out of level by before it is worth saying so. Below
# this the difference it makes to a height is under a tenth of a millimetre
# across a crate, and a tool that complains about it is a tool nobody reads.
LEVEL_OK = np.radians(0.5)


def floor_plane(surface, down, drop):
    """The floor under `surface`, squared to gravity rather than to the crate.

    Two separate corrections, and it matters that they are separate. `drop` is
    where zero is — a tape measurement, the one thing an accelerometer cannot
    supply. True vertical is which way "up" runs, and taking it from gravity
    rather than from the surface normal is what stops a crate lid that is out
    of level from tilting the whole room's idea of height with it.
    """
    down = np.asarray(down, dtype=float).reshape(3)
    up = -down / np.linalg.norm(down)
    # A point the surface actually passes through: its own foot from the lens.
    on_surface = surface.offset * surface.normal
    return BoardPlane(up, float(up @ (on_surface + float(drop) * (-up))))


def report(surface, down, spread, drop=None, log=print):
    """What gravity says, and what it would change."""
    log("")
    log("gravity in the camera's frame")
    log("  down  %+.4f %+.4f %+.4f   (samples agree to %.2f%% of g)"
        % (down[0], down[1], down[2], spread * 100))
    if surface is None:
        log("")
        log("no vision.surface — use Camera > Measure Surface + Box first")
        return None
    out = level_error(surface.normal, down)
    log("")
    log("the measured surface")
    log("  %s" % surface.describe())
    log("  out of level by %.2f deg%s"
        % (np.degrees(out),
           "" if out <= LEVEL_OK else "  <- worth looking at"))
    if out > LEVEL_OK:
        # The cost, in the units the mistake is made in: a height read along
        # a normal that is `out` from vertical is short by that cosine, and a
        # box measured at one end of a crate is not the height of the same box
        # measured at the other.
        log("     a 100 mm height reads %.1f mm short of the true vertical one,"
            % (100.0 * (1.0 - np.cos(out))))
        log("     and sliding a box 300 mm along it changes its true height by"
            "  %.1f mm" % (300.0 * np.tan(out)))
    if drop is None:
        log("")
        log("  --floor MM writes vision.floor, and heights are read from the"
            " floor instead")
        return None
    return floor_plane(surface, down, drop)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--config", default=DEFAULT_PATH)
    ap.add_argument("--floor", type=float, default=None, metavar="MM",
                    help="millimetres from the floor up to the surface the "
                         "boxes stand on, measured with a tape. Writes "
                         "vision.floor")
    ap.add_argument("--samples", type=int, default=60,
                    help="accelerometer readings to average (default 60)")
    args = ap.parse_args()

    config = CellConfig.load(args.config)
    surface = BoardPlane.from_dict(config.vision.get("surface"))
    print("%s, camera %s" % (args.config, config.vision.get("source")))
    try:
        down, spread = read_gravity(config.vision.get("serial"), args.samples)
    except ImuError as exc:
        raise SystemExit(str(exc))

    floor = report(surface, down, spread,
                   None if args.floor is None else args.floor / 1000.0)
    if floor is None:
        return
    config.vision["floor"] = dict(
        floor.to_dict(),
        below_surface=round(args.floor / 1000.0, 4),
        measured=datetime.now().astimezone().isoformat(timespec="seconds"))
    config.save_vision(args.config)
    print("")
    print("wrote vision.floor: %s" % floor.describe())
    print("the Camera tab now reads heights above the floor")


if __name__ == "__main__":
    main()
