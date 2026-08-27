#!/usr/bin/env python3
"""
What the printed ChArUco board says about the table, and about the box on it.

The operator workflow now lives in the main Camera panel under
`Measure Surface + Box`.  This module owns the measurement primitives shared
by that dialog and their offline checks; it does not own a launcher.

Two pictures, in this order, with the camera left alone in between:

    1   the box on the table, board nowhere in sight — keeps its rim
    n   offer the next rim, when the one outlined is not the box meant
    2   the board lying where the box stood — keeps the surface
    a   write the surface to vision.surface, and the measured size to
        vision.box_size where the pair measured one
    r   start again from step 1
    s   keep what is on the screen: captures/board_<date>_<time>.jpg
    q   or ESC (the window's close button too, where the build reports it)

Staged, rather than both at once, because the two things this needs cannot
share a picture. The board is a large bright rectangle lying on exactly the
surface being measured, and no rule of shape can tell it from a box: scaling
a rectangle leaves a rectangle, so the sheet read at the rim's height is a
tidy rectangle too — a bigger one, which is worse, because bigger is how the
box was being chosen. Taking them one at a time removes the question instead
of answering it, and costs one keypress from somebody already standing there.

What must not move between the two is the camera, which here is bolted to the
frame. What must be shared is the surface: the board goes *where the box
stood*, not beside it.

Nothing is typed. The opening comes from the rim's four corners meeting the
plane; the rim's height above that plane comes from the depth stream along the
edge of the picture the box was captured in. `--height` exists for a sensor
that has nothing to say there, and is not otherwise wanted.

Step one offers the rims nearest the lens first, which on a cell whose box
stands on a crate is the box rather than the crate it stands on — that is what
`n` is for when it guesses wrong, and a bare crate is a box too.

Lay the board flat on the surface the boxes stand on, beside the box rather
than under it, with the whole sheet in view. Print it at 100% and check the
100 mm line before trusting anything here: a sheet scaled by a printer is a
ruler with the wrong millimetres, and every number below is that ruler.

Why this exists. The detector places a box at whatever depth makes an opening
of the *configured* size fit the rim it found, so a size typed 5% large reads
5% too far away and moves the arms 5% too far sideways — right directions,
wrong distances, and nothing on screen to say so. The board's squares are
25 mm because they were drawn 25 mm, so it can contradict both. `--box` meets
the four rim rays with the plane the rim stands on and measures the opening
outright; the size stops being typed.

Nothing is written until `a` is pressed, like the flange fit: the default is a
report, and `--once` has no way to write at all. Nothing here moves an arm
either — it has no way to command one.

A RealSense is opened by one process at a time. Stop the panel's live view
before running this, or it gets "no RealSense opened" while the GUI holds
the lens.
"""

import argparse
import os
import time
from datetime import datetime
from collections import deque

import cv2
import numpy as np

from ..config import DEFAULT_PATH, REPO_ROOT, CellConfig
from ..vision.camera import CameraError
from ..vision.charuco import BoardError, depth_agreement, find_board, \
    measure_opening, printed_board
from ..vision.detect import DetectionError, roi_bounds
from ..vision.rim import (
    find_bright_quads, find_rim_quad, measure_near_edge, why_nothing,
)
from .snap import caption, card, closes_are_reported, label, open_camera


WINDOW = "ur5dual — board"
KEYS = "n next rim    a apply    s save    r reset    q quit"
FLASH = 2.5

FOUND_BGR = (120, 200, 120)
KEPT_BGR = (200, 160, 90)          # what was captured, not what is in view
MISSING_BGR = (0, 119, 204)        # style.AMBER
RIM_BGR = (40, 170, 240)          # amber, in OpenCV's B-G-R order

# What a live measurement has to look like before `a` will write it. Both are
# about the same thing: one reading is a sample, and a size written from a
# sample is a size that happened to be on screen when somebody pressed a key.
LIVE_MIN_SAMPLES = 5
LIVE_MAX_SPREAD = 0.005
# Only the recent past. A box that was moved has to stop counting, or the
# median averages where it is with where it was.
LIVE_MEMORY = 30

# Pixels a quadrilateral must cover before it is looked at. The rim finder's
# own default of 5000 is for a crate; a 200 mm carton at this cell's range is
# well under it.
RIM_MIN_AREA = 1500
# How far a candidate's two diagonals may differ, as a fraction of its mean
# side, before it is not a rectangle lying where the box stands. Perspective
# is already undone by the time this is measured — these are metres on the
# plane — so a real rim is square to well within this and a shadow, a
# reflection or a chessboard square read at the wrong height is not.
RECTANGLE_TOLERANCE = 0.08
# And the shortest side worth calling a box. The board's own 25 mm squares are
# rectangles on a plane too, and this is what keeps them out of the answer.
MIN_OPENING = 0.040


def rim_candidates(frame, roi=None, notes=None):
    """Every quadrilateral in the picture that could be a rim, ungated.

    Deliberately not `OpenBoxDetector`. That gate accepts a quad only when a
    pose solved from the *configured* size agrees with measured depth to
    30 mm — which at this cell's range is 4% of the size, and the size is the
    very number this tool exists to correct. A cell whose box is 190 mm and
    whose config says 200 mm has every box refused, so the instrument that
    would find the right size cannot run until somebody has already guessed
    it closely enough not to need it.

    The board removes the reason for the gate. Scale comes from the plane, so
    a candidate no longer has to agree with an assumed size to be measured —
    it only has to be a quadrilateral, and `best_opening` then judges it by
    whether it is a rectangle where the box actually stands.
    """
    bounds = roi_bounds(np.asarray(frame.color).shape, roi)
    found = []
    quad, _edges = find_rim_quad(frame.color, bounds, min_area=RIM_MIN_AREA,
                                 notes=notes)
    if quad is not None:
        found.append(np.asarray(quad, dtype=float).reshape(4, 2))
    # A pale carton often has too few edge pixels to win on area against
    # whatever it is standing on, and the board beside it is all edges.
    found.extend(np.asarray(q, dtype=float).reshape(4, 2)
                 for q in find_bright_quads(frame.color, bounds))
    return found


def rank_rims(frame, plane_free_candidates):
    """Every candidate, the one standing highest first.

    Area used to decide this, on the reasoning that the largest quadrilateral
    in view is the box when the box is alone. The box is not alone. On this
    cell it is a 200x100 carton standing on a stacking crate, and the crate's
    lid is a rectangle nine times its area: measured on one frame, 91764 px of
    crate against 9714 px of carton, so step one kept the crate every time
    while the Camera tab locked the carton — 186 px apart, with nothing on
    screen saying the two tools disagreed.

    Depth settles it without any size being assumed, which is the property
    this whole tool is built on. A box put on a crate is nearer the lens than
    the crate, because the lens looks down: the same frame reads 452 mm at the
    carton's near rim against 499 mm at the crate's, and the clutter behind
    the cell reads 1963 to 2198 mm and sorts itself to the back.

    Nearest, not largest, is therefore the better guess — but it is still a
    guess, because a crate with nothing on it is a perfectly good box and so
    is the carton on top of it. Which one is meant is a fact about the job,
    not about the picture, so the order is offered rather than imposed and the
    operator steps through it. Candidates depth cannot reach at all keep their
    old area order, at the back, where a rim that the sensor has nothing to
    say about belongs.
    """
    ranked = []
    for quad in plane_free_candidates:
        quad = np.asarray(quad, dtype=float).reshape(4, 2)
        area = abs(cv2.contourArea(quad.astype(np.float32)))
        edge = measure_near_edge(frame.depth, frame.intrinsics,
                                 quad[0], quad[1])
        near = None if edge is None else float(np.linalg.norm(edge["centre"]))
        ranked.append((near is None, near or 0.0, -area, quad))
    ranked.sort(key=lambda item: item[:3])
    return [item[3] for item in ranked]


def best_rim(frame, plane_free_candidates):
    """The one `rank_rims` puts first, or None when there is nothing."""
    ranked = rank_rims(frame, plane_free_candidates)
    return ranked[0] if ranked else None


class Staged:
    """One box capture, one board capture, and the size the pair implies.

    Two presses rather than one picture, because the two things this needs
    cannot be in the same picture without being confused for each other: the
    board is a large bright rectangle lying on exactly the surface being
    measured, and telling it from a box by its shape is impossible — scaling
    a rectangle leaves a rectangle. Taking the box first and then putting the
    board where the box was removes the question instead of answering it, and
    costs an operator one extra keypress they were standing there for anyway.

    What must not change between the two is the camera, which in this cell is
    bolted to the frame. What must be the same is the surface: the board goes
    where the box stood, not beside it.
    """

    def __init__(self, asked_height=None):
        self.box = None              # (corners, frame) as captured
        self.board = None            # BoardPose, taken where the box stood
        self.asked_height = asked_height

    @property
    def stage(self):
        return 1 if self.box is None else 2

    def take_box(self, frame, corners):
        self.box = (np.asarray(corners, dtype=float).reshape(4, 2), frame)

    def take_board(self, pose):
        self.board = pose

    def reset(self):
        self.box = self.board = None

    def height(self):
        """How far the rim stood off the board, measured or asked for.

        Measured wins. The rim's height is one more number that would
        otherwise be typed, and a box 20 mm shorter than the config says puts
        the measuring plane 20 mm out — which at this cell's 61 degrees is
        36 mm of size error. `measure_near_edge` reads the depth stream along
        the rim of the picture the box was captured in, and the board supplies
        the datum, so neither half of it is assumed.
        """
        if self.box is None or self.board is None:
            return None, "nothing captured yet"
        corners, frame = self.box
        edge = measure_near_edge(frame.depth, frame.intrinsics,
                                 corners[0], corners[1])
        if edge is not None:
            stands = self.board.height_of(edge["centre"])
            if 0.005 <= stands <= 0.600:
                return stands, "measured"
            return (self.asked_height,
                    "depth put the rim %.0f mm off the board, which is not a "
                    "box standing on it" % (stands * 1000))
        if self.asked_height is None:
            return None, ("no depth at the rim of the captured picture, so "
                          "nothing says how high it stood — pass --height")
        return self.asked_height, "asked for"

    def opening(self):
        """The rim's real size, or why it cannot be had yet."""
        if self.box is None:
            return None, None, "capture the box first — press 1"
        if self.board is None:
            return None, None, ("now put the board where the box stood and "
                                "press 2")
        height, how = self.height()
        if height is None:
            return None, None, how
        corners, frame = self.box
        try:
            return (measure_opening(self.board.plane, frame.intrinsics,
                                    corners, height), height, how)
        except (BoardError, DetectionError, ValueError) as exc:
            return None, height, str(exc)


def _median_or_none(values):
    return float(np.median(values)) if values else None


def read_board(camera, frames, looking_for_box=False, height=0.0,
               board=None, roi=None, log=print):
    """Every frame's answer, so the spread between them can be reported.

    One picture is a sample. The board is steady and the lens is not, and the
    difference between those two is the only evidence available here about how
    much of a millimetre to believe.
    """
    planes, distances, tilts, corners, fits = [], [], [], [], []
    biases, spreads, middles = [], [], []
    refusals, box_refusals, seen_boxes = [], [], []
    last_frame = last_pose = None
    for _ in range(frames):
        frame = camera.read()
        last_frame = frame
        try:
            pose = find_board(frame, board)
        except BoardError as exc:
            refusals.append(str(exc))
            continue
        last_pose = pose
        plane = pose.plane
        planes.append(plane)
        distances.append(plane.distance)
        tilts.append(plane.tilt)
        corners.append(len(pose.corners))
        fits.append(pose.reprojection)
        try:
            bias, spread, _said = depth_agreement(frame, pose)
            biases.append(bias)
            spreads.append(spread)
        except BoardError:
            pass
        # The middle of the board, along the lens axis. `distance` above is
        # measured square to the surface, which on a 60 deg rig is half of it
        # and reads like a fault to anyone who knows how high the camera is.
        middles.append(float(plane.depth_at(
            frame.intrinsics, [pose.corners.mean(axis=0)])[0]))
        if not looking_for_box:
            continue
        # Only what is in this one picture. Measuring the box needs the board
        # to be somewhere else at the time, which is a sequence rather than a
        # frame — the window does that, and this only says whether a rim is
        # there to be captured.
        notes = {}
        rim = best_rim(frame, rim_candidates(frame, roi, notes))
        if rim is None:
            box_refusals.append(why_nothing(notes))
            continue
        seen_boxes.append(rim)
    return {
        "planes": planes, "distance": _median_or_none(distances),
        "distance_spread": float(np.std(distances)) if distances else None,
        "middle": _median_or_none(middles),
        "tilt": _median_or_none(tilts),
        "corners": _median_or_none(corners), "fit": _median_or_none(fits),
        "bias": _median_or_none(biases), "noise": _median_or_none(spreads),
        "rims": len(seen_boxes), "refusals": refusals,
        "box_refusals": box_refusals, "frames": frames,
        "last_frame": last_frame, "last_pose": last_pose,
        "last_box": seen_boxes[-1] if seen_boxes else None,
    }


def annotate(answer, path, log=print):
    """The last picture with what was found drawn on it.

    There is no live view in a command-line tool, so "did it see the box"
    is otherwise a question with no way to ask it. Green is the board, amber
    is the box rim; a picture with green and no amber says the box is the
    problem, and one with neither says the lens is.
    """
    frame = answer.get("last_frame")
    if frame is None or frame.color is None:
        raise SystemExit("no picture to draw on — the camera gave no colour")
    canvas = np.ascontiguousarray(np.asarray(frame.color)).copy()
    pose = answer.get("last_pose")
    if pose is not None:
        for corner in np.rint(pose.corners).astype(int):
            cv2.circle(canvas, tuple(corner), 4, (80, 220, 80), -1,
                       cv2.LINE_AA)
        cv2.putText(canvas, "board: %d corners" % len(pose.corners),
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (80, 220, 80), 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "board: not found", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 230), 2,
                    cv2.LINE_AA)
    box = answer.get("last_box")
    if box is not None:
        cv2.polylines(canvas, [np.rint(box).astype(np.int32)], True,
                      (40, 170, 240), 3, cv2.LINE_AA)
        cv2.putText(canvas, "box rim", (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (40, 170, 240), 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "box rim: not found", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 230), 2,
                    cv2.LINE_AA)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not cv2.imwrite(path, canvas):
        raise SystemExit("could not write %s" % path)
    log("\nwrote %s" % path)


def _draw_live(frame, seen, staged, configured, rate, flash):
    """The picture, what was found in it, and which step is being waited on."""
    view = np.ascontiguousarray(np.asarray(frame.color)).copy()
    for corner in np.rint(seen.get("board_corners", [])).astype(int):
        cv2.circle(view, tuple(corner), 4, FOUND_BGR, -1, cv2.LINE_AA)
    if seen.get("rim") is not None:
        cv2.polylines(view, [np.rint(seen["rim"]).astype(np.int32)], True,
                      RIM_BGR, 2, cv2.LINE_AA)
    if staged.box is not None:
        # What was kept, not what is in front of the lens now. Once the box has
        # been taken away this is the only sight of what the answer rests on.
        cv2.polylines(view, [np.rint(staged.box[0]).astype(np.int32)], True,
                      KEPT_BGR, 2, cv2.LINE_AA)

    label(view, KEYS, (8, 20))
    row = 40
    if staged.stage == 1:
        rims = seen.get("rims") or []
        # Which of them, and how many there were. A tool that silently picked
        # one of ten and drew it like the only one is how step 1 came to keep
        # a crate while the Camera tab kept the carton standing on it.
        which = ("press 1 to keep this rim (%d of %d, n for next)"
                 % (seen.get("rim_index", 0) + 1, len(rims))
                 if seen.get("rim") is not None and len(rims) > 1
                 else "press 1 to keep this rim"
                 if seen.get("rim") is not None
                 else seen.get("rim_why") or "no rim in view")
        label(view, "step 1 - box on the table, no board: %s" % which,
              (8, row), FOUND_BGR if seen.get("rim") is not None else MISSING_BGR)
    else:
        label(view, "step 2 - board where the box stood: %s"
              % ("press 2 to keep this surface" if seen.get("pose") is not None
                 else seen.get("why") or "no board in view"),
              (8, row), FOUND_BGR if seen.get("pose") is not None else MISSING_BGR)
    row += 20
    if staged.box is not None:
        label(view, "kept: box rim", (8, row), KEPT_BGR)
        row += 20
    if staged.board is not None:
        plane = staged.board.plane
        label(view, "kept: surface %.0f mm square, tilt %.1f deg, %d corners"
              " -- a saves it"
              % (plane.distance * 1000, np.degrees(plane.tilt),
                 len(staged.board.corners)), (8, row), KEPT_BGR)
        row += 20
        # The cost of the thing the surface exists to fix, in the units of the
        # mistake: how far a box may slide before camera Z alone calls it a
        # change of height.
        label(view, "   without it, sliding the box 100 mm reads as %.0f mm "
              "of height change" % (100.0 * np.sin(plane.tilt)), (8, row))
        row += 20

    opening, height, how = staged.opening()
    if opening is not None:
        label(view, "opening %.1f x %.1f mm   rim stood %.0f mm off (%s)"
              % (opening.length * 1000, opening.width * 1000, height * 1000,
                 how), (8, row), FOUND_BGR)
        row += 20
        label(view, "configured %.0f x %.0f mm  ->  x%.3f along, x%.3f across"
              "   -- a applies"
              % (configured[0] * 1000, configured[1] * 1000,
                 opening.length / configured[0], opening.width / configured[1]),
              (8, row))
        row += 20
    elif how:
        label(view, how, (8, row), MISSING_BGR)
        row += 20
    if flash:
        label(view, flash, (8, row), FOUND_BGR)
    caption(view, "%d x %d   %.0f fps" % (view.shape[1], view.shape[0], rate))
    return view


def _look(frame, board, staged, roi, choice=0):
    """Only what the step being waited on needs, because both cost real time.

    Running the board finder and the rim finder on every frame halves the rate
    for a picture in which one of the two is deliberately absent.
    """
    seen = {}
    if staged.stage == 1:
        notes = {}
        ranked = rank_rims(frame, rim_candidates(frame, roi, notes))
        seen["rims"] = ranked
        # The operator's standing choice, not a fresh guess every frame: a
        # pick that reset itself whenever the ranking reshuffled would be
        # unpressable, and the ranking does reshuffle, because depth at a rim
        # is a measurement and measurements move.
        seen["rim_index"] = choice % len(ranked) if ranked else 0
        seen["rim"] = ranked[seen["rim_index"]] if ranked else None
        if seen["rim"] is None:
            seen["rim_why"] = why_nothing(notes)
        return seen
    try:
        pose = find_board(frame, board)
        seen["pose"] = pose
        seen["board_corners"] = pose.corners
    except BoardError as exc:
        seen["why"] = str(exc)
    return seen


def live(config, vision, board, height, configured, path, roi=None,
         log=print):
    """The two captures on screen, and `a` to write what they measure."""
    if not os.environ.get("DISPLAY") and os.name != "nt":
        raise SystemExit("no display to open a window on (DISPLAY is unset).\n"
                         "Run it on the cell's screen, or use --once over ssh.")
    camera = open_camera(vision, log)
    staged = Staged(height)
    # Which of the ranked rims the operator is looking at. Nearest first is a
    # good guess and not a reliable one — a bare crate is as much a box as the
    # carton on top of it — so `n` steps through the rest.
    choice = 0
    flash, flashed_at = "", 0.0
    watch_close, sized, painted = None, False, 0
    rate, last, written = 0.0, None, []
    trouble, shown, seen = "", None, {}
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    log("live — " + KEYS)
    try:
        while True:
            try:
                frame = camera.read()
                if frame.color is None:
                    trouble = "this source sends no colour image"
                else:
                    shown, trouble = frame, ""
                    seen = _look(frame, board, staged, roi, choice)
            except CameraError as exc:
                trouble = str(exc)

            now = time.monotonic()
            if last is not None and now > last:
                rate = (0.9 * rate + 0.1 / (now - last) if rate
                        else 1.0 / (now - last))
            last = now

            if shown is None:
                view = card(trouble or "waiting for the first frame")
            else:
                view = _draw_live(shown, seen, staged, configured, rate,
                                  flash if now - flashed_at < FLASH else "")

            cv2.imshow(WINDOW, view)
            if watch_close is None:
                watch_close = closes_are_reported(log, WINDOW)
            if shown is not None and not sized:
                cv2.resizeWindow(WINDOW, view.shape[1], view.shape[0])
                sized = True
            key = cv2.waitKey(1) & 0xFF
            painted += 1
            if (watch_close and painted > 1
                    and cv2.getWindowProperty(WINDOW,
                                              cv2.WND_PROP_VISIBLE) < 1):
                break
            if key in (27, ord("q")):
                break
            if key == ord("n"):
                rims = seen.get("rims") or []
                if len(rims) < 2:
                    flash = "no other rim in view"
                else:
                    choice = (choice + 1) % len(rims)
                    flash = "rim %d of %d" % (choice + 1, len(rims))
                flashed_at = now
            if key == ord("1"):
                if seen.get("rim") is None:
                    flash = seen.get("rim_why") or "no rim to keep"
                else:
                    staged.take_box(shown, seen["rim"])
                    flash = "box kept — now put the board where it stood"
                flashed_at = now
            if key == ord("2"):
                if seen.get("pose") is None:
                    flash = seen.get("why") or "no board to keep"
                else:
                    staged.take_board(seen["pose"])
                    flash = "surface kept"
                flashed_at = now
            if key == ord("r"):
                staged.reset()
                flash, flashed_at = "cleared — back to step 1", now
            if key == ord("s"):
                if shown is None:
                    log("nothing to save yet")
                    continue
                where = os.path.join(
                    REPO_ROOT, "captures",
                    time.strftime("board_%Y%m%d_%H%M%S.jpg"))
                os.makedirs(os.path.dirname(where), exist_ok=True)
                cv2.imwrite(where, view)
                written.append(where)
                flash, flashed_at = "saved %s" % os.path.basename(where), now
                log("wrote %s" % where)
            if key == ord("a"):
                flash, flashed_at = _apply_live(config, staged, path, log), now
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyWindow(WINDOW)
        for _ in range(4):
            cv2.waitKey(1)
    return written


def _apply_live(config, staged, path, log=print):
    """Write what has been captured: the surface always, the size if measured.

    Two different answers under one key because they come from the same two
    presses and an operator is finishing one job. The surface is the more
    valuable of the two and needs only step 2: it is what lets the cell ask
    "did the box change height" and mean it. Without a plane that question is
    asked of camera Z, which on a rig looking down at an angle answers yes
    every time the box is slid sideways along a table it never left.
    """
    wrote = []
    if staged.board is not None:
        plane = staged.board.plane
        config.vision["surface"] = dict(
            plane.to_dict(),
            corners=len(staged.board.corners),
            measured=datetime.now().astimezone().isoformat(timespec="seconds"))
        wrote.append("surface at %.1f deg" % np.degrees(plane.tilt))
    opening, height, how = staged.opening()
    if opening is not None:
        size = [round(opening.length, 4), round(opening.width, 4),
                round(float(height), 4)]
        config.vision["box_size"] = size
        wrote.append("box %.1f x %.1f x %.1f mm"
                     % (size[0] * 1000, size[1] * 1000, size[2] * 1000))
    if not wrote:
        return how
    config.save_vision(path)
    log("wrote %s to %s" % (" and ".join(wrote), path))
    return "wrote " + " and ".join(wrote) + " to cell.yaml"


def report(answer, configured, height, log=print):
    """The numbers, and what each of them is evidence of."""
    log("")
    log("the board")
    if answer["distance"] is None:
        log("  not found in any frame")
        for reason in sorted(set(answer["refusals"]))[:3]:
            log("    %s" % reason)
        return None
    log("  %.0f of 35 corners, fits to %.2f px"
        % (answer["corners"], answer["fit"]))
    if answer["corners"] < 24:
        log("    fewer than two thirds of them: the sheet is partly out of "
            "frame, at a glancing angle, or too far to resolve its 18 mm "
            "codes. The tilt below is the number that suffers.")
    log("  surface %.1f mm square to it; the middle of the board is %.1f mm "
        "along the lens axis"
        % (answer["distance"] * 1000, (answer["middle"] or 0.0) * 1000))
    log("  tilted %.1f deg from square on" % np.degrees(answer["tilt"]))
    if answer["distance_spread"]:
        log("  frame to frame it moves %.1f mm, which is the noise floor for "
            "everything below" % (answer["distance_spread"] * 1000))
    # planar.py prices tilt: a height error costs its tangent in lateral miss.
    log("  a box 10 mm off this surface's height would land %.1f mm out"
        % (10.0 * np.tan(answer["tilt"])))

    if answer["bias"] is not None:
        log("")
        log("the depth stream, at corners whose distance is known exactly")
        log("  reads %+.1f mm %s, %.1f mm of spread"
            % (answer["bias"] * 1000,
               "long" if answer["bias"] > 0 else "short",
               (answer["noise"] or 0.0) * 1000))

    if not answer["rims"] and not answer["box_refusals"]:
        return None
    log("")
    log("a rim, in the same picture as the board")
    if answer["rims"]:
        log("  found in %d of %d frames" % (answer["rims"], answer["frames"]))
    else:
        for reason in sorted(set(answer["box_refusals"]))[:3]:
            log("  %s" % reason)
    log("  its size is not measured here: that needs the board standing where"
        " the box stood, which is two pictures rather than one. Run without "
        "--once and press 1 then 2.")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--config", default=DEFAULT_PATH)
    ap.add_argument("--source", choices=("realsense", "sim"), default=None,
                    help="override vision.source for this run")
    ap.add_argument("--once", action="store_true",
                    help="one printed report instead of the live window, for "
                         "a session with no display")
    ap.add_argument("--frames", type=int, default=10,
                    help="pictures the --once report answers from (default 10)")
    ap.add_argument("--box", action="store_true",
                    help="--once only: also say whether a rim is in view "
                         "(measuring it is the window's two-step job)")
    ap.add_argument("--height", type=float, default=None, metavar="MM",
                    help="millimetres the rim stands above the board. Only "
                         "needed when the depth stream cannot measure it, "
                         "which it normally can")
    ap.add_argument("--scale-check", type=float, default=100.0,
                    metavar="MM",
                    help="what the sheet's '100 mm scale check' line really "
                         "measures, for a print a printer shrank (default 100)")
    ap.add_argument("--save", nargs="?", const="", default=None,
                    metavar="PATH",
                    help="write the last picture with the board and the box "
                         "rim drawn on it (default: captures/board_<time>.jpg)")
    args = ap.parse_args()

    config = CellConfig.load(args.config)
    vision = dict(config.vision)
    if args.source:
        vision["source"] = args.source
    configured = tuple(vision.get("box_size") or (0.60, 0.40, 0.20))
    # None means "measure it". The configured height is not a default here:
    # it is one of the numbers being corrected, and quietly standing it in
    # would put the measuring plane wherever the old guess was.
    height = None if args.height is None else float(args.height) / 1000.0
    try:
        board = printed_board(args.scale_check)
    except BoardError as exc:
        raise SystemExit(str(exc))

    print("%s, camera %s %dx%d" % (args.config, vision.get("source"),
                                   vision.get("width", 0),
                                   vision.get("height", 0)))
    if abs(args.scale_check - 100.0) > 0.05:
        scale = args.scale_check / 100.0
        print("printed at %.1f%%: squares are %.2f mm, not 25 mm. Nothing "
              "below can check that number — a board declared the wrong size "
              "fits its own corners perfectly and simply sits at the wrong "
              "distance. Only the depth line can contradict it."
              % (scale * 100, 25.0 * scale))
    if args.box:
        print("box configured %s mm"
              % "x".join("%.0f" % (v * 1000) for v in configured))

    roi = tuple(vision["roi"]) if vision.get("roi") else None
    if not args.once:
        live(config, vision, board, height, configured, args.config, roi)
        return

    camera = open_camera(vision)
    try:
        answer = read_board(camera, max(1, args.frames), args.box,
                            height or 0.0, board, roi)
    except CameraError as exc:
        raise SystemExit(str(exc))
    finally:
        camera.close()

    report(answer, configured, height or 0.0)
    if args.save is not None:
        annotate(answer, args.save or os.path.join(
            REPO_ROOT, "captures",
            time.strftime("board_%Y%m%d_%H%M%S.jpg")))


if __name__ == "__main__":
    main()
