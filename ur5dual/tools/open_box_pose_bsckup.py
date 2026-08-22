#!/usr/bin/env python3
"""
The open-box detector on a screen, live, with the numbers printed.

    scripts/ur5dual-open-box-pose                     # live, at the frame rate
    scripts/ur5dual-open-box-pose --source realsense  # override the config
    scripts/ur5dual-open-box-pose --size 600 400      # the opening, in mm
    scripts/ur5dual-open-box-pose --measure           # let depth pick the size
    scripts/ur5dual-open-box-pose --no-roi            # search the whole picture
    scripts/ur5dual-open-box-pose --npz frame.npz     # one saved frame, no window
    scripts/ur5dual-open-box-pose --replay frame.npz  # that frame, as a live loop

    q or ESC   quit             s   save this frame beside the captures
    m          measure the size from depth and use it
    r          forget the lock and acquire again
    d          depth instead of the lens        space   pause

This is the Camera tab's detector, on a terminal instead of a pendant: it
opens the camera `vision.source` names, runs `OpenBoxDetector` — the same
class, the same tracker, the same settings out of config/cell.yaml — and
prints what the tab would have shown. A disagreement between the two would be
a disagreement about nothing, which is exactly what a diagnostic tool must
never introduce.

What was here before is gone: RANSAC plane segmentation, DBSCAN clustering,
an open3d dependency, a synthetic demo scene and a CSV logger. The plane
approach was measured against this cell's own workshop and did not survive it
— the largest plane in view is the floor, and cropping to the box only moves
the problem to the far inner wall, which a steep camera sees more of than the
box's own floor. The CSV went with it because the service already writes one
for every frame the tab sees, in the same shape, and two loggers is one too
many to keep honest.

Both hold: a RealSense is opened by one process at a time, so stop the panel's
live view before opening this, and `--replay` needs no lens at all.
"""

import argparse
import os
import time

import numpy as np

from ..config import DEFAULT_PATH, CellConfig
from ..vision.camera import CameraError
from ..vision.detect import AGREE_MM, DetectionError, OpenBoxDetector, \
    detect_opening_quad, size_check, solve_opening_pnp
from ..vision.rim import STANDARD_SIZES, choose_size
from .snap import DEFAULT_DIR, load, opencv, open_camera, save_npz, unused

WINDOW = "ur5dual — open-box pose"
KEYS = "q quit   s save   m measure size   r reset   d depth   space pause"

INK = (255, 255, 255)
GOOD = (120, 255, 120)
BUSY = (0, 200, 255)
RIM = (0, 255, 0)
WALL = (255, 120, 0)
FLOOR = (0, 140, 255)

# How far the pose's near rim may sit from the measured one before the size it
# was given stops being believable. Thirty millimetres is loose for a right
# answer — this cell agrees to about five — and tight enough that the nearest
# wrong standard crate, out by hundreds, can never slip through. Shared with
# the Camera tab, so both call the same distance wrong.
TOLERANCE_MM = AGREE_MM


# ── what goes on the picture ──────────────────────────────────────────────
def label(image, text, at, colour=INK, scale=0.55):
    """One line, outlined in black so a pale box under it cannot eat it."""
    cv2 = opencv()
    for width, ink in ((4, (0, 0, 0)), (1, colour)):
        cv2.putText(image, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, ink,
                    width, cv2.LINE_AA)
    return image


def project(intrinsics, points):
    """Camera-frame metres to pixels, or None if anything is behind the lens."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if np.any(points[:, 2] <= 1e-6):
        return None
    return np.stack([
        points[:, 0] * intrinsics.fx / points[:, 2] + intrinsics.cx,
        points[:, 1] * intrinsics.fy / points[:, 2] + intrinsics.cy,
    ], axis=1)


def draw_box(image, frame, found):
    """The solved box as a wireframe, with its axes at the opening's centre.

    The rim is drawn in green and the floor in orange, and it is the floor
    that is worth looking at: the rim was fitted to the corners and will lie
    on them however wrong the pose is, while the floor is a prediction made
    from the size and can be seen to be wrong.
    """
    cv2 = opencv()
    pixels = project(frame.intrinsics, found.landmarks_3d())
    if pixels is None:
        return image
    pixels = pixels.astype(int)
    cv2.polylines(image, [pixels[:4]], True, RIM, 2)
    cv2.polylines(image, [pixels[4:]], True, FLOOR, 2)
    for i in range(4):
        cv2.line(image, tuple(pixels[i]), tuple(pixels[i + 4]), WALL, 2)

    axis = min(found.size) * 0.28
    origin = project(frame.intrinsics, found.centre)
    ends = project(frame.intrinsics,
                   [found.centre + found.rotation[:, i] * axis
                    for i in range(3)])
    if origin is not None and ends is not None:
        start = tuple(origin[0].astype(int))
        for end, colour in zip(ends.astype(int),
                               ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            cv2.line(image, start, tuple(end), colour, 2)
    return image


def depth_picture(depth, z_max=3.0):
    """Depth in colour, for deciding whether the rim had any depth at all."""
    cv2 = opencv()
    ramp = np.clip(np.asarray(depth) / max(z_max, 1e-6), 0, 1)
    view = cv2.applyColorMap((ramp * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    view[np.asarray(depth) <= 0] = 0        # no reading is black, not near
    return view


def overlay(frame, found, notes, show_depth, fps, cost, size, measured):
    """One finished frame for the window."""
    view = (depth_picture(frame.depth) if show_depth
            else np.ascontiguousarray(frame.color).copy())
    if found is not None:
        draw_box(view, frame, found)
    roi = notes.get("roi")
    if roi is not None:
        cv2 = opencv()
        cv2.rectangle(view, (int(roi[0]), int(roi[1])),
                      (int(roi[2]), int(roi[3])), (255, 255, 255), 1)

    state = notes.get("state", "—")
    label(view, "%s   %.0f fps   %.0f ms" % (state, fps, cost), (8, 22),
          GOOD if found is not None else BUSY)
    label(view, "opening %.0f x %.0f mm%s"
          % (size[0] * 1000, size[1] * 1000,
             "   (measured)" if measured else ""), (8, 44))
    if found is not None:
        centre = found.centre * 1000
        label(view, "xyz %+.0f %+.0f %+.0f mm   rmse %.1f px"
              % (*centre, found.reprojection_error), (8, 66))
        label(view, size_check(found), (8, 88),
              GOOD if agrees(found) else BUSY)
    else:
        label(view, notes.get("reason", state)[:70], (8, 66), BUSY)
    label(view, KEYS, (8, view.shape[0] - 12), scale=0.45)
    return view


def agrees(found, tolerance_mm=TOLERANCE_MM):
    return (found.depth_disagree is not None
            and abs(found.depth_disagree) * 1000 <= tolerance_mm)


# ── the size, measured rather than assumed ────────────────────────────────
def measure_size(frame, corners, tolerance_mm=TOLERANCE_MM):
    """Which standard crate this is, decided by depth. None if none fits.

    Every candidate produces a pose from the same four corners; only one of
    them puts the near rim where the sensor says it is. On this cell the right
    one lands within a few millimetres and the next-nearest wrong one is out
    by hundreds, so this is a decision rather than a preference.
    """
    return choose_size(corners, frame, solve_opening_pnp,
                       tolerance=tolerance_mm / 1000.0)


def report_sizes(frame, corners, log=print):
    """Every candidate opening and how far its pose lands from measured depth."""
    log("    %-16s %s" % ("opening (mm)", "pose vs measured depth"))
    for millimetres in STANDARD_SIZES:
        length, width = (v / 1000.0 for v in millimetres)
        size, off, _ = choose_size(corners, frame, solve_opening_pnp,
                                   candidates=[(length, width)],
                                   tolerance=TOLERANCE_MM / 1000.0)
        if off is None:
            continue
        log("    %-16s %+8.1f mm%s" % ("%d x %d" % millimetres, off * 1000,
                                       "   <- fits" if size else ""))


# ── one saved frame ───────────────────────────────────────────────────────
def run_once(frame, detector, args, log=print):
    """Report a single frame the way the old tool did, and stop."""
    notes = {}
    if args.measure:
        try:
            corners, _ = _corners_only(frame, detector)
        except DetectionError as e:
            log("[!] %s" % e)
            return 1
        log("[+] candidate openings, judged by depth:")
        report_sizes(frame, corners, log)
        size, off, _ = measure_size(frame, corners)
        if size is None:
            log("[!] no standard opening agrees with depth — pass --size")
            return 1
        detector.box_size = (size[0], size[1], detector.box_size[2])
        log("[+] measured %.0f x %.0f mm (off %+.1f mm)"
            % (size[0] * 1000, size[1] * 1000, off * 1000))

    found = detector.find(frame, notes)
    if found is None:
        log("[!] nothing found: %s" % notes.get("reason", notes.get("state")))
        return 1

    transform = found.matrix()
    log("")
    log("6D POSE — camera frame, origin at the centre of the opening")
    log(str(np.round(transform, 4)))
    log("")
    log("opening        : %.0f x %.0f mm"
        % (found.size[0] * 1000, found.size[1] * 1000))
    log("translation mm : %s" % np.round(transform[:3, 3] * 1000, 1))
    log("distance       : %.4f m" % float(np.linalg.norm(transform[:3, 3])))
    log("reprojection   : %.2f px" % found.reprojection_error)
    log("depth at centre: %.4f m" % found.depth_center)
    log("%s" % size_check(found))

    if args.vis and frame.color is not None:
        cv2 = opencv()
        cv2.imwrite(args.vis, overlay(frame, found, notes, False, 0.0, 0.0,
                                      found.size, args.measure))
        log("[+] wrote %s" % args.vis)
    return 0


def _corners_only(frame, detector):
    """The raw corners of this frame, with no tracker in the way."""
    return detect_opening_quad(frame.color, detector.roi)


# ── the live loop ─────────────────────────────────────────────────────────
class Replay:
    """One saved frame, handed out over and over at a chosen rate.

    For tuning the size, the ROI and the tracker without a lens, and for
    seeing what the loop costs per frame on this machine.
    """

    def __init__(self, path, fps=30):
        self.path, self.fps = path, max(1, int(fps))
        self.frame = load(path)
        self._due = 0.0

    def open(self):
        return self

    def read(self):
        now = time.monotonic()
        if now < self._due:
            time.sleep(self._due - now)
        self._due = max(now, self._due) + 1.0 / self.fps
        return self.frame

    def close(self):
        pass

    @property
    def description(self):
        return "replay %s @%d" % (os.path.basename(self.path), self.fps)


def run_live(camera, detector, args, log=print):
    """Read, detect, draw, repeat — until q, or until --frames have gone by."""
    cv2 = opencv()
    headless = args.no_window or (not os.environ.get("DISPLAY")
                                  and os.name != "nt")
    if headless and not args.no_window:
        log("no display to open a window on — printing only (Ctrl-C to stop)")
    if not headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    frames, fps, last, said = 0, 0.0, None, 0.0
    view, show_depth, paused, measured = None, False, False, args.measure
    log("%s — %s" % (camera.description, KEYS if not headless else "Ctrl-C"))
    try:
        while True:
            try:
                frame = camera.read()
            except CameraError as e:
                log("camera: %s" % e)
                break
            frames += 1

            now = time.monotonic()
            if last is not None and now > last:
                gap = now - last
                fps = 1.0 / gap if not fps else 0.9 * fps + 0.1 / gap
            last = now

            notes, found = {}, None
            if not paused:
                started = time.perf_counter()
                found = detector.find(frame, notes)
                cost = (time.perf_counter() - started) * 1000.0

                # The size is measured once, from the first frame that offers
                # corners, and then held. Re-deciding every frame would let a
                # noisy frame swap a 600 for a 594 under a running program.
                if args.measure and not measured and found is not None:
                    size, off, _ = measure_size(frame, found.corners)
                    if size is not None:
                        detector.box_size = (size[0], size[1],
                                             detector.box_size[2])
                        detector.reset()
                        measured = True
                        log("measured opening %.0f x %.0f mm (off %+.1f mm)"
                            % (size[0] * 1000, size[1] * 1000, off * 1000))

                if found is not None and now - said >= args.report:
                    log("xyz mm %+7.1f %+7.1f %+7.1f   %.0fx%.0f   "
                        "rmse %.1f px   %s   %s   %.0f fps"
                        % (*(found.centre * 1000), found.size[0] * 1000,
                           found.size[1] * 1000, found.reprojection_error,
                           found.state, size_check(found), fps))
                    said = now
                elif found is None and now - said >= args.report * 4:
                    log("still looking: %s"
                        % notes.get("reason", notes.get("state", "")))
                    said = now

                if not headless and frame.color is not None:
                    view = overlay(frame, found, notes, show_depth, fps, cost,
                                   detector.box_size, measured)

            if headless:
                if args.frames and frames >= args.frames:
                    break
                continue

            shown = view if view is not None else np.zeros((240, 640, 3),
                                                           np.uint8)
            if paused:
                shown = shown.copy()
                label(shown, "PAUSED", (8, 110), BUSY)
            cv2.imshow(WINDOW, shown)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("d"):
                show_depth = not show_depth
            elif key == ord("r"):
                detector.reset()
                measured = False
                log("lock cleared")
            elif key == ord("m"):
                if found is None:
                    log("nothing to measure yet")
                else:
                    size, off, _ = measure_size(frame, found.corners)
                    if size is None:
                        log("no standard opening agrees with depth "
                            "(closest is %s)"
                            % ("out of reach" if off is None
                               else "%+.0f mm out" % (off * 1000)))
                    else:
                        detector.box_size = (size[0], size[1],
                                             detector.box_size[2])
                        detector.reset()
                        measured = True
                        log("measured opening %.0f x %.0f mm (off %+.1f mm)"
                            % (size[0] * 1000, size[1] * 1000, off * 1000))
            elif key == ord("s"):
                path = unused(os.path.join(
                    args.out, "pose_%s.npz" % time.strftime("%Y%m%d_%H%M%S")))
                save_npz(path, frame)
                written = [path]
                if view is not None and cv2.imwrite(path[:-4] + ".jpg", view):
                    written.append(path[:-4] + ".jpg")
                log("saved " + "  ".join(written))
            if args.frames and frames >= args.frames:
                break
    except KeyboardInterrupt:
        log("")
    finally:
        camera.close()
        if not headless:
            cv2.destroyWindow(WINDOW)
            for _ in range(4):
                cv2.waitKey(1)
        log("closed (%d frames)" % frames)
    return 0


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_PATH)
    p.add_argument("--source", choices=("realsense", "sim"), default=None,
                   help="override vision.source from the config")
    p.add_argument("--npz", default=None,
                   help="one saved frame, reported and then done")
    p.add_argument("--replay", default=None,
                   help="a saved frame, fed to the live loop over and over")
    p.add_argument("--size", type=float, nargs=2, metavar=("L", "W"),
                   default=None, help="the opening in mm, e.g. --size 600 400")
    p.add_argument("--measure", action="store_true",
                   help="let depth choose the opening from the standard sizes")
    p.add_argument("--roi", type=int, nargs=4, default=None,
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="search only this window (default: vision.roi)")
    p.add_argument("--no-roi", action="store_true",
                   help="search the whole picture, whatever the config says")
    p.add_argument("--fps", type=int, default=30, help="--replay rate")
    p.add_argument("--report", type=float, default=1.0,
                   help="seconds between printed poses (default 1)")
    p.add_argument("--frames", type=int, default=0,
                   help="stop after this many frames (0 = no limit)")
    p.add_argument("--no-window", action="store_true",
                   help="print only, for ssh")
    p.add_argument("--out", default=DEFAULT_DIR,
                   help="where s writes (default captures/)")
    p.add_argument("--vis", default=None,
                   help="--npz: write the overlay to this file")
    args = p.parse_args()

    vision = dict(CellConfig.load(args.config).vision)
    if args.source:
        vision["source"] = args.source
    if args.size:
        vision["box_size"] = [args.size[0] / 1000.0, args.size[1] / 1000.0,
                              (list(vision.get("box_size") or []) +
                               [0.20, 0.20, 0.20])[2]]
    if args.no_roi:
        vision["roi"] = None
    elif args.roi:
        vision["roi"] = args.roi

    detector = OpenBoxDetector(
        box_size=tuple(vision.get("box_size") or (0.60, 0.40, 0.20)),
        roi=vision.get("roi"),
        smoothing=float(vision.get("smoothing", 0.20)),
        max_reprojection=float(vision.get("max_reprojection", 4.0)),
        max_corner_jump=float(vision.get("max_corner_jump", 35.0)),
        confirm_frames=1 if args.npz else int(vision.get("confirm_frames", 4)),
        hold_frames=int(vision.get("hold_frames", 15)))

    if args.npz:
        return run_once(load(args.npz), detector, args)
    camera = Replay(args.replay, args.fps).open() if args.replay \
        else open_camera(vision)
    return run_live(camera, detector, args)


if __name__ == "__main__":
    raise SystemExit(main())
