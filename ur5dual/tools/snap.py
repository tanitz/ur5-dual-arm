#!/usr/bin/env python3
"""
The camera in a window, and `s` to keep what is on the screen.

    scripts/ur5dual-snap                  # live; s saves, q quits
    scripts/ur5dual-snap --source sim     # no lens needed
    scripts/ur5dual-snap --out /tmp/shots # where the JPEGs go
    scripts/ur5dual-snap --once           # one picture, no window, for ssh

    s   save what is on the screen: box_<date>_<time>.jpg, and beside it
        the same frame's depth as .npz (--ply as well, or --no-depth for
        the photograph alone)
    q   or ESC (the window's close button too, where the build reports it)

The camera is the whole of it. Nothing here looks for a box, moves an arm or
writes to the cell config: the detector lives in the Camera tab, where its
answer is used, and a viewer that ran it would spend a quarter of a second of
every frame — this Jetson's cost for `find_box` — deciding something a
photograph does not need. Frames come straight off the lens instead, so `s`
answers at the frame rate.

Which camera, and how big, come from `vision` in config/cell.yaml, so the view
is the Camera tab's view.

What is kept beside the photograph is depth and the four numbers of the lens,
not three coordinates per pixel: `Intrinsics.deproject_grid` turns the one
into the other exactly, so the .npz *is* the point cloud, at a sixth of the
size, and it loads straight back into a `Frame` for the detector to be run
over again. `--ply` writes the cloud out as points for something else to open.
Both are in the camera's own frame — +Z out of the lens, +X right, +Y down —
because `vision.camera_to_world` in this cell is still uncalibrated.

A RealSense is opened by one process at a time. Stop the panel's live view (or
the app) before opening this, or it gets "no RealSense opened" while the GUI
holds the lens.
"""

import argparse
import json
import os
import time

import numpy as np

from ..config import DEFAULT_PATH, REPO_ROOT, CellConfig
from ..vision.camera import CameraError, Frame, Intrinsics, make_camera

DEFAULT_DIR = os.path.join(REPO_ROOT, "captures")
WINDOW = "ur5dual — camera"

# How many frames `--once` throws away before the one it keeps. A RealSense
# opens with its auto-exposure still converging: the first frames come back
# dark. The window never shows them because it is on its tenth frame before an
# eye is on it; one shot has to wait on purpose.
WARMUP = 10

# How long a save stays announced on the picture. Long enough to read after
# looking down at the keyboard, short enough not to sit over the next shot.
FLASH = 1.5

# librealsense hands the device back to the kernel a moment after the process
# that held it exits, so "busy" in the second after the panel was closed is a
# race and not a refusal. Waited out rather than reported.
BUSY_TRIES, BUSY_WAIT = 3, 1.0

INK_BGR = (255, 255, 255)
SAVED_BGR = (120, 200, 120)
TROUBLE_BGR = (0, 119, 204)       # style.AMBER

KEYS = "s save    q quit"


def opencv():
    """The window and the JPEG encoder, with the install line if it is absent."""
    try:
        import cv2
    except ImportError as e:
        raise SystemExit("this needs OpenCV for the window and the JPEG (%s).\n"
                         "    pip3 install opencv-python" % e)
    return cv2


# ── files ─────────────────────────────────────────────────────────────────
def out_path(out=None, when=None):
    """Where the picture goes: the name given, or a dated one under captures/.

    A directory is taken as a directory, and any other extension is replaced:
    a file called .png that holds JPEG bytes is a file that opens wrong in
    half the tools that meet it.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(when))
    if not out:
        return os.path.join(DEFAULT_DIR, "box_%s.jpg" % stamp)
    if out.endswith(os.sep) or os.path.isdir(out):
        return os.path.join(out, "box_%s.jpg" % stamp)
    root, ext = os.path.splitext(out)
    return out if ext.lower() in (".jpg", ".jpeg") else root + ".jpg"


def unused(path):
    """The same name, or the next free one beside it.

    A dated name is only unique to the second, and `s` pressed twice while
    something interesting is on the table is exactly when both pictures
    matter. The second one must not land on the first.
    """
    root, ext = os.path.splitext(path)
    candidate, n = path, 1
    while os.path.exists(candidate):
        n += 1
        candidate = "%s_%d%s" % (root, n, ext)
    return candidate


def save_jpeg(path, image, quality=95):
    """Write BGR pixels to `path`, making the folder if it is not there yet.

    `Frame.color` is already BGR — a RealSense sends bgr8 and the panel is the
    one that flips it for Qt — so it goes to `imwrite` untouched. Flipping it
    here as well would save every picture in the wrong colour.
    """
    cv2 = opencv()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ok = cv2.imwrite(path, np.ascontiguousarray(image),
                     [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise SystemExit("could not write %s" % path)
    return path


def open_camera(vision, log=print):
    """The camera named in the config, opened, or a plain reason why not."""
    camera = make_camera(vision)
    for attempt in range(BUSY_TRIES):
        try:
            return camera.open()
        except CameraError as e:
            if "busy" not in str(e).lower() or attempt == BUSY_TRIES - 1:
                raise SystemExit("%s\n(--source sim uses a simulated picture, "
                                 "if the lens is the problem)" % e)
            log("the lens is busy — waiting for the other process to let go")
            time.sleep(BUSY_WAIT)


def save_npz(path, frame):
    """Depth, colour and the lens, in the form that reconstructs the cloud.

    Depth goes as millimetres in a uint16, which is not a compression: a
    RealSense reports z16 counts scaled by its depth unit, and 1 mm is that
    unit. What would lose something is float32 metres, which cannot hold every
    integer millimetre out to the far end of the range.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    depth_mm = np.clip(np.asarray(frame.depth) * 1000.0, 0, 65535)
    np.savez_compressed(
        path,
        depth_mm=depth_mm.astype(np.uint16),
        color=(np.zeros((0, 0, 3), np.uint8) if frame.color is None
               else np.asarray(frame.color)),
        intrinsics=json.dumps(frame.intrinsics.to_dict()),
        taken=time.strftime("%Y-%m-%dT%H:%M:%S"),
        frame_of="camera: +Z out of the lens, +X right, +Y down, metres")
    return path


def load(path):
    """An .npz written here, back as the `Frame` it came from.

    The point of saving depth rather than points: this replays into
    `find_box` exactly as the lens delivered it, so a detection can be argued
    about days later without the box still being on the table.
    """
    held = np.load(path)
    k = Intrinsics.from_dict(json.loads(str(held["intrinsics"])))
    color = held["color"] if held["color"].size else None
    return Frame(held["depth_mm"].astype(float) / 1000.0, k, color=color)


# One vertex of a binary PLY: three floats and three bytes, in the order the
# header below promises them.
PLY_VERTEX = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])

PLY_HEADER = """ply
format binary_little_endian 1.0
comment written by ur5dual-snap
comment camera frame, metres: +Z out of the lens, +X right, +Y down
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""


def save_ply(path, frame):
    """The cloud as points, for something that opens clouds.

    Pixels with no depth are left out rather than written at zero: a sensor
    saying "I could not see" is not a point on the lens, and a viewer cannot
    tell the difference once they are in the file.
    """
    depth = np.asarray(frame.depth)
    keep = depth > 0
    points = frame.intrinsics.deproject_grid(depth)[keep]
    vertices = np.empty(len(points), dtype=PLY_VERTEX)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    if frame.color is None:
        vertices["red"] = vertices["green"] = vertices["blue"] = 200
    else:
        # BGR off the camera, RGB in the file, which is what PLY names
        bgr = np.asarray(frame.color)[keep]
        vertices["red"], vertices["green"], vertices["blue"] = bgr[:, ::-1].T
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write((PLY_HEADER % len(vertices)).encode("ascii"))
        f.write(vertices.tobytes())
    return path


def write_set(frame, out=None, quality=95, depth=True, ply=False, log=print):
    """The photograph, and beside it whatever else that frame held.

    One stem for all of them, taken from the photograph's own free name, so a
    picture and its cloud cannot drift apart in a folder.
    """
    written = [save_jpeg(unused(out_path(out)), frame.color, quality)]
    stem = os.path.splitext(written[0])[0]
    if depth:
        written.append(save_npz(stem + ".npz", frame))
    if ply:
        written.append(save_ply(stem + ".ply", frame))
    log("saved %s  (%dx%d)%s"
        % (written[0], frame.color.shape[1], frame.color.shape[0],
           "".join("  + " + os.path.splitext(p)[1] for p in written[1:])))
    return written


# ── what is written over the picture ──────────────────────────────────────
def label(image, text, at, colour=INK_BGR):
    """One line, dark-outlined so it survives a pale table under it."""
    cv2 = opencv()
    for width, ink in ((3, (0, 0, 0)), (1, colour)):
        cv2.putText(image, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.45, ink,
                    width, cv2.LINE_AA)
    return image


def caption(image, text):
    """The line along the bottom: what this picture is, in pixels and fps."""
    return label(image, text, (8, image.shape[0] - 10))


def card(text):
    """A black card with one line on it, for a camera that has sent nothing."""
    view = np.zeros((240, 640, 3), dtype=np.uint8)
    return label(view, text, (8, 120))


# ── the live window ───────────────────────────────────────────────────────
def closes_are_reported(log=print):
    """Whether this build can say the window was shut with its own button.

    Asked rather than assumed, because the answer belongs to the backend and
    not to the window: the GTK3 build on this Jetson answers -1 — "no such
    property" — for a window that is open and on the screen, and answers the
    same -1 after that window has been destroyed. A loop that reads -1 as
    "the window is gone" ends before it has drawn a single frame, and what an
    operator sees is a camera that never opened.
    """
    cv2 = opencv()
    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 0:
        return True
    log("(this OpenCV does not report the window's close button — press q)")
    return False


def live(vision, out=None, quality=95, depth=True, ply=False, log=print):
    """The camera on screen until `q`, writing a picture on every `s`.

    Returns the paths written, so a test can press keys at it and check what
    landed on disk rather than what was drawn.
    """
    cv2 = opencv()
    if not os.environ.get("DISPLAY") and os.name != "nt":
        raise SystemExit("no display to open a window on (DISPLAY is unset).\n"
                         "Run it on the cell's screen, or take one picture "
                         "with --once over ssh.")
    camera = open_camera(vision, log)

    written = []
    shown = None                 # the last frame that had pixels in it
    trouble = ""                 # what the lens said instead of a frame
    flash, flashed_at = "", 0.0
    watch_close, sized, painted = None, False, 0
    rate, last = 0.0, None
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    log("live — " + KEYS)
    try:
        while True:
            # A read is one frame off the lens and nothing else, so it costs a
            # thirtieth of a second and belongs on this thread. The panel's
            # camera runs on its own for the detector's sake, not the lens's.
            try:
                frame = camera.read()
                if frame.color is None:
                    trouble = "this source sends no colour image"
                else:
                    # the whole frame, not just its colour: `s` writes the
                    # depth that came with the picture, and the two have to
                    # be the same reading of the same moment
                    shown, trouble = frame, ""
            except CameraError as e:
                trouble = str(e)

            now = time.monotonic()
            if last is not None and now > last:
                # smoothed, because a number that flickers between 28 and 31
                # is a number nobody can read
                rate = 0.9 * rate + 0.1 / (now - last) if rate else 1.0 / (now - last)
            last = now

            if shown is None:
                view = card(trouble or "waiting for the first frame")
            else:
                view = np.ascontiguousarray(shown.color).copy()
                label(view, KEYS, (8, 20))
                label(view, camera.description, (8, 40))
                if flash and now - flashed_at < FLASH:
                    label(view, flash, (8, 60), SAVED_BGR)
                elif trouble:
                    label(view, trouble, (8, 60), TROUBLE_BGR)
                # how much of the picture the depth camera could answer for,
                # which is the number that decides whether the cloud saved
                # next to it is worth having
                caption(view, "%d x %d   %.0f fps   depth %.0f%%"
                        % (view.shape[1], view.shape[0], rate,
                           100.0 * np.mean(shown.depth > 0)))

            cv2.imshow(WINDOW, view)
            if watch_close is None:
                watch_close = closes_are_reported(log)
            if shown is not None and not sized:
                # WINDOW_NORMAL opens at whatever size the backend likes; the
                # picture's own size is the one an operator is judging
                cv2.resizeWindow(WINDOW, view.shape[1], view.shape[0])
                sized = True
            key = cv2.waitKey(1) & 0xFF
            painted += 1
            # the window's own close button, which sends no key at all. Not
            # asked on the first pass: a window is created before it is
            # mapped, and a backend that reports "not visible yet" would be
            # answering about the wrong moment.
            if (watch_close and painted > 1
                    and cv2.getWindowProperty(WINDOW,
                                              cv2.WND_PROP_VISIBLE) < 1):
                break
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                if shown is None:
                    log("nothing to save yet")
                    continue
                kept = write_set(shown, out, quality, depth, ply, log)
                written.extend(kept)
                # the name once, then only what else went with it: three
                # full paths across the picture is a caption nobody reads
                flash = "saved %s%s" % (
                    os.path.basename(kept[0]),
                    "".join("  + " + os.path.splitext(p)[1] for p in kept[1:]))
                flashed_at = now
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        cv2.destroyWindow(WINDOW)
        # GTK only takes the window down on the next few event loops
        for _ in range(4):
            cv2.waitKey(1)
    return written


# ── one picture, no window ────────────────────────────────────────────────
def grab(camera, warmup=WARMUP):
    """The frame worth keeping, not the first one the sensor hands over."""
    frame = camera.read()
    for _ in range(max(0, int(warmup))):
        frame = camera.read()
    return frame


def capture(vision, out=None, quality=95, warmup=WARMUP, depth=True,
            ply=False, log=print):
    """One frame, written without a window. Returns the paths written."""
    camera = open_camera(vision, log)
    try:
        frame = grab(camera, warmup)
    finally:
        camera.close()

    if frame.color is None:
        raise SystemExit("this source sends no colour image — nothing to save "
                         "as a JPEG")

    return write_set(frame, out, quality, depth, ply, log)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="file or folder to write (default captures/, dated)")
    ap.add_argument("--config", default=DEFAULT_PATH)
    ap.add_argument("--source", choices=("realsense", "sim"), default=None,
                    help="override vision.source from the config")
    ap.add_argument("--quality", type=int, default=95,
                    help="JPEG quality 1-100 (default 95)")
    ap.add_argument("--no-depth", dest="depth", action="store_false",
                    help="the photograph alone — no .npz beside it")
    ap.add_argument("--ply", action="store_true",
                    help="also write the cloud as points, for a viewer "
                         "elsewhere (~4 MB a frame)")
    ap.add_argument("--once", action="store_true",
                    help="no window: take one picture and stop, for ssh")
    ap.add_argument("--warmup", type=int, default=WARMUP,
                    help="--once: frames discarded before the one kept "
                         "(default %d)" % WARMUP)
    args = ap.parse_args()

    vision = dict(CellConfig.load(args.config).vision)
    if args.source:
        vision["source"] = args.source
    run = capture if args.once else live
    run(vision, out=args.out, quality=args.quality, depth=args.depth,
        ply=args.ply, **({"warmup": args.warmup} if args.once else {}))


if __name__ == "__main__":
    main()
