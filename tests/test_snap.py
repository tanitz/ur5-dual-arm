"""
The capture tool: does a JPEG of what the lens sees land on disk.

Everything here runs against the simulated camera, so it needs no lens and
touches no arm. What is being checked is the part that is easy to get quietly
wrong — the file is a real JPEG of the frame's own size, the colours are not
reversed on the way out, and what is written over the picture for the operator
does not end up written into the file, and the cloud kept beside a picture is
the same reading of the same moment — reconstructed to the millimetre, and
still good enough for the detector to be run over again days later.

The live window is driven with OpenCV's own window calls stubbed out and the
keys typed from here, so `s` is tested by what it writes rather than by
somebody watching a window. `getWindowProperty` is stubbed at -1 because that
is what the GTK3 build on this Jetson answers for a window that is open: the
loop once read that as "the window was closed" and quit before drawing a
single frame, which looked exactly like a camera that never opened.
"""

import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from ur5dual.tools import snap
from ur5dual.vision.camera import CameraError, Frame, Intrinsics, SimCamera
from ur5dual.vision.detect import find_box

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


SIM = {"source": "sim"}
work = tempfile.mkdtemp(prefix="ur5dual-snap-")


print("where the picture goes")
check("a dated name under captures/ when nothing is asked for",
      snap.out_path().startswith(snap.DEFAULT_DIR)
      and snap.out_path().endswith(".jpg"), snap.out_path())
check("a folder is a folder", os.path.dirname(snap.out_path(work)) == work,
      snap.out_path(work))
check("a name is kept", snap.out_path("/tmp/lid.jpg") == "/tmp/lid.jpg")
check("and another extension is not — the bytes are JPEG either way",
      snap.out_path("/tmp/lid.png") == "/tmp/lid.jpg")
check(".jpeg is left alone", snap.out_path("/tmp/lid.jpeg") == "/tmp/lid.jpeg")

taken = os.path.join(work, "twice.jpg")
open(taken, "w").close()
check("two pictures in the same second do not become one",
      snap.unused(taken) == os.path.join(work, "twice_2.jpg"),
      os.path.basename(snap.unused(taken)))


print("\none picture, no window")
plain = os.path.join(work, "box.jpg")
written = snap.capture(SIM, out=plain, warmup=1, log=lambda text: None)
check("the photograph and its depth are written, under one stem",
      written == [plain, plain[:-4] + ".npz"]
      and all(os.path.exists(p) for p in written),
      str([os.path.basename(p) for p in written]))
alone = snap.capture(SIM, out=os.path.join(work, "alone.jpg"), warmup=1,
                     depth=False, log=lambda text: None)
check("and the photograph alone when that is what was asked for",
      len(alone) == 1, str([os.path.basename(p) for p in alone]))
read_back = cv2.imread(plain)
check("and opens as an image of the frame's own size",
      read_back is not None and read_back.shape[:2] == (480, 640),
      "" if read_back is None else str(read_back.shape))

# The box the SimCamera draws is a lighter grey than its table, and it is
# drawn BGR. If a flip crept in on the way to the file the two would swap
# channels, so compare the file against the frame the camera actually gave.
frame = snap.grab(snap.make_camera(SIM).open(), warmup=1)
check("the pixels are the camera's, not a channel-swapped copy",
      float(np.mean(np.abs(read_back.astype(float)
                           - frame.color.astype(float)))) < 4.0)


print("\nthe depth kept beside the picture")
sim_frame = snap.grab(snap.make_camera(SIM).open(), warmup=1)
back = snap.load(plain[:-4] + ".npz")
worst_mm = float(np.max(np.abs(back.depth - sim_frame.depth))) * 1000
check("it reconstructs to the millimetre the sensor works in",
      worst_mm < 1.0, "worst %.3f mm" % worst_mm)
check("with the lens it was taken through",
      back.intrinsics.to_dict() == sim_frame.intrinsics.to_dict())
check("and the colour of the same moment",
      back.color is not None
      and np.array_equal(back.color, sim_frame.color))
# A saved colour/depth frame can be run through the same known-opening
# detector later. The physical size is configuration, just as it is live.
again = find_box(back, box_size=(0.30, 0.22, 0.08))
check("the detector runs over it again and finds the same box",
      again is not None and abs(again.size[0] - 0.30) < 0.01
      and abs(again.size[1] - 0.22) < 0.01,
      "" if again is None else again.describe())


print("\nthe cloud, for something that opens clouds")
ply = snap.save_ply(os.path.join(work, "cloud.ply"), sim_frame)
head, body = open(ply, "rb").read().split(b"end_header\n", 1)
head = head.decode("ascii")
said = int([l for l in head.splitlines()
            if l.startswith("element vertex")][0].split()[-1])
kept = int(np.count_nonzero(sim_frame.depth > 0))
check("one vertex per pixel the sensor could answer for", said == kept,
      "%d said, %d with depth" % (said, kept))
check("and the file holds exactly that many", len(body) == said * 15,
      "%d bytes" % len(body))
check("it says which frame the numbers are in",
      "camera frame, metres" in head)
points = np.frombuffer(body, dtype=snap.PLY_VERTEX)
want = sim_frame.intrinsics.deproject_grid(sim_frame.depth).reshape(-1, 3)
check("the first point is where the lens puts that pixel",
      abs(points[0]["z"] - want[0][2]) < 1e-4
      and abs(points[0]["x"] - want[0][0]) < 1e-4,
      "%.4f vs %.4f m" % (points[0]["z"], want[0][2]))
# the simulated box is (165, 150, 140) in BGR; PLY names its channels RGB, so
# a cloud with the two swapped would come out blue in every viewer
on_box = int(np.flatnonzero(
    np.all(sim_frame.color.reshape(-1, 3) == (165, 150, 140), axis=1))[0])
vertex = points[on_box]
check("and the colour is RGB, as PLY says it is",
      (vertex["red"], vertex["green"], vertex["blue"]) == (140, 150, 165),
      str((vertex["red"], vertex["green"], vertex["blue"])))


print("\nwhat a source with no colour gets")


class DepthOnly(SimCamera):
    """A camera that answers with depth and nothing else."""

    def read(self):
        k = Intrinsics.from_fov(64, 48, 87.0)
        return Frame(np.full((48, 64), 0.7), k, color=None)


was = snap.make_camera
snap.make_camera = lambda cfg: DepthOnly()
try:
    snap.capture(SIM, out=os.path.join(work, "none.jpg"), warmup=0,
                 log=lambda text: None)
    check("it is refused rather than written empty", False)
except SystemExit as e:
    check("it is refused, in words that name the problem",
          "colour" in str(e), str(e))
finally:
    snap.make_camera = was


print("\nthe live window, with s pressed at it")
os.environ.setdefault("DISPLAY", ":0")           # the window is stubbed out
WINDOW_CALLS = ("namedWindow", "imshow", "resizeWindow", "waitKey",
                "getWindowProperty", "destroyWindow")


class Spy(SimCamera):
    """The simulated camera, with a note of whether it was let go."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.closed = False

    def close(self):
        self.closed = True


def press(keys, visible=-1.0, never_quit=False, camera=None, **kw):
    """Run the window with `keys` typed into it and nothing drawn on screen.

    `visible` is what `getWindowProperty(WND_PROP_VISIBLE)` answers: -1 here
    by default, the number this Jetson's build gives for a window that is
    open and on the screen.
    """
    cam = camera or Spy()
    real = {name: getattr(cv2, name) for name in WINDOW_CALLS}
    made = snap.make_camera
    script = [ord(k) for k in keys]
    deadline = time.monotonic() + 10.0

    def wait_key(delay):
        time.sleep(0.005)
        if time.monotonic() > deadline:
            return ord("q")
        if script:
            return script.pop(0)
        return 255 if never_quit else ord("q")       # 255: -1 & 0xFF, no key

    cv2.namedWindow = lambda *a, **k: None
    cv2.imshow = lambda *a, **k: None
    cv2.resizeWindow = lambda *a, **k: None
    cv2.waitKey = wait_key
    cv2.getWindowProperty = lambda *a, **k: visible
    cv2.destroyWindow = lambda *a, **k: None
    snap.make_camera = lambda cfg: cam
    try:
        return snap.live(SIM, log=lambda text: None, **kw), cam
    finally:
        for name, fn in real.items():
            setattr(cv2, name, fn)
        snap.make_camera = made


shots, cam = press("sq", out=os.path.join(work, "live.jpg"))
check("s writes the picture and its depth, and q comes back",
      len(shots) == 2 and shots[0].endswith(".jpg")
      and shots[1].endswith(".npz"),
      str([os.path.basename(p) for p in shots]))
live_read = cv2.imread(shots[0]) if shots else None
check("of the frame's own size",
      live_read is not None and live_read.shape[:2] == (480, 640))
check("the camera is let go on the way out", cam.closed)

# The keys, the source and the frame rate are drawn for the operator, on a
# copy. A file with "s save   q quit" burnt into the corner is not a
# photograph of the cell.
check("and the writing on the window stays out of the file",
      float(np.mean(np.abs(live_read.astype(float)
                           - cam.read().color.astype(float)))) < 4.0)

shots, _ = press("ssq", out=os.path.join(work, "twin.jpg"))
check("s twice keeps both, and each picture keeps its own depth",
      [os.path.basename(p) for p in shots]
      == ["twin.jpg", "twin.npz", "twin_2.jpg", "twin_2.npz"],
      str([os.path.basename(p) for p in shots]))

shots, _ = press("sq", ply=True, out=os.path.join(work, "three.jpg"))
check("--ply puts the cloud beside them, under the same stem",
      [os.path.basename(p) for p in shots]
      == ["three.jpg", "three.npz", "three.ply"],
      str([os.path.basename(p) for p in shots]))

shots, _ = press("sq", depth=False, out=os.path.join(work, "flat.jpg"))
check("--no-depth is the photograph and nothing else", len(shots) == 1,
      str([os.path.basename(p) for p in shots]))

shots, _ = press("s\x1b", out=os.path.join(work, "esc.jpg"))
check("ESC quits as well as q", len(shots) == 2, str(len(shots)))


print("\nthe close button, on the two kinds of build there are")
began = time.monotonic()
shots, _ = press("", visible=0.0, never_quit=True,
                 out=os.path.join(work, "shut.jpg"))
shut_in = time.monotonic() - began
check("a build that reports a closed window is believed",
      shut_in < 3.0 and not shots, "%.1f s" % shut_in)
began = time.monotonic()
shots, _ = press("sq", visible=-1.0, out=os.path.join(work, "quiet.jpg"))
check("and one that reports nothing at all is not read as a closed window",
      len(shots) == 2, "%.1f s, %s"
      % (time.monotonic() - began, [os.path.basename(p) for p in shots]))


print("\nwhen something else still has the lens")
snap.BUSY_WAIT = 0.01            # the wait is real, the test's patience is not


class Busy(SimCamera):
    """A lens somebody else has, for the first `refusals` attempts.

    librealsense answers "Device or resource busy" for a moment after the
    process that held the camera exits, which is exactly the moment an
    operator closes the panel and runs this — so a refusal that clears is
    waited out, and one that does not is reported.
    """

    def __init__(self, refusals=0, **kw):
        super().__init__(**kw)
        self.refusals, self.tries = refusals, 0

    def open(self):
        self.tries += 1
        if self.tries > self.refusals:
            return self
        raise CameraError("no RealSense opened: Device or resource busy")


held = Busy(refusals=2)
shots, _ = press("sq", camera=held, out=os.path.join(work, "waited.jpg"))
check("a lens that is let go is waited for, not reported",
      held.tries == 3 and len(shots) == 2, "%d tries" % held.tries)

try:
    press("q", camera=Busy(refusals=99), out=os.path.join(work, "never.jpg"))
    check("one that is never let go is refused", False)
except SystemExit as e:
    check("one that is never let go is refused, in words that name it",
          "busy" in str(e) and "--source sim" in str(e))

shutil.rmtree(work, ignore_errors=True)

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
