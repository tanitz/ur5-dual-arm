"""
The camera, running on its own thread.

Edge detection, solvePnP, temporal filtering and CSV logging happen here, away
from the thread that draws the panel. Even a short camera stall must never be
a stall in which the pendant cannot respond to STOP.

So the camera lives here, in a thread of its own, and everything else reads
the last thing it saw. The panel paints it, and a `FIND` step asks for a fresh
one. Neither waits on a lens.

The service is owned by the app rather than by the camera tab, for the same
reason the executor is: a program that wants a detection must not depend on
which tab an operator happened to leave open.
"""

import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .camera import CameraError, make_camera
from .detect import DetectionError, OpenBoxDetector


class DetectionLog:
    """One CSV row per Camera-tab frame, including rejected raw answers."""

    BASE = ["time_iso", "elapsed_s", "frame", "state", "found",
            "raw_reprojection_px", "depth_center_m", "message"]
    CORNERS = [f"{kind}_{axis}{i}" for kind in ("raw", "filtered")
               for i in range(4) for axis in ("u", "v")]
    POSES = [f"{kind}_{name}" for kind in ("raw", "filtered")
             for name in (["x_m", "y_m", "z_m"] +
                          [f"r{r}{c}" for r in range(3) for c in range(3)])]
    FIELDS = BASE + CORNERS + POSES

    def __init__(self, path, metadata):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.file.write("# metadata=" + json.dumps(
            metadata, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.started = time.monotonic()
        self.rows = 0

    @staticmethod
    def _corners(row, kind, corners):
        if corners is not None:
            for i, (u, v) in enumerate(np.asarray(corners).reshape(4, 2)):
                row[f"{kind}_u{i}"] = "%.4f" % u
                row[f"{kind}_v{i}"] = "%.4f" % v

    @staticmethod
    def _pose(row, kind, transform):
        if transform is None:
            return
        transform = np.asarray(transform)
        for axis, value in zip("xyz", transform[:3, 3]):
            row[f"{kind}_{axis}_m"] = "%.8f" % value
        for r in range(3):
            for c in range(3):
                row[f"{kind}_r{r}{c}"] = "%.8f" % transform[r, c]

    def write(self, reading):
        notes = reading.notes
        found = reading.detection
        row = {
            "time_iso": datetime.now().astimezone().isoformat(
                timespec="milliseconds"),
            "elapsed_s": "%.4f" % (time.monotonic() - self.started),
            "frame": reading.pass_no,
            "state": notes.get("state", "ERROR" if reading.error else ""),
            "found": int(found is not None),
            "raw_reprojection_px": ("" if notes.get("raw_reprojection") is None
                                      else "%.6f" % notes["raw_reprojection"]),
            "depth_center_m": ("" if not notes.get("depth_center")
                               else "%.6f" % notes["depth_center"]),
            "message": reading.error or notes.get("reason", ""),
        }
        self._corners(row, "raw", notes.get("raw_corners"))
        self._corners(row, "filtered", notes.get("corners"))
        self._pose(row, "raw", notes.get("raw_pose"))
        self._pose(row, "filtered", None if found is None else found.matrix())
        self.writer.writerow(row)
        self.rows += 1
        if self.rows % 30 == 0:
            self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.flush()
            self.file.close()


class Reading:
    """What the camera last saw, and when."""

    def __init__(self, frame=None, detection=None, error=None, stamp=None,
                 pass_no=0, notes=None):
        self.frame = frame
        self.detection = detection
        self.error = error
        self.stamp = stamp or time.monotonic()
        # which pass of the loop produced it, so a caller can tell a reading
        # taken after its request from one that was already in flight
        self.pass_no = int(pass_no)
        # what the detector passed over on the way to its answer
        self.notes = dict(notes or {})

    def why_not(self):
        """Why there is no detection, in the terms an operator can act on."""
        if self.detection is not None:
            return ""
        if self.error:
            return self.error
        return self.notes.get("reason") or self.notes.get("state") or \
            "no rectangular opening found inside ROI"

    @property
    def age(self):
        return time.monotonic() - self.stamp

    @property
    def found(self):
        return self.detection is not None


class VisionService:
    """One camera, read continuously, with the newest answer kept."""

    def __init__(self, config=None, log=None):
        self.config = dict(config or {})
        self.log = log or (lambda text: None)
        self.camera = None
        self.detector = None
        self.csv_log = None
        self.thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._reading = Reading(error="not started")
        self._passes = 0                     # how many have *started*
        self._new = threading.Condition(self._lock)

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if self.running:
            return self
        try:
            self.camera = make_camera(self.config).open()
        except CameraError as e:
            self._set(Reading(error=str(e)))
            self.log("camera: %s" % e)
            return self
        self.detector = OpenBoxDetector(**self.settings())
        if self.config.get("log_enabled", False):
            directory = Path(self.config.get("log_dir", "logs"))
            path = directory / datetime.now().strftime(
                "camera_openbox_%Y%m%d_%H%M%S_%f.csv")
            try:
                self.csv_log = DetectionLog(path, {
                    "camera": self.camera.description,
                    "detector": self.settings(),
                })
                self.log("camera log: %s" % self.csv_log.path)
            except OSError as exc:
                self.csv_log = None
                self.log("camera log disabled: %s" % exc)
        self._stop.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.log("camera: %s" % self.camera.description)
        return self

    def stop(self):
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        self.detector = None
        if self.csv_log is not None:
            self.csv_log.close()
            self.log("camera log saved: %s (%d frames)" %
                     (self.csv_log.path, self.csv_log.rows))
            self.csv_log = None

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    # -- what it saw -------------------------------------------------------
    def _set(self, reading):
        with self._new:
            self._reading = reading
            self._new.notify_all()

    @property
    def latest(self):
        with self._new:
            return self._reading

    def refit(self):
        """Forget the temporal lock, so the opening is acquired afresh."""
        if self.detector is not None:
            self.detector.reset()

    def settings(self):
        """Settings for the known-size opening detector and its tracker."""
        return {
            "box_size": tuple(self.config.get("box_size") or (0.60, 0.40, 0.20)),
            "box_sizes": tuple(tuple(v) for v in
                               (self.config.get("box_sizes") or ())),
            # When enabled, inspect pale small-box faces as well as the main
            # crate rim and let the independent depth edge choose the size.
            # The CLI exposes the same detector mode as --measure.
            "auto_size": bool(self.config.get("auto_size", False)),
            # None means the whole picture, which is what a cell without a
            # `vision.roi` wants; a default window here would put one back
            "roi": tuple(self.config["roi"]) if self.config.get("roi") else None,
            "smoothing": float(self.config.get("smoothing", 0.20)),
            "max_reprojection": float(
                self.config.get("max_reprojection", 4.0)),
            "max_corner_jump": float(self.config.get("max_corner_jump", 35.0)),
            "confirm_frames": int(self.config.get("confirm_frames", 4)),
            "hold_frames": int(self.config.get("hold_frames", 15)),
        }

    # -- the loop ----------------------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            with self._new:
                self._passes += 1
                started_as = self._passes
            try:
                frame = self.camera.read()
                notes = {}
                detection = self.detector.find(frame, notes=notes)
                reading = Reading(frame, detection, pass_no=started_as,
                                  notes=notes)
                if self.csv_log is not None:
                    self.csv_log.write(reading)
                self._set(reading)
            except (CameraError, DetectionError) as e:
                reading = Reading(error=str(e), pass_no=started_as)
                if self.csv_log is not None:
                    self.csv_log.write(reading)
                self._set(reading)
            except Exception as e:                   # a lens is not worth a crash
                reading = Reading(error="%s: %s" % (type(e).__name__, e),
                                  pass_no=started_as)
                if self.csv_log is not None:
                    self.csv_log.write(reading)
                self._set(reading)
            if self._stop.wait(float(self.config.get("period", 0.0))):
                break

    def fresh(self, timeout=3.0):
        """A reading from a pass that *began* after this call.

        Not merely the next one to finish: when this is called there is
        usually a pass already in flight, taken before whatever the caller
        just changed — most importantly, before an arm moved out of the way.
        Answering with that one is answering with a picture of the arm.
        """
        if not self.running:
            return self.latest
        with self._new:
            after = self._passes
            deadline = time.monotonic() + timeout
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    return Reading(
                        error="the camera did not answer in %.0f s" % timeout)
                if self._reading.pass_no <= after:
                    self._new.wait(left)
                    continue
                # Initial temporal lock deliberately needs several agreeing
                # frames. A Once/FIND request wants the first confirmed pose,
                # not the intermediate LOCKING 1/4 reading.
                state = str(self._reading.notes.get("state", ""))
                # A held pose is useful to keep the overlay from flashing,
                # but it is deliberately stale and must never move a robot.
                # FIND waits for a newly accepted TRACKING/RELOCKED answer.
                if (state.startswith("LOCKING") or
                        (self._reading.detection is not None and
                         (state.startswith("HOLD") or
                          state.startswith("REJECT")))):
                    after = self._reading.pass_no
                    self._new.wait(left)
                    continue
                return self._reading
