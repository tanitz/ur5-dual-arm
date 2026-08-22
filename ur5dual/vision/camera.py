"""
Where a frame comes from.

One class per source, the same surface on each, and `make_camera` decides
which — the same shape `robot/backends.py` uses for the servo backends and
for the same reason: the detector, the panel and the program step never learn
which kind of camera answered.

`SimCamera` is not a stub. It renders a depth image of a plane with a box on
it, from the pose it was told to put the box at, so the detector can be tested
against a number that is known rather than against a number a human read off a
screen. Every millimetre it is wrong by is a millimetre of the detector's own
error, and nothing else.

A RealSense is optional. It is imported when one is opened and not before, so
a Jetson with no camera plugged in still runs the panel, the tests and the
simulated source.
"""

import math

import numpy as np


class CameraError(RuntimeError):
    pass


class Intrinsics:
    """A pinhole camera, in the only four numbers the maths needs."""

    def __init__(self, width, height, fx, fy, cx=None, cy=None):
        self.width, self.height = int(width), int(height)
        self.fx, self.fy = float(fx), float(fy)
        self.cx = float(width / 2.0 if cx is None else cx)
        self.cy = float(height / 2.0 if cy is None else cy)

    @classmethod
    def from_fov(cls, width, height, hfov_deg):
        """What a datasheet gives you: a field of view, not a focal length."""
        fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(width, height, fx, fx)

    def deproject(self, u, v, depth):
        """One pixel and its depth -> a point in the camera's own frame.

        +Z out of the lens, +X right, +Y down: the convention every RealSense
        and every OpenCV routine already uses, so nothing has to be flipped on
        the way in.
        """
        return np.array([(u - self.cx) * depth / self.fx,
                         (v - self.cy) * depth / self.fy,
                         depth], dtype=float)

    def deproject_grid(self, depth):
        """The whole depth image at once, as an (h, w, 3) array of points.

        Zero depth means the sensor had nothing to say about that pixel, and
        stays zero here rather than becoming a point on the lens.
        """
        depth = np.asarray(depth, dtype=float)
        v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        return np.dstack([x, y, depth])

    def to_dict(self):
        return {"width": self.width, "height": self.height,
                "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    @classmethod
    def from_dict(cls, d):
        return cls(d["width"], d["height"], d["fx"], d["fy"],
                   d.get("cx"), d.get("cy"))


class Frame:
    """One reading: metres of depth, and the colour image if there was one."""

    def __init__(self, depth, intrinsics, color=None, stamp=None):
        self.depth = np.asarray(depth, dtype=float)
        self.intrinsics = intrinsics
        self.color = color
        self.stamp = stamp

    @property
    def valid(self):
        return bool(np.any(self.depth > 0))


class Camera:
    """The surface every source answers."""

    def open(self):
        return self

    def read(self):
        raise NotImplementedError

    def close(self):
        pass

    @property
    def description(self):
        return self.__class__.__name__


class SimCamera(Camera):
    """A plane with a box on it, drawn rather than photographed.

    The box is given in the camera's own frame: `centre` is the middle of its
    top face and `yaw` turns it about the plane's normal. That is exactly what
    the detector is asked to recover, so a test can compare the two directly.
    """

    def __init__(self, intrinsics=None, plane_z=0.70, box_size=(0.18, 0.12),
                 box_height=0.08, centre=None, yaw=0.0, noise=0.0, seed=0):
        self.intrinsics = intrinsics or Intrinsics.from_fov(640, 480, 87.0)
        self.plane_z = float(plane_z)
        self.box_size = tuple(float(v) for v in box_size)
        self.box_height = float(box_height)
        self.centre = np.array([0.0, 0.0] if centre is None else centre,
                               dtype=float)
        self.yaw = float(yaw)
        self.noise = float(noise)
        self._rng = np.random.default_rng(seed)

    def place(self, centre=None, yaw=None):
        """Move the box the camera is looking at. What a test drives."""
        if centre is not None:
            self.centre = np.asarray(centre, dtype=float)
        if yaw is not None:
            self.yaw = float(yaw)
        return self

    def read(self):
        k = self.intrinsics
        v, u = np.mgrid[0:k.height, 0:k.width]

        # the table: a plane square to the lens at plane_z
        depth = np.full((k.height, k.width), self.plane_z, dtype=float)

        # the box's top face is one plane nearer, and the pixels it covers are
        # found by asking where each ray meets that height
        top_z = self.plane_z - self.box_height
        x = (u - k.cx) * top_z / k.fx
        y = (v - k.cy) * top_z / k.fy
        dx, dy = x - self.centre[0], y - self.centre[1]
        cos, sin = math.cos(-self.yaw), math.sin(-self.yaw)
        along = dx * cos - dy * sin
        across = dx * sin + dy * cos
        on_box = ((np.abs(along) <= self.box_size[0] / 2.0)
                  & (np.abs(across) <= self.box_size[1] / 2.0))
        depth[on_box] = top_z

        if self.noise > 0:
            depth += self._rng.normal(0.0, self.noise, depth.shape)

        # A colour image too, because the panel shows both and a source that
        # answered with only half of what a real one gives would leave the
        # other pane untested until a lens arrived.
        color = np.full((k.height, k.width, 3), 90, dtype=np.uint8)
        color[on_box] = (165, 150, 140)
        return Frame(depth, k, color=color, stamp=None)

    @property
    def description(self):
        return "simulated %dx%d, box %.0fx%.0f mm at %.0f mm" % (
            self.intrinsics.width, self.intrinsics.height,
            self.box_size[0] * 1000, self.box_size[1] * 1000,
            self.plane_z * 1000)


class RealSenseCamera(Camera):
    """An Intel RealSense, opened only when one is actually wanted."""

    def __init__(self, width=640, height=480, fps=30, serial=None, warmup=15):
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.serial = serial
        self.warmup = max(0, int(warmup))
        self.pipeline = None
        self.intrinsics = None
        self._align = None
        self._scale = 1.0

    def open(self):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise CameraError(
                "pyrealsense2 is not installed (%s). On this Jetson:\n"
                "    sudo apt install -y python3-pip\n"
                "    pip3 install pyrealsense2\n"
                "A camera also needs udev rules to be reachable without root "
                "— librealsense's 99-realsense-libusb.rules. Until then the "
                "Camera tab's 'sim' source works and the detector can be set "
                "up without a lens." % e)
        config = rs.config()
        if self.serial:
            config.enable_device(str(self.serial))
        config.enable_stream(rs.stream.depth, self.width, self.height,
                             rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height,
                             rs.format.bgr8, self.fps)
        self.pipeline = rs.pipeline()
        try:
            profile = self.pipeline.start(config)
        except Exception as e:
            self.pipeline = None
            raise CameraError("no RealSense opened: %s" % e)
        self._align = rs.align(rs.stream.color)
        self._scale = profile.get_device().first_depth_sensor().get_depth_scale()
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        k = stream.get_intrinsics()
        self.intrinsics = Intrinsics(k.width, k.height, k.fx, k.fy, k.ppx, k.ppy)
        # Auto exposure starts dark/noisy enough that the edge detector sees
        # a different scene for its first few frames. Do not let those frames
        # become the Camera tab's first failed Once/FIND result.
        for _ in range(self.warmup):
            self.pipeline.wait_for_frames()
        return self

    def read(self):
        if self.pipeline is None:
            raise CameraError("camera is not open")
        frames = self._align.process(self.pipeline.wait_for_frames())
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth:
            raise CameraError("no depth in this frame")
        return Frame(np.asanyarray(depth.get_data()) * self._scale,
                     self.intrinsics,
                     color=np.asanyarray(color.get_data()) if color else None)

    def close(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None

    @property
    def description(self):
        return "RealSense %dx%d @%d" % (self.width, self.height, self.fps)


def make_camera(cfg=None):
    """The camera `vision.source` names, or the simulated one.

    A cell with no camera plugged in gets the simulated source rather than an
    error, for the same reason a cell with no robots still opens the panel:
    the thing being set up is usually the thing that is not there yet.
    """
    cfg = cfg or {}
    source = cfg.get("source", "sim")
    if source == "realsense":
        return RealSenseCamera(cfg.get("width", 640), cfg.get("height", 480),
                               cfg.get("fps", 30), cfg.get("serial"),
                               cfg.get("warmup", 15))
    # The simulated box is the box the detector is looking for. Two defaults
    # that disagree mean a cell fresh out of the box reports "nothing found"
    # against its own scenery, which is the least useful first impression a
    # camera page could give.
    return SimCamera(plane_z=float(cfg.get("sim_plane_z", 0.70)),
                     box_size=cfg.get("box_size") or (0.30, 0.22),
                     box_height=float(cfg.get("sim_box_height",
                                              (list(cfg.get("box_size") or []) +
                                               [0.08, 0.08, 0.08])[2])))
