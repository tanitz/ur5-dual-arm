"""
Which way is down, asked of the earth rather than of a sheet of paper.

`charuco.py` measures the surface a box stands on, and everything built on it
— the height line, a pick's approach — is measured along that surface's own
normal. That is the right datum for "how tall is the box", and it is silent
about the question underneath it: whether the surface is *level*. A crate lid
resting on a table that is 2 degrees out is a perfectly good plane, and a
height measured against it is a perfectly good number that is not a height
above the floor.

Nothing in a picture can settle that. A camera cannot see gravity, and a
board lying on a slope looks exactly like a board lying flat. The D435i has
an accelerometer, so on this cell it does not have to be seen: at rest the
only force on it is gravity, and the direction it reads is the vertical the
whole room is built to.

Deliberately its own pipeline, opened and closed for one reading. Adding a
motion stream to `RealSenseCamera` would put an IMU in the path of every
frame the detector takes, for a number that changes when somebody moves the
bracket and at no other time. A camera that has been bolted down has one
answer here for as long as it stays bolted.

One reading at rest, and only at rest. An accelerometer measures the sum of
gravity and whatever else is accelerating it, and cannot tell them apart, so
a camera on a moving arm reads a vertical that is wrong for exactly as long
as the arm is moving. `read_gravity` returns the spread across its samples so
a caller can see that it was still, rather than trusting that it was.
"""

import numpy as np


# What the samples may disagree by, as a fraction of g, and still be one
# reading of a camera at rest. Measured noise on a D435i lying still is well
# under a thousandth of g; a bracket being knocked is far over this.
STILL = 0.02

# Earth's, near enough for a direction. Only used to say whether what came
# back is plausibly gravity alone rather than gravity plus a shove.
G = 9.81


class ImuError(RuntimeError):
    pass


def down_from_samples(samples):
    """The unit vector gravity points along, in the camera's frame.

    An accelerometer at rest reads the *reaction* to gravity — the push of
    whatever is holding it up — so what comes back points up and the direction
    wanted is its negative. Getting that sign wrong turns every height in the
    cell upside down while leaving every magnitude right, which is the kind of
    mistake that survives a review and fails on the floor.
    """
    samples = np.asarray(samples, dtype=float).reshape(-1, 3)
    if len(samples) < 2:
        raise ImuError("one sample cannot show whether the camera was still")
    mean = samples.mean(axis=0)
    magnitude = float(np.linalg.norm(mean))
    if magnitude < 1e-6:
        raise ImuError("the accelerometer read nothing at all")
    spread = float(np.max(np.linalg.norm(samples - mean, axis=1))) / G
    if spread > STILL:
        raise ImuError(
            "the camera was moving while this was read (samples differ by "
            "%.1f%% of g); read it again with the rig at rest" % (spread * 100))
    if abs(magnitude - G) / G > 0.10:
        raise ImuError(
            "the accelerometer read %.2f m/s^2, which is not gravity alone — "
            "the camera is being accelerated or the stream is not the "
            "accelerometer" % magnitude)
    return -mean / magnitude, spread


def level_error(normal, down):
    """Radians between a surface's normal and true vertical.

    Sign-free on purpose: a plane's normal may be stored pointing either way
    round — `BoardPlane` keeps it facing the lens — and how level a surface is
    cannot depend on which side of it somebody measured from.
    """
    normal = np.asarray(normal, dtype=float).reshape(3)
    down = np.asarray(down, dtype=float).reshape(3)
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    down = down / max(np.linalg.norm(down), 1e-12)
    return float(np.arccos(np.clip(abs(normal @ down), -1.0, 1.0)))


def read_gravity(serial=None, samples=60, timeout_ms=2000):
    """Gravity's direction in the camera's frame, from the D435i's IMU.

    Returns `(down, spread)`. Its own pipeline, opened and closed here — see
    the module docstring for why this is not part of the frame path.
    """
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ImuError("pyrealsense2 is not installed (%s)" % exc)

    config = rs.config()
    if serial:
        config.enable_device(str(serial))
    try:
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
    except Exception as exc:
        raise ImuError("this build cannot open an accelerometer: %s" % exc)

    pipeline = rs.pipeline()
    try:
        pipeline.start(config)
    except Exception as exc:
        raise ImuError(
            "no accelerometer opened: %s\n(a D435 without the 'i' has no IMU, "
            "and only one process may hold the camera — stop the Camera tab's "
            "live view first)" % exc)
    try:
        readings = []
        # The first frames after a motion stream starts are the ones taken
        # while the sensor was still settling, and they are the ones furthest
        # from gravity — which would be read here as a camera that moved.
        for index in range(samples + 10):
            frames = pipeline.wait_for_frames(timeout_ms)
            motion = frames.first_or_default(rs.stream.accel)
            if not motion:
                continue
            if index < 10:
                continue
            value = motion.as_motion_frame().get_motion_data()
            readings.append([value.x, value.y, value.z])
    finally:
        pipeline.stop()
    if len(readings) < 2:
        raise ImuError("the accelerometer sent nothing in %d frames" % samples)
    return down_from_samples(readings)
