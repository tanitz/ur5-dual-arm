"""Minimal object-centric midpoint hold, independent of coupling.py.

This is deliberately a small commissioning path.  It captures one virtual
object frame midway between two live TCPs, freezes the two grasp transforms,
and feeds the resulting *unchanged* targets to both robot controllers from one
clock.  It never translates or rotates the object.  Passing this test proves
that the dual servo backends can start and hold a closed-chain grasp before any
directional motion is attempted.
"""

import threading
import time

import numpy as np

from ..robot.backends import make_backend
from ..geometry.kinematics import inv, mat_to_pose, pose_to_mat


class MidpointHoldError(RuntimeError):
    pass


class MidpointObject:
    """One virtual midpoint frame and fixed transforms to both TCPs."""

    def __init__(self):
        self.pose_world = None
        self.grasps = {}
        self.targets_base = {}

    def capture(self, cell):
        if tuple(cell.connected_ids) != ("A", "B"):
            raise MidpointHoldError("both arms A and B must be connected")

        tcp_world = {a: cell.arms[a].tcp_matrix_world() for a in ("A", "B")}
        object_world = tcp_world["A"].copy()
        object_world[:3, 3] = (
            tcp_world["A"][:3, 3] + tcp_world["B"][:3, 3]
        ) / 2.0

        self.pose_world = mat_to_pose(object_world)
        self.grasps = {a: inv(object_world) @ tcp_world[a] for a in ("A", "B")}
        self.targets_base = {
            a: cell.arms[a].world_to_base(
                mat_to_pose(object_world @ self.grasps[a]))
            for a in ("A", "B")
        }
        return self

    def reconstruction_errors(self, cell):
        """Pose-matrix error between captured and regenerated TCP targets."""
        return {
            a: float(np.max(np.abs(
                pose_to_mat(self.targets_base[a])
                - pose_to_mat(cell.arms[a].tcp_pose())
            )))
            for a in ("A", "B")
        }

    def targets_for_pose(self, cell, pose_world):
        object_world = pose_to_mat(pose_world)
        return {
            a: cell.arms[a].world_to_base(
                mat_to_pose(object_world @ self.grasps[a]))
            for a in ("A", "B")
        }


class MidpointHoldLoop:
    """Feed one stationary midpoint-object target to both arms."""

    def __init__(self, cell, held_object, max_motion=0.003):
        self.cell = cell
        self.object = held_object
        self.max_motion = float(max_motion)
        self.backends = {}
        self.samples = 0
        self.max_tcp_motion = {"A": 0.0, "B": 0.0}

    def _stop_all(self):
        for backend in self.backends.values():
            try:
                backend.shutdown()
            except Exception:
                pass
        self.backends = {}
        self.cell.halt()

    def run(self, seconds=3.0):
        if seconds <= 0:
            raise MidpointHoldError("hold duration must be positive")
        reason = self.cell.not_ready_reason(("A", "B"))
        if reason:
            raise MidpointHoldError(reason)
        if not self.cell.feeds_fresh():
            raise MidpointHoldError("robot state feeds are stale")

        start = {
            a: self.cell.arms[a].tcp_matrix_world()[:3, 3].copy()
            for a in ("A", "B")
        }
        baseline_force = {
            a: self.cell.force_vector(a).copy() for a in ("A", "B")
        }
        rate = float(self.cell.config.motion["servo_rate_hz"])
        dt = 1.0 / rate

        try:
            for a in ("A", "B"):
                # Cartesian targets always, whatever motion.control says. This
                # is the commissioning path that proves two backends can hold
                # a grasp at all, and it has to stay the simplest thing that
                # can work — putting our own IK in front of it would mean a
                # failure here could be either the servo stream or the
                # kinematics, which is exactly the ambiguity it exists to
                # remove.
                backend = make_backend(self.cell.arms[a].cfg.ip,
                                       self.cell.config.motion, control="pose")
                backend.start()
                self.backends[a] = backend

            deadline = time.monotonic() + float(seconds)
            next_tick = time.monotonic()
            while time.monotonic() < deadline:
                reason = self.cell.not_ready_reason(("A", "B"))
                if reason:
                    raise MidpointHoldError(reason)
                if not self.cell.feeds_fresh():
                    raise MidpointHoldError("robot state feed became stale")

                for a in ("A", "B"):
                    force_rise = float(np.linalg.norm(
                        self.cell.force_vector(a) - baseline_force[a]))
                    limit = float(self.cell.config.motion["max_force_rise"])
                    if force_rise > limit:
                        raise MidpointHoldError(
                            "arm %s force rose %.0f N (limit %.0f N)"
                            % (a, force_rise, limit))

                # One clock, one stationary virtual object, two derived TCPs.
                for a in ("A", "B"):
                    self.backends[a].servo_pose(self.object.targets_base[a])

                for a in ("A", "B"):
                    moved = float(np.linalg.norm(
                        self.cell.arms[a].tcp_matrix_world()[:3, 3] - start[a]))
                    self.max_tcp_motion[a] = max(self.max_tcp_motion[a], moved)
                    if moved > self.max_motion:
                        raise MidpointHoldError(
                            "arm %s moved %.1f mm during a stationary hold "
                            "(limit %.1f mm)" %
                            (a, moved * 1000.0, self.max_motion * 1000.0))
                self.samples += 1

                next_tick += dt
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
        finally:
            self._stop_all()
        return self


class MidpointServoSession:
    """Non-blocking commissioning loop used by the Object-tab test controls.

    It starts at the captured midpoint and permits only tiny world-frame
    translations.  Rotation is intentionally absent: first prove that all
    three translation axes agree on the real cell.
    """

    MAX_STEP = 0.002       # 2 mm per button press
    READY_SAMPLES = 125    # one second stable at the configured 125 Hz

    def __init__(self, cell, held_object):
        self.cell = cell
        self.object = held_object
        self.backends = {}
        self.error = None
        self.state = "idle"
        self.samples = 0
        self.drift = 0.0
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._target_pose = np.array(held_object.pose_world, dtype=float)
        self._captured_relative = None
        self._force_baseline = None

    @property
    def ready_for_step(self):
        return self.state == "holding" and self.samples >= self.READY_SAMPLES

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return self
        self.error = None
        self.state = "starting"
        self.samples = 0
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="ur5dual-midpoint-test")
        self._thread.start()
        return self

    def stop(self, wait=False):
        self._running = False
        if (wait and self._thread is not None
                and self._thread is not threading.current_thread()):
            self._thread.join(timeout=1.0)

    def step_world(self, axis, distance):
        if not self.ready_for_step:
            raise MidpointHoldError("hold must be stable for one second first")
        axis = int(axis)
        distance = float(distance)
        if axis not in (0, 1, 2):
            raise MidpointHoldError("translation axis must be X, Y, or Z")
        if abs(distance) > self.MAX_STEP + 1e-12:
            raise MidpointHoldError("one test step is limited to %.1f mm"
                                    % (self.MAX_STEP * 1000.0))
        with self._lock:
            self._target_pose[axis] += distance
        return np.array(self._target_pose, dtype=float)

    def _stop_all(self):
        for backend in self.backends.values():
            try:
                backend.shutdown()
            except Exception:
                pass
        self.backends = {}
        try:
            self.cell.halt()
        except Exception:
            pass

    def _run(self):
        rate = float(self.cell.config.motion["servo_rate_hz"])
        dt = 1.0 / rate
        try:
            reason = self.cell.not_ready_reason(("A", "B"))
            if reason:
                raise MidpointHoldError(reason)
            if not self.cell.feeds_fresh():
                raise MidpointHoldError("robot state feeds are stale")

            self._captured_relative = self.cell.relative_transform(max_age=None)
            self._force_baseline = {
                a: self.cell.force_vector(a).copy() for a in ("A", "B")
            }
            for a in ("A", "B"):
                # Cartesian targets always, whatever motion.control says. This
                # is the commissioning path that proves two backends can hold
                # a grasp at all, and it has to stay the simplest thing that
                # can work — putting our own IK in front of it would mean a
                # failure here could be either the servo stream or the
                # kinematics, which is exactly the ambiguity it exists to
                # remove.
                backend = make_backend(self.cell.arms[a].cfg.ip,
                                       self.cell.config.motion, control="pose")
                backend.start()
                self.backends[a] = backend

            self.state = "holding"
            next_tick = time.monotonic()
            while self._running:
                reason = self.cell.not_ready_reason(("A", "B"))
                if reason:
                    raise MidpointHoldError(reason)
                if not self.cell.feeds_fresh():
                    raise MidpointHoldError("robot state feed became stale")

                for a in ("A", "B"):
                    rise = float(np.linalg.norm(
                        self.cell.force_vector(a) - self._force_baseline[a]))
                    limit = float(self.cell.config.motion["max_force_rise"])
                    if rise > limit:
                        raise MidpointHoldError(
                            "arm %s force rose %.0f N (limit %.0f N)"
                            % (a, rise, limit))

                relative = self.cell.relative_transform()
                if relative is not None:
                    self.drift = float(np.linalg.norm(
                        relative[:3, 3] - self._captured_relative[:3, 3]))
                    limit = float(self.cell.config.motion["max_pair_drift"])
                    if self.drift > limit:
                        raise MidpointHoldError(
                            "TCP relationship drifted %.1f mm (limit %.1f mm)"
                            % (self.drift * 1000.0, limit * 1000.0))

                with self._lock:
                    pose = np.array(self._target_pose, dtype=float)
                targets = self.object.targets_for_pose(self.cell, pose)
                for a in ("A", "B"):
                    self.backends[a].servo_pose(targets[a])
                self.object.pose_world = pose
                self.samples += 1

                next_tick += dt
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
        except Exception as exc:
            self.error = exc
            self.state = "error"
        finally:
            self._running = False
            self._stop_all()
            if self.state != "error":
                self.state = "stopped"
