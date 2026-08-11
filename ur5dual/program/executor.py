"""
Running a two-arm program.

Steps execute on a worker thread so the GUI stays responsive, and every
transition is reported through callbacks rather than touched directly — the
executor knows nothing about Qt.

The awkward part of driving UR controllers over sockets is that `movel`
returns the moment it is sent, not when the arm arrives. There is no
completion event to wait on, so arrival is inferred: the joints have stopped
and the TCP is where it was asked to be. `_wait_until_arrived` is that check,
and BARRIER is the same check applied to both arms at once.
"""

import threading
import time

import numpy as np

from ..coupling import Coordinator, CouplingError, HeldObject
from ..geometry.kinematics import pose_distance

IDLE_SPEED = 0.01        # rad/s below which a joint counts as stopped
ARRIVE_LIN = 0.002       # m
ARRIVE_ANG = 0.02        # rad
SETTLE_TIME = 0.15       # s of continuous stillness before believing it


class ProgramError(RuntimeError):
    pass


class Executor:
    """Runs a Program against a Cell. One at a time."""

    def __init__(self, cell, points, simulate=False):
        self.cell = cell
        self.points = points
        self.simulate = simulate

        self.object = HeldObject()
        self.coordinator = None

        self.thread = None
        self._pause = threading.Event()
        self._stop = threading.Event()
        self.running = False
        self.current = -1

        # callbacks: (index, step) / (text) / (ok, message)
        self.on_step = None
        self.on_log = None
        self.on_finished = None

    # -- plumbing ----------------------------------------------------------
    def log(self, text):
        if self.on_log:
            self.on_log(text)
        else:
            self.cell.log(text)

    def start(self, program):
        if self.running:
            raise ProgramError("a program is already running")
        problems = program.validate(self.points)
        if problems:
            raise ProgramError("; ".join(problems))
        self._stop.clear()
        self._pause.clear()
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(program,),
                                       daemon=True)
        self.thread.start()

    def pause(self):
        self._pause.set()
        self.cell.halt()
        self.log("paused")

    def resume(self):
        self._pause.clear()
        self.log("resumed")

    def stop(self):
        self._stop.set()
        if self.coordinator is not None:
            self.coordinator.abort()
        self.cell.halt()

    @property
    def paused(self):
        return self._pause.is_set()

    # -- the loop ----------------------------------------------------------
    def _run(self, program):
        ok, message = True, "finished"
        try:
            while True:
                for i, step in enumerate(program.steps):
                    if self._stop.is_set():
                        raise ProgramError("stopped")
                    while self._pause.is_set():
                        if self._stop.is_set():
                            raise ProgramError("stopped")
                        time.sleep(0.05)
                    if not step.enabled:
                        continue
                    self.current = i
                    if self.on_step:
                        self.on_step(i, step)
                    self.log("%3d  %s" % (i + 1, step.describe()))
                    self._execute(step)
                if not program.loop:
                    break
        except (ProgramError, CouplingError, OSError, KeyError, ValueError) as e:
            ok, message = False, str(e)
            self.cell.halt()
        finally:
            self._shutdown_coordinator()
            self.running = False
            self.current = -1
            self.log("program %s: %s" % ("finished" if ok else "aborted", message))
            if self.on_finished:
                self.on_finished(ok, message)

    def _execute(self, step):
        kind = step.kind
        if kind == "MOVE_ARM":
            self._move_arm(step)
        elif kind == "GRIP":
            self._grip(step)
        elif kind == "BARRIER":
            self._wait_arms_idle(self.cell.connected_ids)
        elif kind == "DELAY":
            self._sleep(float(step.get("seconds", 0)))
        elif kind == "ATTACH":
            self._attach(step)
        elif kind == "DETACH":
            self._detach()
        elif kind == "MOVE_OBJ":
            self._move_object(step)
        elif kind == "ROTATE_OBJ":
            self._rotate_object(step)
        else:
            raise ProgramError("cannot execute %r" % kind)

    # -- single-arm steps --------------------------------------------------
    def _move_arm(self, step):
        arm_id = step.get("arm")
        arm = self.cell.arms[arm_id]
        if not arm.connected:
            raise ProgramError("arm %s is not connected" % arm_id)
        target = self.points.get(step.get("point"))

        reachable, why = self.cell.check_reachable(arm_id, target)
        if not reachable:
            raise ProgramError(why)

        speed = float(step.get("speed") or self.cell.config.limits["object_lin_speed"])
        if self.simulate:
            self._sleep(0.2)
            return
        if step.get("motion", "movej") == "movel":
            arm.movel_world(target, vel=speed)
        else:
            arm.movej_world(target, vel=speed)
        self._wait_until_arrived(arm_id, target)

    def _grip(self, step):
        arms = ("A", "B") if step.get("arm") == "both" else (step.get("arm"),)
        out = int(step.get("output", 0))
        state = bool(step.get("state", True))
        for a in arms:
            if self.cell.arms[a].connected and not self.simulate:
                self.cell.arms[a].motion.set_digital_out(out, state)
        self._sleep(float(step.get("settle", 0.4)))

    # -- coupled steps -----------------------------------------------------
    def _attach(self, step):
        arm_ids = tuple(a for a in ("A", "B") if self.cell.arms[a].connected)
        if len(arm_ids) < 2 and step.get("origin", "midpoint") == "midpoint":
            raise ProgramError("two connected arms are needed to take hold "
                               "with a midpoint origin")
        self.object = HeldObject(step.get("object", "object"))
        self.object.capture(self.cell, arm_ids, step.get("origin", "midpoint"))
        self.log("     grip span %.1f mm" % ((self.object.span() or 0) * 1000))

        self.coordinator = Coordinator(self.cell, simulate=self.simulate)
        self.coordinator.start(self.object)

    def _detach(self):
        self._shutdown_coordinator()
        self.object.release()

    def _shutdown_coordinator(self):
        if self.coordinator is not None:
            self.coordinator.shutdown()
            self.coordinator = None

    def _move_object(self, step):
        target = self.points.get(step.get("point"))
        for a in self.object.arm_ids:
            pose = self.object.tcp_world(a, target)
            reachable, why = self.cell.check_reachable(
                a, np.concatenate([pose[:3, 3], [0, 0, 0]]))
            if not reachable:
                raise ProgramError("carrying there puts %s" % why)
        self.coordinator.move_object(self.object, target,
                                     step.get("lin_speed"), step.get("ang_speed"))

    def _rotate_object(self, step):
        self.coordinator.rotate_object(self.object,
                                       step.get("axis", "z"),
                                       np.radians(float(step.get("angle_deg", 0))),
                                       step.get("ang_speed"),
                                       # programs written before world rotation
                                       # existed meant the object's own axes,
                                       # and must keep meaning that
                                       step.get("frame", "object"))

    # -- waiting -----------------------------------------------------------
    def _sleep(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._stop.is_set():
                raise ProgramError("stopped")
            time.sleep(0.02)

    def _wait_until_arrived(self, arm_id, target_world, timeout=60.0):
        arm = self.cell.arms[arm_id]
        deadline = time.monotonic() + timeout
        still_since = None
        while True:
            if self._stop.is_set():
                raise ProgramError("stopped")
            if not arm.ready():
                raise ProgramError("arm %s: %s / %s" % (arm_id, arm.robot_mode(),
                                                        arm.safety_mode()))
            st = arm.state()
            moving = float(np.max(np.abs(st["qd_actual"]))) > IDLE_SPEED
            d_t, d_r = pose_distance(arm.tcp_pose_world(), target_world)
            arrived = d_t < ARRIVE_LIN and d_r < ARRIVE_ANG

            if not moving and arrived:
                still_since = still_since or time.monotonic()
                if time.monotonic() - still_since >= SETTLE_TIME:
                    return
            else:
                still_since = None

            if time.monotonic() > deadline:
                raise ProgramError(
                    "arm %s did not arrive: still %.1f mm / %.2f deg away"
                    % (arm_id, d_t * 1000, np.degrees(d_r)))
            time.sleep(0.02)

    def _wait_arms_idle(self, arm_ids, timeout=60.0):
        deadline = time.monotonic() + timeout
        still_since = None
        while True:
            if self._stop.is_set():
                raise ProgramError("stopped")
            moving = False
            for a in arm_ids:
                st = self.cell.arms[a].state()
                if float(np.max(np.abs(st["qd_actual"]))) > IDLE_SPEED:
                    moving = True
            if not moving:
                still_since = still_since or time.monotonic()
                if time.monotonic() - still_since >= SETTLE_TIME:
                    return
            else:
                still_since = None
            if time.monotonic() > deadline:
                raise ProgramError("arms never settled at the barrier")
            time.sleep(0.02)
