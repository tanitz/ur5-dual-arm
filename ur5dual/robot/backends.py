"""
Servo backends — the thing that makes two arms move *together*.

Firing `movel` at both controllers is not coordination. Each one plans its
own trajectory with its own timing, and the tens of milliseconds they differ
by become millimetres of relative error. Two arms gripping one bottle then
fight each other until something gives.

The fix is to stop letting the controllers plan. A servo backend accepts a
fresh target ~125 times a second and tracks it, so both arms are driven from
one clock in one Python loop and their skew is network jitter, not planner
disagreement.

Two implementations, chosen by `motion.backend` in cell.yaml:

  RtdeBackend      ur_rtde's servoL/servoJ. Less code, well travelled, and
                   already installed — but it uploads its own control program
                   and has never been tried against this cell's PolyScope
                   3.7.2, which is old.

  UrScriptBackend  a streaming program uploaded to the controller which sits
                   in a loop reading poses off a socket and servoj()s to
                   them. No dependency, works on any 3.x firmware, and the
                   whole protocol is visible in this file.

Both satisfy the same four calls, so the coordinator neither knows nor cares
which one it has.
"""

import socket
import struct
import threading
import time

import numpy as np


class BackendError(RuntimeError):
    pass


class ServoBackend:
    """Interface every backend implements.

    servo_pose() is called at the loop rate with the next target; it must
    return promptly and never block on robot motion.

    servo_joints() is the same call with the IK already done on this side.
    Two arms holding one box want it: a Cartesian target lets each controller
    pick its own joint solution, and the elbow one of them flips mid-carry is
    the box. A backend built for poses cannot accept joints and vice versa —
    the stream program uploaded to the robot differs — so `control` is fixed
    when the backend is made rather than per call.
    """

    name = "abstract"
    control = "pose"

    # A backend that has stopped being consumed while the loop goes on feeding
    # it. Only the streaming one can tell — ur_rtde and the simulator both fail
    # loudly on their own — but the coordinator asks every backend, because
    # "the targets are going nowhere" must never be a thing only one of them
    # can notice.
    stalled = False

    def start(self):
        raise NotImplementedError

    def servo_pose(self, pose_base):
        raise NotImplementedError

    def servo_joints(self, q):
        raise NotImplementedError

    def stop_motion(self):
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError

    def send(self, target):
        """Whichever of the two this backend was built for."""
        if self.control == "joint":
            return self.servo_joints(target)
        return self.servo_pose(target)


# ── ur_rtde ───────────────────────────────────────────────────────────────
class RtdeBackend(ServoBackend):
    """servoL through ur_rtde's RTDEControlInterface.

    Connecting uploads ur_rtde's control program and takes the robot over;
    nothing moves until servo_pose() is called, but the teach pendant will
    show a program running.
    """

    name = "rtde"

    def __init__(self, ip, motion_cfg, control="pose"):
        self.ip = ip
        self.cfg = motion_cfg
        self.control = control
        self.ctrl = None
        self.dt = 1.0 / float(motion_cfg["servo_rate_hz"])

    def start(self):
        try:
            import rtde_control
        except ImportError as e:
            raise BackendError("ur_rtde is not importable: %s" % e)
        try:
            self.ctrl = rtde_control.RTDEControlInterface(self.ip)
        except Exception as e:                     # ur_rtde raises RuntimeError
            raise BackendError("RTDE connect to %s failed: %s" % (self.ip, e))
        if not self.ctrl.isConnected():
            raise BackendError("RTDE reports not connected to %s" % self.ip)
        return self

    def servo_pose(self, pose_base):
        self.ctrl.servoL(list(map(float, pose_base)),
                         float(self.cfg["servo_speed"]),
                         float(self.cfg["servo_accel"]),
                         self.dt,
                         float(self.cfg["servo_lookahead"]),
                         float(self.cfg["servo_gain"]))

    def servo_joints(self, q):
        # servoJ takes the same speed and acceleration arguments as servoL and
        # ignores them for the same reason: the trajectory is the stream of
        # targets, not something the controller plans. Only dt, lookahead and
        # gain do anything.
        self.ctrl.servoJ(list(map(float, q)),
                         float(self.cfg["servo_speed"]),
                         float(self.cfg["servo_accel"]),
                         self.dt,
                         float(self.cfg["servo_lookahead"]),
                         float(self.cfg["servo_gain"]))

    def stop_motion(self):
        if self.ctrl is not None:
            self.ctrl.servoStop()

    def shutdown(self):
        if self.ctrl is not None:
            try:
                self.ctrl.servoStop()
                self.ctrl.stopScript()
            except Exception:
                pass
            self.ctrl = None


# ── streaming URScript ────────────────────────────────────────────────────
# Two things on the robot, and the split between them is the whole design.
#
# `servo_thread` calls servoj on a global target and nothing else. servoj
# blocks for exactly one control cycle, so that thread — and therefore the
# arm — runs at a clean 125 Hz whatever the socket is doing. When no fresh
# target arrives it simply servos the last one again, which is an arm holding
# station rather than an arm being dropped and re-caught.
#
# The main loop only talks to the socket, and it *asks* for each target rather
# than being pushed them. That is what the first version of this got wrong: it
# read, ran get_inverse_kin, and servoj'd in one sequence, and every one of
# those costs the controller at least a control cycle. Fed at 125 Hz it could
# consume maybe half of that, so the unread targets piled up in the socket and
# the robot fell further behind every second, executing poses from further and
# further in the past. From the outside that is an arm that will not move.
# With the robot asking, it can never be sent more than it is ready for and no
# queue can form; the target simply refreshes at whatever rate the controller
# manages, and the servo thread interpolates between them at full rate.
# Laid out as ur_rtde's own control program is — globals and thread at the top
# level of the script, the loop after them, no enclosing `def`. That shape is
# what actually runs on a CB3, which matters more than tidiness here: a `global`
# declared inside a function is the one construct in this file that no shipped
# UR program uses, and if a controller rejects it the failure lands *after*
# socket_open. The stream connects, the panel reports both backends up, and the
# arms never move — which is indistinguishable from every other reason they
# might not, and cost a day of looking in the wrong places.
#
# textmsg puts a line in the pendant's own log, so whether the program reached
# its loop can be answered at the robot rather than inferred from here.
_STREAM_BODY = """\
global ur5dual_q = get_actual_joint_positions()
global ur5dual_run = True

thread ur5dual_servo():
  while ur5dual_run:
    servoj(ur5dual_q, t={dt}, lookahead_time={lookahead}, gain={gain})
  end
end

socket_open("{host}", {port}, "srv")
textmsg("ur5dual: servo stream open")
ur5dual_handle = run ur5dual_servo()
ur5dual_misses = 0
while ur5dual_run:
  socket_send_int(1, "srv")
  raw = socket_read_binary_integer(6, "srv", {timeout})
  if raw[0] == 6:
    ur5dual_misses = 0
{apply}
  else:
    ur5dual_misses = ur5dual_misses + 1
    if ur5dual_misses > {misses}:
      ur5dual_run = False
    end
  end
end
join ur5dual_handle
stopj({jacc})
textmsg("ur5dual: servo stream closed")
"""

# get_inverse_kin runs on the controller, so no IK is needed on this side;
# passing the previous joint solution as the seed keeps it on the same branch
# instead of flipping the elbow mid-move. It is also the expensive half of the
# loop, which is why `joint` control below is the better path when the Python
# kinematics have been checked against these robots.
_APPLY_POSE = """\
    ur5dual_target = p[raw[1] / {scale}, raw[2] / {scale}, raw[3] / {scale},
                       raw[4] / {scale}, raw[5] / {scale}, raw[6] / {scale}]
    ur5dual_q = get_inverse_kin(ur5dual_target, qnear=ur5dual_q)"""

# The same stream, with the IK already done on the Jetson. Six joint angles
# arrive instead of six pose components and go straight to the servo target,
# so the controller never chooses a branch and never has the chance to change
# its mind about one. Radians fit the fixed-point encoding with room to spare:
# +-2pi at 1e-5 resolution is 0.0006 degrees, far finer than the arm resolves.
_APPLY_JOINT = """\
    ur5dual_q = [raw[1] / {scale}, raw[2] / {scale}, raw[3] / {scale},
                 raw[4] / {scale}, raw[5] / {scale}, raw[6] / {scale}]"""


def _stream_script(apply_block):
    """The one template with the one line that differs pasted in.

    Assembled rather than formatted in two passes: the apply block carries its
    own {scale} placeholders, and str.format does not recurse into what it has
    just substituted.
    """
    return _STREAM_BODY.replace("{apply}", apply_block)


_STREAM_SCRIPT = _stream_script(_APPLY_POSE)
_STREAM_SCRIPT_JOINT = _stream_script(_APPLY_JOINT)

_SCALE = 100000.0          # poses go over as fixed-point ints, 1e-5 resolution


class UrScriptBackend(ServoBackend):
    """Our own streaming controller. No third-party dependency, and every
    byte of the protocol is in this file.

    The wire is request-response: the robot sends one 32-bit int to ask, and
    this side answers with the six that make up the newest target. So
    `servo_pose` and `servo_joints` do not touch the socket at all — they park
    a value, and a pump thread hands over whatever is parked when the robot
    next asks. Two things follow from that, and both matter:

      the coordinator never blocks   a 125 Hz loop that called sendall() into
                                     a controller reading more slowly would
                                     eventually block on the send buffer, with
                                     two arms holding a workpiece

      the robot is never behind      it is answered once per request, so the
                                     newest target is always the next one it
                                     sees. There is no queue to fall behind in
    """

    name = "urscript"

    # How long the robot waits for one answer, in seconds — the units URScript
    # counts socket timeouts in, which the first version of this had as
    # milliseconds and so waited 32 seconds where it meant 32 ms.
    READ_TIMEOUT = 0.1
    # Consecutive unanswered requests before the robot stops and ends the
    # program. Twenty of them is two seconds of silence: long enough that
    # bringing the second arm's stream up does not kill the first one's, short
    # enough that a coordinator that has died leaves a stopped robot.
    MAX_MISSES = 20

    def __init__(self, ip, motion_cfg, control="pose",
                 listen_host=None, listen_port=0):
        self.ip = ip
        self.cfg = motion_cfg
        self.control = control
        self.dt = 1.0 / float(motion_cfg["servo_rate_hz"])
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.srv = None
        self.conn = None
        self._lock = threading.Lock()
        self._target = None       # newest target, already fixed-point
        self._pump = None
        self._running = False
        self._fault = None        # what broke the stream, raised at the caller
        self._started_at = 0.0
        self.served = 0           # targets the robot has actually asked for
        self.last_served = 0.0

    def _local_ip_towards_robot(self):
        """Which of our addresses the robot can dial back on."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((self.ip, 30002))
            return s.getsockname()[0]
        finally:
            s.close()

    def start(self, connect_timeout=10.0):
        from .transport import send_script

        host = self.listen_host or self._local_ip_towards_robot()
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, self.listen_port))
        self.srv.listen(1)
        port = self.srv.getsockname()[1]
        self.srv.settimeout(connect_timeout)

        script = (_STREAM_SCRIPT_JOINT if self.control == "joint"
                  else _STREAM_SCRIPT)
        send_script(self.ip, script.format(
            host=host, port=port, scale=_SCALE, dt=self.dt,
            lookahead=float(self.cfg["servo_lookahead"]),
            gain=float(self.cfg["servo_gain"]),
            jacc=2.0,
            timeout=self.READ_TIMEOUT,
            misses=self.MAX_MISSES,
        ))

        try:
            self.conn, _ = self.srv.accept()
        except socket.timeout:
            self.srv.close()
            self.srv = None
            raise BackendError(
                "robot %s never connected back to %s:%d — check that the "
                "program is allowed to run (Remote Control on 3.13+) and that "
                "no firewall sits between them" % (self.ip, host, port))
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # so the pump notices `shutdown` rather than sitting in recv forever
        self.conn.settimeout(0.5)
        self._fault = None
        self._started_at = time.monotonic()
        self._running = True
        self._pump = threading.Thread(target=self._serve, daemon=True,
                                      name="ur5dual-urscript-%s" % self.ip)
        self._pump.start()
        return self

    # -- the pump ----------------------------------------------------------
    def _serve(self):
        """Answer each request with the newest target, one for one.

        Lockstep by construction: the robot has at most one request out at a
        time, so however slowly it gets through a cycle there is never a
        backlog of stale targets for it to work through first.
        """
        pending = b""
        while self._running:
            try:
                chunk = self.conn.recv(64)
            except socket.timeout:
                continue
            except OSError as e:
                self._break("servo stream to %s broke: %s" % (self.ip, e))
                return
            if not chunk:
                self._break("robot %s closed the servo stream — its program "
                            "stopped" % self.ip)
                return
            pending += chunk
            while len(pending) >= 4 and self._running:
                pending = pending[4:]
                with self._lock:
                    target = self._target
                if target is None:
                    continue      # nothing to say yet; the robot holds station
                try:
                    self.conn.sendall(struct.pack("!6i", *target))
                except OSError as e:
                    self._break("servo stream to %s broke: %s" % (self.ip, e))
                    return
                self.served += 1
                self.last_served = time.monotonic()

    def _break(self, message):
        """Record why the stream died, for the next caller to trip over.

        A stream that has stopped being read must not look like one that is
        working. Parking a target succeeds whether or not anything is
        listening, so without this the coordinator would go on happily
        computing poses for a robot that had left.
        """
        if self._running:
            self._fault = BackendError(message)

    # A second without the robot asking for anything. It asks every control
    # cycle when it is well, so this is a hundred missed requests — far past a
    # slow cycle, and reached whether the program stopped, never got as far as
    # its read loop, or was replaced by something else on the pendant.
    STALL_AFTER = 1.0
    # Before the first target is served, though, the clock has to be much
    # looser: this stream comes up before the second arm's does, and nothing is
    # parked for either of them until the coordinator's feed thread starts. A
    # second of that is ordinary, and calling it a stall would kill the loop on
    # its first cycle every time.
    STARTUP_GRACE = 5.0

    @property
    def stalled(self):
        """True once the robot has stopped asking for targets."""
        if self._fault is not None:
            return True
        if self.last_served:
            return time.monotonic() - self.last_served > self.STALL_AFTER
        return (bool(self._started_at)
                and time.monotonic() - self._started_at > self.STARTUP_GRACE)

    # -- targets -----------------------------------------------------------
    def _set_target(self, values):
        if self._fault is not None:
            raise self._fault
        vals = tuple(int(round(float(v) * _SCALE)) for v in values)
        with self._lock:
            if self.conn is None:
                raise BackendError("servo stream is not running")
            self._target = vals

    def servo_pose(self, pose_base):
        if self.control != "pose":
            raise BackendError("this stream was started for joint targets")
        self._set_target(pose_base)

    def servo_joints(self, q):
        if self.control != "joint":
            raise BackendError("this stream was started for Cartesian targets")
        self._set_target(q)

    def stop_motion(self):
        # Leaving the last target parked is the stop: the robot's servo thread
        # keeps servoing that one pose, which is an arm standing still under
        # control rather than an arm let go of.
        pass

    def shutdown(self):
        from .transport import send_script
        self._running = False
        if self._pump is not None:
            self._pump.join(timeout=1.0)
            self._pump = None
        with self._lock:
            for sock in (self.conn, self.srv):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            self.conn = self.srv = None
            self._target = None
        try:
            send_script(self.ip, "stopj(2.0)")     # replaces the loop program
        except OSError:
            pass


# ── nothing at all ────────────────────────────────────────────────────────
class SimBackend(ServoBackend):
    """Accepts every target and sends none of them.

    What makes `sim` mode worth having rather than a separate code path: the
    coordinated loop still runs its clock, still solves the closed chain, still
    checks its guards, still ramps a jog — and the only thing that differs is
    that the last step goes nowhere. A mode built by skipping the loop instead
    would test the arithmetic and leave the part that goes wrong untested.

    Keeps the last target so the caller can hand it to a viewer.
    """

    name = "sim"

    def __init__(self, ip, motion_cfg, control="pose", sink=None):
        self.ip = ip
        self.cfg = motion_cfg
        self.control = control
        # The simulated arm this drives, when there is one. Writing the target
        # back into it is what closes the loop: the next cycle reads the arm
        # and finds it where the last cycle put it, exactly as it would find a
        # real one. Without this the solver would re-seed from a stationary
        # arm every cycle and the model would never move.
        self.sink = sink
        self.last = None
        self.count = 0

    def start(self):
        return self

    def servo_pose(self, pose_base):
        self.last = list(map(float, pose_base))
        self.count += 1
        if self.sink is not None:
            self.sink.go_to_pose(self.last)

    def servo_joints(self, q):
        self.last = list(map(float, q))
        self.count += 1
        if self.sink is not None:
            self.sink.set_joints(self.last)

    def stop_motion(self):
        pass

    def shutdown(self):
        pass


def make_backend(ip, motion_cfg, control=None, drive_robots=True, sink=None):
    kind = str(motion_cfg.get("backend", "rtde")).lower()
    control = str(control or motion_cfg.get("control", "pose")).lower()
    if control not in ("pose", "joint"):
        raise BackendError("unknown motion control %r — use pose or joint"
                           % control)
    if not drive_robots:
        return SimBackend(ip, motion_cfg, control, sink)
    if kind == "rtde":
        return RtdeBackend(ip, motion_cfg, control)
    if kind == "urscript":
        return UrScriptBackend(ip, motion_cfg, control)
    raise BackendError("unknown motion backend %r — use rtde or urscript" % kind)
