"""The streaming URScript backend, with a fake robot on the other end.

No controller and no network beyond loopback. What this has to prove is the
one property the old push-based stream did not have: that a robot which gets
through its loop more slowly than we produce targets is handed the *newest*
one, not the next in a queue it can never catch up with. That failure had no
symptom on this side — every send succeeded — and on the robot it looked like
an arm that would not move.

    python3 tests/test_urscript_stream.py
"""

import os
import socket
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.robot import transport
from ur5dual.robot.backends import _SCALE, BackendError, UrScriptBackend

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


class FakeRobot:
    """The other end of the wire: asks for a target, reads six ints, repeats.

    Deliberately slower than the 125 Hz the coordinator runs at, because that
    is the case being tested — a real controller spends a control cycle on the
    socket read and another inside servoj.
    """

    def __init__(self, period=0.05):
        self.period = period
        self.sock = None
        self.received = []
        self.stop = threading.Event()
        self.thread = None
        self.script = ""

    def install(self):
        """Stand in for send_script: parse the port out and dial back."""
        real = transport.send_script

        def fake(ip, script, timeout=5.0):
            self.script = script
            port = int(script.split('socket_open("')[1]
                       .split('", ')[1].split(",")[0])
            self.thread = threading.Thread(target=self._run, args=(port,),
                                           daemon=True)
            self.thread.start()

        transport.send_script = fake
        return real

    def _run(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        self.sock.settimeout(1.0)
        while not self.stop.is_set():
            try:
                self.sock.sendall(struct.pack("!i", 1))
                buf = b""
                while len(buf) < 24:
                    chunk = self.sock.recv(24 - len(buf))
                    if not chunk:
                        return
                    buf += chunk
            except OSError:
                return
            self.received.append([v / _SCALE for v in struct.unpack("!6i", buf)])
            time.sleep(self.period)

    def close(self):
        self.stop.set()
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass


cfg = {"servo_rate_hz": 125.0, "servo_lookahead": 0.1, "servo_gain": 300.0,
       "servo_accel": 1.2, "servo_speed": 0.5}

print("the program that gets uploaded")
robot = FakeRobot()
original_send_script = robot.install()
backend = UrScriptBackend("127.0.0.1", cfg, control="joint")
backend.start(connect_timeout=5.0)

check("servoj runs in its own thread, so the socket cannot slow the arm",
      "thread ur5dual_servo():" in robot.script
      and "servoj(ur5dual_q" in robot.script)
# Globals and thread at the top level, as ur_rtde's own control program has
# them. A `global` inside a `def` is the one construct here that no shipped UR
# program uses, and a controller that rejects it fails after socket_open —
# stream up, panel happy, arms motionless.
check("nothing is wrapped in a def, so no global is declared inside one",
      "def " not in robot.script
      and robot.script.startswith("global ur5dual_q"))
check("the read timeout is in seconds, the units URScript counts in",
      "socket_read_binary_integer(6, \"srv\", 0.1)" in robot.script)
check("the robot asks for each target rather than being pushed them",
      "socket_send_int(1, \"srv\")" in robot.script)
check("joint control does no inverse kinematics on the controller",
      "get_inverse_kin" not in robot.script)
# The second arm's stream is built after this one and nothing is parked for
# either until the feed thread starts. Calling that a stall killed the loop on
# its first cycle.
check("a stream that has not been asked for anything yet is not a stall",
      not backend.stalled)

print("\nfeeding it faster than it can read")
# 125 Hz in, 20 Hz out: the ratio a real controller imposes, and the ratio the
# old push stream turned into an ever-growing backlog.
for i in range(250):
    backend.servo_joints([i / 1000.0] * 6)
    time.sleep(0.008)
time.sleep(0.15)

check("the robot got far fewer targets than were produced",
      0 < len(robot.received) < 100, "%d of 250" % len(robot.received))
newest = robot.received[-1][0] if robot.received else None
check("and the last one it was handed is the last one produced, not the "
      "next in a queue", newest is not None and abs(newest - 0.249) < 5e-4,
      "%.3f" % (newest if newest is not None else -1))
check("every target it did get was one that was actually commanded",
      all(abs(round(r[0] * 1000) / 1000.0 - r[0]) < 1e-6
          for r in robot.received))
check("the backend counted what it served", backend.served == len(robot.received),
      "%d" % backend.served)

print("\nwhen the robot goes away")
robot.close()
time.sleep(0.3)
raised = None
try:
    for _ in range(20):
        backend.servo_joints([0.0] * 6)
        time.sleep(0.02)
except BackendError as e:
    raised = str(e)
check("parking a target stops silently succeeding", raised is not None,
      (raised or "")[:60])
check("and the stream reports itself stalled", backend.stalled)

backend._running = False
try:
    backend.shutdown()
except Exception:                                          # noqa: BLE001
    pass
transport.send_script = original_send_script

print("\npose control still asks the controller to solve")
robot2 = FakeRobot(period=0.02)
robot2.install()
pose_backend = UrScriptBackend("127.0.0.1", cfg, control="pose")
pose_backend.start(connect_timeout=5.0)
check("the pose stream runs get_inverse_kin, seeded from the last answer",
      "get_inverse_kin(ur5dual_target, qnear=ur5dual_q)" in robot2.script)
pose_backend.servo_pose([0.4, 0.1, 0.3, 0.0, 3.14, 0.0])
time.sleep(0.2)
check("and a Cartesian target arrives intact",
      bool(robot2.received) and abs(robot2.received[-1][0] - 0.4) < 1e-4)
raised = None
try:
    pose_backend.servo_joints([0.0] * 6)
except BackendError as e:
    raised = str(e)
check("a stream started for poses refuses joint targets", raised is not None)
robot2.close()
pose_backend._running = False
try:
    pose_backend.shutdown()
except Exception:                                          # noqa: BLE001
    pass
transport.send_script = original_send_script

print("\n%d failed" % fail)
sys.exit(1 if fail else 0)
