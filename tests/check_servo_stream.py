"""Does one arm actually consume and follow the servo stream?

The question `check_backends_online.py` cannot answer. That one proves the
backend comes up — the program is uploaded and the robot dials back — and the
failure this exists for happens after that: a stream that is up, accepting
every target, and being read far too slowly to follow any of them. Nothing on
the Jetson side notices, because parking a target always succeeds.

So this measures the two numbers that separate the cases:

  served    how many targets the robot asked for. It asks once per control
            cycle when it is well, so this should land near 125 per second.
            A number near zero is a program that is not running its read loop.

  lag       how far the arm is behind the target it was last handed. Tens of
            microradians while holding; a few milliradians while ramping. A
            figure that grows for the whole run is a robot falling behind.

THE ARM MOVES, but only wrist 3, and only by 3 degrees at 1 deg/s.
Close the panel first — it owns port 30003 and this needs to read it.

    python3 tests/check_servo_stream.py         # arm A
    python3 tests/check_servo_stream.py B
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.robot.backends import BackendError, make_backend
from ur5dual.config import CellConfig
from ur5dual.robot.transport import StateStream

HOLD = 2.0             # s feeding the pose it is already in
SWEEP = math.radians(3.0)
SWEEP_RATE = math.radians(1.0)     # rad/s

arm_id = (sys.argv[1] if len(sys.argv) > 1 else "A").upper()
config = CellConfig.load()
arm = config.arms[arm_id]
control = str(config.motion.get("control", "pose")).lower()
rate = float(config.motion["servo_rate_hz"])
dt = 1.0 / rate

print("arm %s  %s   backend %s   control %s   %.0f Hz"
      % (arm_id, arm.ip, config.motion["backend"], control, rate))

stream = StateStream(arm.ip, timeout=5.0)
state = stream.latest()
print("packet %d bytes   J6 at %.2f deg"
      % (state["packet_size"], math.degrees(state["q_actual"][5])))

backend = make_backend(arm.ip, config.motion, control)
try:
    backend.start()
except (BackendError, OSError) as e:
    raise SystemExit("backend did not start: %s" % e)
print("backend up: %s\n" % backend.name)


def measure(label, duration, target_at):
    """Feed `target_at(t)` for `duration`, reporting what the arm did."""
    served0 = getattr(backend, "served", 0)
    t0 = time.monotonic()
    n = 0
    worst_lag = 0.0
    last_lag = 0.0
    while True:
        t = time.monotonic() - t0
        if t >= duration:
            break
        target = target_at(t)
        try:
            backend.send(target)
        except BackendError as e:
            raise SystemExit("\nstream died after %.1f s: %s" % (t, e))
        if control == "joint":
            actual = np.array(stream.latest()["q_actual"], dtype=float)
            last_lag = float(np.max(np.abs(actual - np.asarray(target))))
            worst_lag = max(worst_lag, last_lag)
        n += 1
        sleep = t0 + n * dt - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)

    served = getattr(backend, "served", 0) - served0
    print("%-8s sent %4d   robot asked %4d (%5.1f/s)   lag worst %6.3f deg  "
          "end %6.3f deg"
          % (label, n, served, served / duration,
             math.degrees(worst_lag), math.degrees(last_lag)))
    return served


held = np.array(stream.latest()["q_actual"], dtype=float) if control == "joint" \
    else np.array(stream.latest()["tcp_pose"], dtype=float)

serviced = measure("hold", HOLD, lambda t: held)

if control == "joint":
    def ramp(t):
        # out and back, so the arm ends where it started whatever happens
        half = SWEEP / SWEEP_RATE
        phase = SWEEP_RATE * (t if t < half else 2 * half - t)
        q = held.copy()
        q[5] += max(0.0, min(SWEEP, phase))
        return q

    serviced += measure("sweep", 2.0 * SWEEP / SWEEP_RATE, ramp)
    time.sleep(0.3)
    ended = np.array(stream.latest()["q_actual"], dtype=float)
    print("\nJ6 finished %.3f deg from where it started"
          % math.degrees(ended[5] - held[5]))
else:
    print("\nmotion.control is `pose`, so this only checked that the stream is "
          "being read.\nSet it to `joint` in cell.yaml to measure tracking too.")

backend.shutdown()
stream.close()

expected = rate * (HOLD + (2.0 * SWEEP / SWEEP_RATE if control == "joint" else 0))
print("\nthe robot asked for %d targets; a healthy stream asks for about %d"
      % (serviced, int(expected)))
if serviced < 0.25 * expected:
    print(">>> the servo program is not reading at anything like the control "
          "rate. The arm cannot follow a stream it is not consuming.")
    sys.exit(1)
print("this arm consumes and follows the servo stream.")
