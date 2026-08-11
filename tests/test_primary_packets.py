"""Reading the primary interface, against a controller made of bytes.

The bug this exists for: KINEMATICS_INFO carries one checksum *per joint*, and
reading it as a single number starts the doubles twenty bytes early. Nothing
raises. Twenty bytes into a run of doubles is still a run of doubles — 1e-315
here, 1e130 there — so the calibration was silently garbage, was silently
rejected for being garbage, and the only visible symptom was an arm 3 mm out
in a completely different part of the program.

So the packet is built here to the documented layout and read back. A parser
that drifts by a single byte cannot pass.

    python3 tests/test_primary_packets.py
"""

import os
import socket
import struct
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.robot.transport.primary import (
    CARTESIAN_INFO, KINEMATICS_INFO, ROBOT_STATE, read_geometry,
)

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


def subpackage(sub_type, body):
    return struct.pack("!iB", len(body) + 5, sub_type) + body


def robot_state(*subpackages):
    body = b"".join(subpackages)
    return struct.pack("!iB", len(body) + 5, ROBOT_STATE) + body


TCP_POSE = [0.4, -0.1, 0.3, 0.1, 3.0, 0.2]
TCP_OFFSET = [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]
CHECKSUMS = [11, 22, 33, 44, 55, 66]
THETA = [1e-3, -2e-3, 3e-3, -4e-3, 5e-3, -6e-3]
A = [1e-5, -2e-5, 3e-5, -4e-5, 5e-5, -6e-5]
D = [1e-4, -2e-4, 3e-4, -4e-4, 5e-4, -6e-4]
ALPHA = [1e-6, -2e-6, 3e-6, -4e-6, 5e-6, -6e-6]
STATUS = 1


def kinematics_body(checksums=CHECKSUMS, status=STATUS):
    return (struct.pack("!6I", *checksums)
            + struct.pack("!24d", *(THETA + A + D + ALPHA))
            + struct.pack("!I", status))


class FakeController:
    """Serves one robot state and then goes quiet, as a controller does."""

    def __init__(self, packet):
        self.packet = packet
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            conn, _ = self.srv.accept()
        except OSError:
            return
        try:
            conn.sendall(self.packet)
            conn.recv(1)                    # hold it open until the reader goes
        except OSError:
            pass
        finally:
            conn.close()

    def close(self):
        try:
            self.srv.close()
        except OSError:
            pass


def read_from(packet, monkeypatched_port):
    """read_geometry against loopback, with the port swapped in."""
    import ur5dual.robot.transport.primary as P
    was = P.PRIMARY_PORT
    P.PRIMARY_PORT = monkeypatched_port
    try:
        return read_geometry("127.0.0.1", timeout=3.0)
    finally:
        P.PRIMARY_PORT = was


print("a complete robot state")
full = robot_state(
    subpackage(CARTESIAN_INFO, struct.pack("!12d", *(TCP_POSE + TCP_OFFSET))),
    subpackage(KINEMATICS_INFO, kinematics_body()),
)
server = FakeController(full)
pose, offset, calibration = read_from(full, server.port)
server.close()

check("the TCP pose comes back", pose is not None
      and np.allclose(pose, TCP_POSE, atol=1e-12))
check("and the tool offset alongside it", offset is not None
      and np.allclose(offset, TCP_OFFSET, atol=1e-12))
check("the calibration subpackage is found", calibration is not None)

if calibration:
    # The size is the tell. 225 bytes is six checksums; 205 would be one, and
    # that difference is the whole bug.
    check("all six per-joint checksums are read",
          calibration["checksums"] == CHECKSUMS, str(calibration["checksums"]))
    check("theta lands on theta and not twenty bytes early",
          np.allclose(calibration["theta"], THETA, atol=1e-15),
          np.array2string(np.array(calibration["theta"]), precision=6))
    check("a, d and alpha follow in that order",
          np.allclose(calibration["a"], A, atol=1e-18)
          and np.allclose(calibration["d"], D, atol=1e-18)
          and np.allclose(calibration["alpha"], ALPHA, atol=1e-18))
    check("the calibration status is read from the end",
          calibration["status"] == STATUS, str(calibration["status"]))
    check("every value is the size a correction should be",
          max(abs(v) for f in ("theta", "a", "d", "alpha")
              for v in calibration[f]) < 0.5)

print("\na subpackage laid out differently is refused, not misread")
# One checksum instead of six: what this parser used to assume. Twenty bytes
# short, and every double in it would still unpack without complaint.
short = robot_state(
    subpackage(CARTESIAN_INFO, struct.pack("!12d", *(TCP_POSE + TCP_OFFSET))),
    subpackage(KINEMATICS_INFO,
               struct.pack("!I", CHECKSUMS[0])
               + struct.pack("!24d", *(THETA + A + D + ALPHA))
               + struct.pack("!I", STATUS)),
)
server = FakeController(short)
_, _, calibration = read_from(short, server.port)
server.close()
check("a short KINEMATICS_INFO yields nothing rather than nonsense",
      calibration is None)

print("\nfirmware that sends no calibration at all")
bare = robot_state(
    subpackage(CARTESIAN_INFO, struct.pack("!12d", *(TCP_POSE + TCP_OFFSET))))
server = FakeController(bare)
pose, offset, calibration = read_from(bare, server.port)
server.close()
check("the tool offset still comes back", pose is not None and offset is not None)
check("and the calibration is simply absent", calibration is None)

print("\nsubpackages this project does not read are stepped over")
padded = robot_state(
    subpackage(0, b"\x00" * 42),
    subpackage(1, b"\x00" * 246),
    subpackage(CARTESIAN_INFO, struct.pack("!12d", *(TCP_POSE + TCP_OFFSET))),
    subpackage(KINEMATICS_INFO, kinematics_body()),
    subpackage(9, b"\x00" * 48),
)
server = FakeController(padded)
pose, _, calibration = read_from(padded, server.port)
server.close()
check("a real robot state's worth of blocks is walked correctly",
      pose is not None and calibration is not None
      and np.allclose(calibration["theta"], THETA, atol=1e-15))

print("\n%d failed" % fail)
sys.exit(1 if fail else 0)
