"""
The primary interface, port 30001 — used here for the two things the
real-time feed does not carry.

The TCP offset configured on the pendant: the difference between the flange
(`tool0` in the URDF, what RViz draws) and the point the controller calls the
TCP (what every pose on the wire refers to). Knowing it is what lets the two
views be compared.

And the robot's own kinematic calibration. Every UR is measured at the factory
and the corrections stored in its controller, because the published DH table
describes the design and not the machine — a few millimetres and a fraction of
a degree apart, which is nothing for a single arm working off its own poses and
everything for two arms whose joint targets are computed here. Without these
numbers our forward kinematics is wrong by exactly the amount nobody can see.
"""

import socket
import struct

PRIMARY_PORT = 30001

ROBOT_STATE = 16
ROBOT_MODE_DATA = 0
CARTESIAN_INFO = 4
KINEMATICS_INFO = 5

# A KINEMATICS_INFO subpackage, byte for byte:
#
#   4   message size            5   uint32 checksum, one per joint    24
#   1   package type (5)      29   theta/a/d/alpha, six doubles each 192
#                            221   uint32 calibration status           4
#                                                             total  225
#
# The checksum being an array and not a single number is the whole reason this
# is spelled out. Reading it as one uint32 starts the doubles 20 bytes early,
# and 20 bytes into a run of little numbers is still a valid run of doubles —
# denormals and 1e130s, but nothing raises and nothing is obviously wrong until
# you notice the arm is 3 mm out. The size is checked below so that a firmware
# which lays this out differently is rejected rather than misread.
_KIN_CHECKSUMS = 6
_KIN_BODY = _KIN_CHECKSUMS * 4 + 24 * 8      # without the trailing status
_KIN_FIXED = 5 + _KIN_BODY


def read_geometry(ip, timeout=5.0, max_packets=60):
    """(tcp_pose, tcp_offset, calibration) from one connection.

    The calibration arrives in the first full robot state after connecting and
    not again, so it is read here rather than asked for separately — and all
    three describe the same arm at the same instant, which is the only way the
    tool offset and the DH corrections can be checked against each other.

    Any of the three may be None on firmware that does not send it.
    """
    pose = offset = calibration = None
    with socket.create_connection((ip, PRIMARY_PORT), timeout=timeout) as s:
        buf = b""
        for _ in range(max_packets):
            while len(buf) < 5:
                chunk = s.recv(4096)
                if not chunk:
                    return pose, offset, calibration
                buf += chunk
            length, ptype = struct.unpack("!iB", buf[:5])
            if length <= 0:
                return pose, offset, calibration
            while len(buf) < length:
                chunk = s.recv(4096)
                if not chunk:
                    return pose, offset, calibration
                buf += chunk
            pkt, buf = buf[:length], buf[length:]
            if ptype != ROBOT_STATE:
                continue

            off = 5
            while off + 5 <= len(pkt):
                sub_len, sub_type = struct.unpack("!iB", pkt[off:off + 5])
                if sub_len <= 0:
                    break
                body = pkt[off + 5: off + sub_len]
                if sub_type == CARTESIAN_INFO and pose is None:
                    n = len(body) // 8
                    vals = struct.unpack("!%dd" % n, body[:n * 8])
                    pose = list(vals[:6])
                    offset = list(vals[6:12]) if n >= 12 else None
                elif sub_type == KINEMATICS_INFO and calibration is None:
                    calibration = _parse_kinematics(sub_len, body)
                off += sub_len

            # The calibration is only ever in the first complete robot state,
            # which is the one that also carries the Cartesian info — so having
            # the pose means this packet was that one and there is nothing left
            # to wait for. Reading on until `max_packets` instead would spend
            # six seconds per arm at the primary interface's 10 Hz, on every
            # connect, for a field this firmware may simply not send.
            if pose is not None:
                break
    return pose, offset, calibration


def _parse_kinematics(sub_len, body):
    """The DH corrections, as four six-vectors, or None.

    These are *deltas* on the published table on every controller seen so far,
    but nothing here assumes that — whoever uses them checks the result against
    the pose the controller reports and keeps whichever table wins. A number
    read out of the wrong offset then costs nothing but a log line.
    """
    if sub_len < _KIN_FIXED:
        return None
    vals = struct.unpack("!%dI24d" % _KIN_CHECKSUMS, body[:_KIN_BODY])
    doubles = vals[_KIN_CHECKSUMS:]
    out = {
        "checksums": [int(v) for v in vals[:_KIN_CHECKSUMS]],
        "theta": list(doubles[0:6]),
        "a": list(doubles[6:12]),
        "d": list(doubles[12:18]),
        "alpha": list(doubles[18:24]),
        "status": None,
    }
    if sub_len >= _KIN_FIXED + 4:
        out["status"] = struct.unpack("!I", body[_KIN_BODY:_KIN_BODY + 4])[0]
    return out


def read_tcp_offset(ip, timeout=5.0, max_packets=40):
    """Return (tcp_pose, tcp_offset), both 6-vectors, or (None, None).

    tcp_pose is the same quantity the real-time feed reports; it comes back
    alongside because the two arriving together is what makes the offset
    interpretable.
    """
    pose, offset, _ = read_geometry(ip, timeout, max_packets)
    return pose, offset


def read_mode_flags(ip, timeout=5.0, max_packets=60):
    """{"program_running": bool, "control_mode": int} or None.

    control_mode 1 is freedrive. The real-time feed cannot answer this — its
    robot mode stays RUNNING throughout, and BACKDRIVE means something else
    entirely (brakes released for manual movement at startup). This is the
    flag that actually says whether the arm can be pushed by hand.
    """
    with socket.create_connection((ip, PRIMARY_PORT), timeout=timeout) as s:
        buf = b""
        for _ in range(max_packets):
            while len(buf) < 5:
                chunk = s.recv(4096)
                if not chunk:
                    return None
                buf += chunk
            length, ptype = struct.unpack("!iB", buf[:5])
            if length <= 0:
                return None
            while len(buf) < length:
                chunk = s.recv(4096)
                if not chunk:
                    return None
                buf += chunk
            pkt, buf = buf[:length], buf[length:]
            if ptype != ROBOT_STATE:
                continue
            off = 5
            while off + 5 <= len(pkt):
                sub_len, sub_type = struct.unpack("!iB", pkt[off:off + 5])
                if sub_len <= 0:
                    break
                if sub_type == ROBOT_MODE_DATA and sub_len >= 22:
                    fields = struct.unpack("!Q7B2B", pkt[off + 5:off + 22])
                    return {"program_running": bool(fields[6]),
                            "control_mode": int(fields[9])}
                off += sub_len
    return None
