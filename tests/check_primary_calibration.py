"""What each controller actually says about its own kinematics.

`check_chain_online.py` reports that arm B is 3.18 mm and 0.85 degrees away
from where its controller says its tool is. That is the size and the shape of a
factory calibration — a fraction of a degree at the wrist, multiplied into
millimetres by a 200 mm tool — so the numbers exist somewhere in that
controller. This finds out whether it sends them, and in what form.

Two things get printed and neither of them moves anything:

  the subpackage map    every block in the first robot state, by type and
                        length. KINEMATICS_INFO is type 5, and if it is not in
                        that list this firmware does not offer it at all.

  every reading of it   the published table, the corrections added to it, the
                        corrections taken whole — each scored against the pose
                        the controller reports for the joints it is sitting at.
                        Field order is included in the sweep because the four
                        six-vectors are only distinguishable by which one makes
                        the arithmetic come out right, and a candidate that
                        lands under a tenth of a millimetre out of forty-odd is
                        not landing there by luck.

Close the panel first — this reads port 30003, which these controllers serve to
one client at a time.

    python3 tests/check_primary_calibration.py
"""

import itertools
import os
import socket
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.geometry import kinematics as K
from ur5dual.geometry import ur_kinematics as UK
from ur5dual.config import ARM_IDS, CellConfig
from ur5dual.robot.transport import read_geometry, read_state
from ur5dual.robot.transport.primary import _KIN_FIXED, PRIMARY_PORT, ROBOT_STATE

SUBPACKAGE_NAMES = {
    0: "ROBOT_MODE_DATA", 1: "JOINT_DATA", 2: "TOOL_DATA",
    3: "MASTERBOARD_DATA", 4: "CARTESIAN_INFO", 5: "KINEMATICS_INFO",
    6: "CONFIGURATION_DATA", 7: "FORCE_MODE_DATA", 8: "ADDITIONAL_INFO",
    9: "NEEDED_FOR_CALIB", 10: "SAFETY_DATA", 11: "TOOL_COMM_INFO",
    12: "TOOL_MODE_INFO", 13: "SINGULARITY_INFO",
}

FIELDS = ("theta", "a", "d", "alpha")


def subpackage_map(ip, timeout=5.0, max_packets=40):
    """[(type, length)] of the first full robot state this controller sends."""
    with socket.create_connection((ip, PRIMARY_PORT), timeout=timeout) as s:
        buf = b""
        for _ in range(max_packets):
            while len(buf) < 5:
                chunk = s.recv(4096)
                if not chunk:
                    return []
                buf += chunk
            length, ptype = struct.unpack("!iB", buf[:5])
            if length <= 0:
                return []
            while len(buf) < length:
                chunk = s.recv(4096)
                if not chunk:
                    return []
                buf += chunk
            pkt, buf = buf[:length], buf[length:]
            if ptype != ROBOT_STATE:
                continue
            out, off = [], 5
            while off + 5 <= len(pkt):
                sub_len, sub_type = struct.unpack("!iB", pkt[off:off + 5])
                if sub_len <= 0:
                    break
                out.append((sub_type, sub_len))
                off += sub_len
            if len(out) > 3:            # the first state, not a later delta
                return out
    return []


def score(q, tcp_pose, tool, dh):
    d_p, d_r = UK.tcp_disagreement(q, tcp_pose, tool, dh)
    return d_p, d_r


config = CellConfig.load()
for arm_id in ARM_IDS:
    arm = config.arms[arm_id]
    if not arm.enabled:
        continue
    print("=" * 68)
    print("arm %s   %s" % (arm_id, arm.ip))

    blocks = subpackage_map(arm.ip)
    print("\n  what the first robot state contains")
    for sub_type, sub_len in blocks:
        print("    %-3d %-18s %4d bytes%s"
              % (sub_type, SUBPACKAGE_NAMES.get(sub_type, "?"), sub_len,
                 "   <-- the calibration" if sub_type == 5 else ""))
    if not any(t == 5 for t, _ in blocks):
        print("    no KINEMATICS_INFO: this firmware does not send its "
              "calibration,\n    so the published table is all there is and "
              "Cartesian targets are\n    the only honest way to drive this arm")

    _, tcp_offset, calibration = read_geometry(arm.ip)
    state = read_state(arm.ip)
    q = np.array(state["q_actual"], dtype=float)
    tool = None if tcp_offset is None else K.pose_to_mat(tcp_offset)

    if calibration is None:
        print("\n  nothing to interpret — %.2f mm out on the published table"
              % (score(q, state["tcp_pose"], tool, UK.UR5_DH)[0] * 1000))
        continue

    kin_len = dict(blocks).get(5)
    print("\n  the calibration as it arrived   %d bytes, expected %d%s"
          % (kin_len, _KIN_FIXED + 4,
             "" if kin_len == _KIN_FIXED + 4 else
             "   <-- LAYOUT DIFFERS, the numbers below are being misread"))
    print("    status    %s   (1 = calibrated)" % calibration.get("status"))
    print("    checksums %s" % calibration["checksums"])
    for name in FIELDS:
        vals = np.array(calibration[name], dtype=float)
        print("    %-6s %s" % (name, np.array2string(vals, precision=8,
                                                     suppress_small=False)))
    # No plausibility check on the magnitudes, deliberately. A calibrated arm
    # reports d2 and d3 as tens of metres — a DH chain cannot express a small
    # change in the angle between two parallel joints, so the fit escapes into
    # huge offsets that cancel. The forward kinematics is exact regardless, and
    # a size check here would reject the only correct table on offer.
    if calibration.get("status") == 0:
        print("    status 0: this controller holds no calibration, so the "
              "numbers above\n    are the published table echoed back — "
              "agreeing with it proves nothing")
    big = [f for f in FIELDS
           if float(np.max(np.abs(np.array(calibration[f])))) > 1.0]
    if big:
        print("    %s run to tens of metres, which is the parallel-joint DH\n"
              "    degeneracy and expected on a calibrated arm — the frames "
              "cancel" % "/".join(big))

    print("\n  every way of reading it, best first")
    results = [("published table, calibration ignored",
                score(q, state["tcp_pose"], tool, UK.UR5_DH))]
    for order in itertools.permutations(FIELDS):
        # the four six-vectors, relabelled — which of them is theta and which
        # is alpha is a convention, and the robot is the one that knows
        relabelled = {order[i]: calibration[FIELDS[i]] for i in range(4)}
        label = "as sent" if order == FIELDS else "read as " + "/".join(order)
        results.append(("corrections added (%s)" % label,
                        score(q, state["tcp_pose"], tool,
                              UK.with_corrections(relabelled))))
        results.append(("taken whole (%s)" % label,
                        score(q, state["tcp_pose"], tool,
                              UK.as_dh(relabelled))))
    results.sort(key=lambda row: row[1][0])
    for label, (d_p, d_r) in results[:6]:
        print("    %8.3f mm  %7.3f deg   %s"
              % (d_p * 1000, np.degrees(d_r), label))
    published = [r for r in results if r[0].startswith("published")][0]
    print("    %8.3f mm  %7.3f deg   %s   <-- what is in use now"
          % (published[1][0] * 1000, np.degrees(published[1][1]), published[0]))

    best_label, (best_p, _) = results[0]
    print()
    if best_p < 0.0005:
        print("  >>> '%s' reproduces this arm to %.3f mm.\n"
              "      That is the reading to lock in." % (best_label, best_p * 1000))
    elif best_p < published[1][0] * 0.5:
        print("  >>> '%s' more than halves the error (%.2f -> %.2f mm) but does\n"
              "      not close it, so something else is wrong too — most likely\n"
              "      the tool offset on the pendant."
              % (best_label, published[1][0] * 1000, best_p * 1000))
    else:
        print("  >>> no reading of these numbers explains the %.2f mm. The gap\n"
              "      is not this arm's DH calibration; check Installation ->\n"
              "      TCP on the pendant against the tool actually fitted."
              % (published[1][0] * 1000))

print("=" * 68)
print("Read-only: no program was uploaded and nothing was commanded.")
