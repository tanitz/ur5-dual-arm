"""
The local channel that lets RViz draw the cell without becoming a second
client of either robot.

The control panel already owns the real-time feeds in REAL mode.  Forwarding
the joint angles it has read over loopback lets RViz follow the real arms
without also opening port 30003, which older controllers cannot serve to two
clients reliably.  In SIM the same channel carries the computed joint angles.

A datagram per servo cycle carries them across. UDP rather than a topic or a
socket stream, for three reasons that all matter here:

  the panel stays out of ROS   importing rclpy into a PyQt process to publish
                               six floats would drag a second executor and a
                               second spin loop into the GUI thread

  a dropped frame is nothing   this is a viewer. Losing one cycle out of 125
                               is invisible, and waiting for delivery of a
                               pose that is already stale is worse than
                               skipping it

  neither end blocks           the panel sends whether or not RViz is running,
                               and the viewer starts whether or not the panel
                               is, in any order, with no connect step between
                               them

The receiver treats silence as "no control panel is running" and lets the
viewer fall back to the live robots.  A longer lease is used only during the
SIM-to-REAL connection handover, when the GUI can be blocked waiting for a
controller and cannot send its normal heartbeat.
"""

import json
import socket
import time

# Loopback only. These joint angles drive a picture, and a picture that can be
# driven from another machine is a picture that can be made to lie about where
# two robots are standing.
SIM_VIEW_HOST = "127.0.0.1"
SIM_VIEW_PORT = 30099

# Past this with no datagram, the viewer stops believing the simulation and
# goes back to the arms. Four servo cycles at 125 Hz would be too tight for a
# GUI that pauses to repaint; half a second is short enough that letting go of
# a jog does not leave the model drifting on stale frames.
SIM_VIEW_STALE = 0.5

ARM_JOINT_COUNT = 6


class SimViewSender:
    """Sends commanded joint angles to whatever is drawing the cell.

    Fire and forget by construction: `send` never raises and never blocks, so
    a viewer that is not running, or a loopback that is momentarily full,
    cannot disturb the loop that is driving two robots.
    """

    def __init__(self, host=SIM_VIEW_HOST, port=SIM_VIEW_PORT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sent = 0
        self.failures = 0

    def send(self, joints_by_arm, moving=True, mode="sim", lease=0.0):
        payload = {a: [float(v) for v in q]
                   for a, q in joints_by_arm.items()
                   if q is not None and len(q) == ARM_JOINT_COUNT}
        if not payload and mode not in ("sim", "real"):
            return False
        try:
            # Stamped by the sender, not by whoever reads it. Freshness has to
            # mean "this is what the cell is doing now", and a receiver that
            # timed its own reads would call a backlog fresh — draining two
            # seconds of queued frames after the panel had already closed, and
            # drawing every one of them as current.
            self.sock.sendto(
                json.dumps({"t": time.time(), "joints": payload,
                            "moving": bool(moving), "mode": mode,
                            "lease": max(0.0, float(lease))}).encode(),
                self.addr)
            self.sent += 1
            return True
        except OSError:
            self.failures += 1
            return False

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class SimViewReceiver:
    """Reads those datagrams, keeping only the newest.

    Drains the socket every time it is asked rather than one packet per call:
    a viewer redrawing at 50 Hz is being sent 125 packets a second, and
    consuming them one at a time would draw an ever-older pose while the
    backlog grew.
    """

    def __init__(self, host=SIM_VIEW_HOST, port=SIM_VIEW_PORT,
                 stale_after=SIM_VIEW_STALE):
        self.stale_after = stale_after
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # The default receive buffer is left alone deliberately. Shrinking it
        # to "only keep the newest" is the wrong instinct: a full UDP receive
        # buffer drops the datagrams still arriving, not the ones already in
        # it, so a small buffer plus a consumer that pauses to redraw loses
        # precisely the frames that matter and keeps the stale ones. Backlog
        # is handled where it belongs — every poll drains the queue, and the
        # sender's timestamp decides what counts as current.
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.joints = {}
        self.received = 0
        self._sent_at = 0.0       # the sender's clock, not ours
        self._lease_until = 0.0
        self.mode = None

    def poll(self):
        """Newest joint angles, or None if nothing recent has arrived."""
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            try:
                message = json.loads(data.decode())
                joints = {a: list(q) for a, q in message["joints"].items()
                          if len(q) == ARM_JOINT_COUNT}
                sent_at = float(message.get("t", 0.0))
                mode = message.get("mode", "sim")
                lease = max(0.0, float(message.get("lease", 0.0)))
                if mode not in ("sim", "real"):
                    continue
            except (ValueError, KeyError, TypeError, AttributeError):
                continue          # a malformed frame is a dropped frame
            if sent_at >= self._sent_at:
                # out-of-order delivery is possible on any datagram socket, and
                # an older frame overwriting a newer one would show the cell
                # stepping backwards
                self.joints = joints
                self.mode = mode
                self._lease_until = max(self._lease_until, sent_at + lease)
                self.received += 1
                self._sent_at = sent_at
        return self.joints if self.fresh else None

    @property
    def fresh(self):
        return self.active and bool(self.joints)

    @property
    def active(self):
        """A control panel owns the robot feeds, even before its first sample."""
        return self.mode in ("sim", "real") and \
            time.time() < max(self._sent_at + self.stale_after,
                              self._lease_until)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
