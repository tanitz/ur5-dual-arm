"""
The robot's 125 Hz real-time feed, port 30003.

A packet is a 4-byte big-endian length followed by big-endian doubles.
Everything this project reads up to safety_mode (byte 812) has sat at the
same offset since PolyScope 3.0; the tail has grown with later releases, so
fields past that point are read only when the packet is long enough and come
back None otherwise rather than as a garbage number.
"""

import select
import socket
import struct
import threading
import time

RT_PORT = 30003


def _recv_packet(sock):
    buf = b""
    while len(buf) < 4:
        chunk = sock.recv(4 - len(buf))
        if not chunk:
            raise ConnectionError("robot closed the real-time connection")
        buf += chunk
    size = struct.unpack("!i", buf[:4])[0]
    while len(buf) < size:
        chunk = sock.recv(size - len(buf))
        if not chunk:
            raise ConnectionError("robot closed the real-time connection")
        buf += chunk
    return buf[:size]


def parse(pkt):
    size = struct.unpack("!i", pkt[:4])[0]

    def dbl(off, n=1):
        return struct.unpack("!%dd" % n, pkt[off: off + n * 8])

    def opt(off):
        return int(dbl(off)[0]) if size >= off + 8 else None

    return {
        "packet_size": size,
        "time": dbl(4)[0],
        "q_target": dbl(12, 6),
        "q_actual": dbl(252, 6),
        "qd_actual": dbl(300, 6),
        "tcp_pose": dbl(444, 6),      # actual TCP in the arm's own base frame
        "tcp_speed": dbl(492, 6),
        "tcp_force": dbl(540, 6),
        "digital_in_bits": int(dbl(684)[0]),
        "motor_temp": dbl(692, 6),
        "robot_mode": dbl(756)[0],
        "safety_mode": dbl(812)[0],
        # Three axes of accelerometer in the tool flange. Standing still it
        # reads one thing only — gravity — which makes it the one instrument
        # in the cell that knows which way is up without being told. Every
        # other statement about the world frame in this project is a number
        # somebody typed.
        "tool_accel": dbl(868, 3) if size >= 868 + 24 else None,
        "digital_out_bits": opt(1044),
    }


def read_state(ip, timeout=5.0):
    """One sample, one connection. For scripts and health checks."""
    with socket.create_connection((ip, RT_PORT), timeout=timeout) as s:
        return parse(_recv_packet(s))


class StateStream:
    """Persistent feed. `latest()` drains whatever the controller has queued
    and returns the newest sample, so a slow caller never falls behind.

    Safe to call from several threads, and it has to be: the GUI redraws from
    one thread while the coordinated servo loop checks the guards from
    another. Without the lock the two race on the same non-blocking socket —
    one sees select() say readable, the other drains it first, and the loser's
    recv raises BlockingIOError. That surfaces as an arm whose readout
    silently stops updating, which is a dangerous thing to be wrong about
    when two arms are holding one object.
    """

    def __init__(self, ip, timeout=5.0, stale_after=0.15):
        self.ip = ip
        self.timeout = timeout
        self.stale_after = stale_after
        self._lock = threading.Lock()
        self.reconnects = 0
        self.on_reconnect = None      # callable(ip, reason)
        self._open()

    def _open(self):
        self.sock = socket.create_connection((self.ip, RT_PORT),
                                             timeout=self.timeout)
        self.buf = b""
        self.state = parse(_recv_packet(self.sock))
        self.sock.setblocking(False)
        self._last_advance = time.monotonic()
        self._robot_clock = self.state["time"]

    @property
    def age(self):
        """Seconds since this feed last produced a genuinely new sample."""
        return time.monotonic() - self._last_advance

    def _drain(self):
        while select.select([self.sock], [], [], 0)[0]:
            try:
                chunk = self.sock.recv(65536)
            except BlockingIOError:
                break                # another reader got there first
            if not chunk:
                raise ConnectionError("robot closed the real-time connection")
            self.buf += chunk

        while len(self.buf) >= 4:
            size = struct.unpack("!i", self.buf[:4])[0]
            if size < 4 or len(self.buf) < size:
                break
            self.state = parse(self.buf[:size])
            self.buf = self.buf[size:]
            if self.state["time"] > self._robot_clock + 1e-9:
                self._robot_clock = self.state["time"]
                self._last_advance = time.monotonic()

    def latest(self):
        """Newest sample, reconnecting if the feed has gone quiet.

        A stalled feed is worse than a broken one. The socket stays open, no
        exception is raised, and every reader keeps getting the last sample it
        saw — so the screen shows an arm standing still that is not, and the
        coordinated loop computes targets from a pose that has moved on. It
        has been seen in the field: with another client already attached, the
        older controller stops serving this one mid-packet and never resumes.

        So staleness is treated as a fault. If the robot's own clock has not
        advanced for `stale_after`, the connection is thrown away and remade.
        The default of 150 ms is about nineteen missed samples at 125 Hz —
        loose enough that a scheduling hiccup does not trigger it, tight
        enough that a pose used by the coordinated loop is never far out of
        date.
        """
        with self._lock:
            try:
                self._drain()
            except (OSError, ConnectionError) as e:
                self._reconnect_locked("stream error: %s" % e)
                return self.state

            if self.age > self.stale_after:
                self._reconnect_locked(
                    "no new data for %.1f s (%d bytes stuck in the buffer)"
                    % (self.age, len(self.buf)))
            return self.state

    def _reconnect_locked(self, reason):
        try:
            self.sock.close()
        except OSError:
            pass
        try:
            self._open()
            self.reconnects += 1
            if self.on_reconnect is not None:
                self.on_reconnect(self.ip, reason)
        except OSError as e:
            # leave the old sample in place and try again on the next call
            self._last_advance = time.monotonic()
            raise ConnectionError("%s — and reconnect failed: %s" % (reason, e))

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
