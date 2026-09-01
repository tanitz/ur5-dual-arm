"""The wire, for one link.

Two classes behind one small interface, because the executor's RECV loop must
not care which it has: `poll()` is one non-blocking look for whatever the step
is waiting for, and the loop around it — pause, stop, timeout — belongs to the
executor, where every other wait in this cell already lives. A blocking
`recv()` here would be a wait the STOP button cannot reach.

Nothing reconnects on its own. A machine that dropped the connection is a fact
the operator needs told, and a link that silently redialled would turn a
cable pulled out of a socket into a program that runs half a cycle late.
"""

import select
import socket

from .links import (
    is_modbus, item_matches, message_text, payload_bytes, unescape,
)

# What one recv asks for. Bigger than any handshake and smaller than anything
# that would make a slow machine's reply arrive in pieces that matter.
CHUNK = 4096
# The longest a poll may block for. Not zero: a bare non-blocking poll spun
# by the executor's 20 ms loop wakes the scheduler for nothing, and 5 ms is
# below what anybody can measure on a handshake.
POLL_WAIT = 0.005


class LinkError(RuntimeError):
    """A link that is not there, or that went away mid-program."""


class RawLink:
    """A TCP or UDP socket carrying strings or byte frames.

    Messages are split on the link's terminator, which is what makes `MC1
    DONE` a thing that can be waited for at all — the alternative is comparing
    against whatever happened to be in the kernel buffer at the moment the
    step looked, which is a different string on a busy line every time.

    A link with no terminator hands back whatever arrived in one read, which
    is the honest answer for a machine that frames its own packets.
    """

    def __init__(self, link):
        self.link = dict(link)
        self.sock = None
        self._buffer = b""
        self._datagrams = []

    # -- the connection ----------------------------------------------------
    @property
    def connected(self):
        return self.sock is not None

    def open(self):
        host = (self.link.get("host") or "").strip()
        port = int(self.link.get("port", 0))
        timeout = float(self.link.get("timeout", 3.0)) or 3.0
        try:
            if self.link.get("kind") == "udp":
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(timeout)
                # Connected UDP: the cell hears this machine and nothing else,
                # and a port nobody is listening on comes back as a refusal
                # rather than as silence that looks like a slow reply.
                sock.connect((host, port))
            else:
                sock = socket.create_connection((host, port), timeout)
            sock.setblocking(False)
        except OSError as exc:
            raise LinkError("%s: cannot reach %s:%d — %s"
                            % (self.link.get("name"), host, port, exc))
        self.sock = sock
        self._buffer = b""
        self._datagrams = []
        return self

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self._buffer = b""
        self._datagrams = []

    # -- sending -----------------------------------------------------------
    def send(self, payload):
        if self.sock is None:
            raise LinkError("%s is not open" % self.link.get("name"))
        try:
            self.sock.sendall(payload)
        except OSError as exc:
            raise LinkError("%s: sending failed — %s"
                            % (self.link.get("name"), exc))
        return len(payload)

    def send_item(self, item):
        return self.send(payload_bytes(self.link, item))

    def send_text(self, text):
        """An unnamed payload, for the tab's Test button and nothing else.

        Program lines send named items on purpose. This exists so that setting
        a link up can be proved from the tab that set it up, without first
        writing a program to find out whether the address was typed right.
        """
        encoding = self.link.get("encoding") or "utf-8"
        body = unescape(str(text)) + unescape(self.link.get("terminator", ""))
        return self.send(body.encode(encoding, "replace"))

    # -- receiving ---------------------------------------------------------
    def _pump(self):
        """Take whatever has arrived, without waiting for more."""
        if self.sock is None:
            raise LinkError("%s is not open" % self.link.get("name"))
        ready, _, _ = select.select([self.sock], [], [], POLL_WAIT)
        if not ready:
            return False
        try:
            chunk = self.sock.recv(CHUNK)
        except BlockingIOError:
            return False
        except OSError as exc:
            raise LinkError("%s: the connection failed — %s"
                            % (self.link.get("name"), exc))
        if not chunk:
            # TCP only, and it means the far end hung up. Reported rather than
            # retried: a program waiting on a machine that has closed the
            # socket will wait forever otherwise.
            self.close()
            raise LinkError("%s closed the connection" % self.link.get("name"))
        if self.link.get("kind") == "udp":
            self._datagrams.append(chunk)
        else:
            self._buffer += chunk
        return True

    def poll(self):
        """The next whole message, or None if nothing has arrived yet."""
        self._pump()
        if self.link.get("kind") == "udp":
            return self._datagrams.pop(0) if self._datagrams else None
        end = unescape(self.link.get("terminator", "")).encode(
            self.link.get("encoding") or "utf-8", "replace")
        if not end:
            if not self._buffer:
                return None
            whole, self._buffer = self._buffer, b""
            return whole
        cut = self._buffer.find(end)
        if cut < 0:
            return None
        message, self._buffer = self._buffer[:cut], self._buffer[cut + len(end):]
        return message

    def drain(self):
        """Throw away anything already waiting.

        Called before a SEND that expects an answer. A reply left over from
        the last cycle satisfies the next RECV the instant it looks, and a
        handshake that passes on stale data is worse than one that times out:
        the arm moves, and nothing on screen says why.
        """
        while self._pump():
            pass
        self._buffer = b""
        self._datagrams = []


class ModbusLink:
    """A Modbus TCP client, over pymodbus.

    Modbus is its own protocol rather than a payload format, which is why it
    is a class and not a branch in `RawLink`: there are no messages to split
    on a terminator, only registers to write and read back.

    pymodbus is imported where it is used rather than at the top of the
    module, so that a cell with no Modbus device — or an install without the
    library — still opens its panel and still talks to its TCP machines.
    """

    _READ = {"coil": "read_coils", "discrete": "read_discrete_inputs",
             "holding": "read_holding_registers", "input": "read_input_registers"}
    _WRITE = {"coil": "write_coil", "holding": "write_register"}

    def __init__(self, link):
        self.link = dict(link)
        self.client = None

    @property
    def connected(self):
        return self.client is not None

    def open(self):
        host = (self.link.get("host") or "").strip()
        port = int(self.link.get("port", 502))
        timeout = float(self.link.get("timeout", 3.0)) or 3.0
        try:
            from pymodbus.client import ModbusTcpClient as Client
        except ImportError:
            raise LinkError("%s needs pymodbus — pip install pymodbus"
                            % self.link.get("name"))
        client = Client(host, port=port, timeout=timeout)
        if not client.connect():
            raise LinkError("%s: cannot reach %s:%d"
                            % (self.link.get("name"), host, port))
        self.client = client
        return self

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:           # pymodbus raises its own on a dead socket
                pass
        self.client = None

    def _device(self):
        return int(self.link.get("unit", 1))

    def read(self, item):
        """One register or coil, as a number."""
        if self.client is None:
            raise LinkError("%s is not open" % self.link.get("name"))
        area = item.get("area", "holding")
        call = getattr(self.client, self._READ[area])
        try:
            reply = call(int(item.get("register", 0)), count=1,
                         device_id=self._device())
        except Exception as exc:
            raise LinkError("%s: reading %s %s failed — %s"
                            % (self.link.get("name"), area,
                               item.get("register"), exc))
        if reply.isError():
            raise LinkError("%s: %s %s answered with an error — %s"
                            % (self.link.get("name"), area,
                               item.get("register"), reply))
        if area in ("coil", "discrete"):
            return int(bool(reply.bits[0]))
        return int(reply.registers[0])

    def write(self, item, value=None):
        """Set a coil or a holding register to what the item says."""
        if self.client is None:
            raise LinkError("%s is not open" % self.link.get("name"))
        area = item.get("area", "holding")
        if area not in self._WRITE:
            raise LinkError("%s: %s is read-only in Modbus"
                            % (self.link.get("name"), area))
        number = int(item.get("value", 0) if value is None else value)
        call = getattr(self.client, self._WRITE[area])
        payload = bool(number) if area == "coil" else number
        try:
            reply = call(int(item.get("register", 0)), payload,
                         device_id=self._device())
        except Exception as exc:
            raise LinkError("%s: writing %s %s failed — %s"
                            % (self.link.get("name"), area,
                               item.get("register"), exc))
        if reply.isError():
            raise LinkError("%s: %s %s refused the write — %s"
                            % (self.link.get("name"), area,
                               item.get("register"), reply))
        return number

    def poll(self, item):
        """Has this register reached what the item is waiting for?

        Returns (matched, value). A Modbus device is asked rather than heard
        from, so `poll` is a read and the executor's loop sets the rate.
        """
        value = self.read(item)
        return value == int(item.get("value", 0)), value


def open_link(link):
    """The right client for a link record, already connected."""
    client = ModbusLink(link) if is_modbus(link) else RawLink(link)
    return client.open()


def raw_matches(link, item, raw):
    """Does a message off a raw link satisfy this item? Also returns the text."""
    text = message_text(link, raw)
    return item_matches(link, item, text), text
