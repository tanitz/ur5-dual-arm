"""
Talking to the machine on the other end of the cable.

Nothing here needs a machine. A socket on 127.0.0.1 answers the same calls a
labeller or a PLC would, so every path except the Modbus wire itself runs
exactly as it will on the cell -- which is the point, because the alternative
is finding out with an arm already reaching into a fixture.

What is actually being checked is the promise the tabs make to a program: a
line carries two *names*, and everything that could go wrong with the address
behind them is caught either on the tab or by the checker, before anything
moves.
"""

import os
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.cell import Cell
from ur5dual.config import CellConfig
from ur5dual.comms import (
    LinkError, LinkLibrary, LinkService, escape, make_item, make_link, unescape,
)
from ur5dual.comms.links import (
    check_item, check_link, item_matches, payload_bytes,
)
from ur5dual.program.executor import Executor, ProgramError
from ur5dual.program.steps import PointLibrary, Program, Step

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


def until(predicate, timeout=3.0):
    """Wait for something a thread is about to do, without a bare sleep."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _refused(action, fragment):
    """Did this fail, and say the thing that makes the failure actionable?"""
    try:
        action()
    except LinkError as exc:
        return fragment in str(exc)
    return False


class Machine:
    """The far end, for the length of one test.

    Speaks TCP or UDP and does nothing clever: it keeps what it was sent and
    says what it is told to say. Clever is what the cell is being tested for.
    """

    def __init__(self, kind="tcp"):
        self.kind = kind
        self.heard = []
        self.conn = None
        self.peer = None
        self._stop = threading.Event()
        family = socket.SOCK_STREAM if kind == "tcp" else socket.SOCK_DGRAM
        self.sock = socket.socket(socket.AF_INET, family)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        if kind == "tcp":
            self.sock.listen(1)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        self.sock.settimeout(0.1)
        while not self._stop.is_set():
            try:
                if self.kind == "tcp":
                    if self.conn is None:
                        self.conn, _ = self.sock.accept()
                        self.conn.settimeout(0.1)
                        continue
                    data = self.conn.recv(4096)
                    if not data:
                        return
                    self.heard.append(data)
                else:
                    data, self.peer = self.sock.recvfrom(4096)
                    self.heard.append(data)
            except (socket.timeout, OSError):
                continue

    def say(self, payload):
        """Push a message at whoever is connected."""
        if isinstance(payload, str):
            payload = payload.encode()
        if self.kind == "tcp":
            until(lambda: self.conn is not None)
            self.conn.sendall(payload)
        else:
            until(lambda: self.peer is not None)
            self.sock.sendto(payload, self.peer)

    def stop(self):
        self._stop.set()
        for sock in (self.conn, self.sock):
            try:
                if sock:
                    sock.close()
            except OSError:
                pass


def library(machine, name="MC1", kind=None, **over):
    link = make_link(name, kind or machine.kind, "127.0.0.1", machine.port,
                     timeout=1.0,
                     data=[make_item("START", "send", "ST"),
                           make_item("DONE", "recv", "DN"),
                           make_item("COUNT", "recv", "", match="prefix")],
                     **over)
    return LinkLibrary([link])


# ── how a link is written down ────────────────────────────────────────────
print("\nthe record")

check("a terminator survives being typed and read back",
      unescape(escape("\r\n")) == "\r\n", repr(escape("\r\n")))
check("and a payload with a non-ASCII character is not mangled by it",
      unescape("มอเตอร์\\r\\n") == "มอเตอร์\r\n")

link = make_link("MC1", "tcp", "10.1.68.200", 2000)
check("a string payload is sent with the link's terminator on it",
      payload_bytes(link, make_item("START", "send", "ST")) == b"ST\r\n",
      repr(payload_bytes(link, make_item("START", "send", "ST"))))
# Sending has to mean exactly what waiting matches. A payload taken literally
# while the thing waiting for it read `\t` as a tab is a handshake that can
# never complete, and it looks identical on screen to one that can.
tabbed = make_item("SPLIT", "both", "A\\tB")
check("an escape in a payload is one character on the wire",
      payload_bytes(link, tabbed) == b"A\tB\r\n",
      repr(payload_bytes(link, tabbed)))
check("and the same character when it is what is being waited for",
      item_matches(link, tabbed, "A\tB")
      and not item_matches(link, tabbed, "A\\tB"))
framed = make_link("F1", "tcp", "10.1.68.200", 2000, format="hex")
check("and a byte frame is sent as it stands, terminator left off",
      payload_bytes(framed, make_item("GO", "send", "02 41 03")) == b"\x02A\x03",
      repr(payload_bytes(framed, make_item("GO", "send", "02 41 03"))))

check("an exact match is the whole message",
      item_matches(link, make_item("D", "recv", "DN"), "DN")
      and not item_matches(link, make_item("D", "recv", "DN"), "DNX"))
check("prefix and contains are what a machine appending a serial needs",
      item_matches(link, make_item("D", "recv", "DN", match="prefix"), "DN0042")
      and item_matches(link, make_item("D", "recv", "42", match="contains"),
                       "DN0042"))

# ── what the tab refuses to accept ────────────────────────────────────────
print("\nthe tab, before anything is dialled")

check("a port that is not a port is caught",
      any("not a port" in p for p in
          check_link(make_link("X", "tcp", "1.2.3.4", 99999))),
      str(check_link(make_link("X", "tcp", "1.2.3.4", 99999))))
check("and an address nobody typed",
      any("no address" in p for p in check_link(make_link("X", "tcp", "", 2000))))

modbus = make_link("PLC1", "modbus", "10.1.68.210", 502)
check("a Modbus input register may not be written",
      any("read-only" in p for p in
          check_item(modbus, make_item("R", "send", 1, area="input",
                                       register=5))),
      str(check_item(modbus, make_item("R", "send", 1, area="input",
                                       register=5))))
check("a coil holds 0 or 1 and says so",
      any("coil is 0 or 1" in p for p in
          check_item(modbus, make_item("R", "send", 7, area="coil"))))
check("and odd hex digits are a typo rather than a frame",
      any("even number" in p for p in
          check_item(framed, make_item("G", "send", "02 4"))))

two = LinkLibrary([make_link("MC1", "tcp", "1.2.3.4", 2000),
                   make_link("MC1", "udp", "1.2.3.4", 2001)])
check("two machines with one name is refused — a line naming it is ambiguous",
      any("more than one link called" in p for p in two.check()),
      str(two.check()))

# ── the library, through cell.yaml and back ───────────────────────────────
print("\nthrough the file")

with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "cell.yaml")
    cfg = CellConfig.load()
    cfg.path = path
    lib = cfg.link_library()
    lib.add(make_link("MC1", "tcp", "10.1.68.200", 2000,
                      data=[make_item("START", "send", "ST"),
                            make_item("DONE", "recv", "DN")]))
    lib.add(make_link("PLC1", "modbus", "10.1.68.210", 502, unit=3,
                      data=[make_item("RUN", "send", 1, area="coil",
                                      register=10)]))
    cfg.set_link_library(lib)
    cfg.save_comms(path)

    back = CellConfig.load(path).link_library()
    check("both machines come back with their names", back.names() == ["MC1", "PLC1"],
          str(back.names()))
    check("and the terminator comes back as bytes, not as backslash-r",
          unescape(back.get("MC1")["terminator"]) == "\r\n")
    check("the Modbus device id survives", back.get("PLC1")["unit"] == 3)
    check("and so does what each carries",
          [i["name"] for i in back.items_for("MC1")] == ["START", "DONE"])
    check("a send picker offers only what may be sent",
          [i["name"] for i in back.items_for("MC1", "send")] == ["START"])
    check("and a wait picker only what may be waited for",
          [i["name"] for i in back.items_for("MC1", "recv")] == ["DONE"])

# ── the wire ──────────────────────────────────────────────────────────────
print("\nTCP, against a machine that is actually there")

machine = Machine("tcp")
try:
    service = LinkService(library(machine), log=lambda _t: None)
    check("a link that answers says so", "answering" in service.probe("MC1"),
          service.probe("MC1"))

    service.send("MC1", "START")
    check("a named datum arrives with its terminator",
          until(lambda: machine.heard == [b"ST\r\n"]), str(machine.heard))

    got, _value = service.attempt("MC1", "DONE")
    check("nothing has come back yet, and that is not a failure", got is False)

    machine.say("DN\r\n")
    check("and once it does, the wait is over",
          until(lambda: service.attempt("MC1", "DONE")[0]))

    machine.say("DN\r\nDN\r\n")
    check("two messages in one write are two messages",
          until(lambda: service.attempt("MC1", "DONE")[0])
          and until(lambda: service.attempt("MC1", "DONE")[0]))

    machine.say("HALF")
    check("half a message is not a message",
          service.attempt("MC1", "DONE")[0] is False)
    machine.say("WAY\r\n")
    got, text = service.attempt("MC1", "")
    check("and the other half completes it", got and text == "HALFWAY",
          repr(text))

    machine.say("DN\r\n")
    until(lambda: bool(machine.heard))
    service.send("MC1", "START")
    check("a reply left over from last cycle is thrown away before a send",
          service.attempt("MC1", "DONE")[0] is False)

    check("a datum set up to be received may not be sent",
          _refused(lambda: service.send("MC1", "DONE"), "recv only"))
    check("and one nobody named is named as missing",
          _refused(lambda: service.send("MC1", "NOPE"), "no data item"))
finally:
    machine.stop()

print("\nUDP, which is a different protocol and not a Modbus one")

machine = Machine("udp")
try:
    service = LinkService(library(machine), log=lambda _t: None)
    service.send("MC1", "START")
    check("a datagram carries the same payload a stream would",
          until(lambda: machine.heard == [b"ST\r\n"]), str(machine.heard))
    machine.say("DN\r\n")
    check("and one datagram is one message, terminator or not",
          until(lambda: service.attempt("MC1", "DONE")[0]))
finally:
    machine.stop()

print("\nwhen the machine goes away")

machine = Machine("tcp")
service = LinkService(library(machine), log=lambda _t: None)
service.send("MC1", "START")
until(lambda: bool(machine.heard))
machine.stop()
check("a machine that hangs up is reported, not waited on forever",
      _refused(lambda: [service.attempt("MC1", "DONE") for _ in range(50)],
               "closed the connection"))

dead = LinkService(LinkLibrary([make_link("MC1", "tcp", "127.0.0.1", 1,
                                          timeout=0.5)]),
                   log=lambda _t: None)
check("and one that was never there names the address it tried",
      "127.0.0.1:1" in dead.probe("MC1"), dead.probe("MC1"))

print("\ntaking up an edit")

machine = Machine("tcp")
try:
    lib = library(machine)
    service = LinkService(lib, log=lambda _t: None)
    service.send("MC1", "START")
    check("the socket is open", service.is_open("MC1"))

    lib.get("MC1").setdefault("data", []).append(make_item("EXTRA", "recv", "X"))
    service.reload(lib)
    check("naming another datum does not hang up on the machine",
          service.is_open("MC1"))

    lib.get("MC1")["port"] = machine.port + 1
    service.reload(lib)
    check("correcting the port does — the next step redials",
          not service.is_open("MC1"))
finally:
    machine.stop()

# ── Modbus, which is asked rather than heard from ─────────────────────────
print("\nModbus TCP, against a device that is actually there")

try:
    from pymodbus.server import ModbusTcpServer
    from pymodbus.simulator import DataType, SimData, SimDevice
except ImportError:
    print("  ---- pymodbus is not installed; the Modbus wire is untested")
else:
    import asyncio

    spare = socket.socket()
    spare.bind(("127.0.0.1", 0))
    mb_port = spare.getsockname()[1]
    spare.close()

    bits = [SimData(0, count=64, datatype=DataType.BITS)]
    regs = [SimData(0, count=64, datatype=DataType.REGISTERS)]
    device = SimDevice(1, simdata=(bits, bits, regs, regs))

    async def _serve():
        await ModbusTcpServer(device,
                              address=("127.0.0.1", mb_port)).serve_forever()

    threading.Thread(target=lambda: asyncio.run(_serve()), daemon=True).start()

    plc = LinkLibrary([make_link(
        "PLC1", "modbus", "127.0.0.1", mb_port, timeout=2.0, unit=1,
        data=[make_item("RUN", "send", 1, area="coil", register=10),
              make_item("STOP", "send", 0, area="coil", register=10),
              make_item("RUNNING", "recv", 1, area="coil", register=10),
              make_item("SPEED", "send", 250, area="holding", register=20),
              make_item("AT_SPEED", "recv", 250, area="holding", register=20),
              make_item("SENSOR", "recv", 1, area="discrete", register=5)])])
    service = LinkService(plc, log=lambda _t: None)
    check("a Modbus device that answers says so",
          until(lambda: "answering" in service.probe("PLC1"), 8.0),
          service.probe("PLC1"))

    check("a coil nobody has set does not satisfy a wait",
          service.attempt("PLC1", "RUNNING") == (False, 0),
          str(service.attempt("PLC1", "RUNNING")))
    service.send("PLC1", "RUN")
    check("and once the cell writes it, it does",
          service.attempt("PLC1", "RUNNING") == (True, 1))
    service.send("PLC1", "STOP")
    check("a coil written back to 0 stops satisfying it",
          service.attempt("PLC1", "RUNNING") == (False, 0))

    service.send("PLC1", "SPEED")
    check("a holding register carries a number rather than a flag",
          service.attempt("PLC1", "AT_SPEED") == (True, 250)
          and service.read("PLC1", "AT_SPEED") == 250)

    check("a discrete input can be read, though nothing may write it",
          service.read("PLC1", "SENSOR") == 0)
    check("and a raw link has nothing to read, which is said rather than "
          "guessed at",
          _refused(lambda: LinkService(
              LinkLibrary([make_link("MC1", "tcp", "127.0.0.1", 1)])
          ).read("MC1", "X"), "listened to, not read"))
    service.close_all()

# ── what a program says, and what it does ─────────────────────────────────
print("\nthe checker, before anything moves")

cfg = CellConfig.load()
lib = cfg.link_library()
lib.add(make_link("MC1", "tcp", "10.1.68.200", 2000,
                  data=[make_item("START", "send", "ST"),
                        make_item("DONE", "recv", "DN")]))
lib.add(make_link("PLC1", "modbus", "10.1.68.210", 502,
                  data=[make_item("RUN", "send", 1, area="coil", register=10)]))
cfg.set_link_library(lib)

program = Program("t")
program.steps = [Step("SEND", link="MC1", item="START"),
                 Step("RECV", link="MC1", item="DONE", timeout=5.0)]
check("a line that names a machine and a datum it has is clean",
      program.validate(PointLibrary(), cfg) == [],
      str(program.validate(PointLibrary(), cfg)))


def problem(step):
    one = Program("t")
    one.steps = [step]
    found = one.validate(PointLibrary(), cfg)
    return found[0] if found else ""


check("a machine that was deleted off the tab is caught on paper",
      "no machine called" in problem(Step("SEND", link="GONE", item="X")),
      problem(Step("SEND", link="GONE", item="X")))
check("and it says which tab to go and set one up on",
      "Communication tab" in problem(Step("SEND", link="GONE", item="X")))
check("so is a datum the machine does not carry",
      "no data item called" in problem(Step("SEND", link="MC1", item="NOPE")))
check("and one pointed the wrong way",
      "set up to send only" in problem(Step("RECV", link="MC1", item="START")))
check("Modbus has no free text to fall back on, and says which tab to use",
      "pick one of its data items" in problem(Step("SEND", link="PLC1", item="")),
      problem(Step("SEND", link="PLC1", item="")))
check("a SEND with nothing to send is caught",
      "nothing to send" in problem(Step("SEND", link="MC1", item="", text="")))
check("and a negative timeout",
      "waits for nothing" in problem(Step("RECV", link="MC1", item="DONE",
                                          timeout=-1)))
check("a line naming no machine at all is caught without a library",
      "names no machine" in problem(Step("RECV", link="", item="DONE")))

print("\nrunning it, against a machine that is actually there")

machine = Machine("tcp")
try:
    cell = Cell(CellConfig.load(), simulated=True)
    executor = Executor(cell, PointLibrary())
    executor.links = LinkService(library(machine), log=lambda _t: None)

    executor._execute(Step("SEND", link="MC1", item="START"))
    check("a SEND step reaches the wire",
          until(lambda: machine.heard == [b"ST\r\n"]), str(machine.heard))

    waiting = threading.Thread(
        target=lambda: executor._execute(
            Step("RECV", link="MC1", item="DONE", into="reply", timeout=5.0)),
        daemon=True)
    waiting.start()
    time.sleep(0.05)
    check("a RECV holds the program while nothing has arrived",
          waiting.is_alive())
    machine.say("DN\r\n")
    waiting.join(3.0)
    check("and lets go the moment it does", not waiting.is_alive())
    check("what arrived is left where an IF below can test it",
          executor.vars.get("reply") == 1.0, str(executor.vars))

    machine.say("42.5\r\n")
    executor._execute(Step("RECV", link="MC1", item="", into="reading",
                           timeout=5.0))
    check("a machine that answers with a number is stored as that number",
          executor.vars.get("reading") == 42.5, str(executor.vars))

    began = time.monotonic()
    try:
        executor._execute(Step("RECV", link="MC1", item="DONE", timeout=0.3))
        check("a machine that never answers fails the program", False,
              "it carried on instead")
    except ProgramError as exc:
        check("a machine that never answers fails the program",
              "never sent DONE" in str(exc), str(exc))
    check("and does it at the timeout rather than whenever",
          0.3 <= time.monotonic() - began < 2.0,
          "%.2f s" % (time.monotonic() - began))

    executor.simulate = True
    machine.heard.clear()
    executor._execute(Step("SEND", link="MC1", item="START"))
    executor._execute(Step("RECV", link="MC1", item="DONE", timeout=0.1))
    check("a simulated run touches no wire and waits for nothing",
          machine.heard == [], str(machine.heard))

    executor.simulate = False
    executor.links = None
    try:
        executor._execute(Step("SEND", link="MC1", item="START"))
        check("a cell with no machines set up says so", False, "it went ahead")
    except ProgramError as exc:
        check("a cell with no machines set up says so",
              "Communication tab" in str(exc), str(exc))
finally:
    machine.stop()

print("\nFAILURES: %d" % fail)
sys.exit(1 if fail else 0)
