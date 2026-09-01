"""What a link is, and the library of the ones this cell has.

Three protocols, kept in one vocabulary because a program should not care
which it is talking to:

    tcp        a raw stream socket. What goes down it is a string or a run of
               bytes, and a message is whatever arrives between terminators.
    udp        the same payloads as datagrams: no connection, no ordering, and
               no acknowledgement that anybody heard.
    modbus     Modbus TCP — a different protocol, not a way of using a socket.
               What goes down it is a number written to a register address,
               and reading is asking for that address back.

Both carry *named data items*, and the name is the whole point. `MC1 START` is
a thing an operator can read on a program line six months later; `10.1.68.200
:2000 "ST\\r\\n"` is not, and neither is `holding 40001 = 1`. The address is
set up once on the tab and the program only ever picks a name off a list.

Everything here is plain dicts and lists, so the whole library drops into
`cell.yaml` and comes back out of it without a schema in between.
"""

# The raw sockets carry whatever bytes an item names; Modbus carries numbers
# at addresses and is a protocol in its own right, not a third way of holding
# a socket open. `udp` is a real choice rather than a completeness exercise: a
# machine that announces its state and does not care who hears is a UDP
# machine, and so is one that must never block the cell waiting for an ACK it
# was never going to send.
RAW_KINDS = ("tcp", "udp")
MODBUS_KINDS = ("modbus",)
KINDS = RAW_KINDS + MODBUS_KINDS

KIND_LABEL = {"tcp": "TCP", "udp": "UDP", "modbus": "Modbus TCP"}

# how a raw payload is written down. `string` is what a machine speaking ASCII
# wants; `hex` is for the ones that speak a frame with an STX in it, which no
# text field can hold without lying about what it contains.
FORMATS = ("string", "hex")

# which way a data item goes. A machine's "cycle done" is not something this
# cell may send, and offering it in a SEND picker is offering a program that
# cannot work — so the direction is part of the item and the pickers respect it.
DIRECTIONS = ("send", "recv", "both")

# how a received message is judged against what the item says to expect.
# `exact` is the honest default; the other two exist because real machines
# append a sequence number or a checksum nobody asked for.
MATCHES = ("exact", "contains", "prefix")

# the four Modbus tables. The last two are read-only in the protocol itself,
# which is why a `send` item may not name one.
AREAS = ("coil", "discrete", "holding", "input")
READ_ONLY_AREAS = ("discrete", "input")
AREA_LABEL = {"coil": "coil (0x)", "discrete": "discrete in (1x)",
              "holding": "holding reg (4x)", "input": "input reg (3x)"}

DEFAULT_PORT = {"tcp": 2000, "udp": 2000, "modbus": 502}
DEFAULT_TIMEOUT = 3.0
# 65535 registers is the whole address space; a number outside it is a typo
# rather than an exotic device.
REGISTER_MAX = 65535
REGISTER_VALUE_MAX = 65535

# Payloads and terminators are stored in *escaped notation* — the four
# characters `\r\n`, not a carriage return and a line feed. One rule, and it
# has to be one: a field showing a literal CR is a field an operator cannot
# tell from an empty one, and a value escaped on the way in and again on the
# way out comes back as `\\r\\n` and goes on the wire as five wrong bytes.
#
# So: `unescape` at the moment it becomes bytes, and nowhere else. `escape` is
# for the other direction — turning a raw string *into* the notation — and the
# dialogs never need it, because what they are showing is already in it.
_ESCAPES = (("\\", "\\\\"), ("\r", "\\r"), ("\n", "\\n"), ("\t", "\\t"))


def escape(text):
    """The bytes a terminator means, as something a text field can hold.

    `\\r\\n` has to survive a round trip through a QLineEdit, and a literal
    carriage return in one is invisible — an operator cannot tell a field
    holding CR LF from a field holding nothing at all.
    """
    out = str(text or "")
    for raw, shown in _ESCAPES:
        out = out.replace(raw, shown)
    return out


def unescape(text):
    """The other direction: what the operator typed, as what it means.

    Hand-rolled rather than `unicode_escape`, which decodes through Latin-1 and
    so turns any non-ASCII character in a payload into mojibake on the wire.
    """
    out, i = [], 0
    text = str(text or "")
    table = {"r": "\r", "n": "\n", "t": "\t", "\\": "\\", "0": "\0"}
    while i < len(text):
        char = text[i]
        if char == "\\" and i + 1 < len(text) and text[i + 1] in table:
            out.append(table[text[i + 1]])
            i += 2
        else:
            out.append(char)
            i += 1
    return "".join(out)


def parse_hex(text):
    """`02 41 03` or `024103`, as bytes. Raises ValueError on anything else."""
    cleaned = "".join(str(text or "").split()).replace("0x", "").replace(",", "")
    if not cleaned:
        return b""
    if len(cleaned) % 2:
        raise ValueError("hex needs an even number of digits, got %d"
                         % len(cleaned))
    return bytes.fromhex(cleaned)


def make_item(name, direction="both", value="", match="exact",
              area="holding", register=0, **extra):
    """One named datum on a link.

    The same shape for both families, because the picker on a program line is
    the same picker: a raw item carries `value` as text, a Modbus one carries
    it as the number to write or wait for, and `area`/`register` say where.
    """
    item = {"name": str(name).strip(),
            "direction": direction if direction in DIRECTIONS else "both",
            "value": value,
            "match": match if match in MATCHES else "exact",
            "area": area if area in AREAS else "holding",
            "register": int(register)}
    item.update({k: v for k, v in extra.items() if v is not None})
    return item


def make_link(name, kind="tcp", host="127.0.0.1", port=None, format="string",
              encoding="utf-8", terminator="\\r\\n", timeout=DEFAULT_TIMEOUT,
              unit=1, data=None, **extra):
    """One named machine.

    `terminator` is stored escaped — the way it is typed and the way it reads
    in `cell.yaml`. `unit` is the Modbus device id and is ignored by the raw
    kinds, which is cheaper than two record shapes for one list.
    """
    kind = kind if kind in KINDS else "tcp"
    link = {"name": str(name).strip(),
            "kind": kind,
            "host": str(host).strip(),
            "port": int(DEFAULT_PORT[kind] if port is None else port),
            "format": format if format in FORMATS else "string",
            "encoding": str(encoding or "utf-8"),
            "terminator": str(terminator or ""),
            "timeout": float(timeout),
            "unit": int(unit),
            "data": [dict(d) for d in (data or [])]}
    link.update({k: v for k, v in extra.items() if v is not None})
    return link


def is_modbus(link):
    return (link or {}).get("kind") in MODBUS_KINDS


def address(link):
    """How a link reads on one line of a table."""
    if not link:
        return ""
    text = "%s:%d" % (link.get("host", ""), int(link.get("port", 0)))
    if is_modbus(link):
        text += "  #%d" % int(link.get("unit", 1))
    return text


def describe_item(link, item):
    """What one data item says it carries, for the tables and the tooltips."""
    if not item:
        return ""
    if is_modbus(link):
        return "%s %d = %s" % (item.get("area", "holding"),
                               int(item.get("register", 0)),
                               item.get("value", 0))
    # Already in escaped notation, so it is shown rather than escaped again.
    shown = str(item.get("value", ""))
    if item.get("direction") != "send" and item.get("match", "exact") != "exact":
        return "%s %r" % (item.get("match"), shown)
    return repr(shown)


class LinkLibrary:
    """The links this cell has, in the order they were added.

    A list rather than a dict, and deliberately: the tables show them in the
    order somebody set them up, and a cell with MC1, MC2 and MC3 reads wrong
    the moment something sorts it alphabetically and MC10 lands between them.
    """

    def __init__(self, links=None):
        self.links = [dict(link) for link in (links or [])]

    # -- reading -----------------------------------------------------------
    def names(self, kinds=None):
        return [link["name"] for link in self.of_kind(kinds)]

    def of_kind(self, kinds=None):
        """Every link, or only the ones one tab is responsible for."""
        if kinds is None:
            return list(self.links)
        return [link for link in self.links if link.get("kind") in kinds]

    def get(self, name):
        for link in self.links:
            if link.get("name") == name:
                return link
        return None

    def items_for(self, name, direction=None):
        """The data items on one link, filtered by which way they go.

        `direction` of None is every item, which is what a table shows; "send"
        and "recv" are what the two step editors offer, and an item marked
        `both` answers to either.
        """
        link = self.get(name)
        if link is None:
            return []
        items = list(link.get("data") or [])
        if direction is None:
            return items
        return [i for i in items
                if i.get("direction", "both") in (direction, "both")]

    def item(self, link_name, item_name):
        for item in self.items_for(link_name):
            if item.get("name") == item_name:
                return item
        return None

    # -- writing -----------------------------------------------------------
    def add(self, link):
        if self.get(link["name"]) is not None:
            raise ValueError("there is already a link called %r" % link["name"])
        self.links.append(dict(link))
        return link

    def replace(self, name, link):
        """Edit one link in place, keeping its position in the list.

        A rename that collides with another link is refused here rather than
        allowed and sorted out later: two links called MC1 make every program
        line naming MC1 ambiguous, and nothing downstream can tell which was
        meant.
        """
        for i, existing in enumerate(self.links):
            if existing.get("name") != name:
                continue
            if link["name"] != name and self.get(link["name"]) is not None:
                raise ValueError("there is already a link called %r"
                                 % link["name"])
            self.links[i] = dict(link)
            return link
        raise KeyError(name)

    def remove(self, name):
        before = len(self.links)
        self.links = [l for l in self.links if l.get("name") != name]
        return len(self.links) != before

    def set_items(self, name, items):
        link = self.get(name)
        if link is None:
            raise KeyError(name)
        link["data"] = [dict(i) for i in items]
        return link

    # -- io ----------------------------------------------------------------
    def to_list(self):
        return [dict(link) for link in self.links]

    @classmethod
    def from_list(cls, data):
        """Read the block out of `cell.yaml`, dropping what cannot be a link.

        A malformed entry is skipped rather than raised on. The panel has to
        start: a cell whose control GUI refuses to open because somebody
        hand-edited one line of a comms block is a cell nobody can fix from
        the pendant.
        """
        links = []
        for entry in (data or []):
            if not isinstance(entry, dict) or not entry.get("name"):
                continue
            links.append(make_link(**{k: v for k, v in entry.items()
                                      if k != "data"},
                                   data=[make_item(**item)
                                         for item in (entry.get("data") or [])
                                         if isinstance(item, dict)
                                         and item.get("name")]))
        return cls(links)

    # -- checking ----------------------------------------------------------
    def check(self):
        """Everything wrong with the library that can be seen without a wire."""
        problems = []
        seen = set()
        for link in self.links:
            name = link.get("name", "")
            if not name:
                problems.append("a link has no name")
                continue
            if name in seen:
                problems.append("there is more than one link called %r" % name)
            seen.add(name)
            problems.extend(check_link(link))
        return problems


def check_link(link):
    """What is wrong with one link, named by the link rather than by a line."""
    problems = []
    name = link.get("name") or "?"
    if link.get("kind") not in KINDS:
        problems.append("%s: %r is not one of %s"
                        % (name, link.get("kind"), " ".join(KINDS)))
    if not (link.get("host") or "").strip():
        problems.append("%s: no address to connect to" % name)
    port = int(link.get("port", 0) or 0)
    if not 1 <= port <= 65535:
        problems.append("%s: port %d is not a port" % (name, port))
    if float(link.get("timeout", 0) or 0) < 0:
        problems.append("%s: a negative timeout waits for nothing" % name)
    if is_modbus(link) and not 0 <= int(link.get("unit", 1)) <= 255:
        problems.append("%s: device id must be 0 to 255" % name)

    seen = set()
    for item in (link.get("data") or []):
        item_name = item.get("name") or ""
        if not item_name:
            problems.append("%s: a data item has no name" % name)
            continue
        if item_name in seen:
            problems.append("%s: there is more than one %r" % (name, item_name))
        seen.add(item_name)
        problems.extend(check_item(link, item))
    return problems


def check_item(link, item):
    """What is wrong with one data item on one link."""
    problems = []
    where = "%s %s" % (link.get("name") or "?", item.get("name") or "?")
    direction = item.get("direction", "both")
    if direction not in DIRECTIONS:
        problems.append("%s: %r is not a direction" % (where, direction))

    if is_modbus(link):
        area = item.get("area", "holding")
        if area not in AREAS:
            problems.append("%s: %r is not a Modbus table" % (where, area))
        elif area in READ_ONLY_AREAS and direction in ("send", "both"):
            problems.append(
                "%s: %s is read-only in Modbus — a coil or a holding register "
                "is what this cell may write" % (where, AREA_LABEL[area]))
        register = int(item.get("register", 0) or 0)
        if not 0 <= register <= REGISTER_MAX:
            problems.append("%s: register %d is outside 0-%d"
                            % (where, register, REGISTER_MAX))
        try:
            value = int(item.get("value", 0))
        except (TypeError, ValueError):
            problems.append("%s: %r is not a number to write or wait for"
                            % (where, item.get("value")))
        else:
            if area == "coil" and value not in (0, 1):
                problems.append("%s: a coil is 0 or 1, not %d" % (where, value))
            elif not 0 <= value <= REGISTER_VALUE_MAX:
                problems.append("%s: %d is outside a 16-bit register"
                                % (where, value))
        return problems

    if item.get("match") not in MATCHES:
        problems.append("%s: %r is not a way of matching" % (where,
                                                             item.get("match")))
    if link.get("format") == "hex":
        try:
            parse_hex(item.get("value", ""))
        except ValueError as exc:
            problems.append("%s: %s" % (where, exc))
    elif not str(item.get("value", "")):
        # An empty string is a legitimate thing to *wait for* only if the
        # terminator alone is the signal, which is a machine nobody has met.
        problems.append("%s: nothing to send or wait for" % where)
    return problems


def payload_bytes(link, item):
    """What goes on the wire for one item, terminator included.

    Payload and terminator are both unescaped here, and this is the only place
    either of them is: sending has to mean exactly what waiting matches, and a
    payload taken literally while the thing waiting for it took `\\t` as a tab
    is a handshake that can never complete and looks identical on screen to
    one that can.

    The terminator is appended to a string and never to hex: a machine
    speaking framed bytes has its own end marker inside the frame, and adding
    CR LF to it is corrupting a packet rather than finishing a line.
    """
    value = item.get("value", "")
    if link.get("format") == "hex":
        return parse_hex(value)
    encoding = link.get("encoding") or "utf-8"
    return (unescape(str(value))
            + unescape(link.get("terminator", ""))).encode(encoding)


def message_text(link, raw):
    """A message off the wire, as the text an item is compared against."""
    if link.get("format") == "hex":
        return raw.hex(" ")
    try:
        text = raw.decode(link.get("encoding") or "utf-8", "replace")
    except LookupError:                 # an encoding name that is not one
        text = raw.decode("utf-8", "replace")
    return text.strip("\r\n")


def item_matches(link, item, text):
    """Does what arrived satisfy what this item said to wait for?"""
    want = str(item.get("value", ""))
    if link.get("format") == "hex":
        try:
            want = parse_hex(want).hex(" ")
        except ValueError:
            return False
        text = text.lower()
        want = want.lower()
    else:
        want = unescape(want).strip("\r\n")
    mode = item.get("match", "exact")
    if mode == "contains":
        return want in text
    if mode == "prefix":
        return text.startswith(want)
    return text == want
