"""The one object that owns the open connections.

Held by the app rather than by a tab, for the reason the camera is: a program
that hands a machine its start signal must not depend on which panel an
operator happened to leave open, and a socket reopened every time a sidebar
closed would drop the reply it was about to be sent.

Connections open lazily, on the first step that uses one. A cell configured
with four machines and running a program that talks to one dials one number —
which also means a link left set up for a machine that is switched off does
not stop every other program in the cell from starting.
"""

import threading

from .client import LinkError, ModbusLink, RawLink, raw_matches
from .links import (
    KIND_LABEL, LinkLibrary, address, describe_item, is_modbus, message_text,
)


# What makes one socket a different socket. `data` is deliberately not here:
# the items a machine carries are names for payloads, and renaming one is no
# reason to hang up on a machine mid-cycle.
_DIALLED = ("kind", "host", "port", "timeout", "unit", "format", "encoding",
            "terminator")


def _dialled(link):
    return tuple((link or {}).get(key) for key in _DIALLED)


class LinkService:
    """Named machines, and whatever is currently connected to them."""

    def __init__(self, library=None, log=None):
        self.library = library or LinkLibrary()
        self.log = log or (lambda _text: None)
        self._clients = {}
        # The executor runs on a worker thread and the tabs' Test buttons run
        # on the GUI thread. They reach the same dict and, for a link under
        # test while a program has it open, the same socket.
        self._lock = threading.RLock()

    # -- the library -------------------------------------------------------
    def reload(self, library=None):
        """Take up an edited library, hanging up on what is no longer the same
        machine.

        Only the fields that decide what the socket *is* are compared, against
        the record each open client was dialled with. Naming another datum on
        a machine must not drop the connection to it mid-cycle; correcting its
        port must, and on the next step rather than at the next restart.
        """
        with self._lock:
            if library is not None:
                self.library = library
            for name, client in list(self._clients.items()):
                fresh = self.library.get(name)
                if fresh is None or _dialled(fresh) != _dialled(client.link):
                    self._close(name)
                    self.log("%s: address changed, will redial on its next use"
                             % name)
        return self.library

    def names(self, kinds=None):
        return self.library.names(kinds)

    # -- connections -------------------------------------------------------
    def _client(self, name):
        """The open client for a link, dialling it if this is the first use."""
        with self._lock:
            client = self._clients.get(name)
            if client is not None and client.connected:
                return client
            link = self.library.get(name)
            if link is None:
                raise LinkError("there is no link called %r — set one up "
                                "on the Communication tab" % name)
            client = ModbusLink(link) if is_modbus(link) else RawLink(link)
            client.open()
            self._clients[name] = client
            self.log("%s: connected to %s" % (name, address(link)))
            return client

    def _close(self, name):
        client = self._clients.pop(name, None)
        if client is not None:
            client.close()
        return client is not None

    def close(self, name):
        with self._lock:
            if self._close(name):
                self.log("%s: closed" % name)

    def close_all(self):
        with self._lock:
            for name in list(self._clients):
                self._close(name)

    def is_open(self, name):
        client = self._clients.get(name)
        return client is not None and client.connected

    # -- what a step does --------------------------------------------------
    def send(self, name, item_name=None, text=None, flush=True):
        """Put one named item — or one literal payload — on the wire.

        `flush` throws away anything already waiting first, and defaults to
        on: a reply left over from the last cycle would satisfy the next RECV
        the instant it looked, and a handshake that passes on stale data moves
        an arm for a reason nothing on screen shows.
        """
        link = self.library.get(name)
        if link is None:
            raise LinkError("there is no link called %r" % name)
        client = self._client(name)
        if is_modbus(link):
            item = self._item(name, item_name, "send")
            written = client.write(item)
            self.log("%s -> %s (%s)" % (name, item["name"],
                                        describe_item(link, item)))
            return written
        if flush:
            client.drain()
        if item_name:
            item = self._item(name, item_name, "send")
            client.send_item(item)
            self.log("%s -> %s %s" % (name, item["name"],
                                      describe_item(link, item)))
            return item.get("value")
        client.send_text(text or "")
        self.log("%s -> %r" % (name, text or ""))
        return text

    def attempt(self, name, item_name=None):
        """One non-blocking look for what a RECV is waiting for.

        Returns `(got, value)`. `got` is False when nothing has arrived yet,
        which is not a failure — the executor's loop is what decides how long
        to keep asking, because that loop is the one the STOP button can
        reach.

        An item name of nothing means "whatever comes", which is how a reading
        is taken off a machine that reports a number rather than a handshake.
        """
        link = self.library.get(name)
        if link is None:
            raise LinkError("there is no link called %r" % name)
        client = self._client(name)
        if is_modbus(link):
            item = self._item(name, item_name, "recv")
            return client.poll(item)

        raw = client.poll()
        if raw is None:
            return False, None
        if not item_name:
            return True, message_text(link, raw)
        item = self._item(name, item_name, "recv")
        matched, text = raw_matches(link, item, raw)
        if not matched:
            # Said out loud rather than dropped. A machine answering something
            # nobody is waiting for is the single most common reason a
            # handshake times out, and it is invisible otherwise.
            self.log("%s <- %r (waiting for %s)"
                     % (name, text, describe_item(link, item)))
        return matched, text

    def read(self, name, item_name):
        """One Modbus register, now. Raw links have nothing to ask."""
        link = self.library.get(name)
        if link is None:
            raise LinkError("there is no link called %r" % name)
        if not is_modbus(link):
            raise LinkError("%s is a %s link — it is listened to, not read"
                            % (name, KIND_LABEL.get(link.get("kind"), "?")))
        return self._client(name).read(self._item(name, item_name, "recv"))

    def _item(self, link_name, item_name, direction):
        item = self.library.item(link_name, item_name)
        if item is None:
            raise LinkError("%s has no data item called %r — add it on the tab"
                            % (link_name, item_name))
        if item.get("direction", "both") not in (direction, "both"):
            raise LinkError("%s %s is set up to %s only"
                            % (link_name, item_name, item.get("direction")))
        return item

    # -- proving a link, from the tab that set it up -----------------------
    def probe(self, name):
        """Open a link and say what happened, in one line for the message bar.

        The point is to fail here, on a tab, with a cursor in the field that
        is wrong — rather than three lines into a program with an arm already
        somewhere.
        """
        link = self.library.get(name)
        if link is None:
            return "there is no link called %r" % name
        try:
            self._client(name)
        except LinkError as exc:
            return str(exc)
        return "%s: %s at %s is answering" % (
            name, KIND_LABEL.get(link.get("kind"), link.get("kind")),
            address(link))
