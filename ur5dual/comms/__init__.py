"""Talking to the machines the cell stands next to.

A link is a named address — `MC1` at 10.1.68.200:2000 — and a handful of named
data items on it, so that a program says `SEND MC1 START` rather than carrying
an IP address and a byte string on the line. The names are set up once on the
Communication tab and the program only ever picks from them, as the
same bargain the point library makes for places.

`links.py` is what a link *is* and how it is written down; `client.py` is the
socket; `service.py` is the one object that owns the open connections and is
what the executor is handed.
"""

from .links import (
    AREAS, DIRECTIONS, FORMATS, KINDS, MATCHES, MODBUS_KINDS, RAW_KINDS,
    LinkLibrary, escape, make_item, make_link, unescape,
)
from .client import LinkError, open_link
from .service import LinkService

__all__ = [
    "AREAS", "DIRECTIONS", "FORMATS", "KINDS", "MATCHES", "MODBUS_KINDS",
    "RAW_KINDS", "LinkError", "LinkLibrary", "LinkService", "escape",
    "make_item", "make_link", "open_link", "unescape",
]
