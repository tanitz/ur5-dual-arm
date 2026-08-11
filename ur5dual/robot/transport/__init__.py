"""Raw socket access to a UR controller — one module per interface."""

from .dashboard import Dashboard
from .primary import read_geometry, read_mode_flags, read_tcp_offset
from .rt_stream import StateStream, read_state
from .script import ScriptSender, send_script

__all__ = ["Dashboard", "StateStream", "read_state", "ScriptSender",
           "send_script", "read_geometry", "read_tcp_offset",
           "read_mode_flags"]
