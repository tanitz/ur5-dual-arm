"""
The secondary client interface, port 30002: send URScript, the robot runs it.

Sending a `def ... end` block replaces whatever program is running, which is
how a streaming controller is installed and how it is torn down again.
"""

import socket
import threading

SECONDARY_PORT = 30002


def send_script(ip, script, timeout=5.0):
    """One-shot: connect, send, disconnect."""
    if not script.endswith("\n"):
        script += "\n"
    with socket.create_connection((ip, SECONDARY_PORT), timeout=timeout) as s:
        s.sendall(script.encode())


class ScriptSender:
    """Persistent connection, for loops where reconnecting per command would
    cost more than the command itself.

    The controller drops this socket on a protective stop or a program load,
    so a failed write reconnects once and retries before giving up.
    """

    def __init__(self, ip, timeout=5.0):
        self.ip = ip
        self.timeout = timeout
        self.sock = socket.create_connection((ip, SECONDARY_PORT), timeout=timeout)
        self._lock = threading.Lock()

    def send(self, script):
        if not script.endswith("\n"):
            script += "\n"
        data = script.encode()
        # one lock so two threads cannot interleave halves of two scripts
        # into the same socket and hand the controller a syntax error
        with self._lock:
            try:
                self.sock.sendall(data)
            except OSError:
                self._reconnect_locked()
                self.sock.sendall(data)

    def reconnect(self):
        with self._lock:
            self._reconnect_locked()

    def _reconnect_locked(self):
        try:
            self.sock.close()
        except OSError:
            pass
        self.sock = socket.create_connection((self.ip, SECONDARY_PORT),
                                             timeout=self.timeout)

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
