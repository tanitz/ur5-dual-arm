"""
The dashboard server, port 29999: power, brakes, safety state, and the
version string. Line-oriented text, one reply per command.

From PolyScope 3.13 onwards most of these are refused unless the robot is in
Remote Control mode, which is what `is_in_remote_control` is for. On older
firmware that command is not understood and the check reports None.
"""

import socket

DASHBOARD_PORT = 29999


class Dashboard:
    def __init__(self, ip, timeout=5.0):
        self.ip = ip
        self.sock = socket.create_connection((ip, DASHBOARD_PORT), timeout=timeout)
        self.sock.settimeout(timeout)
        self._recv()                      # the greeting banner

    def _recv(self):
        return self.sock.recv(4096).decode(errors="replace").strip()

    def send(self, cmd):
        self.sock.sendall((cmd + "\n").encode())
        return self._recv()

    # -- state -------------------------------------------------------------
    def polyscope_version(self):
        return self.send("PolyscopeVersion")

    def robot_mode(self):
        return self.send("robotmode")

    def safety_mode(self):
        return self.send("safetymode")

    def program_state(self):
        return self.send("programState")

    def is_in_remote_control(self):
        """True / False, or None on firmware that predates the concept."""
        reply = self.send("is in remote control").strip().lower()
        if reply in ("true", "false"):
            return reply == "true"
        return None

    # -- actions -----------------------------------------------------------
    def power_on(self):
        return self.send("power on")

    def power_off(self):
        return self.send("power off")

    def brake_release(self):
        return self.send("brake release")

    def unlock_protective_stop(self):
        return self.send("unlock protective stop")

    def close_popup(self):
        return self.send("close popup")

    def stop(self):
        return self.send("stop")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
