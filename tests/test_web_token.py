"""The browser token: fresh each run, fixed on request, or off for a trusted LAN."""

import os
import socket
import sys
import tempfile
import urllib.error
import urllib.request

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication

from ur5dual.cell import Cell
from ur5dual.config import CellConfig
from ur5dual.gui.app import MainWindow, lan_address

# 127.0.0.2 is loopback exactly as 127.0.0.1 is, but it is not one of the three
# names _start_web reads as local, so it takes the LAN branch without offering
# a port to the network for the length of the test.
LAN = "127.0.0.2"


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def serve(token=None, host=LAN):
    config = CellConfig.load()
    config.path = os.path.join(tempfile.mkdtemp(prefix="ur5dual-token-"), "cell.yaml")
    port = free_port()
    window = MainWindow(cell=Cell(config, simulated=True), connect_on_start=False,
                        web_host=host, web_port=port, web_token=token)
    return window, "http://%s:%d" % (host, port)


def status(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


app = QApplication.instance() or QApplication([])
windows = []

# Left out, every run mints its own token — which is why a bookmark goes stale.
first, _ = serve(); windows.append(first)
second, _ = serve(); windows.append(second)
assert first.web_server.token and second.web_server.token
assert first.web_server.token != second.web_server.token

# A fixed token is used verbatim, so one URL keeps working across restarts.
fixed, url = serve("cellkey"); windows.append(fixed)
assert fixed.web_server.token == "cellkey"
assert status(url + "/api/state") == 401
assert status(url + "/api/state?access_token=cellkey") == 200

# "none" drops the check entirely for a trusted LAN.
open_run, open_url = serve("none"); windows.append(open_run)
assert open_run.web_server.token is None
assert status(open_url + "/api/state") == 200

# Empty and "off" say the same thing; a local run never had a token to begin with.
assert serve("")[0].web_server.token is None
assert serve(" OFF ")[0].web_server.token is None
assert serve(None, host="127.0.0.1")[0].web_server.token is None

# The logged URL has to be one another machine can type: a bound address is
# itself, and a wildcard bind resolves to the interface facing the default
# route rather than to a hostname nothing on the LAN can look up.
assert lan_address("172.101.99.34") == "172.101.99.34"
assert lan_address(LAN) == LAN
wildcard = lan_address("0.0.0.0")
assert wildcard.count(".") == 3 and not wildcard.startswith("127."), wildcard

for window in windows:
    window.web_server.stop()
print("web token modes: ok")
