"""The browser jog key: held through a slow desktop, dropped when the browser goes."""

import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication

from ur5dual.cell import Cell
from ur5dual.config import CellConfig
from ur5dual.gui.app import WEB_JOG_GRACE, MainWindow


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


app = QApplication.instance() or QApplication([])
config = CellConfig.load()
config.path = os.path.join(tempfile.mkdtemp(prefix="ur5dual-jog-"), "cell.yaml")
port = free_port()
window = MainWindow(cell=Cell(config, simulated=True), connect_on_start=False,
                    web_host="127.0.0.1", web_port=port)
base = "http://127.0.0.1:%d" % port

# Count the simulated speed commands. A web heartbeat used to maintain only
# ownership and never refresh speedl/speedj, so a "held" key stopped moving
# after the controller watchdog expired.
speed_calls = []
original_speed = window.cell.arms["A"].motion.speed


def counted_speed(*args, **kwargs):
    speed_calls.append(time.monotonic())
    return original_speed(*args, **kwargs)


window.cell.arms["A"].motion.speed = counted_speed


def send(payload, out):
    """Post off-thread, the way a browser does; the caller pumps Qt meanwhile."""
    def run():
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            base + "/api/command", data=body,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                out.append((response.status, ""))
        except urllib.error.HTTPError as exc:
            out.append((exc.code, json.load(exc).get("detail", "")))
        except Exception as exc:                                # pragma: no cover
            out.append(("ERR", repr(exc)))
    thread = threading.Thread(target=run)
    thread.start()
    return thread


def hold(client, seconds, stall_at=None, stall_for=0.0, beat=True):
    """Replay press + the 100ms heartbeat, optionally freezing Qt mid-hold."""
    out, threads = [], []
    threads.append(send({"action": "jog_press", "client_id": client,
                         "target": "A", "row": 0, "sign": 1}, out))
    while not out:                       # the beat waits for the press, as app.js does
        app.processEvents()
        time.sleep(0.005)
    start, beats, stalled = time.monotonic(), 0, False
    while time.monotonic() - start < seconds:
        app.processEvents()
        now = time.monotonic() - start
        if stall_at is not None and not stalled and now >= stall_at:
            stalled = True
            time.sleep(stall_for)        # Qt's loop frozen, as one slow tick
        if beat and now >= (beats + 1) * 0.1:
            beats += 1
            threads.append(send({"action": "jog_heartbeat",
                                 "client_id": client}, out))
        time.sleep(0.005)
    for thread in threads:               # keep pumping until every reply lands
        while thread.is_alive():
            app.processEvents()
            time.sleep(0.005)
        thread.join()
    app.processEvents()
    return out


def release(client):
    out = [] ; thread = send({"action": "jog_release", "client_id": client}, out)
    while thread.is_alive():
        app.processEvents()
        time.sleep(0.005)
    thread.join(); app.processEvents()
    return out


# An ordinary hold: every call is accepted and the key stays with the browser.
answers = hold("browser-1", 0.9)
assert all(code == 200 for code, _ in answers), answers
assert window._web_jog_owner == "browser-1"
assert len(speed_calls) >= 5, "web hold did not refresh simulated motion"
release("browser-1")

# A desktop that freezes past the old 0.35s window used to revoke the key and
# then refuse every queued heartbeat with 400. The stamp is taken when the
# request arrives, so a browser that never stopped pressing keeps it.
answers = hold("browser-1", 1.4, stall_at=0.25, stall_for=0.5)
assert all(code == 200 for code, _ in answers), answers
assert window._web_jog_owner == "browser-1"

# A second browser is still refused while the first one is live.
denied = []
thread = send({"action": "jog_press", "client_id": "browser-2",
               "target": "A", "row": 0, "sign": 1}, denied)
while thread.is_alive():
    app.processEvents()
    time.sleep(0.005)
thread.join()
assert denied == [(400, "jog is owned by another browser")], denied
release("browser-1")

# A browser that goes away mid-hold loses the key inside the grace window.
hold("browser-1", 0.3, beat=False)
assert window._web_jog_owner == "browser-1"
deadline = time.monotonic() + WEB_JOG_GRACE + 0.5
while window._web_jog_owner and time.monotonic() < deadline:
    app.processEvents()
    time.sleep(0.02)
assert window._web_jog_owner is None

window.web_server.stop()
print("web jog key: ok")
