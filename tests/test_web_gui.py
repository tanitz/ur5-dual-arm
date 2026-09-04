"""A browser command crosses into Qt and desktop changes cross back out."""

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
from ur5dual.comms import make_item, make_link
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
config.path = os.path.join(tempfile.mkdtemp(prefix="ur5dual-web-"), "cell.yaml")
config.ui.update({"sidebar_panel": "jog", "sidebar_open": True})
port = free_port()
window = MainWindow(cell=Cell(config, simulated=True), connect_on_start=False,
                    web_host="127.0.0.1", web_port=port)
window.programs_dir = tempfile.mkdtemp(prefix="ur5dual-web-programs-")
base = "http://127.0.0.1:%d" % port


def get_state():
    with urllib.request.urlopen(base + "/api/state", timeout=2) as response:
        return json.load(response)


def post(payload):
    """Issue HTTP off-thread while pumping the Qt command receiver."""
    answer = {}

    def request():
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(base + "/api/command", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=4) as response:
                answer["value"] = json.load(response)
        except Exception as exc:
            answer["error"] = exc

    thread = threading.Thread(target=request)
    thread.start()
    deadline = time.monotonic() + 5.0
    while thread.is_alive() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)
    thread.join(timeout=0.1)
    if "error" in answer:
        raise answer["error"]
    assert not thread.is_alive() and answer["value"]["ok"] is True
    return answer["value"]


def refuse(payload):
    """The other half of post(): a command the desktop must turn down."""
    try:
        post(payload)
    except urllib.error.HTTPError as exc:
        return json.loads(exc.read().decode("utf-8"))["detail"]
    raise AssertionError("the desktop accepted %r" % payload.get("action"))


def machine(name, kind="tcp", host="10.1.68.200", port=2000, unit=1, data=()):
    """One machine the shape the browser's Com tab sends it."""
    return {"name": name, "kind": kind, "host": host, "port": port,
            "format": "string", "encoding": "utf-8", "terminator": "\\r\\n",
            "timeout": 1.0, "unit": unit, "data": [dict(d) for d in data]}


def datum(name, direction="send", value="ST", register=0):
    return {"name": name, "direction": direction, "value": value,
            "match": "exact", "area": "holding", "register": register}


assert get_state()["ready"] is True
post({"action": "camera_mode", "mode": "depth"})
assert window.panels["camera"].mode == "depth"
assert get_state()["camera"]["mode"] == "depth"
post({"action": "camera_auto_size", "enabled": True})
assert window.cell.config.vision["auto_size"] is True
assert get_state()["camera"]["auto_size"] is True
post({"action": "camera_box", "box_mm": [200, 100, 70]})
assert window.cell.config.vision["auto_size"] is False
assert get_state()["camera"]["box_mm"] == [200, 100, 70]

post({"action": "program_set", "program": {
    "name": "from_web", "loop": False, "steps": [],
}})
assert window.panels["program"].program.name == "from_web"
post({"action": "program_save"})
assert os.path.isfile(os.path.join(window.programs_dir, "from_web.json"))

# ---- Com: what a browser edits is what the desktop tab shows -------------
net = window.panels["net"]


def rows(table, columns=3):
    return [[table.item(r, c).text() for c in range(columns)]
            for r in range(table.rowCount())]


# A machine and its data, added from the browser.
post({"action": "comm_set", "links": [machine("MC1", data=[
    datum("START", "send", "ST"), datum("DONE", "recv", "DN")])]})
assert window.link_library().names() == ["MC1"]
assert rows(net.links_table) == [["MC1", "TCP", "10.1.68.200:2000"]]
assert net.items_strip.text() == "Data on MC1"
assert [row[:2] for row in rows(net.items_table)] == [
    ["START", "\u2192 send"], ["DONE", "\u2190 recv"]]
assert "'ST'" in net.items_table.item(0, 2).text()
# The tab's own status line is the one both surfaces read.
assert net.status.text() == "Communication updated from web"
assert get_state()["comm"]["status"] == "Communication updated from web"
assert get_state()["comm"]["links"][0]["data"][0]["name"] == "START"

# Renamed, moved onto Modbus, and still carrying what it carried.
post({"action": "comm_set", "links": [machine(
    "PLC1", kind="modbus", port=502, unit=3,
    data=[datum("START", "send", 7, register=12)])]})
assert window.link_library().names() == ["PLC1"]
assert rows(net.links_table) == [["PLC1", "Modbus TCP", "10.1.68.200:502  #3"]]
assert rows(net.items_table) == [["START", "\u2192 send", "holding 12 = 7"]]

# A second machine, and the desktop's own selection choosing whose data shows.
post({"action": "comm_set", "links": [
    machine("PLC1", kind="modbus", port=502, unit=3,
            data=[datum("START", "send", 7, register=12)]),
    machine("SCAN1", kind="udp", host="10.1.68.201", port=5000,
            data=[datum("CODE", "recv", "*")])]})
assert [row[0] for row in rows(net.links_table)] == ["PLC1", "SCAN1"]
assert net.links_table.item(1, 1).text() == "UDP"
net.links_table.selectRow(1)
app.processEvents()
assert net.items_strip.text() == "Data on SCAN1"
assert rows(net.items_table)[0][0] == "CODE"

# Nonsense typed into the browser's editor is refused, and refused whole:
# from_list drops what it cannot read, and a silent drop here would delete a
# machine or a datum the operator is looking at.
settled = window.link_library().to_list()
assert refuse({"action": "comm_set", "links": [machine("")]}) \
    == "every machine needs a name"
assert refuse({"action": "comm_set", "links": [
    {"name": "MC9", "kind": "tcp", "host": "10.1.68.9", "port": 2000,
     "data": [{"direction": "send", "value": "ST"}]}]}) \
    == "every data item needs a name"
assert refuse({"action": "comm_set", "links": [
    dict(machine("MC9"), data="START")]}) \
    == "a machine's data must be a JSON list"
assert "more than one" in refuse({"action": "comm_set", "links": [
    machine("PLC1"), machine("PLC1")]})
assert "not a port" in refuse({"action": "comm_set",
                               "links": [machine("PLC1", port=70000)]})
assert window.link_library().to_list() == settled
assert [row[0] for row in rows(net.links_table)] == ["PLC1", "SCAN1"]

# And the other direction: a machine set up on the desktop reaches the browser.
window.link_library().add(make_link("LABEL1", "tcp", "10.1.68.202", 3000,
                                    data=[make_item("PRINT", "send", "P1")]))
window.links_changed()
net.refresh()
net._say("added LABEL1")
window._publish_web_state()
shared = get_state()["comm"]
assert [link["name"] for link in shared["links"]] == ["PLC1", "SCAN1", "LABEL1"]
assert shared["links"][2]["data"][0]["name"] == "PRINT"
assert shared["status"] == "added LABEL1"

# Test opens the socket off the Qt thread; its one line comes back to both.
closed = free_port()
post({"action": "comm_set", "links": [machine(
    "MC1", host="127.0.0.1", port=closed, data=[datum("START", "send", "ST")])]})
assert refuse({"action": "comm_test", "name": "MC2"}) == "machine not found"
net.status.setText("")
post({"action": "comm_test", "name": "MC1"})
deadline = time.monotonic() + 5.0
while not net.status.text() and time.monotonic() < deadline:
    app.processEvents()
    time.sleep(0.02)
assert "MC1" in net.status.text() and "cannot reach" in net.status.text()
assert get_state()["comm"]["status"] == net.status.text()

post({"action": "comm_save"})
assert os.path.isfile(config.path)
assert "MC1" in open(config.path, encoding="utf-8").read()
assert net.status.text() == "saved 1 machine(s)"
assert get_state()["comm"]["status"] == "saved 1 machine(s)"

window.panels["camera"]._set_mode("lens")
window._publish_web_state()
assert get_state()["camera"]["mode"] == "lens"

post({"action": "jog_press", "client_id": "test-browser",
      "target": "A", "row": 0, "sign": 1})
assert window._web_jog_owner == "test-browser"
deadline = time.monotonic() + WEB_JOG_GRACE + 0.5
while window._web_jog_owner and time.monotonic() < deadline:
    app.processEvents()
    time.sleep(0.02)
assert window._web_jog_owner is None

window.close()
app.processEvents()
print("desktop/web synchronization: ok")
