"""Web transport and its boundary to the desktop-owned state."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from ur5dual.web import WebHub, create_app


seen, jog_touches = [], []
hub = WebHub(lambda command: seen.append(command) or {"accepted": True},
             jog_touch=lambda command: jog_touches.append(command.copy()))
hub.publish({"program": {"running": False}, "tabs": [
    "program", "points", "camera", "comm", "jog",
]}, camera=b"jpeg")
client = TestClient(create_app(hub))

state = client.get("/api/state")
assert state.status_code == 200
assert state.json()["program"]["running"] is False
assert state.json()["tabs"] == ["program", "points", "camera", "comm", "jog"]
assert "net" not in state.json()["tabs"] and "modbus" not in state.json()["tabs"]

camera = client.get("/api/camera.jpg")
assert camera.status_code == 200 and camera.content == b"jpeg"
assert camera.headers["content-type"] == "image/jpeg"

with client.websocket_connect("/ws/camera") as camera_socket:
    assert camera_socket.receive_bytes() == b"jpeg"
    hub.publish({"program": {"running": False}}, camera=b"jpeg-2")
    assert camera_socket.receive_bytes() == b"jpeg-2"
assert hub.camera_streaming() is False

result = client.post("/api/command", json={"action": "program_pause"})
assert result.status_code == 200 and result.json()["ok"] is True
assert result.json()["state"]["revision"] == hub.snapshot()["revision"]
assert result.json()["state"]["program"]["running"] is False
assert seen == [{"action": "program_pause"}]
assert client.post("/api/command", json={}).status_code == 400

index = client.get("/")
assert index.status_code == 200 and "Dual UR5 control" in index.text
assert index.headers["cache-control"] == "no-store, max-age=0"
assert 'data-tab="comm"' in index.text and 'class="workspace"' in index.text
assert 'id="swapSide"' in index.text
assert "app.js?v=20260828-status-fast" in index.text

# The two-column shape, and the side and tab each browser remembers for
# itself, live in the static files — checked here rather than in a browser.
sheet = client.get("/assets/app.css")
assert sheet.status_code == 200 and ".side-left .workspace" in sheet.text
script = client.get("/assets/app.js").text
assert "localStorage.ur5dualWebSide" in script
assert "localStorage.ur5dualWebTab" in script
# Browsers gate crypto.randomUUID to secure contexts, so a plain http:// LAN page
# must not depend on it or the script dies before it opens the socket.
assert "crypto.randomUUID ?" in script and "getRandomValues" in script
assert "/ws/camera" in script and "binaryType='blob'" in script

with client.websocket_connect("/ws") as websocket:
    pushed = websocket.receive_json()
    assert pushed["revision"] == hub.snapshot()["revision"]

    press = {"action": "jog_press", "client_id": "browser-1",
             "session_id": "press-1", "target": "A", "row": 0, "sign": 1}
    websocket.send_json(press)
    accepted = websocket.receive_json()
    assert accepted["type"] == "jog_ack" and accepted["action"] == "jog_press"
    websocket.send_json({"action": "jog_heartbeat", "client_id": "browser-1",
                         "session_id": "press-1"})
    websocket.send_json({"action": "jog_release", "client_id": "browser-1",
                         "session_id": "press-1"})
    released = websocket.receive_json()
    assert released["type"] == "jog_ack" and released["action"] == "jog_release"

assert seen[-2:] == [press, {"action": "jog_release", "client_id": "browser-1",
                            "session_id": "press-1"}]
assert [item["action"] for item in jog_touches] == [
    "jog_press", "jog_heartbeat"]

# Closing a page is a release even when pointerup never reached the server.
with client.websocket_connect("/ws") as websocket:
    websocket.receive_json()
    websocket.send_json({"action": "jog_press", "client_id": "browser-2",
                         "session_id": "press-2", "target": "A",
                         "row": 1, "sign": -1})
    assert websocket.receive_json()["type"] == "jog_ack"
assert seen[-1] == {"action": "jog_release", "client_id": "browser-2",
                    "session_id": "press-2"}

secure = TestClient(create_app(hub, token="secret"))
assert secure.get("/api/state").status_code == 401
assert secure.get("/api/state", headers={
    "X-UR5Dual-Token": "secret",
}).status_code == 200
assert secure.get("/api/camera.jpg?access_token=secret").status_code == 200
with secure.websocket_connect("/ws/camera?access_token=secret") as camera_socket:
    assert camera_socket.receive_bytes() == b"jpeg-2"

print("web transport: ok")
