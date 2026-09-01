"""Small FastAPI host for the shared desktop/browser control surface.

The HTTP thread never touches Qt or a robot. Commands cross the callback
boundary into the Qt main thread; state and camera JPEGs cross back as plain
immutable data. This keeps one owner for the hardware while allowing any
number of browser displays to observe the same state.
"""

import asyncio
import copy
import os
import secrets
import threading
import time


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class WebHub:
    """Thread-safe state store shared by Qt and the ASGI server."""

    def __init__(self, command=None, jog_touch=None):
        self.command = command
        self.jog_touch = jog_touch
        self._lock = threading.Lock()
        self._state = {"revision": 0, "ready": False}
        self._camera = None
        self._camera_revision = 0
        self._camera_clients = 0
        self._revision = 0

    def publish(self, state, camera=None):
        with self._lock:
            self._revision += 1
            value = copy.deepcopy(state)
            value["revision"] = self._revision
            value["ready"] = True
            self._state = value
            if camera is not None:
                self._camera = bytes(camera)
                self._camera_revision += 1
        return self._revision

    def publish_camera(self, camera):
        """Publish a frame without rebuilding or revising the control state."""
        if camera is None:
            return self._camera_revision
        with self._lock:
            self._camera = bytes(camera)
            self._camera_revision += 1
            return self._camera_revision

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._state)

    def camera(self):
        with self._lock:
            return self._camera

    def camera_snapshot(self):
        """Return one immutable frame and its independent stream revision."""
        with self._lock:
            return self._camera_revision, self._camera

    def camera_subscribe(self, connected):
        with self._lock:
            self._camera_clients = max(
                0, self._camera_clients + (1 if connected else -1))

    def camera_streaming(self):
        with self._lock:
            return self._camera_clients > 0

    def dispatch(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("command must be a JSON object")
        if not payload.get("action"):
            raise ValueError("command needs an action")
        if self.command is None:
            raise RuntimeError("the desktop command bridge is not ready")
        return self.command(payload)

    def touch_jog(self, payload):
        """Record stream liveness without queueing work on the UI thread."""
        if self.jog_touch is not None:
            self.jog_touch(payload)


def create_app(hub, static_dir=STATIC_DIR, token=None):
    """Build the ASGI app lazily so the desktop can run without web extras."""
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket
        from fastapi.responses import FileResponse, Response
        from fastapi.staticfiles import StaticFiles
        from starlette.websockets import WebSocketDisconnect
    except ImportError as exc:
        raise RuntimeError(
            "web UI needs: python3 -m pip install -r requirements-web.txt"
        ) from exc

    app = FastAPI(title="Dual UR5 control", docs_url="/api/docs",
                  redoc_url=None)

    def authorize(x_ur5dual_token: str = Header(default=None),
                  access_token: str = Query(default=None)):
        supplied = x_ur5dual_token or access_token
        if token and (not supplied or not secrets.compare_digest(token, supplied)):
            raise HTTPException(status_code=401, detail="invalid web access token")

    @app.get("/api/state", dependencies=[Depends(authorize)])
    def state():
        return hub.snapshot()

    @app.post("/api/command", dependencies=[Depends(authorize)])
    def command(payload: dict):
        try:
            result = hub.dispatch(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        # The command bridge publishes after applying the action. Returning
        # that exact snapshot lets its button update immediately instead of
        # waiting for the independently paced observer WebSocket.
        return {"ok": True, "result": result, "state": hub.snapshot()}

    @app.get("/api/camera.jpg", dependencies=[Depends(authorize)])
    def camera():
        picture = hub.camera()
        if picture is None:
            return Response(status_code=204)
        return Response(picture, media_type="image/jpeg", headers={
            "Cache-Control": "no-store, max-age=0",
        })

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket):
        supplied = websocket.query_params.get("access_token")
        if token and (not supplied or not secrets.compare_digest(token, supplied)):
            await websocket.close(code=1008, reason="invalid web access token")
            return
        await websocket.accept()
        revision = -1
        send_lock = asyncio.Lock()
        commands = asyncio.Queue()
        active_jog = None

        async def send_json(value):
            async with send_lock:
                await websocket.send_json(value)

        async def publish_state():
            nonlocal revision
            while True:
                current = hub.snapshot()
                if current.get("revision") != revision:
                    revision = current.get("revision")
                    await send_json(current)
                await asyncio.sleep(0.08)

        async def run_commands():
            while True:
                payload = await commands.get()
                try:
                    result = await asyncio.to_thread(hub.dispatch, payload)
                    await send_json({"type": "jog_ack",
                                     "action": payload.get("action"),
                                     "session_id": payload.get("session_id"),
                                     "result": result})
                except Exception as exc:
                    try:
                        await send_json({"type": "jog_error",
                                         "action": payload.get("action"),
                                         "session_id": payload.get("session_id"),
                                         "detail": str(exc)})
                    except Exception:
                        pass
                finally:
                    commands.task_done()

        publisher = asyncio.create_task(publish_state())
        worker = asyncio.create_task(run_commands())
        try:
            while True:
                payload = await websocket.receive_json()
                if not isinstance(payload, dict):
                    await send_json({"type": "jog_error",
                                     "detail": "jog message must be an object"})
                    continue
                action = payload.get("action")
                if action not in ("jog_press", "jog_heartbeat", "jog_release"):
                    await send_json({"type": "jog_error", "action": action,
                                     "detail": "websocket accepts jog messages only"})
                    continue
                client = str(payload.get("client_id") or "")
                session = str(payload.get("session_id") or client)
                if not client or not session:
                    await send_json({"type": "jog_error", "action": action,
                                     "detail": "jog message needs client_id and session_id"})
                    continue
                payload["client_id"] = client
                payload["session_id"] = session
                if action in ("jog_press", "jog_heartbeat"):
                    hub.touch_jog(payload)
                if action == "jog_press":
                    active_jog = (client, session)
                    await commands.put(payload)
                elif action == "jog_release":
                    await commands.put(payload)
                    if active_jog == (client, session):
                        active_jog = None
                # A heartbeat is deliberately not queued. Its arrival time is
                # all the UI timer needs in order to keep refreshing motion.
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            publisher.cancel()
            if active_jog is not None:
                client, session = active_jog
                await commands.put({"action": "jog_release",
                                    "client_id": client,
                                    "session_id": session})
            try:
                await asyncio.wait_for(commands.join(), timeout=3.5)
            except asyncio.TimeoutError:
                pass
            worker.cancel()
            await asyncio.gather(publisher, worker, return_exceptions=True)

    @app.websocket("/ws/camera")
    async def camera_websocket(websocket: WebSocket):
        """Binary JPEG stream, isolated from state and safety commands."""
        supplied = websocket.query_params.get("access_token")
        if token and (not supplied or not secrets.compare_digest(token, supplied)):
            await websocket.close(code=1008, reason="invalid web access token")
            return
        await websocket.accept()
        revision = -1
        hub.camera_subscribe(True)
        try:
            while True:
                current_revision, picture = hub.camera_snapshot()
                if picture is not None and current_revision != revision:
                    revision = current_revision
                    await websocket.send_bytes(picture)
                await asyncio.sleep(0.03)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.camera_subscribe(False)

    app.mount("/assets", StaticFiles(directory=static_dir), name="assets")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(static_dir, "index.html"), headers={
            "Cache-Control": "no-store, max-age=0",
        })

    return app


class WebServer:
    """Run uvicorn beside Qt and stop it with the desktop window."""

    def __init__(self, hub, host="127.0.0.1", port=8765, token=None):
        self.hub = hub
        self.host = host
        self.port = int(port)
        self.token = token
        self.server = None
        self.thread = None

    def start(self):
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError(
                "web UI needs: python3 -m pip install -r requirements-web.txt"
            ) from exc
        config = uvicorn.Config(create_app(self.hub, token=self.token), host=self.host,
                                port=self.port, log_level="warning",
                                access_log=False)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run,
                                       name="ur5dual-web", daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 3.0
        while not self.server.started and self.thread.is_alive():
            if time.monotonic() >= deadline:
                raise RuntimeError("web server did not start")
            time.sleep(0.01)
        if not self.thread.is_alive():
            raise RuntimeError("web server stopped during startup")
        return "http://%s:%d" % (self.host, self.port)

    def stop(self):
        if self.server is not None:
            self.server.should_exit = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
