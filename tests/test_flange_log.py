"""
The flange log's arithmetic and its refusals — no robot, no sockets.

Three of these are guards rather than sums, and they are the reason the file
is worth anything. A sample taken while an arm was still moving, a sample
taken from the simulator, and a file holding samples from two different DH
tables all look exactly like good data once they are written down: six angles
and three coordinates, in range, in the right units. Nothing downstream could
ever tell. So they have to be refused at the prompt, and that is what is
checked here.

    python3 tests/test_flange_log.py
"""

import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.config import CellConfig
from ur5dual.geometry import ur_kinematics as UK
from ur5dual.sim_view import SimViewSender
from ur5dual.tools import flange_log as FL

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


class FakeRx:
    """Stands in for SimViewReceiver, which would want a real UDP port."""

    def __init__(self, joints, mode="real", active=True):
        self.joints, self.mode, self.active = joints, mode, active

    def poll(self):
        return self.joints if self.active and self.joints else None


def panel(joints, mode="real", active=True):
    feed = object.__new__(FL.PanelFeed)
    feed.receivers = [FakeRx(joints, mode, active)]
    feed._lock = threading.Lock()       # no drain thread; these fakes are static
    feed.dh = {a: UK.UR5_DH for a in ("A", "B")}
    feed.dh_source = {a: FL.PUBLISHED_DH for a in ("A", "B")}
    return feed


class ScriptedFeed:
    """Two readings in sequence, so `steady_joints` has something to compare."""

    source = "a script"

    def __init__(self, first, second):
        self.readings = [first, second]
        self.dh = {a: UK.UR5_DH for a in ("A", "B")}
        self.dh_source = {a: FL.PUBLISHED_DH for a in ("A", "B")}

    def joints(self):
        return self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]

    def complaint(self, missing):
        return "missing " + " ".join(missing)


Q_A = np.radians([-19.5, -98.4, 105.2, -96.3, -90.0, 0.1])
Q_B = np.radians([12.0, -80.0, -95.0, 4.0, 88.0, -30.0])
STILL = {"A": Q_A, "B": Q_B}

print("a pose is only a measurement once both arms have stopped")
joints, why = FL.steady_joints(ScriptedFeed(STILL, STILL), settle=0.0)
check("a standing cell is recorded", joints is not None and why is None)

drifted = {"A": Q_A + np.radians([0, 0, 0.5, 0, 0, 0]), "B": Q_B}
joints, why = FL.steady_joints(ScriptedFeed(STILL, drifted), settle=0.0)
check("half a degree of motion is refused", joints is None and "moved" in why, why or "")

joints, why = FL.steady_joints(ScriptedFeed({"A": Q_A}, {"A": Q_A}), settle=0.0)
check("one arm alone is refused", joints is None and "B" in why, why or "")

# encoder noise is not motion; refusing it would make the prompt unusable
noisy = {"A": Q_A + np.radians([0.0, 0.01, 0.0, -0.01, 0.0, 0.0]), "B": Q_B}
joints, why = FL.steady_joints(ScriptedFeed(STILL, noisy), settle=0.0)
check("0.01 deg of noise still records", joints is not None, why or "")

print("simulated joint angles are not measurements")
check("SIM is not read at all", panel(
    {"A": list(Q_A), "B": list(Q_B)}, mode="sim").joints() == {})
check("SIM says why", "SIM" in panel({}, mode="sim").complaint(["A"]))
check("REAL is read", set(panel({"A": list(Q_A), "B": list(Q_B)}).joints()) == {"A", "B"})
check("a panel that went away says so",
      "stopped publishing" in panel({}, active=False).complaint(["A", "B"]))

print("listening for the panel must never take the channel from RViz")
holder = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
holder.bind(("127.0.0.1", 0))
taken = holder.getsockname()[1]
# Linux hands every datagram to the *newest* SO_REUSEADDR socket on a port, so
# binding here would leave RViz with silence — which it reads as "no panel is
# running" and answers by opening its own feed to each robot.
check("a port RViz already holds is not free", not FL.channel_is_free("127.0.0.1", taken))
try:
    FL.PanelFeed(ports=(taken,))
    check("PanelFeed refuses to bind on top of it", False)
except OSError as e:
    check("PanelFeed refuses to bind on top of it", True, str(e))
holder.close()
check("a free port is free", FL.channel_is_free("127.0.0.1", taken))

print("a prompt nobody is typing at must not blind the feed")
# The failure this reproduces: the program sits in input() while somebody
# reads a gauge, nothing drains the socket, and a minute of datagrams fills
# the receive buffer. A full UDP buffer drops what is arriving and keeps what
# it has, so the newest frame available is the one from the moment it filled —
# and the panel, still publishing, reads as gone. It showed up as "the first
# reading works, then you have to restart the program".
probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
probe.bind(("127.0.0.1", 0))
spare = probe.getsockname()[1]
probe.close()

feed = FL.PanelFeed(ports=(spare,))
tx = SimViewSender(port=spare, tap_port=None)
stale = {"A": [0.0] * 6, "B": [0.0] * 6}
for _ in range(6000):                       # far more than the buffer holds
    tx.send(stale, mode="real")
time.sleep(0.4)                             # longer than a frame stays fresh
tx.send({"A": list(Q_A), "B": list(Q_B)}, mode="real")
time.sleep(0.15)

got = feed.joints()
check("the newest frame comes back, not the queue",
      bool(got) and np.allclose(got["A"], Q_A, atol=1e-9))
check("a panel that never stopped still reads as live", feed.live)
feed.close()
check("closing stops the thread that was draining it",
      feed._thread is None and not feed.receivers)
tx.close()

print("what a sample holds")
feed = ScriptedFeed(STILL, STILL)
sample = FL.build_sample(412.5, STILL, feed.dh)
check("the gauge reading, in mm", sample["gap_mm"] == 412.5)
check("six angles per arm, in degrees",
      len(sample["A"]["joints_deg"]) == 6
      and np.allclose(np.radians(sample["A"]["joints_deg"]), Q_A, atol=1e-6))
check("three coordinates per arm, in mm",
      np.allclose(np.array(sample["B"]["xyz_mm"]) / 1000.0,
                  UK.fk(Q_B)[:3, 3], atol=1e-6))
check("nothing else is recorded", set(sample) == {"t", "gap_mm", "A", "B"}
      and set(sample["A"]) == {"joints_deg", "xyz_mm"})
check("the record is the flange, not the TCP — 82.3 mm of tool length apart",
      abs(np.linalg.norm(np.array(sample["A"]["xyz_mm"]) / 1000.0
                         - UK.fk_tcp(Q_A, np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                                    [0, 0, 1, 0.0823], [0, 0, 0, 1]],
                                                   dtype=float))[:3, 3]) - 0.0823) < 1e-6)
check("it is JSON, with no numpy left in it",
      json.loads(json.dumps(sample))["A"] == sample["A"])

print("the gap cell.yaml implies, for comparison at the prompt")
cfg = CellConfig()
cfg.set_custom_mount()
cfg.arms["A"].set_base([0.0, 0.30, 1.0], [0.0, 0.0, 0.0])
cfg.arms["B"].set_base([0.0, -0.30, 1.0], [0.0, 0.0, 0.0])
gap = FL.configured_gap_mm(cfg, {"A": Q_A, "B": Q_A}, feed.dh)
check("parallel bases 600 mm apart, both arms held the same, is 600 mm",
      abs(gap - 600.0) < 1e-6, "%.4f mm" % gap)
check("it needs both arms", FL.configured_gap_mm(cfg, {"A": Q_A}, feed.dh) is None)

print("two DH tables must not end up in one file")
published = FL.meta_for(feed)
own = dict(published, kinematics={"A": "this robot's own calibration",
                                  "B": "this robot's own calibration"})
data = {"meta": {}, "samples": []}
check("an empty file takes either", FL.kinematics_clash(data, published) is None)
data["meta"] = published
data["samples"] = [sample]
check("the same table appends", FL.kinematics_clash(data, published) is None)
clash = FL.kinematics_clash(data, own)
# the message names no flag: two programs write these files now, and they
# spell "somewhere else" differently on the command line
check("a different table is refused",
      clash is not None and "another file" in clash, clash or "")

print("the file survives being written to")
work = tempfile.mkdtemp()
try:
    path = os.path.join(work, "flange_log.json")
    check("a missing file starts empty", FL.load_log(path)["samples"] == [])
    FL.save_log(path, {"meta": published, "samples": [sample]})
    again = FL.load_log(path)
    check("what went in comes back", again["samples"] == [sample]
          and again["meta"] == published)
    check("no temporary file left behind", os.listdir(work) == ["flange_log.json"])

    with open(path, "w") as f:
        json.dump({"A_1": [0, 0, 0, 0, 0, 0]}, f)
    try:
        FL.load_log(path)
        check("config/points.json is not a flange log", False)
    except SystemExit as e:
        check("config/points.json is not a flange log", "--file" in str(e))
finally:
    shutil.rmtree(work)

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
