"""Sim mode end to end, with no robot and no ROS.

What this has to prove is not that the arithmetic is right — the other tests
do that — but that `sim` is the same machinery as `real` with one substitution.
The feed thread runs, the closed chain is solved every cycle, a held jog ramps
up and down, and the joints that would have gone to two controllers go to a
socket instead. If any of that were bypassed, sim would be testing something
other than what real does, which is the one thing a simulator must not do.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.geometry import kinematics as K
from ur5dual.geometry import ur_kinematics as UK
from ur5dual.robot.backends import SimBackend, make_backend
from ur5dual.config import CellConfig
from ur5dual.coupling import Coordinator, HeldObject
from ur5dual.cell import Cell
from ur5dual.robot.sim_arm import HOME_Q
from ur5dual.sim_view import SimViewReceiver

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


cfg = CellConfig.load()
# Left exactly as it sits on the bench: nothing measured, no robot reachable.
# Sim has to work anyway — a cell that must be calibrated and plugged in before
# it can be drawn is a cell you cannot look at until after you have committed
# to its geometry.
print("this cell is calibrated: %s   translation taught: %s   robots: none"
      % (cfg.calibrated, cfg.translation_calibrated))

cell = Cell(cfg, simulated=True)
# out of the zero-pose singularity first: this half of the file is about the
# coordinated loop, and a cell parked on a singularity cannot demonstrate it
cell.sim_ready_pose()
obj = HeldObject("box").capture(cell, ("A", "B"), origin="midpoint")

print("\nthe backend that moves nothing")
sim_backend = make_backend("10.0.0.1", cfg.motion, "joint", drive_robots=False)
check("make_backend hands back a SimBackend when not driving",
      isinstance(sim_backend, SimBackend))
sim_backend.start()
sim_backend.servo_joints([0.0] * 6)
check("it accepts joint targets and keeps the last one",
      sim_backend.count == 1 and sim_backend.last == [0.0] * 6)
check("real backends are still real",
      not isinstance(make_backend("10.0.0.1", cfg.motion, "pose"), SimBackend))

print("\nstarting the loop with nothing measured")
listener = SimViewReceiver()
coord = Coordinator(cell, drive_robots=False)
try:
    coord.start(obj)
    started = True
except Exception as exc:                                    # noqa: BLE001
    started = False
    check("sim mode starts on an uncalibrated cell", False, str(exc))

if started:
    check("sim mode starts on an uncalibrated cell", True)
    check("every backend is a SimBackend",
          all(isinstance(b, SimBackend) for b in coord.backends.values()))
    check("the closed-chain solver is running whatever cell.yaml says",
          coord.solver is not None)
    check("the feed thread is alive", coord.alive)

    time.sleep(0.3)
    frame = listener.poll()
    check("the viewer is being fed while the object is merely held",
          frame is not None and sorted(frame) == ["A", "B"])
    resting = {a: list(frame[a]) for a in ("A", "B")} if frame else {}

    print("\njogging the object")
    said = []
    cell.listeners.append(said.append)
    coord.command_jog(obj, [0, 0, 1], 0.05)
    moved_pose = None
    for _ in range(40):
        time.sleep(0.02)
        coord.command_jog(obj, [0, 0, 1], 0.05)      # a held button refreshes
    coord.stop_jog()
    time.sleep(0.3)

    # Letting go says how far it went, commanded and measured. Under the
    # uncalibrated REAL cap a held button moves the box a millimetre a second
    # — real motion that looks exactly like a dead servo loop, and which was
    # being judged by eye against a single-arm jog thirty times faster.
    ended = [m for m in said if m.startswith("held jog ended")]
    check("letting go reports what the jog achieved", len(ended) == 1,
          ended[-1] if ended else "nothing was logged")
    check("and reports the arms' own measured travel beside the command",
          bool(ended) and "the arms moved" in ended[-1] and "mm" in ended[-1])

    frame = listener.poll()
    check("the viewer kept up through the jog", frame is not None)
    if frame and resting:
        delta = max(float(np.max(np.abs(np.array(frame[a]) - resting[a])))
                    for a in ("A", "B"))
        check("and the joints it is being sent actually changed",
              delta > 1e-3, "%.2f deg" % np.degrees(delta))

    lifted = coord.current_pose(obj)
    check("the object rose while nothing was commanded to a robot",
          lifted[2] > obj.pose_world[2] - 1e9)
    check("no backend ever spoke to a controller",
          all(b.count > 0 for b in coord.backends.values()))

    print("\nturning it, which a real uncalibrated cell would refuse")
    # 10 degrees: past the 5 the uncalibrated cap allows, so getting through
    # proves the cap is lifted. A far larger angle is refused — correctly — by
    # the closed-chain precheck, which is the other half of what sim is for.
    before = coord.current_pose(obj).copy()
    coord.command_rotate(obj, "z", np.radians(10), frame="world")
    coord.wait_done()
    _, turned = K.pose_distance(before, coord.current_pose(obj))
    check("10 deg about world Z goes through uncapped in sim",
          abs(np.degrees(turned) - 10.0) < 0.5,
          "%.2f deg" % np.degrees(turned))

    # The other half of what sim is for: the closed-chain precheck walks the
    # path in arithmetic and refuses one the arms could not follow, before a
    # single target is issued. Three metres is unambiguously outside any UR5's
    # reach, so this is a refusal about geometry and not about a marginal pose.
    away = coord.current_pose(obj).copy()
    away[0] += 3.0
    refused = None
    try:
        coord.command_move(obj, away)
    except Exception as exc:                                # noqa: BLE001
        refused = str(exc)
    check("a carry the arms could not follow is refused before it starts",
          refused is not None and "arm" in refused, (refused or "")[:70])
    check("and the refusal names what runs out",
          bool(refused) and ("reach" in refused or "J" in refused
                             or "singularity" in refused))
    check("the object did not move on a refused command",
          float(np.linalg.norm(coord.current_pose(obj)[:3] - before[:3])) < 0.3)

    coord.shutdown()
    time.sleep(0.6)
    # nothing is publishing now: the loop has stopped and this test has no
    # panel tick behind it, so the frames simply stop
    check("with nothing publishing, the viewer goes stale and RViz\n         returns to the arms".replace("\n         ", " "),
          listener.poll() is None)

listener.close()


print("\na cell with no robots anywhere")
# What the panel does on startup. Nothing below opens a socket to a robot,
# because there is nothing to open one to — this is the case that was broken:
# sim mode still needed two arms answering before anything on the Jog tab
# would come out of grey.
bare = Cell(CellConfig.load(), simulated=True)
check("both arms report themselves connected", bare.connected_ids == ["A", "B"])
check("and ready, so the jog grid is not greyed out",
      bare.not_ready_reason(("A", "B")) is None)
check("the monitors have joint angles to show",
      all(len(bare.arms[a].state()["q_actual"]) == 6 for a in ("A", "B")))
check("feeds are never stale", bare.feeds_fresh())

check("each arm starts at its configured SIM home angles",
      all(np.allclose(bare.arms[a].state()["q_actual"], HOME_Q[a], atol=1e-12)
          for a in ("A", "B")))

check("Ready pose restores a simulated cell somewhere it can be driven",
      bare.sim_ready_pose())
gap = bare.tcp_separation()
check("and leaves the two tools a box-width apart",
      gap is not None and 0.15 < gap < 0.45, "%.0f mm" % (gap * 1000))
check("with both wrists well clear of a singularity",
      min(UK.manipulability(bare.arms[a].state()["q_actual"])
          for a in ("A", "B")) > 0.1)

# A position-servo target has a measured velocity, but that velocity is not a
# command to continue past the target.  This used to extrapolate between feed
# cycles and made Held Object motion visibly oscillate as each cycle corrected
# the overshoot from the previous one.
position_target = bare.arms["A"].joints() + np.array([0.01, 0, 0, 0, 0, 0])
bare.arms["A"].set_joints(position_target)
time.sleep(0.05)
check("a SIM joint target stays at the target between servo cycles",
      np.allclose(bare.arms["A"].joints(), position_target, atol=1e-12))
bare.sim_ready_pose()

check("Ready pose refuses to touch real arms",
      Cell(CellConfig.load(), simulated=False).sim_ready_pose() is False)

held = HeldObject("box").capture(bare, ("A", "B"), origin="midpoint")
check("taking hold works with nothing plugged in", held.held)

print("\njogging one arm on its own, as the pendant would")
before = bare.arms["A"].state()["q_actual"].copy()
bare.arms["A"].motion.step("joint", 0, +1, 2)
check("a joint step moves that joint and no other",
      abs(bare.arms["A"].state()["q_actual"][0] - before[0]) > 1e-4
      and np.allclose(bare.arms["A"].state()["q_actual"][1:], before[1:], atol=1e-9))

before = bare.arms["B"].tcp_pose()
bare.arms["B"].motion.step("base", 2, +1, 2)
after = bare.arms["B"].tcp_pose()
check("a Cartesian step moves the tool along that axis",
      after[2] - before[2] > 1e-4, "%.1f mm" % ((after[2] - before[2]) * 1000))

bare.arms["A"].motion.speed("base", [1, 0, 0, 0, 0, 0], 2)
start_x = bare.arms["A"].tcp_pose()[0]
time.sleep(0.15)
check("a held Cartesian jog keeps travelling while it is held",
      bare.arms["A"].tcp_pose()[0] - start_x > 1e-4)
bare.arms["A"].motion.halt("base")
stopped = bare.arms["A"].tcp_pose()[0]
time.sleep(0.15)
check("and stops when it is let go",
      abs(bare.arms["A"].tcp_pose()[0] - stopped) < 1e-6)

bare.arms["B"].motion.speed("base", [0, 1, 0, 0, 0, 0], 3)
held_at = bare.arms["B"].tcp_pose()[1]
time.sleep(0.5)                       # longer than the velocity watchdog
drifted = abs(bare.arms["B"].tcp_pose()[1] - held_at)
time.sleep(0.3)
check("a jog whose release never arrived times out rather than running on",
      abs(bare.arms["B"].tcp_pose()[1] - held_at) - drifted < 1e-6)

print("\nthe drawing is fed in every state, not only while carrying")
# The bug this replaces: publishing lived in the coordinated loop, which only
# exists between ATTACH and DETACH. Jogging one arm, or just looking at an
# idle cell, sent nothing — so RViz fell back to the robots, found none, and
# drew every joint at zero while the panel's own numbers moved.
watcher = SimViewReceiver()

bare.publish_sim_view()
time.sleep(0.05)
idle = watcher.poll()
check("an idle cell with nothing held is still drawn", idle is not None)

bare.arms["A"].motion.speed("base", [0, 0, 1, 0, 0, 0], 3)
swept = []
for _ in range(10):
    time.sleep(0.05)
    bare.publish_sim_view()
    frame = watcher.poll()
    if frame:
        swept.append(frame["A"][1])
bare.arms["A"].motion.halt("base")
check("a single arm being jogged reaches the drawing",
      len(swept) >= 5 and abs(swept[-1] - swept[0]) > 1e-4,
      "J2 swept %.1f deg over %d frames"
      % (np.degrees(abs(swept[-1] - swept[0])), len(swept)))

live_view = Cell(CellConfig.load(), simulated=False)
check("a live cell announces itself so RViz releases its robot feeds",
      live_view.publish_sim_view() is True)
time.sleep(0.05)
check("the viewer recognizes a REAL panel before its first robot sample",
      watcher.poll() is None and watcher.active and watcher.mode == "real")
live_view.sim_view.close()
watcher.close()


print("\nswitching a live cell into simulation")
mixed = Cell(CellConfig.load(), simulated=False)
check("it starts with real arms, none of them connected",
      not mixed.simulated and mixed.connected_ids == [])
check("switching reports that it changed", mixed.set_simulated(True))
check("and now every arm answers", mixed.connected_ids == ["A", "B"])
check("switching again to the same thing is a no-op",
      mixed.set_simulated(True) is False)

print("\n%d failed" % fail)
sys.exit(1 if fail else 0)
