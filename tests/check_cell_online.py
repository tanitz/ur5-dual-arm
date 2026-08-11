"""
Live check against the real cell — the one test that needs both robots.

Read-only apart from a textmsg the robot echoes back, which is how the
command path gets proven without commanding motion. Run it after wiring,
after a firmware change, or whenever the arms behave as if they cannot hear
you.

    python3 tests/check_cell_online.py
"""

import math
import os
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.robot import motion as M
from ur5dual.cell import Cell
from ur5dual.config import ARM_IDS, CellConfig
from ur5dual.robot.transport import send_script

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


def script_echoes(ip, timeout=4.0):
    """Ask the robot to say something and listen for it on the primary
    interface. Proves the script ran, not merely that the socket accepted it."""
    mark = "URDUAL_%d" % int(time.time() * 1000)
    listener = socket.create_connection((ip, 30001), timeout=5)
    listener.settimeout(0.2)
    stop = time.time() + 0.6
    while time.time() < stop:
        try:
            listener.recv(65536)
        except socket.timeout:
            break
    try:
        send_script(ip, 'def link_check():\n  textmsg("%s")\nend\n' % mark)
        buf, deadline = b"", time.time() + timeout
        while time.time() < deadline:
            try:
                buf += listener.recv(65536)
            except socket.timeout:
                continue
            if mark.encode() in buf:
                return True
        return False
    finally:
        listener.close()


config = CellConfig.load()
cell = Cell(config)
cell.listeners.append(lambda text: print("       . %s" % text))

print("configured cell")
print("  mount %s   column %.2f m   spacing %.2f m   tilt %.0f deg"
      % (config.mount["style"], config.mount["column_height"],
         config.mount["spacing"], config.mount["tilt_deg"]))
for arm_id in ARM_IDS:
    out = config.base_axis_outward(arm_id)
    check("arm %s reaches away from the mast" % arm_id, out > 0,
          "outward %+.3f" % out)

print("connecting")
results = cell.connect(ARM_IDS)
for arm_id in ARM_IDS:
    check("arm %s connected" % arm_id, results.get(arm_id, False),
          config.arms[arm_id].ip)

online = [a for a in ARM_IDS if cell.arms[a].connected]
if not online:
    print("\nno arms reachable — nothing further to check")
    sys.exit(1)

print("per-arm state")
for arm_id in online:
    arm = cell.arms[arm_id]
    state = arm.state()
    print("  arm %s  %s" % (arm_id, arm.polyscope))
    check("arm %s is RUNNING and NORMAL" % arm_id, arm.ready(),
          "%s / %s" % (arm.robot_mode(), arm.safety_mode()))
    check("arm %s real-time packet parses" % arm_id,
          state["packet_size"] >= 1108
          and all(abs(q) < 2 * math.pi for q in state["q_actual"]),
          "%d bytes" % state["packet_size"])
    check("arm %s digital I/O readable" % arm_id,
          state["digital_out_bits"] is not None,
          "DI %s DO %s" % (state["digital_in_bits"], state["digital_out_bits"]))
    check("arm %s TCP offset known" % arm_id, arm.tcp_offset is not None,
          str(arm.tcp_offset))
    # world <-> base must be exactly reversible or every two-arm number is junk
    pose = arm.tcp_pose_world()
    err = float(np.linalg.norm(arm.base_to_world(arm.world_to_base(pose)) - pose))
    check("arm %s world/base round trip" % arm_id, err < 1e-9, "%.2e" % err)
    print("       TCP in world  %s mm" % np.round(pose[:3] * 1000, 1))

print("command path (textmsg echo — no motion)")
for arm_id in online:
    check("arm %s executes URScript" % arm_id,
          script_echoes(config.arms[arm_id].ip))

if len(online) == 2:
    print("pair geometry")
    separation = cell.tcp_separation()
    check("TCP separation is a sane cell-sized number",
          separation is not None and 0.0 < separation < 3.0,
          "%.1f mm" % (separation * 1000))
    relative = cell.relative_transform()
    check("A-to-B transform is a proper rigid transform",
          abs(np.linalg.det(relative[:3, :3]) - 1.0) < 1e-9)
    check("force readings are plausible",
          all(cell.arms[a].tcp_force_magnitude() < 200 for a in online),
          " ".join("%s %.0f N" % (a, cell.arms[a].tcp_force_magnitude())
                   for a in online))
else:
    print("pair geometry")
    print("  -- skipped, only arm %s is online" % online[0])

cell.disconnect()
print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
