"""
Which servo backend this cell can actually use.

The coordinated loop is the only part of the system that needs a controller
resident on the robot, and it is the part most likely to be refused: ur_rtde
uploads its own control program, and these two arms run different PolyScope
versions, one of them from 2018.

This starts each backend, checks it came up, and shuts it down again. No
target is ever sent, so nothing moves — and the joint angles are read before
and after to prove it.

    python3 tests/check_backends_online.py

Write the winner into config/cell.yaml as motion.backend.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.robot.backends import BackendError, RtdeBackend, UrScriptBackend
from ur5dual.config import ARM_IDS, CellConfig
from ur5dual.robot.transport import read_state

MOVED = 0.002        # rad; joint noise sits well under this


def joints(ip):
    return np.array(read_state(ip)["q_actual"])


def try_backend(name, factory, ip, motion_cfg):
    before = joints(ip)
    backend = factory(ip, motion_cfg)
    try:
        backend.start()
    except (BackendError, OSError) as e:
        return False, str(e).strip().splitlines()[0][:110], 0.0
    except Exception as e:                       # ur_rtde raises bare RuntimeError
        return False, "%s: %s" % (type(e).__name__, str(e)[:90]), 0.0
    try:
        time.sleep(1.0)                          # let it settle, send nothing
        after = joints(ip)
        return True, "up", float(np.max(np.abs(after - before)))
    finally:
        backend.shutdown()
        time.sleep(0.5)


config = CellConfig.load()
print("configured backend: %s\n" % config.motion["backend"])

results = {}
for arm_id in ARM_IDS:
    arm = config.arms[arm_id]
    if not arm.enabled:
        continue
    state = read_state(arm.ip)
    print("arm %s  %s   packet %d bytes" % (arm_id, arm.ip, state["packet_size"]))
    for label, factory in (("rtde", RtdeBackend), ("urscript", UrScriptBackend)):
        ok, note, drift = try_backend(label, factory, arm.ip, config.motion)
        results.setdefault(label, []).append(ok)
        print("   %-9s %-4s  %s%s" % (
            label, "OK" if ok else "FAIL", note,
            "   moved %.4f rad" % drift if ok and drift > MOVED else
            "   (nothing moved)" if ok else ""))
    print()

usable = [name for name, oks in results.items() if oks and all(oks)]
if usable:
    print("usable on every enabled arm: %s" % ", ".join(usable))
    if config.motion["backend"] not in usable:
        print("\n>>> config/cell.yaml has motion.backend: %s, which is NOT usable."
              % config.motion["backend"])
        print(">>> change it to:  backend: %s" % usable[0])
    sys.exit(0)

print("no backend came up on every arm — coordinated motion cannot run yet")
sys.exit(1)
