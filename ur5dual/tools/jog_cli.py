#!/usr/bin/env python3
"""
Keyboard jog for one arm, for a terminal with no display — Cartesian
translation and rotation in the base or tool frame, plus per-joint jogging.

    scripts/ur5dual-jog                 # arm A, step mode, base frame
    scripts/ur5dual-jog --arm B
    scripts/ur5dual-jog --mode cont     # continuous hold-to-move
    scripts/ur5dual-jog --frame joint   # the way out of a singularity

The GUI is the better tool when there is a screen: it drives both arms, and
its buttons give real press/release events. This exists for ssh sessions.

Two motion modes:

  STEP  each keypress commands one movel of a fixed increment. Predictable,
        nothing moves unless you press a key. Hold a key and the terminal's
        auto-repeat turns it into steady motion.

  CONT  speedl while the key is held. Smoother, but the terminal only tells
        us a key is *down* by repeating it, and auto-repeat has a start-up
        delay — so motion is held alive for --grace seconds after the last
        repeat. Releasing a key therefore coasts for up to that long before
        the stop is sent. Keep the speed low and SPACE is always an
        immediate stop.

  (gui_ur5.py has no such limit: real button press/release events let it
  stop the instant you let go.)

THE ROBOT MOVES. Clear the workspace and keep the E-stop within reach.
"""

import argparse
import math
import os
import select
import sys
import termios
import time
import tty

import numpy as np

from ..config import DEFAULT_PATH, CellConfig
from ..robot.motion import (
    AXIS_NAMES, CONT_ANG, CONT_LIN, FRAMES, JOINT_NAMES, REFRESH,
    ROBOT_MODE_NAMES, SAFETY_NAMES, STEP_ANG, STEP_LIN, ArmMotion,
)
from ..robot.transport import Dashboard, ScriptSender, StateStream

# ---------------------------------------------------------------- key map --
# axis index: 0,1,2 = X,Y,Z translation   3,4,5 = RX,RY,RZ rotation
# in the joint frame the same twelve keys drive J1..J6.
KEYS = {
    "a": (0, -1), "d": (0, +1),
    "s": (1, -1), "w": (1, +1),
    "f": (2, -1), "r": (2, +1),
    "j": (3, -1), "l": (3, +1),
    "k": (4, -1), "i": (4, +1),
    "u": (5, -1), "o": (5, +1),
    # arrow keys / page keys as an alternative for translation
    "\x1b[D": (0, -1), "\x1b[C": (0, +1),
    "\x1b[B": (1, -1), "\x1b[A": (1, +1),
    "\x1b[6~": (2, -1), "\x1b[5~": (2, +1),
}


def read_keys(fd):
    """Return the list of key tokens currently buffered on stdin."""
    data = os.read(fd, 256).decode("latin-1")
    tokens, i = [], 0
    while i < len(data):
        if data[i] == "\x1b" and i + 1 < len(data) and data[i + 1] == "[":
            j = i + 2
            while j < len(data) and not ("A" <= data[j] <= "Z" or data[j] == "~"):
                j += 1
            tokens.append(data[i:j + 1])
            i = j + 1
        else:
            tokens.append(data[i])
            i += 1
    return tokens


class Jogger:
    def __init__(self, ip, mode, frame, preset, grace):
        self.mode = mode          # "step" | "cont"
        self.frame = frame        # "base" | "tool" | "joint"
        self.preset = preset
        self.grace = grace

        self.stream = StateStream(ip)
        self.sender = ScriptSender(ip)
        self.motion = ArmMotion(self.sender, self.stream)

        self.cmd = np.zeros(6)    # current commanded direction (cont mode)
        self.last_key_time = 0.0
        self.last_send = 0.0
        self.moving = False
        self.stepped = False
        self.message = ""

    def handle_key(self, key, now):
        if key in KEYS:
            axis, sign = KEYS[key]
            if self.mode == "step":
                self.stepped = True
                if now - self.last_send >= self.motion.step_period(self.frame,
                                                                   self.preset):
                    self.motion.step(self.frame, axis, sign, self.preset)
                    self.last_send = now
            else:
                self.cmd[:] = 0
                self.cmd[axis] = sign
                self.last_key_time = now
            return True

        if key == " ":
            self.halt()
            self.message = "stopped"
        elif key == "t":
            self.halt()
            self.frame = FRAMES[(FRAMES.index(self.frame) + 1) % len(FRAMES)]
        elif key == "m":
            self.halt()
            self.mode = "cont" if self.mode == "step" else "step"
        elif key in "1234":
            self.preset = int(key) - 1
        elif key == "-":
            self.preset = max(0, self.preset - 1)
        elif key in "=+":
            self.preset = min(len(STEP_LIN) - 1, self.preset + 1)
        elif key in ("x", "\x1b", "\x03", "\x04"):
            return False
        return True

    def halt(self):
        self.cmd[:] = 0
        self.moving = False
        self.motion.halt(self.frame)

    def run(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setraw(fd)
        print(HELP.replace("\n", "\r\n"))
        try:
            while True:
                now = time.time()
                if select.select([sys.stdin], [], [], 0.03)[0]:
                    self.stepped = False
                    for key in read_keys(fd):
                        if not self.handle_key(key, now):
                            return
                        if self.stepped:
                            break   # one step per batch; auto-repeat paces us

                if self.mode == "cont":
                    active = self.cmd.any() and (now - self.last_key_time) < self.grace
                    if active:
                        if now - self.last_send >= REFRESH:
                            self.motion.speed(self.frame, self.cmd, self.preset)
                            self.last_send = now
                            self.moving = True
                    elif self.moving:
                        self.halt()

                self.draw()
        finally:
            try:
                self.halt()
            except OSError:
                pass
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            self.stream.close()
            self.sender.close()
            print("\r\nstopped, disconnected.")

    def draw(self):
        st = self.stream.latest()
        p = st["tcp_pose"]
        q = st["q_actual"]
        safety = SAFETY_NAMES.get(int(st["safety_mode"]), "?")
        rmode = ROBOT_MODE_NAMES.get(int(st["robot_mode"]), "?")

        if self.frame == "joint":
            amount = ("step %14.1f deg" % math.degrees(STEP_ANG[self.preset])
                      if self.mode == "step" else
                      "speed %12.1f deg/s" % math.degrees(CONT_ANG[self.preset]))
        elif self.mode == "step":
            amount = "step %5.1f mm / %4.1f deg" % (
                STEP_LIN[self.preset] * 1000, math.degrees(STEP_ANG[self.preset]))
        else:
            amount = "speed %5.0f mm/s / %4.1f deg/s" % (
                CONT_LIN[self.preset] * 1000, math.degrees(CONT_ANG[self.preset]))

        names = JOINT_NAMES if self.frame == "joint" else AXIS_NAMES
        active = [names[i] + ("+" if v > 0 else "-")
                  for i, v in enumerate(self.cmd) if v]

        lines = [
            "mode %-4s  frame %-5s  %-30s  [%d]" % (
                self.mode.upper(), self.frame.upper(), amount, self.preset + 1),
            "TCP  x %8.4f  y %8.4f  z %8.4f   rx %7.4f  ry %7.4f  rz %7.4f" % tuple(p),
            "JNT  " + "  ".join("%7.2f" % math.degrees(v) for v in q) + "  deg",
            "%-10s %-16s %-12s %s" % (
                rmode, safety, " ".join(active) or "-", self.message),
        ]
        sys.stdout.write("\r" + "\r\n".join(l.ljust(88) for l in lines))
        sys.stdout.write("\x1b[%dA\r" % (len(lines) - 1))
        sys.stdout.flush()


HELP = """
  cartesian   X  a / d      Y  s / w      Z  f / r      (or arrows, PgUp/PgDn)
             RX  j / l     RY  k / i     RZ  u / o
  joint      J1  a / d     J2  s / w     J3  f / r
             J4  j / l     J5  k / i     J6  u / o
  size        1 2 3 4  presets     - / =  smaller / larger
  toggles     t  base -> tool -> joint    m  step <-> continuous
  SPACE  stop now          x or ESC  quit
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("A", "B"), default="A",
                    help="which arm from config/cell.yaml (default A)")
    ap.add_argument("--ip", default=None,
                    help="override the address from the config")
    ap.add_argument("--config", default=DEFAULT_PATH)
    ap.add_argument("--mode", choices=["step", "cont"], default="step")
    ap.add_argument("--frame", choices=FRAMES, default="base")
    ap.add_argument("--preset", type=int, default=2, choices=[1, 2, 3, 4],
                    help="initial step/speed preset (default 2)")
    ap.add_argument("--grace", type=float, default=0.45,
                    help="continuous mode: seconds of missing key-repeat "
                         "before motion is stopped (default 0.45)")
    args = ap.parse_args()

    if not sys.stdin.isatty():
        raise SystemExit("jog.py needs an interactive terminal.")

    ip = args.ip or CellConfig.load(args.config).arms[args.arm].ip
    with Dashboard(ip) as db:
        rmode, safety = db.robot_mode(), db.safety_mode()
    if "RUNNING" not in rmode:
        raise SystemExit("%s — power on and release the brakes first." % rmode)
    if "NORMAL" not in safety and "REDUCED" not in safety:
        raise SystemExit("%s — clear it on the teach pendant first." % safety)

    Jogger(ip, args.mode, args.frame, args.preset - 1, args.grace).run()


if __name__ == "__main__":
    main()
