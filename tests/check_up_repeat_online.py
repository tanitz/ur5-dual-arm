"""
Is the world error the cell leaning, or the accelerometer lying?

`check_up_online.py` reads gravity once, in whatever posture the arms happen
to be standing in, and turns the answer into a correction that re-levels the
entire cell. One reading has nothing to check itself against, so it cannot
tell a real lean from a biased sensor — and a flange accelerometer reading
10.0 m/s2 where it should read 9.81 is telling you it has a bias.

This says which it is, and needs no jig. Gravity in an arm's own base frame
does not depend on where that arm's joints are: same base, same building, same
vertical. Read it in several genuinely different postures and the spread
across those readings is the error bar on the measurement — measured on this
machine, today, rather than assumed.

  up_in_base repeats to a few tenths of a degree
      the reading is sound. A world error that survives is the cell really
      hanging off vertical, and check_up_online.py --apply is the fix for it.

  up_in_base swings by degrees between postures
      the accelerometer is the limit, not the mounting. A correction that size
      re-levels the world onto sensor noise, and every world jog afterwards
      inherits it.

NOTHING IS COMMANDED TO MOVE. Between samples you move the arms yourself —
pendant, freedrive, or the jog panel. Wrist joints alone will do; what matters
is that the accelerometer points a different way each time, so spread the
postures as widely as the cell allows.

    python3 tests/check_up_repeat_online.py            # 6 postures
    python3 tests/check_up_repeat_online.py -n 10
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.cell import Cell                                    # noqa: E402
from ur5dual.config import CellConfig                            # noqa: E402

# Reading to a few tenths means the geometry is what is being measured; a
# couple of degrees of scatter is the sensor, and a correction that size says
# more about the sensor than about the cell.
SOUND_DEG = 1.0
NOISE_DEG = 2.5


def angle_between(a, b):
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))


def sample(cell, arm_ids):
    """One reading per arm, or None if an arm would not answer."""
    out = {}
    for a in arm_ids:
        up = cell.arms[a].up_in_base()
        if up is None:
            print("  arm %s did not answer — it is still moving, or the "
                  "controller does not send the accelerometer" % a)
            return None
        out[a] = (up, np.array(cell.arms[a].state()["tool_accel"], dtype=float))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("-n", "--samples", type=int, default=6,
                    help="how many postures to read (default 6)")
    args = ap.parse_args()

    np.set_printoptions(precision=3, suppress=True)
    cell = Cell(CellConfig.load())
    cell.listeners.append(lambda t: print("   .", t))
    cell.connect(("A", "B"))
    arm_ids = sorted(cell.connected_ids)
    if not arm_ids:
        raise SystemExit("no arm connected")

    print()
    print("Move the arms to a new posture before each sample, then press "
          "Enter. Hold them still while it reads.")
    readings = {a: [] for a in arm_ids}
    try:
        while len(readings[arm_ids[0]]) < args.samples:
            n = len(readings[arm_ids[0]]) + 1
            input("\n  posture %d of %d — Enter to read " % (n, args.samples))
            got = sample(cell, arm_ids)
            if got is None:
                continue
            for a in arm_ids:
                up, accel = got[a]
                readings[a].append(up)
                print("    arm %s  up in base %s   |accel| %5.2f m/s2   "
                      "tilt %6.2f deg" % (a, up, np.linalg.norm(accel),
                                          angle_between(up, [0, 0, 1])))
    except (KeyboardInterrupt, EOFError):
        print("\n  stopped early")

    counts = {a: len(v) for a, v in readings.items()}
    if min(counts.values()) < 2:
        cell.disconnect()
        raise SystemExit("need at least two postures to say anything")

    print()
    print("spread over %d postures — how much the same measurement moved"
          % min(counts.values()))
    worst = 0.0
    means = {}
    for a in arm_ids:
        m = np.mean(readings[a], axis=0)
        m = m / np.linalg.norm(m)
        means[a] = m
        spread = max(angle_between(u, m) for u in readings[a])
        worst = max(worst, spread)
        print("  arm %s  mean up %s   worst posture %.2f deg off it   "
              "mean tilt %.2f deg" % (a, m, spread, angle_between(m, [0, 0, 1])))

    # the same arithmetic check_up_online.py applies, but once per posture, so
    # the number that gets written into cell.yaml can be seen wobbling
    if len(arm_ids) == 2:
        errors = []
        for i in range(min(counts.values())):
            consensus = sum(cell.arms[a].base_matrix()[:3, :3] @ readings[a][i]
                            for a in arm_ids)
            errors.append(angle_between(consensus, [0, 0, 1]))
        print()
        print("world error per posture  %s deg" % np.round(errors, 2))
        print("  it should be one number; it varies by %.2f deg, and "
              "check_up_online.py writes whichever one it saw"
              % (max(errors) - min(errors)))

    print()
    if worst < SOUND_DEG:
        print("the reading repeats to %.2f deg, so it is measuring the cell. "
              "A world error bigger than that is a real lean." % worst)
    elif worst > NOISE_DEG:
        print("the reading wanders by %.2f deg between postures, so anything "
              "smaller than that is the accelerometer, not the mounting. Do "
              "not level the cell onto it — check the mast against a spirit "
              "level instead." % worst)
    else:
        print("%.2f deg of scatter: borderline. Trust a world error several "
              "times that, and nothing near it." % worst)
    cell.disconnect()


if __name__ == "__main__":
    main()
