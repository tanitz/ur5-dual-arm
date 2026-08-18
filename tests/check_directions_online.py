"""
Teaching the two bases' relative orientation by hand, on the real cell.

SUPERSEDED by `check_hold_online.py`, which keeps the orientations these
controllers already report instead of throwing them away. That buys two
things this cannot offer: the workpiece may be turned — turning it is what
makes the measurement work there, rather than what invalidates a push — and
the base *positions* come out along with their orientations. Reach for this
one only when the workpiece genuinely cannot be rotated in the space
available; it is the weaker measurement, and its refusals are the price of
assuming a pure translation.

The one error nothing else in this project can see. `check_up_online.py` and
`check_geometry_online.py` both measure a direction that points *down* — one
from the flange accelerometers, one from the weight of an unconfigured tool —
and gravity says nothing whatever about rotation *about* the vertical. Two
arms whose relative transform is out by a pure yaw report the same "up", agree
with each other perfectly, and pass both checks. RViz cannot see it either:
the description and the jog code read the same `arms.<id>.base` out of
cell.yaml, so a wrong transform draws a picture that matches itself.

What it looks like on the bench is a world jog along one horizontal axis
closing or opening the gap between the two flanges, while the other two axes
leave it alone. That is the signature, and this is the measurement.

`DirectionCalibration` does the arithmetic already; this is the hand-guiding
around it. Both grippers hold one rigid workpiece, both arms go into
freedrive, and somebody pushes the thing around. One physical motion seen from
two base frames is one equation:

    delta_B = R_BA · delta_A

Three pushes in genuinely different directions determine R_BA, and nothing
about where the bases sit or how far apart they are ever enters — which is why
this needs no jig and no touch-off fixture, and equally why it cannot license
rotating a held object. For that, run the touch-off wizard on the Cell tab.

    python3 tests/check_directions_online.py            # measure and report
    python3 tests/check_directions_online.py --apply    # and write cell.yaml

BOTH ARMS GO INTO FREEDRIVE while this runs. They will not hold the workpiece
up on their own there: take its weight before starting the first push, and
keep the E-stop in reach. Freedrive is switched off again on the way out,
including after Ctrl-C.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.cell import Cell
from ur5dual.config import ARM_IDS, CellConfig
from ur5dual.geometry import calibration as C
from ur5dual.geometry.kinematics import mat_to_rotvec, mat_to_rpy

# how much travel to quote the divergence over. 100 mm is about as far as one
# hand-guided push goes, so the figure is one an operator can check by eye
# against the jog they already tried.
QUOTE_TRAVEL = 0.100            # m

# what to ask for, as opposed to what the solver will accept. calibration.py
# grades a push against slop measured on 80 mm of hand guiding, so a push that
# size or larger is one the tolerances were actually tuned for.
RECOMMENDED_PUSH = (0.080, 0.150)    # m

parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true",
                    help="write a warning-free measured orientation into cell.yaml")
args = parser.parse_args()

np.set_printoptions(precision=3, suppress=True)

config = CellConfig.load()
cell = Cell(config)
cell.listeners.append(lambda text: print("       . %s" % text))

print("this is the translation-only teach; check_hold_online.py measures more")
print("connecting")
cell.connect(ARM_IDS)
if len(cell.connected_ids) < 2:
    raise SystemExit("both arms have to be online for this; got %s"
                     % (", ".join(cell.connected_ids) or "none"))
blocked = cell.not_ready_reason()
if blocked:
    raise SystemExit("cannot enter freedrive: %s" % blocked)


# ── what the configuration currently claims ───────────────────────────────
R_a = config.arms["A"].base_matrix()[:3, :3]
R_b = config.arms["B"].base_matrix()[:3, :3]
R_ba_config = R_b.T @ R_a


def world_error(R_ba):
    """Where arm B really goes when the cell asks for a world direction.

    Arm A is the reference — everything here is relative, so the pair can only
    be placed against A, and A stays wherever `mount.tilt_deg` put it. Taking
    A as truth, the measurement fixes B's real base orientation at
    R_a · R_ba^T, while the cell keeps resolving world directions for B
    through the R_b it has in the file. The mismatch between those two is the
    whole fault, expressed as a rotation of the world frame.
    """
    return R_a @ R_ba.T @ R_b.T


def describe_divergence(R_ba, label):
    """The fault in millimetres of jog, which is how it was noticed."""
    E_b = world_error(R_ba)
    rotvec = mat_to_rotvec(E_b)
    angle = float(np.linalg.norm(rotvec))
    print("  %s" % label)
    if angle < 1e-9:
        print("    the two arms travel along the same world vector — nothing "
              "to correct")
        return angle
    axis = rotvec / angle
    print("    arm B's real base is %.2f deg from what cell.yaml says, about "
          "the world axis %s" % (math.degrees(angle), np.round(axis, 3)))
    print("    of that, %.2f deg is about the vertical — the component "
          "gravity cannot see" % abs(math.degrees(angle * axis[2])))

    # the gap between the flanges, which is what an operator watches. Its
    # direction is taken from where the two TCPs are now, so this is only as
    # good as the pose the arms happen to be in — a diagnostic, not a spec.
    try:
        p_a = cell.arms["A"].tcp_matrix_world()[:3, 3]
        p_b = cell.arms["B"].tcp_matrix_world()[:3, 3]
        gap = p_b - p_a
        ghat = gap / float(np.linalg.norm(gap))
    except (OSError, ConnectionError, RuntimeError):
        ghat = None

    print("    per %.0f mm of commanded world jog:" % (QUOTE_TRAVEL * 1000))
    for i, name in enumerate("XYZ"):
        d = np.zeros(3)
        d[i] = 1.0
        diverge = (E_b - np.eye(3)) @ d * QUOTE_TRAVEL
        line = "      %s   the arms end up %5.1f mm apart" % (
            name, float(np.linalg.norm(diverge)) * 1000)
        if ghat is not None:
            closing = float(ghat @ diverge) * 1000
            line += ("   (flange gap %+5.1f mm — %s)"
                     % (closing, "opens" if closing > 0 else "closes"))
        print(line)
    return angle


print()
print("as cell.yaml stands")
print("  arm A base rpy  %s deg" % np.round(np.degrees(config.arms["A"].rpy), 3))
print("  arm B base rpy  %s deg" % np.round(np.degrees(config.arms["B"].rpy), 3))
print("  it claims a world displacement reaches arm B as R_BA rpy %s deg"
      % np.round(np.degrees(mat_to_rpy(R_ba_config)), 3))


# ── the teach ─────────────────────────────────────────────────────────────
print()
print("Put both grippers on one rigid workpiece and take its weight.")
print()
print("  SLIDE it, do not TURN it. The equation is delta_B = R_BA . delta_A,")
print("  and that only holds while both grippers travel the same physical")
print("  vector — which is what a pure translation of a rigid body is. Turn")
print("  the workpiece and the two tips sweep different arcs, the two")
print("  displacements stop being the same motion, and the push is rejected.")
print()
print("  Aim for %.0f-%.0f mm a push. The residual tolerances below were set"
      % (RECOMMENDED_PUSH[0] * 1000, RECOMMENDED_PUSH[1] * 1000))
print("  against hand-guiding slop on pushes that size; %.0f mm is the bare"
      % (C.MIN_MOVE * 1000))
print("  minimum and leaves the answer riding on encoder noise.")
print()
print("  Only the start and the end are read, so take as long as you like in")
print("  between and let go of the button-less middle. Push along all three")
print("  directions — pushes back and forth along one line determine nothing,")
print("  however many you record, and `spread` below is what says so.")
print()
print("Each push: ENTER to mark the start, slide the workpiece, ENTER again.")
print("Type q at the first prompt when you have enough.")

cal = C.DirectionCalibration()


def freshen_feeds(duration=0.40):
    """Drain samples accumulated while input() was waiting for the operator.

    Port 30003 produces about 125 packets a second.  A human-guided push takes
    long enough to fill the kernel receive buffer; one read can then return a
    perfectly valid but old pose from before the second arm moved.  Polling
    briefly after the object has stopped lets both streams catch up to the
    physical arms before either endpoint is recorded.
    """
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        for arm_id in ARM_IDS:
            cell.arms[arm_id].state()
        time.sleep(0.01)

for arm_id in ARM_IDS:
    cell.arms[arm_id].motion.freedrive_on()
print()
print("!! BOTH ARMS ARE IN FREEDRIVE")

try:
    while True:
        n = len(cal.pairs)
        answer = input("\npush %d — ENTER to start, q to finish  " % (n + 1))
        if answer.strip().lower().startswith("q"):
            if n >= 3:
                break
            print("   only %d motions recorded; the solve needs 3" % n)
            continue
        print("   syncing live poses — keep the workpiece still")
        freshen_feeds()
        cal.start(cell)
        input("   pushing — ENTER when the workpiece has stopped  ")
        print("   syncing live poses — keep the workpiece still")
        freshen_feeds()
        accepted, message = cal.commit(cell)
        print("   %s %s" % ("recorded:" if accepted else "rejected:", message))
        if accepted:
            d_a, d_b = cal.pairs[-1]
            print("            arm A saw %s m in its own base" % np.round(d_a, 4))
            print("            arm B saw %s m in its own base" % np.round(d_b, 4))
            if float(np.linalg.norm(d_a)) < RECOMMENDED_PUSH[0]:
                # accepted, and still worth saying: the solver's own residual
                # tolerances were calibrated on longer pushes, so a short one
                # passes the guards while contributing mostly noise
                print("            (short — %.0f mm or more carries the angle "
                      "far better)" % (RECOMMENDED_PUSH[0] * 1000))
except KeyboardInterrupt:
    print("\ninterrupted")
    cal.reset()
finally:
    for arm_id in ARM_IDS:
        try:
            cell.arms[arm_id].motion.freedrive_off()
        except OSError as e:
            print("!! arm %s did not leave freedrive (%s) — stop the program "
                  "on its pendant" % (arm_id, e))
    print("freedrive off")

if len(cal.pairs) < 3:
    cell.disconnect()
    raise SystemExit("nothing measured")


# ── the answer ────────────────────────────────────────────────────────────
try:
    R_ba, report = cal.solve()
except C.CalibrationError as e:
    cell.disconnect()
    raise SystemExit("cannot solve: %s" % e)

print()
print("solved from %d motions" % report["motions"])
print("  median push      %.0f mm" % report["median_move_mm"])
print("  fit              rms %.2f mm, worst %.2f mm"
      % (report["rms_mm"], report["max_mm"]))
print("  direction spread %.3f   (below %.2f the rotation about one line is "
      "guesswork)" % (report["spread"], C.MIN_DIRECTION_SPREAD))
print("  measured R_BA    rpy %s deg" % np.round(np.degrees(mat_to_rpy(R_ba)), 3))
for w in report["warnings"]:
    print("  !! %s" % w)

print()
print("against the configuration")
angle = describe_divergence(R_ba, "measured vs cell.yaml")

if report["warnings"]:
    print()
    print("  Those warnings are about the measurement, not the cell — a bad")
    print("  set of pushes produces a confident wrong answer. Re-run rather")
    print("  than apply one.")
    if args.apply:
        print()
        print("REFUSED: --apply was requested, but this measurement did not pass. ")
        print("cell.yaml was not changed. Secure both grippers, keep every push ")
        print("to a pure translation, and repeat until there are no warnings.")
        cell.disconnect()
        raise SystemExit(2)

if not args.apply:
    print()
    print("nothing was changed. Re-run with --apply to write this into "
          "arms.B.base")
    cell.disconnect()
    raise SystemExit(0)

before = config.arms["B"].rpy.copy()
cal.apply_to_config(config)
config.save()
print()
print("applied")
print("  arm B base rpy %s -> %s deg"
      % (np.round(np.degrees(before), 3),
         np.round(np.degrees(config.arms["B"].rpy), 3)))
print("  arm B position untouched — these pushes say nothing about it")
print("  mount.style is now custom, so the Cell tab's Apply no longer "
      "overwrites this")
print("  translation_calibrated is set: carrying a workpiece in a straight "
      "line is now trustworthy.")
print("  `calibrated` is NOT set — rotating a held object still needs the")
print("  distance between the bases, which only the touch-off wizard "
      "measures.")
print("saved to %s" % config.path)
print()
print("Restart the panel and RViz so both pick up the new description, then")
print("jog the pair along world X again: the flange gap should now hold to")
print("about the fit residual above.")
cell.disconnect()
