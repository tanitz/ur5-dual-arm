"""
Measuring the whole cell by hand, from a workpiece both arms are holding.

This replaces `check_directions_online.py` and does strictly more. That script
solves

    delta_B = R_BA . delta_A

from how far each tip travelled, which is only true while the workpiece is
translating and nothing else. Two consequences, both felt on the bench: any
push with a couple of degrees of unintended turn in it lands 10 mm outside a
tolerance set at 3 and gets thrown out, and even a clean session can only ever
produce the two bases' *orientation* — where they sit stays whatever the
mounting numbers claimed, because a pure translation never separates the base
offset from the unknown gap between the two grips.

Keeping the orientations the controllers already report fixes both at once.
One placement of the workpiece is then one equation,

    M_i . X = Z . N_i

with M_i and N_i the two TCP poses in their own base frames, X the constant
relationship between the two grips, and Z the answer. This is hand-eye
calibration in its AX=ZB form, and it turns the problem inside out: rotating
the workpiece stops being the mistake that ruins a push and becomes the thing
that makes the measurement work, because rotation is what separates X from Z.

So: slide it, turn it, tip it over, put it down somewhere else. Every
placement is recorded whole and every one of them constrains the answer.

    python3 tests/check_hold_online.py            # measure and report
    python3 tests/check_hold_online.py --apply    # and write cell.yaml

What comes out is arm B's base — position and orientation, the same six
numbers the touch-off wizard measures with a fixture, and no fixture needed.
Against 0.5 mm and 0.1 deg of pose slop the base position lands about 2 mm
out, so this is a real alternative to the wizard rather than a rough stand-in.
It is not better than the arms' own kinematics, which no method here can be.

BOTH ARMS GO INTO FREEDRIVE while this runs. They will not hold the workpiece
up on their own there: take its weight before the first placement, and keep
the E-stop in reach. Freedrive is switched off again on the way out, including
after Ctrl-C.
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
from ur5dual.geometry.kinematics import inv, mat_to_rotvec, mat_to_rpy

# how much travel to quote the divergence over. 100 mm is about as far as one
# hand-guided move goes, so the figure is one an operator can check by eye
# against the jog they already tried.
QUOTE_TRAVEL = 0.100            # m

parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true",
                    help="write a warning-free measured geometry into cell.yaml")
args = parser.parse_args()

np.set_printoptions(precision=3, suppress=True)

config = CellConfig.load()
cell = Cell(config)
cell.listeners.append(lambda text: print("       . %s" % text))

print("connecting")
cell.connect(ARM_IDS)
if len(cell.connected_ids) < 2:
    raise SystemExit("both arms have to be online for this; got %s"
                     % (", ".join(cell.connected_ids) or "none"))
blocked = cell.not_ready_reason()
if blocked:
    raise SystemExit("cannot enter freedrive: %s" % blocked)


# ── what the configuration currently claims ───────────────────────────────
T_ab_config = config.a_to_b()

print()
print("as cell.yaml stands")
print("  arm A base rpy  %s deg" % np.round(np.degrees(config.arms["A"].rpy), 3))
print("  arm B base rpy  %s deg" % np.round(np.degrees(config.arms["B"].rpy), 3))
print("  it puts arm B's base %.1f mm from arm A's, at rpy %s deg in A's frame"
      % (float(np.linalg.norm(T_ab_config[:3, 3])) * 1000,
         np.round(np.degrees(mat_to_rpy(T_ab_config[:3, :3])), 3)))
print("  calibrated: %s" % ("yes" if config.calibrated else
                            "straight lines only" if config.translation_calibrated
                            else "no — these are mounting numbers, not measurements"))


def describe_divergence(T_ab):
    """The fault in millimetres, which is how it gets noticed on the bench."""
    E = T_ab @ inv(T_ab_config)          # the correction, in arm A's frame
    rotvec = mat_to_rotvec(E[:3, :3])
    angle = float(np.linalg.norm(rotvec))
    shift = float(np.linalg.norm(T_ab[:3, 3] - T_ab_config[:3, 3]))

    print("  measured vs cell.yaml")
    print("    arm B's base is %.1f mm from where cell.yaml puts it" % (shift * 1000))
    if angle < 1e-9:
        print("    and turned exactly as cell.yaml says — nothing to correct")
    else:
        axis = rotvec / angle
        print("    and turned %.2f deg from it, about the axis %s"
              % (math.degrees(angle), np.round(axis, 3)))
        print("    of that, %.2f deg is about the vertical — the component "
              "gravity cannot see" % abs(math.degrees(angle * axis[2])))

    # The orientation half is what a world jog exposes, and it is worth
    # quoting on its own: a position error moves both arms' idea of a point
    # together, while an orientation error makes them travel apart, which is
    # the thing that fights.
    R_err = T_ab[:3, :3] @ T_ab_config[:3, :3].T
    print("    per %.0f mm of commanded world jog, the arms end up:"
          % (QUOTE_TRAVEL * 1000))
    for i, name in enumerate("XYZ"):
        d = np.zeros(3)
        d[i] = 1.0
        diverge = (R_err - np.eye(3)) @ d * QUOTE_TRAVEL
        print("      %s   %5.1f mm apart" % (name, float(np.linalg.norm(diverge)) * 1000))
    return angle, shift


# ── the teach ─────────────────────────────────────────────────────────────
print()
print("Put both grippers on one rigid workpiece and take its weight.")
print()
print("  MOVE IT AND TURN IT. Turning is not a mistake here — it is the")
print("  measurement. Sliding alone fixes how the two bases are oriented and")
print("  says nothing about where they are; it is rotation, and only")
print("  rotation, that pins the positions down. Turn it about at least two")
print("  genuinely different axes, and turn it well past 45 degrees: the")
print("  positions come out of those angles, so small turns just multiply the")
print("  noise. `amplification` below is what says whether you did enough.")
print()
print("  What must NOT happen is a gripper shifting its bite on the workpiece.")
print("  That is the one assumption the arithmetic makes, and the residual is")
print("  what tests it.")
print()
print("  Aim for %d placements or more. Between them, move the thing anywhere"
      % C.RECOMMENDED_HELD_POSES)
print("  you like, take as long as you like, and let go in the middle — only")
print("  where it comes to rest is ever read.")
print()
print("Each placement: hold the workpiece still, then press ENTER.")
print("Type q when you have enough.")

cal = C.HeldObjectCalibration()


def freshen_feeds(duration=0.40):
    """Let both feeds catch up to the arms before either pose is recorded.

    `StateStream.latest()` drains whatever the controller has queued and
    returns the newest sample, so this is belt and braces rather than the
    load-bearing part. It costs a fifth of a second per placement and removes
    any question about which packet a reading came from.
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
        n = len(cal.samples)
        answer = input("\nplacement %d — ENTER to record, q to finish  " % (n + 1))
        if answer.strip().lower().startswith("q"):
            if n >= C.MIN_HELD_POSES:
                break
            print("   only %d placements recorded; the solve needs %d"
                  % (n, C.MIN_HELD_POSES))
            continue
        print("   reading both arms — keep the workpiece still")
        freshen_feeds()
        accepted, message = cal.commit(cell)
        print("   %s %s" % ("recorded:" if accepted else "rejected:", message))
except KeyboardInterrupt:
    print("\ninterrupted")
    cal.clear()
finally:
    for arm_id in ARM_IDS:
        try:
            cell.arms[arm_id].motion.freedrive_off()
        except OSError as e:
            print("!! arm %s did not leave freedrive (%s) — stop the program "
                  "on its pendant" % (arm_id, e))
    print("freedrive off")

if len(cal.samples) < C.MIN_HELD_POSES:
    cell.disconnect()
    raise SystemExit("nothing measured")


# ── the answer ────────────────────────────────────────────────────────────
try:
    T_ab, report = cal.solve(seed=T_ab_config)
except C.CalibrationError as e:
    cell.disconnect()
    raise SystemExit("cannot solve: %s" % e)

print()
print("solved from %d placements" % report["placements"])
print("  widest turn      %.0f deg about an axis spread of %.2f"
      % (report["turn_deg"], report["spread"]))
print("  fit              rms %.2f mm, worst %.2f mm, %.2f deg rms"
      % (report["rms_mm"], report["max_mm"], report["tilt_rms_deg"]))
if report["determined"]:
    print("  amplification    %.1f x pose slop into the base position "
          "(about 3 is a well-spread session)" % report["amplification"])
    print("  grip             the two gripper frames come out %.1f mm apart"
          % report["grip_mm"])
    print("                   (a tape measure across the workpiece checks "
          "that, and nothing else here can be checked so directly)")
    print("  measured arm B   %.1f mm from arm A's base, rpy %s deg in A's "
          "frame" % (float(np.linalg.norm(T_ab[:3, 3])) * 1000,
                     np.round(np.degrees(mat_to_rpy(T_ab[:3, :3])), 3)))
for w in report["warnings"]:
    print("  !! %s" % w)

if report["determined"]:
    print()
    print("against the configuration")
    describe_divergence(T_ab)
else:
    # Printing the fitted geometry here would be worse than printing nothing.
    # An under-determined fit still returns six confident-looking numbers, and
    # they can be metres out; the operator has no way to tell them from the
    # good ones by eye, and the warning above is easier to disbelieve when a
    # tidy table of millimetres is sitting next to it.
    print()
    print("  The geometry this produced is not printed. Nothing above")
    print("  determines it, so the numbers would be arbitrary — and arbitrary")
    print("  numbers laid out in millimetres read exactly like measured ones.")


# ── the orientation-only fallback ─────────────────────────────────────────
# A session with no real turning in it cannot place the bases, but the same
# placements still hold every pure translation between them, and those do fix
# the orientation. Offering that is the difference between an operator getting
# half an answer and getting nothing.
fallback = None
if not report["determined"]:
    direction = cal.as_direction_calibration()
    if len(direction.pairs) >= 3:
        try:
            R_ba, drep = direction.solve()
            fallback = (direction, R_ba, drep)
        except C.CalibrationError:
            pass

if fallback is not None:
    direction, R_ba, drep = fallback
    print()
    print("the orientation on its own, from the %d translations in this set"
          % drep["motions"])
    print("  fit              rms %.2f mm, worst %.2f mm"
          % (drep["rms_mm"], drep["max_mm"]))
    print("  direction spread %.3f" % drep["spread"])
    for w in drep["warnings"]:
        print("  !! %s" % w)
    print("  This much IS determined by what you recorded. It makes a world")
    print("  jog reach both arms; it does not license turning a held object.")


# ── writing it ────────────────────────────────────────────────────────────
if report["warnings"]:
    print()
    print("  Those warnings are about the measurement, not the cell — a badly")
    print("  posed set of placements produces a confident wrong answer. Fix")
    print("  the measurement rather than applying one.")

if not args.apply:
    print()
    print("nothing was changed. Re-run with --apply to write this into arms.B")
    cell.disconnect()
    raise SystemExit(0)

if report["warnings"] and not (fallback is not None and not fallback[2]["warnings"]):
    print()
    print("REFUSED: --apply was requested, but this measurement did not pass.")
    print("cell.yaml was not changed. Keep both grips fixed on the workpiece,")
    print("turn it about two or more different axes and well past 45 degrees,")
    print("and record %d placements or more." % C.RECOMMENDED_HELD_POSES)
    cell.disconnect()
    raise SystemExit(2)

before_xyz = config.arms["B"].xyz.copy()
before_rpy = config.arms["B"].rpy.copy()

if report["warnings"]:
    # the full solve did not pass, but the translations inside it did
    direction = fallback[0]
    direction.apply_to_config(config)
    config.save()
    print()
    print("applied — the ORIENTATION ONLY")
    print("  arm B base rpy %s -> %s deg"
          % (np.round(np.degrees(before_rpy), 3),
             np.round(np.degrees(config.arms["B"].rpy), 3)))
    print("  arm B position untouched — this set of placements never")
    print("  determined it. Turn the workpiece about a second axis and re-run")
    print("  to get the positions as well.")
    print("  translation_calibrated is set: carrying a workpiece in a straight")
    print("  line is now trustworthy. `calibrated` is NOT — rotating a held")
    print("  object still needs the distance between the bases.")
else:
    _, report = cal.apply_to_config(config, seed=T_ab_config)
    config.save()
    print()
    print("applied — position AND orientation")
    print("  arm B base xyz %s -> %s m"
          % (np.round(before_xyz, 4), np.round(config.arms["B"].xyz, 4)))
    print("  arm B base rpy %s -> %s deg"
          % (np.round(np.degrees(before_rpy), 3),
             np.round(np.degrees(config.arms["B"].rpy), 3)))
    print("  arm A untouched — everything here is relative, and A is the "
          "reference")
    print("  mount.style is now custom, so the Cell tab's Apply no longer "
          "overwrites this")
    if report["marked_calibrated"]:
        print("  `calibrated` is set: carrying a workpiece and turning it are "
              "both trustworthy now.")
    else:
        print("  translation_calibrated is set, `calibrated` is NOT — the "
              "placements agreed to %.1f mm against a %.1f mm bar, and the "
              "error a base distance carries into a rotation grows with the "
              "angle turned." % (report["max_mm"], C.HELD_TRUST_MM))

print("saved to %s" % config.path)
print()
print("Restart the panel and RViz so both pick up the new description, then")
print("jog the pair along world X: the flange gap should hold to about the fit")
print("residual above.")
cell.disconnect()
