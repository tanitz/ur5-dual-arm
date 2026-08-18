"""
Measuring the cell with a block of known thickness between the two flanges.

For a cell with no grippers. Both axis-6 flanges are brought down flat onto
opposite faces of one parallel-sided block, and that posture is recorded. The
block does the work a fixture would: seated properly, it holds the two flange
faces parallel to each other and exactly its own thickness apart, and those
are things worth knowing about a geometry nobody has measured.

    python3 tests/check_block_online.py --gap 84.88
    python3 tests/check_block_online.py --gap 84.88 --apply
    python3 tests/check_block_online.py --gap 84.88 --resume --apply

Every placement is written to config/block_log.json as it is taken, because
the fit can be redone from the joint angles at any time and a session that
only turns out to be badly posed at the end is otherwise an hour of careful
seating thrown away with the answer. --resume fits what is already in the log
along with this session's placements, which is how a set of postures too
awkward to reach in one sitting gets collected over several; and
`scripts/ur5dual-flange-fit --fit --file config/block_log.json` fits the file
with no robot switched on at all.

The file is a flange log like the one `ur5dual-flange-log` writes, and the
difference between them is recorded in it rather than left to be remembered:
these placements mean the perpendicular distance between two parallel faces
and a gauge reading means centre to centre, so each file says which it holds
and refuses samples of the other kind. It also carries the DH tables the poses
were computed from, which is what lets an offline fit reproduce them — the
controllers' own tables and the published one differ by about the size of what
is being measured.

WHAT ONE PLACEMENT IS WORTH, and it is less than it looks. Seating the flanges
says three things:

    the faces are parallel     two of them — it fixes the direction of arm B's
                               flange axis in arm A's flange frame
    they are `gap` apart       one — the distance along that shared axis

and says nothing whatever about the remaining three: turn either arm's J6 and
the faces stay just as parallel and just as far apart, and the block can slide
anywhere in its own plane without a single reading changing. Three constraints
a placement against six unknowns, so this needs *the block clamped in
genuinely different attitudes* rather than more placements on the same one — a
seated flange points where the face does, so re-clamping the block is the only
thing that buys a new direction, and walking the arms along the same two faces
buys nothing. Eight is the number to aim at, and `spread` below is what says
whether they were different enough.

This is the same arithmetic `scripts/ur5dual-flange-fit --fit` performs on a
log of hand-measured gauge readings, with two differences that matter here.
The gap is not typed per sample, because a block has one thickness; and the
flange poses come from each controller's own kinematic table rather than the
published one, so arm B contributes what it actually knows about itself.

Compare with `check_hold_online.py`, which is the better measurement when
there is anything to bolt or grip with: a rigidly held object pins all six
degrees of freedom every placement instead of three, and needs no block, no
seating, and no assumption that the faces went down flat.

BOTH ARMS GO INTO FREEDRIVE while this runs. Stand the block in a vice or on a
solid rest first — nothing here presses it, and freedrive arms will not hold
it. Freedrive is switched off again on the way out, including after Ctrl-C.
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.cell import Cell
from ur5dual.config import ARM_IDS, REPO_ROOT, CellConfig
from ur5dual.geometry import calibration as C
from ur5dual.geometry.kinematics import inv, mat_to_rotvec, mat_to_rpy
from ur5dual.geometry.ur_kinematics import fk
from ur5dual.tools import flange_log as FL
from ur5dual.tools.flange_fit import dh_from_meta

# The gap this cell is set up to measure with. A number with two decimals in
# it is a measured block rather than a nominal one, which is the right kind of
# number to be here — everything below is only as true as it is.
DEFAULT_GAP_MM = 84.88

# Not config/flange_log.json. That file holds gauge readings taken centre to
# centre; these are perpendicular distances between two seated faces, and one
# file cannot hold both — see the note about models above.
DEFAULT_LOG = os.path.join(REPO_ROOT, "config", "block_log.json")

# How far the seated flanges may be from what the configured geometry predicts
# before saying so at record time. These are gross checks and nothing finer,
# because the geometry doing the predicting is the thing under test — the
# per-placement residuals after the fit are the real test, and they are quoted
# against a transform that was measured rather than assumed.
#
# The two numbers are not equally sharp, and the difference is worth knowing.
# Parallelism compares two directions, so a cell 3 deg out reads about 3 deg
# out and 10 deg means something is actually wrong. The gap compares two
# points about 700 mm from arm B's base, so the same 3 deg of orientation
# error moves it by 700 mm x 3 deg = 40 mm all by itself — which is why the
# bar for it sits where it does, and why it can only catch a flange that is
# not on the block at all rather than one seated badly.
SEATING_ANGLE_WARN = 10.0       # deg
SEATING_GAP_WARN = 0.060        # m

parser = argparse.ArgumentParser()
parser.add_argument("--gap", type=float, default=DEFAULT_GAP_MM,
                    help="block thickness in mm (default %.2f)" % DEFAULT_GAP_MM)
parser.add_argument("--apply", action="store_true",
                    help="write a warning-free measured geometry into cell.yaml")
parser.add_argument("--log", default=DEFAULT_LOG, metavar="PATH",
                    help="where the placements are kept (default %s)"
                         % DEFAULT_LOG)
parser.add_argument("--no-log", dest="log", action="store_const", const=None,
                    help="record nothing to disk")
parser.add_argument("--resume", action="store_true",
                    help="fit the placements already in the log along with "
                         "this session's — for a set of postures collected "
                         "over more than one sitting")
args = parser.parse_args()
if args.resume and not args.log:
    raise SystemExit("--resume reads the log that --no-log is refusing to keep")
GAP = args.gap / 1000.0

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

T_ab_config = config.a_to_b()


def joints_now():
    """Both arms' joint angles. One read each, and everything comes out of it.

    The pose that is fitted and the angles that go in the log are then the
    same instant rather than two reads of a stream that is 125 packets a
    second ahead of whichever one was asked for first.
    """
    return {arm_id: np.asarray(cell.arms[arm_id].state()["q_actual"],
                               dtype=float)
            for arm_id in ARM_IDS}


def flange_matrices(q):
    """Each arm's tool0 in its own base frame, on its own kinematics.

    Deliberately not `tcp_pose()`. That is the flange times whatever tool
    offset the pendant happens to be holding, and a block is laid against the
    flange, not against the TCP — an offset left over from some earlier job
    would be a silent error of exactly one tool length. `arm.dh` is whatever
    table the arm answered with when the cell connected, so arm B contributes
    its own factory calibration rather than the published average.
    """
    return {arm_id: fk(q[arm_id], cell.arms[arm_id].dh) for arm_id in ARM_IDS}


class Log:
    """The placements on disk, one whole write per placement.

    Rewritten and renamed into place after every one, so an interrupted run —
    or a refused one — keeps everything up to the interruption. What it holds
    is joint angles, which is the form nothing downstream can misread: the
    flange poses are worked out again from them and the DH tables named in the
    file's own meta.
    """

    def __init__(self, path, meta):
        self.path = path
        self.meta = meta
        self.data = FL.load_log(path)
        self.existing = len(self.data["samples"])
        # why this session must not be appended to this file, or None. Asked
        # before anything is written and before the arms go anywhere, because
        # the answer does not change once eight placements have been taken.
        self.refusal = (FL.kinematics_clash(self.data, meta)
                        or FL.model_clash(self.data, meta))

    def add(self, gap_mm, q, dh):
        self.data["meta"] = self.meta
        self.data["samples"].append(FL.build_sample(gap_mm, q, dh))
        FL.save_log(self.path, self.data)

    def undo(self):
        self.data["samples"].pop()
        FL.save_log(self.path, self.data)


def seating(F_a, F_b, T_ab):
    """What a geometry says about a pair of flanges: (angle, gap, sideways)."""
    G = inv(F_a) @ T_ab @ F_b               # arm A's flange -> arm B's flange
    # facing means B's flange Z opposes A's, so -G[2,2] is the cosine of how
    # far off parallel they are
    angle = math.degrees(math.acos(max(-1.0, min(1.0, -float(G[2, 2])))))
    return angle, float(G[2, 3]), float(np.linalg.norm(G[:2, 3]))


print()
print("as cell.yaml stands")
print("  it puts arm B's base %.1f mm from arm A's, at rpy %s deg in A's frame"
      % (float(np.linalg.norm(T_ab_config[:3, 3])) * 1000,
         np.round(np.degrees(mat_to_rpy(T_ab_config[:3, :3])), 3)))
print("  calibrated: %s" % ("yes" if config.calibrated else
                            "straight lines only" if config.translation_calibrated
                            else "no — these are mounting numbers, not measurements"))
for arm_id in ARM_IDS:
    arm = cell.arms[arm_id]
    print("  arm %s flange poses come from %s" % (arm_id, arm.dh_source))

cal = C.FlangePairCalibration(model="facing")

log = None
if args.log:
    log = Log(args.log, FL.log_meta(
        "check_block_online",
        {a: cell.arms[a].dh_source for a in ARM_IDS},
        {a: cell.arms[a].dh for a in ARM_IDS}, model=FL.FACING))
    if log.refusal:
        cell.disconnect()
        raise SystemExit("%s\n  (%s)" % (log.refusal, args.log))
    print("  placements are kept in %s (%d already there)"
          % (args.log, log.existing))
    if log.existing and args.resume:
        prior = C.FlangePairCalibration.from_log(
            log.data, model=FL.FACING, dh=dh_from_meta(log.data["meta"]))
        cal.samples.extend(prior.samples)
        print("  --resume: those %d are in this fit too, so the spread below "
              "is over both sittings" % log.existing)
    elif log.existing:
        print("  they are not in this fit — --resume puts them in")

# where this session starts in `cal.samples`, so undo cannot reach behind it
# into placements that came off the disk
resumed = len(cal.samples)

print()
print("Stand the %0.2f mm block in a vice or on a solid rest." % args.gap)
print()
print("  Bring both axis-6 flanges down FLAT on opposite faces of it. Flat is")
print("  the whole measurement: the block only holds the two faces parallel")
print("  while both are fully seated, and a flange resting on an edge or")
print("  bridging a burr reports a parallelism that is not there.")
print()
print("  Then MOVE THE BLOCK and do it again. A placement only constrains the")
print("  geometry along the direction the flange axis points, and a seated")
print("  flange points along the face of the block — so it is where the block")
print("  is clamped that decides what a placement is worth, and moving the")
print("  arms along the same two faces adds nothing at all.")
print()
print("  Re-clamp it square across the cell, then along it, then leaning 20 to")
print("  45 deg out of the vertical, and take a couple of placements at each.")
print("  Faces that all stand upright cover a plane and score 0.00 however")
print("  many there are; the leaning ones are what fills the third direction.")
print("  `spread` below is what grades it, not the count — watch it climb")
print("  after each re-clamp, and stop when it is past %.2f."
      % C.MIN_PAIR_SPREAD)
print()
print("  Turning either wrist about its own axis between placements changes")
print("  nothing that is measured here, so it is not worth doing.")
print()
print("Each placement: seat both flanges, hold still, press ENTER.")
print("Type u to drop the last one%s"
      % (" — it is on disk from the moment it is taken, and a flange that was "
         "not flat is worth dropping." if log is not None
         else ", if a flange was not flat when it went down."))
print("Type q when you have enough (%d minimum, %d wanted)."
      % (C.MIN_PAIR_SAMPLES, C.RECOMMENDED_PAIR_SAMPLES))


def freshen_feeds(duration=0.40):
    """Let both feeds catch up to the arms before either pose is recorded."""
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
        answer = input("\nplacement %d — ENTER to record, u to undo, q to "
                       "finish  " % (n + 1)).strip().lower()
        if answer.startswith("q"):
            if n >= C.MIN_PAIR_SAMPLES:
                break
            print("   only %d placements recorded; the solve needs %d"
                  % (n, C.MIN_PAIR_SAMPLES))
            continue
        if answer.startswith("u"):
            if n <= resumed:
                print("   nothing recorded this session%s"
                      % (" — the %d from the log are not this program's to "
                         "drop" % resumed if resumed else ""))
                continue
            dropped = cal.samples.pop()
            if log is not None:
                log.undo()
            print("   dropped %s (%d left, spread %.2f)"
                  % (dropped.name, len(cal.samples),
                     cal.spread(T_ab_config)))
            continue
        print("   reading both arms — hold the flanges on the block")
        freshen_feeds()
        q = joints_now()
        F = flange_matrices(q)
        F_a, F_b = F["A"], F["B"]

        # What the current geometry makes of this posture. It is the thing
        # under test, so this cannot verify a placement — but a block seated
        # flat against a cell that is a few degrees out should read a few
        # degrees out, and anything wilder is worth looking up from the
        # terminal for.
        angle, gap, sideways = seating(F_a, F_b, T_ab_config)
        sample = cal.add(F_a, F_b, GAP)
        if log is not None:
            log.add(args.gap, q, {a: cell.arms[a].dh for a in ARM_IDS})
        print("   recorded placement %s (%d so far, spread %.2f)"
              % (sample.name, len(cal.samples), cal.spread(T_ab_config)))
        print("            through cell.yaml's own geometry, which is what is "
              "under test: faces %.1f deg off parallel, %.0f mm apart"
              % (angle, gap * 1000))
        if angle > SEATING_ANGLE_WARN:
            print("            !! %.0f deg off parallel is more than a mounting "
                  "error explains — check both flange faces are flat on the "
                  "block and not resting on an edge or a burr" % angle)
        if abs(gap - GAP) > SEATING_GAP_WARN:
            print("            !! %.0f mm from the %.2f mm block is too far to "
                  "be the geometry alone — is one flange actually touching?"
                  % (abs(gap - GAP) * 1000, args.gap))
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

if len(cal.samples) < C.MIN_PAIR_SAMPLES:
    cell.disconnect()
    if log is not None and len(log.data["samples"]) > log.existing:
        raise SystemExit(
            "too few placements to solve — the %d taken are in %s, and "
            "--resume carries on from them"
            % (len(log.data["samples"]), args.log))
    raise SystemExit("nothing measured")


# ── the answer ────────────────────────────────────────────────────────────
try:
    T_ab, report = cal.solve(T_ab_config)
except C.CalibrationError as e:
    cell.disconnect()
    raise SystemExit("cannot solve: %s" % e)

print()
print("solved from %d placements with the %s model"
      % (report["samples"], report["model"]))
print("  fit              rms %.2f mm, worst %.2f mm" % (report["rms_mm"],
                                                         report["max_mm"]))
if report["tilt_rms_deg"] is not None:
    print("  parallelism      %.2f deg rms across the placements"
          % report["tilt_rms_deg"])
print("  direction spread %.3f   (below %.2f the geometry square to the "
      "readings is still whatever cell.yaml said)"
      % (report["spread"], C.MIN_PAIR_SPREAD))
print("  measured arm B   %.1f mm from arm A's base, rpy %s deg in A's frame"
      % (float(np.linalg.norm(T_ab[:3, 3])) * 1000,
         np.round(np.degrees(mat_to_rpy(T_ab[:3, :3])), 3)))
print("  per placement    %s mm"
      % np.round(np.array(report["per_sample_mm"]), 2))
for w in report["warnings"]:
    print("  !! %s" % w)

E = T_ab @ inv(T_ab_config)
rotvec = mat_to_rotvec(E[:3, :3])
angle = float(np.linalg.norm(rotvec))
print()
print("against the configuration")
print("  arm B's base is %.1f mm from where cell.yaml puts it"
      % (float(np.linalg.norm(T_ab[:3, 3] - T_ab_config[:3, 3])) * 1000))
if angle > 1e-9:
    axis = rotvec / angle
    print("  and turned %.2f deg from it, about the axis %s"
          % (math.degrees(angle), np.round(axis, 3)))
    print("  of that, %.2f deg is about the vertical — the component gravity "
          "cannot see" % abs(math.degrees(angle * axis[2])))
R_err = T_ab[:3, :3] @ T_ab_config[:3, :3].T
print("  per 100 mm of commanded world jog, the arms end up:")
for i, name in enumerate("XYZ"):
    d = np.zeros(3)
    d[i] = 1.0
    print("    %s   %5.1f mm apart"
          % (name, float(np.linalg.norm((R_err - np.eye(3)) @ d)) * 100))

if report["warnings"]:
    print()
    print("  Those warnings are about the measurement, not the cell — a badly")
    print("  posed set of placements produces a confident wrong answer. Fix")
    print("  the measurement rather than applying one.")

if log is not None:
    print()
    print("%d placement%s in %s"
          % (len(log.data["samples"]),
             "" if len(log.data["samples"]) == 1 else "s", args.log))
    print("  add to them with --resume, or fit them with no robot switched on:")
    print("  scripts/ur5dual-flange-fit --fit --file %s" % args.log)

if not args.apply:
    print()
    print("nothing was changed. Re-run with --apply to write this into arms.B")
    cell.disconnect()
    raise SystemExit(0)

if report["warnings"]:
    print()
    print("REFUSED: --apply was requested, but this measurement did not pass.")
    print("cell.yaml was not changed. Seat both flanges flat on the block, and")
    print("record %d placements or more with the block clamped in genuinely"
          % C.RECOMMENDED_PAIR_SAMPLES)
    print("different attitudes — the flange axis has to point somewhere new")
    print("each time, and it points wherever the block's face does.")
    if log is not None:
        print()
        print("Nothing was lost: this session is in %s. Run again with"
              % args.log)
        print("--resume and add the attitudes that were missing, and the fit "
              "will")
        print("be over both sittings.")
    cell.disconnect()
    raise SystemExit(2)

before_xyz = config.arms["B"].xyz.copy()
before_rpy = config.arms["B"].rpy.copy()
T_ab, report = cal.apply_to_config(config)
config.save()
print()
print("applied")
print("  arm B base xyz %s -> %s m"
      % (np.round(before_xyz, 4), np.round(config.arms["B"].xyz, 4)))
print("  arm B base rpy %s -> %s deg"
      % (np.round(np.degrees(before_rpy), 3),
         np.round(np.degrees(config.arms["B"].rpy), 3)))
print("  arm A untouched — everything here is relative, and A is the reference")
print("  mount.style is now custom, so the Cell tab's Apply no longer "
      "overwrites this")
if report["marked_calibrated"]:
    print("  `calibrated` is set: carrying a workpiece and turning it are both "
          "trustworthy now.")
else:
    print("  translation_calibrated is set, `calibrated` is NOT — the "
          "placements agreed to %.1f mm against a %.1f mm bar, and the error a "
          "base distance carries into a rotation grows with the angle turned."
          % (report["max_mm"], C.PAIR_TRUST_MM))
print("saved to %s" % config.path)
print()
print("Restart the panel and RViz so both pick up the new description, then")
print("jog the pair along world X: the flange gap should hold to about the fit")
print("residual above.")
cell.disconnect()
