#!/usr/bin/env python3
"""
What the flange gaps you measured say about where arm B really stands.

    scripts/ur5dual-flange-fit                    # what the samples say
    scripts/ur5dual-flange-fit --refine           # correct with the newest one
    scripts/ur5dual-flange-fit --fit              # fit them all
    scripts/ur5dual-flange-fit --refine --apply   # and write config/cell.yaml

Reads config/flange_log.json and config/cell.yaml. Touches no robot, and
without `--apply` writes nothing either — the default is a report.

The two ways to use a reading, and they are not the same size of claim:

  --refine   one pairing corrects the configured geometry in the three
             directions a gauge can see: the two angles that make the faces
             parallel, and the distance between them. The spin of the flanges
             about their common axis and how far one sits sideways of the
             other stay exactly as configured, because nothing was measured
             about them. A correction, not a calibration — the flags in
             cell.yaml are left alone.

  --fit      several pairings, taken with the flange axis pointing in
             genuinely different directions, over-determine the whole
             transform. That is a calibration and is marked as one. Readings
             all taken along one line are refused rather than fitted: they
             produce a transform that explains every reading and is wrong in
             the directions square to them, which is how two arms end up
             driving into each other while every number on screen looks fine.

The report is worth reading before either. It puts each gauge reading beside
what the current geometry predicts, and a cell whose numbers are right shows a
column of small differences; anything else is telling you something. A reading
about a tenth of the configured distance is the usual something — a gap typed
in centimetres — and those samples are named rather than quietly fitted.
"""

import argparse
import json
import os
import sys

import numpy as np

from ..config import ARM_IDS, DEFAULT_PATH, REPO_ROOT, CellConfig
from ..geometry import ur_kinematics as UK
from ..geometry.calibration import (
    CalibrationError, FlangePairCalibration, MIN_PAIR_SAMPLES,
    RECOMMENDED_PAIR_SAMPLES, describe, refine_from_flange_pair,
)
from ..geometry.kinematics import inv, mat_to_pose

DEFAULT_LOG = os.path.join(REPO_ROOT, "config", "flange_log.json")

# A reading this many times smaller than the configured distance was not a
# small mistake. Ten is a change of unit; the band around it is wide because
# the geometry it is being compared against is the thing under suspicion.
SUSPECT_RATIO = (6.0, 16.0)


def load_log(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            "no flange log at %s — record some readings first with "
            "scripts/ur5dual-flange-log" % path)
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
        raise SystemExit("%s is not a flange log" % path)
    if not data["samples"]:
        raise SystemExit("%s has no samples in it yet" % path)
    return data


def dh_from_meta(meta):
    """The DH tables these samples were recorded with, per arm.

    A log written from the controllers' own tables and re-fitted against the
    published one moves every flange by about 3 mm, which is the size of the
    thing being fitted — so the numbers travel with the samples. Logs written
    before they did get the published table, which is what they were computed
    from anyway.
    """
    tables = (meta.get("dh") or {})
    out = {}
    for arm_id in ARM_IDS:
        t = tables.get(arm_id)
        out[arm_id] = ({k: np.asarray(t[k], dtype=float) for k in UK.DH_FIELDS}
                       if t and all(k in t for k in UK.DH_FIELDS)
                       else UK.UR5_DH)
    return out


class Row:
    """One sample, worked out in both frames so it can be talked about."""

    def __init__(self, index, sample, config, dh):
        self.index = index
        self.name = sample.get("t", "S%d" % index)
        self.gap = float(sample["gap_mm"]) / 1000.0
        self.q = {a: np.radians(sample[a]["joints_deg"]) for a in ARM_IDS}
        self.F = {a: UK.fk(self.q[a], dh[a]) for a in ARM_IDS}      # in its base
        world = {a: config.arms[a].base_matrix() @ self.F[a] for a in ARM_IDS}
        v = world["B"][:3, 3] - world["A"][:3, 3]
        self.configured = float(np.linalg.norm(v))
        along = float(v @ world["A"][:3, 2])
        self.along = along
        self.sideways = float(np.linalg.norm(v - along * world["A"][:3, 2]))
        self.tilt = float(np.degrees(np.arccos(
            np.clip(world["A"][:3, 2] @ -world["B"][:3, 2], -1.0, 1.0))))

    @property
    def suspect(self):
        """True when this reading is a different unit rather than a bad pose."""
        if self.gap <= 0:
            return True
        ratio = self.configured / self.gap
        return SUSPECT_RATIO[0] <= ratio <= SUSPECT_RATIO[1]

    def line(self):
        return ("%3d  %8.1f %12.1f %11.1f %10.1f %9.1f   %s"
                % (self.index, self.gap * 1000, self.configured * 1000,
                   self.along * 1000, self.sideways * 1000, self.tilt,
                   "looks like cm" if self.suspect else ""))


def report(rows, kinematics):
    print("      gauge   cell.yaml    along axis   sideways   faces off")
    print("         mm          mm            mm         mm     parallel")
    for row in rows:
        print(row.line())

    err = np.array([r.configured - r.gap for r in rows])
    print("\n%d sample%s: cell.yaml is %.1f mm out rms, %.1f mm at worst"
          % (len(rows), "" if len(rows) == 1 else "s",
             float(np.sqrt((err ** 2).mean()) * 1000),
             float(np.abs(err).max() * 1000)))
    tilt = np.array([r.tilt for r in rows])
    print("the configured geometry has the faces %.1f to %.1f deg off parallel"
          % (tilt.min(), tilt.max()))
    if tilt.min() > 1.0:
        print("  — every sample, by about the same amount, so that is the "
              "geometry rather than the poses: if the flanges were physically "
              "parallel when these were taken, the A-to-B rotation in "
              "cell.yaml is out by roughly %.1f deg" % float(np.median(tilt)))
    suspect = [r.index for r in rows if r.suspect]
    if suspect:
        print("  — sample%s %s read about a tenth of the configured distance, "
              "which is a gap typed in centimetres rather than a cell that is "
              "ten times wrong. --skip-suspect leaves them out of a fit"
              % ("" if len(suspect) == 1 else "s",
                 ", ".join(str(i) for i in suspect)))
    if set(kinematics.values()) == {"the published UR5 table"}:
        print("  — these poses came from the published UR5 table, which is "
              "about 3 mm out on these two arms. Readings taken through "
              "ur5dual-flange-log --direct carry each controller's own "
              "calibration and are worth more")


def show_change(config, T_ab, note):
    """What this would do to arm B, before anybody agrees to it."""
    was = config.a_to_b()
    moved = mat_to_pose(inv(was) @ T_ab)
    print("\n%s" % note)
    print("  cell.yaml now  A->B  %s" % describe(was))
    print("  this measure   A->B  %s" % describe(T_ab))
    print("  arm B moves %.1f mm and turns %.2f deg"
          % (float(np.linalg.norm(moved[:3])) * 1000,
             float(np.degrees(np.linalg.norm(moved[3:])))))


def write(config, T_ab, path):
    config.arms["B"].set_base_matrix(config.arms["A"].base_matrix() @ T_ab)
    config.set_custom_mount()
    config.save(path)
    print("  written to %s — arm A untouched, it is the reference" % path)


def do_refine(config, rows, args):
    row = rows[-1] if args.sample is None else rows[args.sample - 1]
    T_ab = refine_from_flange_pair(config.a_to_b(), row.F["A"], row.F["B"],
                                   row.gap, concentric=args.concentric)
    show_change(config, T_ab, "correcting with sample %d (%.1f mm gap)%s"
                % (row.index, row.gap * 1000,
                   ", flanges assumed concentric" if args.concentric else ""))
    print("  the spin about the common axis and the sideways offset are "
          "unchanged — this reading says nothing about them, so neither does "
          "this correction. cell.yaml's calibration flags stay as they are")
    if args.apply:
        write(config, T_ab, args.config)
    else:
        print("  nothing written — add --apply")


def do_fit(config, rows, args):
    cal = FlangePairCalibration(args.model)
    for row in rows:
        cal.add(row.F["A"], row.F["B"], row.gap, name=str(row.index))
    try:
        T_ab, rep = cal.solve(config.a_to_b())
    except CalibrationError as e:
        raise SystemExit("cannot fit these: %s" % e)

    show_change(config, T_ab, "fitting %d sample%s with the %s model"
                % (rep["samples"], "" if rep["samples"] == 1 else "s",
                   rep["model"]))
    print("  readings agree with it to %.1f mm rms, %.1f mm at worst"
          % (rep["rms_mm"], rep["max_mm"]))
    if rep["tilt_rms_deg"] is not None:
        print("  faces come out %.2f deg from parallel rms" % rep["tilt_rms_deg"])
    print("  direction spread %.2f (needs 0.25; an isotropic set scores 0.58)"
          % rep["spread"])
    for w in rep["warnings"]:
        print("  ! %s" % w)

    if not args.apply:
        print("  nothing written — add --apply")
        return
    try:
        cal.apply_to_config(config)
    except CalibrationError as e:
        raise SystemExit("refusing to write this: %s" % e)
    config.save(args.config)
    print("  written to %s — cell marked %s"
          % (args.config,
             "calibrated" if config.calibrated else
             "translation-calibrated only, because the readings disagree by "
             "more than a rotation could survive"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=DEFAULT_LOG, help="the flange log")
    ap.add_argument("--config", default=DEFAULT_PATH)
    ap.add_argument("--refine", action="store_true",
                    help="correct the geometry with one reading")
    ap.add_argument("--fit", action="store_true",
                    help="fit the whole transform to several readings")
    ap.add_argument("--apply", action="store_true",
                    help="write the result into cell.yaml")
    ap.add_argument("--sample", type=int, default=None,
                    help="which reading --refine uses (default: the newest)")
    ap.add_argument("--last", type=int, default=None,
                    help="use only the last N readings")
    ap.add_argument("--model", choices=("separation", "facing"), default=None,
                    help="what the recorded distance means, when the file "
                         "does not say. separation: measured centre to "
                         "centre, which is what a gauge reads. facing: "
                         "measured between parallel faces, which is what a "
                         "block of known thickness holds them at")
    ap.add_argument("--concentric", action="store_true",
                    help="--refine only: the two flange axes were lined up, "
                         "not merely parallel")
    ap.add_argument("--skip-suspect", action="store_true",
                    help="leave out readings that look like centimetres")
    args = ap.parse_args()

    if args.refine and args.fit:
        raise SystemExit("--refine uses one reading and --fit uses several; "
                         "pick one")

    config = CellConfig.load(args.config)
    data = load_log(args.file)
    meta = data.get("meta") or {}
    kinematics = meta.get("kinematics") or {}
    dh = dh_from_meta(meta)
    # What the samples were taken to mean is the file's business; the flag is
    # for a log that predates the field, and overrules it when given.
    args.model = args.model or meta.get("model") or "separation"

    rows = [Row(i, s, config, dh)
            for i, s in enumerate(data["samples"], 1)]
    if args.last is not None:
        rows = rows[-args.last:]
    if args.skip_suspect:
        kept = [r for r in rows if not r.suspect]
        print("leaving out %d reading%s that look like centimetres"
              % (len(rows) - len(kept), "" if len(rows) - len(kept) == 1 else "s"))
        rows = kept
    if not rows:
        raise SystemExit("no samples left to work with")

    print("%s, against %s" % (args.file, args.config))
    print("%d sample%s, read as %s%s\n"
          % (len(data["samples"]), "" if len(data["samples"]) == 1 else "s",
             args.model,
             "" if meta.get("model") else " (the file does not say which)"))
    report(rows, kinematics)

    if args.refine:
        if args.sample is not None and not 1 <= args.sample <= len(rows):
            raise SystemExit("--sample must be between 1 and %d" % len(rows))
        do_refine(config, rows, args)
    elif args.fit:
        if len(rows) < MIN_PAIR_SAMPLES:
            raise SystemExit(
                "%d reading%s to fit with; %d is the minimum and %d is where a "
                "bad one stops mattering"
                % (len(rows), "" if len(rows) == 1 else "s", MIN_PAIR_SAMPLES,
                   RECOMMENDED_PAIR_SAMPLES))
        do_fit(config, rows, args)
    else:
        print("\n--refine corrects the geometry with the newest reading, "
              "--fit fits them all; neither writes anything without --apply")


if __name__ == "__main__":
    main()
