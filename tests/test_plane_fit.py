"""Checks for collecting and fitting a plane map, against a rendered box.

The camera is `SimCamera`, so every placement has an answer that is known
rather than measured, and the arm is a list of points this file computes —
which is the whole reason the session takes a point and a picture rather than
going and getting them. Nothing here needs a robot or a lens.
"""

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.geometry.kinematics import mat_to_pose as _mat_to_pose
from ur5dual.geometry.kinematics import pose_to_mat as _pose_to_mat
from ur5dual.tools.plane_fit import ENOUGH_CYCLES, PlaneFitError, PlaneFitSession
from ur5dual.vision.planar import PlaneFile, box_on_plane
from ur5dual.vision.service import VisionService


fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name +
          (("  " + detail) if detail else ""))
    if not ok:
        fail += 1


BOX = (0.60, 0.40, 0.20)
PLANE_Z = 1.20
RIM = PLANE_Z - BOX[2]          # the rim is one box height nearer than the table
work = tempfile.mkdtemp(prefix="openbox-planefit-")
rng = np.random.default_rng(19)

vision = VisionService({"source": "sim", "box_size": list(BOX),
                        "sim_plane_z": PLANE_Z, "confirm_frames": 1})
vision.start()


def seen_at(x, y, yaw=0.0):
    """What the camera reports with the box put at a place on the plane.

    The plane's axes are right-handed with +Y turned away from the lens, and
    the simulated camera is told about its box in its own axes, where +Y runs
    down the picture — so the sign of both y and yaw flips on the way in. This
    is the only place in these checks that has to know that, which is the
    point of solving the pairing rather than agreeing one.
    """
    vision.camera.place(centre=(x, -y), yaw=-yaw)
    # The same reset the panel does before every sample: the tracker smooths
    # corners across frames, and a calibration sample must be a measurement
    # rather than a blend with wherever the box was standing a moment ago.
    vision.refit()
    reading = vision.fresh(timeout=5.0)
    assert reading.found, reading.why_not()
    return reading.detection.corners


def touches_at(x, y, yaw=0.0, shuffle=True):
    """Where an arm would be driven, to touch the same four corners."""
    points = np.column_stack([rim_corners(x, y, yaw, BOX), np.full(4, RIM)])
    return points[rng.permutation(4)] if shuffle else points


PLACES = [(-0.15, -0.10, 0.0), (0.15, -0.10, math.radians(14)),
          (0.15, 0.10, math.radians(-11)), (-0.15, 0.10, math.radians(22))]

try:
    # ── the job as the calibration ────────────────────────────────────────
    print("\nthe same map, from pick-and-places instead of touched corners")
    # Two arms stood at opposite faces of the box: their tool centres end up
    # close together, which is exactly the case the old line-between-them
    # frame could not read and this one does not need to.
    HOLD = {"A": (0.012, -0.008), "B": (-0.011, 0.009)}

    def tools_at(x, y, yaw):
        """Two arms stood ready at the box — so they face the way it does.

        The rotation is the point: `stance_frame` reads which way the box is
        facing off the tool's orientation, because two arms posed at a box
        from opposite sides have their tool centres a couple of centimetres
        apart and the line between them says nothing.
        """
        cos, sin = math.cos(yaw), math.sin(yaw)
        return {arm: _pose_to_mat([x + cos * gx - sin * gy,
                                   y + sin * gx + cos * gy, RIM, 0, 0, yaw])
                for arm, (gx, gy) in HOLD.items()}

    # Spread over the patch the box actually travels, and turned through as
    # much of a range as the cell allows: the map is exact everywhere, but
    # what pins the grip offset down is the box being seen facing different
    # ways.
    CYCLES = [(-0.15, -0.10, 0.0), (0.15, -0.10, math.radians(14)),
              (0.15, 0.10, math.radians(-11)), (-0.15, 0.10, math.radians(22)),
              (0.0, 0.0, math.radians(8)), (0.10, 0.05, math.radians(-18))]

    job = PlaneFitSession(BOX, path=os.path.join(work, "job.json"))
    try:
        job.fit_cycles()
        check("fewer than three cannot disagree, and are refused", False)
    except PlaneFitError as e:
        check("fewer than three cannot disagree, and are refused",
              "fewest that can disagree" in str(e), str(e)[:44])

    for x, y, yaw in CYCLES:
        job.ready(tools_at(x, y, yaw))
        job.look(seen_at(x, y, yaw))
    check("every cycle is two presses and keeps both halves",
          len(job.cycles) == len(CYCLES)
          and sorted(job.cycles[0].tools) == ["A", "B"])

    print("\nwhat is wrong with a run of rounds, said while the box is in reach")
    lifted = PlaneFitSession(BOX, path=os.path.join(work, "lifted.json"))
    for i, (x, y, yaw) in enumerate(CYCLES[:3]):
        held = tools_at(x, y, yaw)
        if i == 1:                       # one round with the box still in the air
            held = {a: m.copy() for a, m in held.items()}
            for m in held.values():
                m[2, 3] += 0.240
        lifted.ready(held)
        lifted.look(seen_at(x, y, yaw))
    check("arms stood at different heights were not put in the same pose "
          "relative to the box, and it says so rather than blaming the "
          "plane's axes",
          "not the same one relative to the box" in lifted.why_not_fitting(),
          lifted.why_not_fitting()[:56])
    try:
        lifted.fit_cycles(height=RIM)
        check("and the fit refuses with that reason, not the generic one",
              False)
    except PlaneFitError as e:
        check("and the fit refuses with that reason, not the generic one",
              "not the same one relative to the box" in str(e)
              and "plane's axes" not in str(e), str(e)[:52])

    still = PlaneFitSession(BOX, path=os.path.join(work, "still.json"))
    for x, y, yaw in [(0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.0, 0.01, 0.0)]:
        still.ready(tools_at(x, y, yaw))
        still.look(seen_at(x, y, yaw))
    check("a box put down in the same place each round is one sample thrice",
          "one place seen several times" in still.why_not_fitting(),
          still.why_not_fitting()[:52])

    unturned = PlaneFitSession(BOX, path=os.path.join(work, "unturned.json"))
    for x, y in [(-0.15, -0.10), (0.15, -0.10), (0.15, 0.10)]:
        unturned.ready(tools_at(x, y, 0.0))
        unturned.look(seen_at(x, y, 0.0))
    check("and one that travelled but never turned is named too",
          "never turned" in unturned.why_not_fitting(),
          unturned.why_not_fitting()[:48])
    check("while the run that did all three has nothing said against it",
          job.why_not_fitting() == "", job.why_not_fitting())

    job.save()
    job_path = os.path.join(work, "job_plane.json")
    store, plane_map, offset = job.taught_from_cycles(
        "box_home", job_path, height=RIM)
    check("a map comes out of the job, with no corner ever touched",
          plane_map.rms < 0.010, plane_map.description)

    worst = 0.0
    for x, y, yaw in [(0.0, 0.0, 0.0), (0.06, -0.05, math.radians(19))]:
        got = box_on_plane(seen_at(x, y, yaw), plane_map, BOX)
        worst = max(worst, math.hypot(got.x - x, got.y - y))
    check("and it puts the box within a centimetre of where it was put",
          worst < 0.010, "%.1f mm worst" % (worst * 1000))

    held = store.stance("box_home")
    check("both holds fall out of the same fit, taught by nothing",
          sorted(held) == ["A", "B"])

    # Where those holds sit "on the box" is not a number to check against the
    # truth, and asking it to be one would be asking the wrong question. The
    # offset is fixed only up to a shift unless the box is turned through a
    # wide range between cycles, and the same shift goes into the map, because
    # the map is fitted through that same offset. What the two of them agree
    # on — and all a pick needs them to agree on — is where the arms have to
    # be for a box the camera can see. That is what is checked.
    worst_arm = 0.0
    for x, y, yaw in [(0.0, 0.0, 0.0), (0.08, -0.06, math.radians(19)),
                      (-0.10, 0.07, math.radians(-22))]:
        seen = box_on_plane(seen_at(x, y, yaw), plane_map, BOX)
        want = tools_at(x, y, yaw)
        for arm in held:
            got = store.stance_world("box_home", seen, arm)
            worst_arm = max(worst_arm,
                            float(np.linalg.norm(got[:3, 3] - want[arm][:3, 3])))
    check("and together they put the arms where the box actually needs them",
          worst_arm < 0.005, "%.2f mm worst" % (worst_arm * 1000))

    again = PlaneFitSession.load(job.path)
    check("and the cycles outlive the fit, like the placements do",
          len(again.cycles) == len(CYCLES))

    # ── what a program reads ──────────────────────────────────────────────
    print("\nwhat the fit writes, and what a re-fit does to what was taught")
    check("the map reaches the file a program reads",
          PlaneFile.load(job_path).ready, store.description)
    check("and the reference the last cycle taught came with it",
          "box_home" in PlaneFile.load(job_path).references)
    check("and nothing is stale yet", store.stale() == [], str(store.stale()))

    job.ready(tools_at(0.02, -0.03, math.radians(5)))
    job.look(seen_at(0.02, -0.03, math.radians(5)))
    store, _map, _off = job.taught_from_cycles("box_home", job_path,
                                               height=RIM)
    check("another pick-and-place re-fits and re-teaches in one action",
          "box_home" in store.references and len(job.cycles) == len(CYCLES) + 1)
    check("so nothing is left older than the map it was measured through — "
          "the reference cannot go stale when the fit re-teaches it",
          store.stale() == [], str(store.stale()))

    print("\nand the box the map was fitted with is the box it may be used for")
    check("the box it was fitted with raises nothing",
          store.fits_box(BOX) == "", store.fits_box(BOX))
    deeper = (BOX[0], BOX[1], BOX[2] + 0.030)
    check("a crate 30 mm deeper is named, with what to do about it",
          "fit again with this box" in store.fits_box(deeper),
          store.fits_box(deeper)[:60])
    try:
        store.correction("box_home", seen_at(0.0, 0.0), deeper)
        check("and a FIND through it is refused, not answered", False,
              "it answered anyway")
    except Exception as e:
        check("and a FIND through it is refused, not answered",
              "map is a map of one plane" in str(e), str(e)[:50])
    ok, found = store.correction("box_home", seen_at(0.0, 0.0), BOX)
    check("while the right box still goes straight through",
          np.allclose(ok @ store.reference("box_home").matrix(),
                      found.matrix(), atol=1e-9),
          found.describe())
finally:
    vision.stop()

print("\\nFAILURES: %d" % fail)
sys.exit(1 if fail else 0)
