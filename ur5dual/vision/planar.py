"""
Three numbers instead of six.

A box that always lies flat, always the same way up, always at the same height
above the same surface, has three degrees of freedom and not six. It slides in
x and y and it turns about the surface's normal. The other three are not
merely unused — they are the three this detector is worst at, and carrying
them into a pick spends accuracy on numbers that were never going to change.

Measured on this cell's own capture, at one pixel of corner noise: yaw comes
out to 0.20 deg and the two out-of-plane angles to 1.07 deg each, five times
worse, because yaw is read from the direction of a 368-pixel edge while the
other two are read from foreshortening, which is a second-order effect. Range
is worse still: 3.35 mm against 1 mm across the image, and it is the only one
of the six that a mistyped `box_size` moves at all — feed solvePnP a size 1%
wrong and the whole answer scales by exactly 1% about the optical axis, 11.8
mm of it in range, while the rotation does not move by so much as 4e-16.

So this module never asks for range and never asks for tilt. It asks where a
pixel lands on one known surface, which for a fixed camera and a fixed plane
is a plane-to-plane perspective map: a homography, eight numbers, and *exact*
— not a fit that happens to be close. On ideal data the residual is 0.0000 mm
at every position tried, including positions well outside the ones it was
fitted from, because a homography is the true model of this geometry rather
than an interpolation between samples.

What that buys, beyond accuracy, is that `vision.camera_to_world` stops
mattering. Where the lens is, which way it points, what `box_size` says, what
the lens does to straight lines — all of it collapses into those eight
numbers, and all of it is fitted from the box itself, at the height it
actually sits, over the patch of surface it actually moves on. The calibration
this package could not get below about 9 mm is not made better here; it is
made unnecessary.

The bill comes due in one place, and it is worth knowing before trusting any
of the above. The map is a map of *one plane*. A box whose rim sits higher
than the plane it was fitted at is reported at the wrong place, shifted along
the line of sight by the height error times the tangent of the camera's angle
from vertical — 1.88 mm per millimetre at this cell's 62 deg. Ten millimetres
of variation in box height is twenty-five millimetres of miss. If the boxes
stop being identical, this module is the wrong tool and `detect.py`'s full
pose is the right one.

That tangent is also the one argument for where to bolt the camera, and it
points the opposite way from `detect.py`'s. Full 6-DOF wants an oblique view,
because foreshortening is what fixes tilt: its out-of-plane noise falls from
1.07 to 0.30 deg between a square-on view and 60 deg. This module wants the
view as near overhead as the rig allows, which improves both of its terms at
once — at 35 deg the height penalty is 0.70 mm/mm instead of 1.88 and yaw
noise is 0.35 deg instead of 0.59. Nothing in software is worth as much as
moving the bracket.

Corner order is this package's throughout: front-left, front-right,
back-right, back-left, front being the bottom of the picture. It is assigned
from where the corners sit *in the image*, so a box turned far enough hands
back the same four points under shifted labels. That happens well inside the
travel of a cell that thinks it is safe: `rim.order_corners` leads with the
corner nearest the top left of the picture, a rule with no symmetry in it, so
the relabelling arrives at different angles turning one way and the other —
measured on this cell's geometry, clean out to +70 deg and already rolled by
one at -25 deg.

Nothing detects that from the image, and nothing needs to. A known rectangle
is its own guard: of the four cyclic labellings only two can fit 600x400 at
all, the other two are trying to fit 400x600, and `fit_rectangle` finds that
out by trying all four and keeping the closest. Relabelled corners therefore
come back as the correct placement, not as a rejected one, with a flag saying
the labels were a quarter out. What no shape can settle is the half turn,
which maps the rectangle exactly onto itself, so every angle here is known
modulo 180 deg and differences are wrapped into the nearer half — which is
what makes the -25 deg case, read as +155 deg, correct the same way it would
have been read the other way round.
"""

import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..geometry.kinematics import inv, mat_to_pose, pose_to_mat


class PlaneMapError(RuntimeError):
    pass


# Below this, the samples a map was fitted from do not enclose an area — they
# lie along a line, and a homography through them is exact on all of them and
# unconstrained across the line. It is the same failure `calibrate.spread_of`
# exists to catch, one dimension down. Corners of a box cannot trip it however
# they are arranged, because one rectangle already spans two dimensions; it is
# there for a caller feeding this something other than rim corners.
MIN_SPREAD_RATIO = 0.10

# What a map may fail to explain about its own samples and still be believed.
# This is not a noise threshold — hand-taught places on this cell land within a
# few millimetres, and the simulated camera, which rasterises its box onto
# whole pixels, fits to 2.0 mm. It is there to catch the one mistake that does
# not look like a mistake: plane axes handed the wrong way round, which pairs
# every corner with the corner opposite it. Measured, that fits to 179.9 mm
# against the same samples' 2.0 mm, so anything between the two settles it.
MAX_FIT_ERROR = 0.030

# How far the box a program is looking for may differ from the box a map was
# fitted with, per side. A map is a map of one plane, and the plane it maps is
# the rim of the box it was fitted from — swap in a crate 30 mm deeper and the
# rim it reads is 30 mm off the surface it knows, which at this cell's 62 deg
# is 56 mm of lateral miss with nothing on screen saying so. The opening's
# length and width matter for a second reason: they are the rectangle
# `fit_rectangle` fits, and fitting the wrong one turns the answer.
SIZE_TOLERANCE = 0.002


def wrap_half(angle):
    """An angle into (-90, +90] degrees, in radians.

    A rectangle maps onto itself under a half turn, so its yaw is only ever
    known to within 180 deg and the difference between two of them is only
    ever known to within 180 deg. Wrapping into the nearer half is what makes
    "the box turned 5 deg" and "the box turned 185 deg" the same answer, which
    is what they physically are.
    """
    return (float(angle) + math.pi / 2) % math.pi - math.pi / 2


def spread_of(points):
    """The two principal spans of a set of plane points, in metres.

    Samples along a line fit a homography perfectly and constrain it not at
    all across that line. This is what a caller checks before believing one.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return np.zeros(2)
    centred = points - points.mean(axis=0)
    _u, singular, _vt = np.linalg.svd(centred, full_matrices=False)
    return singular[:2] / np.sqrt(len(points))


def rim_corners(x, y, yaw, size):
    """The four rim corners of a box at (x, y, yaw), in the plane's own axes.

    The inverse of what `fit_rectangle` recovers, and what turns one taught
    box placement into the four correspondences a map is fitted from. Corner
    order and the object frame both match `detect.solve_opening_pnp`, so a
    pose from either source describes the same physical corner.

    Which way the plane's axes point is not free, and getting it wrong is the
    one mistake here that produces a confident answer rather than an obvious
    one. The package labels the rim's near edge — the bottom of the picture —
    "front", and this frame puts front on the -Y side: +X to the picture's
    right, +Y away from the lens, +Z out of the surface toward it. A frame
    handed the other way pairs every corner with the one diagonally opposite,
    which is a reflection, and no rotation fits it. `PlaneMap.from_samples`
    catches that by the residual and says so; it is mentioned here because
    this is the function that decides it.
    """
    length, width = float(size[0]), float(size[1])
    local = np.array([[-length / 2, -width / 2],
                      [ length / 2, -width / 2],
                      [ length / 2,  width / 2],
                      [-length / 2,  width / 2]], dtype=float)
    cos, sin = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = np.array([[cos, -sin], [sin, cos]])
    return local @ rotation.T + np.array([float(x), float(y)])


class PlaneMap:
    """Pixels to one plane, in the eight numbers that describe that map.

    Fitted from places seen twice: where the camera saw a corner, and where
    the arm says that corner was. It reports what it could not explain beside
    the answer, because a homography through four points fits those four
    exactly whatever they were, and a residual computed from the same samples
    that produced it is not evidence.
    """

    def __init__(self, matrix, height=None, samples=0, rms=0.0, worst=0.0):
        matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
        if not np.isfinite(matrix).all():
            raise PlaneMapError("this map has values that are not numbers")
        if abs(matrix[2, 2]) < 1e-12:
            raise PlaneMapError("this map is degenerate")
        # scale is free in a homography; pinning it makes two maps of the same
        # plane compare equal instead of merely proportional
        self.matrix = matrix / matrix[2, 2]
        # the rim height this map was fitted at. Never used in the arithmetic
        # — recorded because it is the one number whose change silently
        # invalidates everything above.
        self.height = None if height is None else float(height)
        self.samples = int(samples)
        self.rms = float(rms)
        self.worst = float(worst)

    # -- fitting -----------------------------------------------------------
    @classmethod
    def from_samples(cls, pixels, plane_xy, height=None):
        """The map carrying these pixels onto these plane points.

        Four correspondences are the fewest that determine a homography, and
        one box placement provides exactly four — its own rim corners. That is
        enough to solve and never enough to check, so three or four placements
        spread over the surface the box actually travels is what a real answer
        looks like: the extra rows are the only thing that can disagree.

        Spread matters for a different reason than it does when extrapolating.
        The map itself is exact everywhere, including far outside the samples;
        what samples crowded into one corner cannot do is average down the few
        millimetres of error in the teaching, or absorb what the lens does to
        straight lines out at the edges of the picture.
        """
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        plane_xy = np.asarray(plane_xy, dtype=float).reshape(-1, 2)
        if len(pixels) != len(plane_xy):
            raise PlaneMapError("every pixel needs the place it stands for")
        if len(pixels) < 4:
            raise PlaneMapError(
                "four correspondences are the fewest that fix a plane map; "
                "got %d — one box placement gives four" % len(pixels))

        spread = spread_of(plane_xy)
        if spread[0] <= 0 or spread[1] / spread[0] < MIN_SPREAD_RATIO:
            raise PlaneMapError(
                "these places lie along a line (spread %.0f x %.0f mm), and a "
                "map through them is exact on them and unconstrained across "
                "them — move the box to a place off that line"
                % (spread[0] * 1000, spread[1] * 1000))

        matrix, _mask = cv2.findHomography(pixels, plane_xy, 0)
        if matrix is None:
            raise PlaneMapError(
                "no plane map fits these %d places" % len(pixels))

        out = cls(matrix, height=height, samples=len(pixels))
        errors = out.residuals(pixels, plane_xy)
        out.rms = float(np.sqrt(np.mean(errors ** 2)))
        out.worst = float(errors.max())
        if out.rms > MAX_FIT_ERROR:
            raise PlaneMapError(
                "no plane map explains these %d places better than %.0f mm, "
                "which is too far wrong to be where they were measured. The "
                "usual cause is the plane's axes running the other way: this "
                "package's corner order puts the rim's near edge — the bottom "
                "of the picture — on the -Y side, and a frame handed the other "
                "way pairs every corner with the one opposite it"
                % (len(pixels), out.rms * 1000))
        return out

    # -- using it ----------------------------------------------------------
    def to_plane(self, pixels):
        """Pixels onto the plane, as (N, 2) metres in the plane's own axes."""
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pixels, self.matrix).reshape(-1, 2)

    def residuals(self, pixels, plane_xy):
        """How far each mapped pixel lands from where it should, in metres."""
        moved = self.to_plane(pixels)
        return np.linalg.norm(
            moved - np.asarray(plane_xy, dtype=float).reshape(-1, 2), axis=1)

    # -- io ----------------------------------------------------------------
    def to_dict(self):
        return {"matrix": [[float(v) for v in row] for row in self.matrix],
                "height": self.height, "samples": self.samples,
                "rms": self.rms, "worst": self.worst}

    @classmethod
    def from_dict(cls, d):
        return cls(d["matrix"], d.get("height"), d.get("samples", 0),
                   d.get("rms", 0.0), d.get("worst", 0.0))

    @property
    def description(self):
        return ("plane map from %d places%s, fits to %.1f mm rms, %.1f mm worst"
                % (self.samples,
                   "" if self.height is None else " at %.0f mm" % (self.height * 1000),
                   self.rms * 1000, self.worst * 1000))


def fit_rectangle(points, length, width):
    """The placement of a known rectangle over four measured corners.

    Kabsch in two dimensions, with the sign of the determinant fixed so the
    answer is a rotation: a reflection fits mirrored corners better than any
    rotation does, and a mirrored placement turns the box the wrong way by
    twice its angle.

    All four cyclic labellings are tried and the closest fitting is kept,
    which is what makes the image-assigned corner order safe. Two of the four
    are the same rectangle under a half turn and fit equally well, so the
    winner is only ever determined to within 180 deg; the caller settles that
    against a taught angle, which is the only thing that knows.

    Returns the centre, the angle, the rms of the fit, and whether the
    labelling that won was an odd one — meaning the corners arrived a quarter
    turn out of the order their labels claimed. That is a routine event, not a
    fault: the image-based labelling rolls partway through this cell's own
    travel. The angle returned is the corrected one either way.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) != 4:
        raise PlaneMapError("a rectangle is fitted over four corners, got %d"
                            % len(points))
    model = np.array([[-length / 2, -width / 2],
                      [ length / 2, -width / 2],
                      [ length / 2,  width / 2],
                      [-length / 2,  width / 2]], dtype=float)

    measured_centre = points.mean(axis=0)
    centred = points - measured_centre
    best = None
    for roll in range(4):
        turned = np.roll(model, roll, axis=0)
        covariance = turned.T @ centred
        u, _s, vt = np.linalg.svd(covariance)
        sign = np.diag([1.0, float(np.sign(np.linalg.det(vt.T @ u.T)))])
        rotation = vt.T @ sign @ u.T
        error = float(np.sqrt(np.mean(np.sum(
            (turned @ rotation.T - centred) ** 2, axis=1))))
        if best is None or error < best[0]:
            best = (error, rotation, roll)

    error, rotation, roll = best
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return measured_centre, yaw, error, bool(roll % 2)


class Placement:
    """Where the box is on its surface: two metres and one angle."""

    def __init__(self, x, y, yaw, fit_error=0.0, side_swapped=False):
        self.x = float(x)
        self.y = float(y)
        # known modulo 180 deg — a rectangle cannot tell a half turn from none
        self.yaw = float(yaw)
        # how far the known rectangle sat from the four mapped corners. The
        # only line here that can say the four corners were not this box.
        self.fit_error = float(fit_error)
        self.side_swapped = bool(side_swapped)

    @property
    def xy(self):
        return np.array([self.x, self.y], dtype=float)

    def matrix(self):
        """The placement as a transform of the plane's own axes."""
        cos, sin = math.cos(self.yaw), math.sin(self.yaw)
        out = np.eye(4)
        out[0, 0], out[0, 1] = cos, -sin
        out[1, 0], out[1, 1] = sin, cos
        out[0, 3], out[1, 3] = self.x, self.y
        return out

    def to_dict(self):
        return {"x": self.x, "y": self.y, "yaw": self.yaw,
                "fit_error": self.fit_error, "side_swapped": self.side_swapped}

    @classmethod
    def from_dict(cls, d):
        return cls(d["x"], d["y"], d["yaw"], d.get("fit_error", 0.0),
                   d.get("side_swapped", False))

    def describe(self):
        return ("box at %.0f %.0f mm, turned %.1f deg, corners fit to %.1f mm%s"
                % (self.x * 1000, self.y * 1000, math.degrees(self.yaw),
                   self.fit_error * 1000,
                   ", labels a quarter out" if self.side_swapped else ""))


def box_on_plane(corners, plane_map, size):
    """Where the box the detector found is standing, in three numbers.

    `corners` are image corners in this package's order, straight from
    `rim.find_rim_quad` or `detect.Detection.corners`; `size` is the opening,
    long side first, as `vision.box_size` gives it.
    """
    mapped = plane_map.to_plane(np.asarray(corners, dtype=float).reshape(4, 2))
    centre, yaw, error, swapped = fit_rectangle(
        mapped, float(size[0]), float(size[1]))
    return Placement(centre[0], centre[1], yaw, error, swapped)


def planar_correction(now, was):
    """The transform carrying everything taught at `was` to where `now` is.

    The same rigid 4x4 a `FIND` already stores and a corrected target is
    already carried by, so nothing downstream learns that three numbers came
    in where six used to. It is built as: undo where the box was, turn about
    the plane's normal, put it where the box is —

        T(now) . Rz(now.yaw - was.yaw) . T(-was)

    which leaves Z alone and rotates only about Z. That is a statement about
    the plane, not a shortcut: this module's coordinates are the plane's own
    axes, so a box turning about the surface it lies on turns about the third
    axis of those. A cell whose surface is not level in world needs its plane
    frame wrapped round this, `P . delta . inv(P)`; a level one does not,
    because a turn about Z commutes with a rise along Z.

    The turn is wrapped into the nearer half, so a rectangle read a half turn
    from the way it was taught corrects by the small angle rather than by the
    large one. Nothing here refuses a large correction — `vision.max_correction`
    is the cell's policy and the executor already applies it, and a second
    limit in a second place is a limit somebody will set twice and disagree
    with once.
    """
    turn = wrap_half(now.yaw - was.yaw)
    cos, sin = math.cos(turn), math.sin(turn)
    rotation = np.array([[cos, -sin], [sin, cos]])
    out = np.eye(4)
    out[:2, :2] = rotation
    out[:2, 3] = now.xy - rotation @ was.xy
    return out


# Where the surface is written down. Not in `cell.yaml`, because what is in
# here is measured rather than chosen: eight numbers nobody can read, and one
# placement per taught pick that is only meaningful beside the pick it was
# taught with. A file that a person is expected to edit and a file that a
# calibration overwrites are different files.
DEFAULT_PLANE_FILE = "config/plane.json"


def _mm(size):
    """A box size as the millimetres a person would say: 200x100x100."""
    return "x".join("%.0f" % (v * 1000) for v in size)


def sized_path(base, box_size):
    """That file's name with the box's size in it.

    A map is a map of one plane, and the plane it maps is the rim of one box —
    so a cell that runs two crates needs two maps, and the one it wants is
    decided by the size already on the Camera tab rather than by remembering
    to load something. `config/plane.json` with a 200x100x100 crate in front
    of the lens becomes `config/plane_200x100x100.json`, and putting the other
    crate back finds the other file rather than refusing the wrong one.

    `PlaneFile.fits_box` still guards what is inside: this decides which file
    to open, and that decides whether what was opened can be believed. A path
    can be typed by hand and a file can be edited.
    """
    if base is None or box_size is None:
        return base
    stem = Path(base)
    return stem.with_name("%s_%s%s" % (stem.stem, _mm(box_size), stem.suffix))


class PlaneFile:
    """The surface: its map, and the places picks were taught against.

    Both live here because neither survives the other. Re-fitting the map
    moves every reference it was measured through, and re-teaching a pick
    without re-reading its reference leaves a correction measured from a
    place the box is no longer taught at. Keeping them in one file is what
    makes "these were true at the same moment" a property of the file rather
    than something an operator has to remember.

    A missing file is not an error. It loads as an empty one, for the reason
    `make_camera` answers with a simulated source rather than refusing: the
    thing being set up is usually the thing that is not there yet, and the
    panel that sets it up has to open first. What refuses is *using* what is
    not there — asking for a map or a reference that was never measured.
    """

    def __init__(self, plane_map=None, references=None, box_size=None,
                 path=None, saved=None, fitted=None, taught=None):
        self.map = plane_map
        self.references = dict(references or {})
        self.box_size = None if box_size is None else tuple(
            float(v) for v in box_size)
        self.path = Path(path) if path else Path(DEFAULT_PLANE_FILE)
        self.saved = saved
        # when the map was fitted, and when each reference was read through
        # it. A reference is only as good as the map it was measured with, so
        # a re-fit does not invalidate the file — it makes some of it older
        # than the rest, and `stale` is the only thing that can say which.
        self.fitted = fitted
        self.taught = dict(taught or {})
        # Where the tool sits when it has hold of the box, expressed in the
        # box's own frame. Stored as the difference rather than as the world
        # pose it was taught at, because the difference is the part that stays
        # true: re-fit the map and every placement moves, including the one a
        # pick was taught against, and a world pose taught beside it does not
        # move with it. This does.
        # Where each arm's tool sits when the two of them have hold of the
        # box, in the box's own frame: {reference: {arm: 4x4}}. Per arm
        # because this cell picks with both, and the two holds are not one
        # number and an offset — they are wherever it was sensible to grip,
        # and the distance between them is the workpiece.
        self.stances = {}

    # -- io ----------------------------------------------------------------
    @classmethod
    def load(cls, path=None):
        path = Path(path or DEFAULT_PLANE_FILE)
        if not path.exists():
            return cls(path=path)
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PlaneMapError("%s cannot be read: %s" % (path, exc))
        try:
            plane_map = (PlaneMap.from_dict(held["map"])
                         if held.get("map") else None)
            references = {name: Placement.from_dict(value)
                          for name, value in (held.get("references") or {}).items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise PlaneMapError(
                "%s is not a plane file this version understands: %s"
                % (path, exc))
        rows = held.get("references") or {}
        taught = {name: value.get("taught") for name, value in rows.items()}
        out = cls(plane_map, references, held.get("box_size"), path,
                  held.get("saved"), held.get("fitted"), taught)
        out.stances = {
            name: {arm: pose_to_mat(pose)
                   for arm, pose in (value.get("stances") or {}).items()}
            for name, value in rows.items() if value.get("stances")}
        return out

    def save(self, path=None):
        path = Path(path or self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.saved = datetime.now().astimezone().isoformat(timespec="seconds")
        with path.open("w", encoding="utf-8") as f:
            json.dump({
                "saved": self.saved,
                "fitted": self.fitted,
                "box_size": None if self.box_size is None else list(self.box_size),
                "map": None if self.map is None else self.map.to_dict(),
                "references": {
                    name: dict(
                        place.to_dict(), taught=self.taught.get(name),
                        stances=(None if name not in self.stances else
                               {arm: [float(v) for v in mat_to_pose(m)]
                                for arm, m in sorted(self.stances[name].items())}))
                    for name, place in sorted(self.references.items())},
            }, f, indent=2)
        self.path = path
        return path

    # -- what it holds -----------------------------------------------------
    @property
    def ready(self):
        return self.map is not None

    def plane_map(self):
        if self.map is None:
            raise PlaneMapError(
                "this cell has no plane map yet, so a detection cannot be "
                "turned into a place on the table — put the box at three or "
                "four places, say where each one was, and fit one into %s"
                % self.path)
        return self.map

    def reference(self, name):
        """Where the box stood when the picks named against it were taught."""
        name = (name or "").strip()
        if name not in self.references:
            known = ", ".join(sorted(self.references)) or "nothing"
            raise PlaneMapError(
                "nothing was taught against %r, so there is no place to "
                "measure how far the box has moved *from* — teach it with the "
                "box standing where the pick was taught (%s has: %s)"
                % (name, self.path, known))
        return self.references[name]

    def set_map(self, plane_map):
        """Put a freshly fitted map in, and stamp when it arrived.

        Everything already taught keeps its numbers and stops being current at
        this moment. Dropping the references instead would be tidier and would
        throw away the only record of where those picks were taught from; `stale`
        names them instead, and re-teaching is one press each.
        """
        self.map = plane_map
        # Microseconds, not milliseconds: `stale` is a comparison between two
        # stamps taken moments apart, and a resolution that can call them
        # equal is a resolution that reports nothing.
        self.fitted = datetime.now().astimezone().isoformat()
        return self.map

    def stale(self):
        """References taught before the map they were measured through."""
        if not self.fitted:
            return []
        return sorted(name for name in self.references
                      if not self.taught.get(name)
                      or self.taught[name] < self.fitted)

    def teach_stance(self, name, tools):
        """Record where the arms are, as distances from the box beneath them.

        `tools` maps arm id to that arm's tool pose in the cell — both of them
        at once, because both are on the box at once and the moment is the
        thing being recorded. Teaching them one at a time would be teaching
        two different moments, and the box need only be nudged between them
        for the pair to describe a workpiece that was never there.
        """
        placement = self.reference(name)
        undo = inv(placement.matrix())
        held = {}
        for arm_id, tool in dict(tools).items():
            tool = np.asarray(tool, dtype=float)
            if tool.shape != (4, 4):
                tool = pose_to_mat(tool)
            held[str(arm_id)] = undo @ tool
        if not held:
            raise PlaneMapError(
                "no arm was holding the box, so there is nothing to record")
        self.stances[str(name).strip()] = held
        return held

    def stance(self, name, arm=None):
        """The taught hold in the box's frame: all of them, or one arm's."""
        held = self.stances.get(str(name or "").strip())
        if held is None or arm is None:
            return held
        return held.get(str(arm))

    def stance_arms(self, name):
        return sorted(self.stances.get(str(name or "").strip()) or {})

    def stance_world(self, name, placement, arm=None):
        """Those holds, carried to where the box is now."""
        held = self.stance(name)
        if not held:
            raise PlaneMapError(
                "nothing was taught as the hold on %r — set the box down with "
                "the arms still on it and press once, then take them out of "
                "the view and press again" % name)
        carried = {a: placement.matrix() @ m for a, m in held.items()}
        if arm is None:
            return carried
        if str(arm) not in carried:
            raise PlaneMapError(
                "arm %s was not holding %r when it was taught (%s was)"
                % (arm, name, ", ".join(sorted(carried)) or "nothing"))
        return carried[str(arm)]

    def teach(self, name, placement):
        """Record where the box is standing, as the zero for a taught pick.

        Called at the moment the pick is taught and not before or after: the
        two numbers are a pair, and a reference read after the box was nudged
        is a correction that will move every pick by the nudge.
        """
        name = str(name).strip()
        self.references[name] = placement
        self.taught[name] = datetime.now().astimezone().isoformat()
        return placement

    def forget(self, name):
        self.taught.pop(str(name).strip(), None)
        self.stances.pop(str(name).strip(), None)
        return self.references.pop(str(name).strip(), None)

    # -- the whole job, for a caller that only wants the answer -------------
    def fits_box(self, box_size):
        """Is this the box the map was fitted with? The reason, or empty.

        Asked at every `FIND` rather than trusted, because the box size is a
        live control on the Camera tab and the map is a file. Changing one
        without the other is a single click, and what it produces is not a
        refusal but an answer — the wrong rectangle fitted through a map of
        the wrong plane, confident and quiet.
        """
        if self.box_size is None or box_size is None:
            return ""
        asked = [float(v) for v in box_size]
        fitted = [float(v) for v in self.box_size]
        if len(asked) != len(fitted):
            asked, fitted = asked[:2], fitted[:2]
        off = max(abs(a - b) for a, b in zip(asked, fitted))
        if off <= SIZE_TOLERANCE:
            return ""
        return ("this map was fitted with a %s box and the cell is now set to "
                "%s — %.0f mm apart. A map is a map of one plane, and that "
                "plane is the rim of the box it was fitted from: fit again "
                "with this box, or put the other one back"
                % (_mm(fitted), _mm(asked), off * 1000))

    def correction(self, name, corners, box_size=None):
        """The transform a pick taught against `name` should be carried by.

        Everything `FIND` needs, from image corners to the 4x4 a corrected
        target multiplies by, against the reference this file already holds.
        """
        size = box_size or self.box_size
        if size is None:
            raise PlaneMapError(
                "this file does not say what size the box is, and no size was "
                "given — pass `vision.box_size`")
        wrong_box = self.fits_box(size)
        if wrong_box:
            raise PlaneMapError(wrong_box)
        found = box_on_plane(corners, self.plane_map(), size)
        return planar_correction(found, self.reference(name)), found

    @property
    def description(self):
        if self.map is None:
            return "no plane map in %s" % self.path
        stale = self.stale()
        return "%s, taught against %s%s" % (
            self.map.description,
            ", ".join(sorted(self.references)) or "nothing yet",
            "" if not stale else
            " — %s older than the map and worth teaching again"
            % ", ".join(stale))


# ── fitting a map from the work itself ────────────────────────────────────
#
# Touching four corners with the arm is one way to say where the box is. Doing
# the job is another, and it is the one that costs nothing: two arms carry the
# box, set it down and are still gripping it. Their poses at that moment are a
# frame rigidly attached to the box — not the box's frame, because nobody
# measured where the arms stand on it, but a frame that differs from it by the
# *same* unknown every time.
#
# That is what makes this solvable with three unknowns instead of eleven. The
# map has eight degrees of freedom and the grip-to-box offset has three, but
# the eight are determined in closed form once the three are guessed: propose
# an offset, and every cycle's rim corners are known in the cell, which is a
# plain homography fit. So only the three are searched, and the residual of
# the inner fit is what scores them.
#
# The offset falls out beside the map, and it is exactly what a hold is. One
# action therefore produces both: where the box is when the camera sees it,
# and where each arm was holding it — with no separate teaching step, because
# the teaching step was the job.

# How far the grip frame may be assumed to sit from the middle of the box
# before the search starts looking there. Two arms taking hold of a box put
# their midpoint near its middle and their line along it; this is the width of
# "near", not a limit on the answer.
STANCE_GUESS = 0.06


def stance_frame(tools):
    """The frame the holding arms put on the box, from their tool poses.

    Origin at the mean of the tool positions, heading from the *orientation*
    of the first of them, both flattened onto the plane.

    The heading used to come from the line between the two tools, and that was
    wrong in a way only a real cell showed: two arms taking hold of a box from
    opposite sides put their tool centres on opposite faces of it, which is a
    few centimetres apart and can be less. Measured here, 23 mm — from which a
    direction is worth nothing, and a guard had to refuse a grip that was
    perfectly good.

    A gripper that has hold of the box turns with it, so its orientation says
    which way the box is facing directly and needs no baseline at all. Which
    arm's is used does not matter: whatever constant sits between that tool
    and the box is absorbed into the offset the fit solves for, the same way
    the position offset is. Both poses are still recorded — each is that
    arm's hold — but only one is asked which way round things are.
    """
    poses = [np.asarray(t, dtype=float) for t in tools]
    if not poses:
        raise PlaneMapError("no arm was holding the box")
    if any(p.shape != (4, 4) for p in poses):
        raise PlaneMapError(
            "a grip frame is built from tool poses, not positions: the way "
            "the tool is turned is what says which way the box is facing")
    middle = np.mean([p[:2, 3] for p in poses], axis=0)
    heading = poses[0][:3, :3] @ np.array([1.0, 0.0, 0.0])
    if float(np.hypot(heading[0], heading[1])) < 1e-6:
        raise PlaneMapError(
            "the tool points straight down the plane's normal, so nothing "
            "about the way it is turned says which way the box faces")
    return _planar_matrix(middle[0], middle[1],
                          math.atan2(heading[1], heading[0]))


def _planar_matrix(x, y, yaw):
    cos, sin = math.cos(yaw), math.sin(yaw)
    out = np.eye(3)
    out[:2, :2] = [[cos, -sin], [sin, cos]]
    out[0, 2], out[1, 2] = x, y
    return out


def _homography(pixels, plane_xy):
    matrix, _mask = cv2.findHomography(np.asarray(pixels, dtype=float),
                                       np.asarray(plane_xy, dtype=float), 0)
    return matrix


def _corner_block(size):
    """The rim's four corners in the box's own frame, as 3xN homogeneous."""
    return np.column_stack([rim_corners(0.0, 0.0, 0.0, size),
                            np.ones(4)]).T


def _cycle_fit(cycles, offset, size, rounds=3):
    """The map implied by one guess at the grip-to-box offset, and its error.

    Which image corner is which is not settled across cycles and cannot be
    assumed to be: `rim.order_corners` labels them by where they sit in the
    picture, and that rolls by one as the box turns — measured on this cell's
    geometry, already at -25 degrees. So the labels are re-read here the same
    way the map itself can settle: fit with the labels as they came, let that
    map say which corner each one really is, and fit again. Twice is enough;
    the first pass only has to be close.
    """
    local = _corner_block(size)
    prepared = [(np.asarray(frame, float) @ offset @ local,
                 np.asarray(corners, dtype=float).reshape(4, 2))
                for frame, corners in cycles]
    rolls = [0] * len(prepared)
    matrix = error = None
    for step in range(max(1, rounds)):
        world, pixels = [], []
        for (block, corners), roll in zip(prepared, rolls):
            world.append(block[:2].T)
            pixels.append(np.roll(corners, roll, axis=0))
        world = np.vstack(world)
        pixels = np.vstack(pixels)
        matrix = _homography(pixels, world)
        if matrix is None:
            return None, float("inf"), None, None
        moved = cv2.perspectiveTransform(
            pixels.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        error = float(np.sqrt(np.mean(np.sum((moved - world) ** 2, axis=1))))
        if step + 1 == rounds:
            break
        settled = True
        for i, (block, corners) in enumerate(prepared):
            want = block[:2].T
            seen = cv2.perspectiveTransform(
                corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
            costs = [np.sum(np.linalg.norm(np.roll(seen, r, axis=0) - want,
                                           axis=1)) for r in range(4)]
            pick = int(np.argmin(costs))
            if pick != rolls[i]:
                rolls[i], settled = pick, False
        if settled:
            break
    return matrix, error, pixels, world


def _search(cost, start, step, shrink=0.55, rounds=60):
    """Coordinate descent with a shrinking step, over three numbers.

    Not an optimiser worth importing SciPy for: the surface is smooth, the
    guess is close because two arms holding a box are near its middle, and
    three parameters at sixty rounds is a few milliseconds. Keeping the
    dependency list at numpy and OpenCV is worth more than the last decimal.
    """
    best = np.asarray(start, dtype=float).copy()
    value = cost(best)
    step = np.asarray(step, dtype=float).copy()
    for _ in range(rounds):
        improved = False
        for i in range(len(best)):
            for direction in (1.0, -1.0):
                trial = best.copy()
                trial[i] += direction * step[i]
                score = cost(trial)
                if score < value:
                    best, value, improved = trial, score, True
                    break
        if not improved:
            step *= shrink
            if float(np.max(step)) < 1e-9:
                break
    return best, value


def fit_from_cycles(cycles, size, height=None):
    """A map, and where the arms were holding the box, from the job itself.

    `cycles` are (grip frame, image corners) — one per pick-and-place, the
    frame taken from the two arms at the moment the box was set down and the
    corners from the picture taken after they came out of the view.

    Returns the map and the grip-to-box offset. Two cycles determine the
    eleven numbers between them and say nothing about whether they are right;
    three is the fewest that can disagree, and disagreement is the only
    evidence a fit ever offers.
    """
    prepared = [(np.asarray(frame, dtype=float).reshape(3, 3),
                 np.asarray(corners, dtype=float).reshape(4, 2))
                for frame, corners in cycles]
    if len(prepared) < 3:
        raise PlaneMapError(
            "three pick-and-places are the fewest that can disagree about "
            "where the arms stand relative to the box; got %d" % len(prepared))

    def cost(p):
        return _cycle_fit(prepared, _planar_matrix(*p), size)[1]

    # Four starts, a quarter turn apart. Which arm ends up on which side of
    # the box is nobody's decision, so the frame the two of them make can
    # point along the box or against it, and a search that only shrinks its
    # step cannot walk half a turn to find out. Trying all four costs four
    # closed-form fits and removes the question.
    best, value = None, float("inf")
    for quarter in range(4):
        start = [0.0, 0.0, quarter * math.pi / 2]
        found, score = _search(cost, start,
                               [STANCE_GUESS, STANCE_GUESS, math.radians(20)])
        if score < value:
            best, value = found, score
    offset = _planar_matrix(*best)
    matrix, _error, pixels, world = _cycle_fit(prepared, offset, size)
    if matrix is None:
        raise PlaneMapError("no plane map fits these pick-and-places")
    return PlaneMap.from_samples(pixels, world, height=height), offset


def placement_of(frame, offset):
    """Where the box was, from the frame the arms put on it and the offset."""
    box = np.asarray(frame, dtype=float) @ np.asarray(offset, dtype=float)
    return Placement(box[0, 2], box[1, 2], math.atan2(box[1, 0], box[0, 0]))


def stances_from_cycle(tools, frame, offset):
    """Where each arm was holding the box, in the box's own frame.

    The offset `fit_from_cycles` recovered puts the box's frame on the one the
    arms made; running their tool poses back through it says where each tool
    sits on the box, which is a hold and needed no teaching of its own. The
    job taught it.
    """
    undo = inv(placement_of(frame, offset).matrix())
    return {str(arm): undo @ np.asarray(pose, dtype=float).reshape(4, 4)
            for arm, pose in dict(tools).items()}
