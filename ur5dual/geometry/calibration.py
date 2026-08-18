"""
Measuring where arm B really is, relative to arm A.

The mounting numbers off a drawing are good to a few millimetres. Two-arm
grasping is not: both grippers hold one rigid object, so any error in the
base-to-base transform becomes a permanent fight between the arms, felt as
wrist force and seen as a protective stop.

So it gets measured instead. Touch both TCPs to the same physical point —
a pointed tip in a fixture, a dimple in a block — and record what each arm
thinks the point is. Do that at several well-spread points and the transform
that maps one arm's readings onto the other's is over-determined, which is
what makes it robust: the solver takes the transform that fits all of them
best, and reports how well it fit so a bad touch cannot pass silently.

The maths is Kabsch/Umeyama: centre both point sets, take the SVD of their
cross-covariance, and read the rotation off it. Reflections are rejected —
without that guard a noisy set can produce a mirrored "solution" that fits
the numbers and is physically impossible.
"""

import math

import numpy as np

from .kinematics import (inv, mat_to_rotvec, mat_to_xyz_rpy, pose_to_mat,
                         rotvec_to_mat)
from .ur_kinematics import UR5_DH, fk

MIN_POINTS = 3
# tests/test_calibration.py measures what these buy against 0.2 mm touch
# repeatability: 4 points lands the base transform within ~0.8 mm, 8 points
# within ~0.4 mm. Six is where the curve stops paying for itself.
RECOMMENDED_POINTS = 6
# below this the points are nearly collinear and the rotation about that line
# is unconstrained — the fit will look good and the transform will be wrong
MIN_SPREAD = 0.05          # m, smallest singular value of the centred cloud


class CalibrationError(RuntimeError):
    pass


class TouchPoint:
    """One physical point, as seen by each arm's own base frame."""

    def __init__(self, name, p_a, p_b):
        self.name = name
        self.p_a = np.asarray(p_a, dtype=float)[:3]
        self.p_b = np.asarray(p_b, dtype=float)[:3]

    def to_dict(self):
        return {"name": self.name,
                "p_a": [float(v) for v in self.p_a],
                "p_b": [float(v) for v in self.p_b]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["p_a"], d["p_b"])


def kabsch(P, Q):
    """Rigid transform T with T @ P ~= Q, for two matched Nx3 point sets."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape or P.shape[0] < MIN_POINTS:
        raise CalibrationError("need at least %d matched points" % MIN_POINTS)

    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    # flip the least-significant axis rather than accept a mirrored solution
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cq - R @ cp
    return T, S


class BaseCalibration:
    """Collect touch points, solve for A-base -> B-base, and grade the fit."""

    def __init__(self):
        self.points = []

    def add_from_cell(self, cell, name=None):
        """Record where each arm currently thinks its TCP is.

        Both TCPs must be touching the same physical point when this is
        called — that is the whole measurement.
        """
        for a in ("A", "B"):
            if not cell.arms[a].connected:
                raise CalibrationError("arm %s is not connected" % a)
        pt = TouchPoint(name or "P%d" % (len(self.points) + 1),
                        cell.arms["A"].tcp_pose()[:3],
                        cell.arms["B"].tcp_pose()[:3])
        self.points.append(pt)
        return pt

    def remove(self, index):
        if 0 <= index < len(self.points):
            self.points.pop(index)

    def clear(self):
        self.points = []

    # -- solving -----------------------------------------------------------
    def spread(self):
        """How three-dimensional the point cloud is, in metres.

        This is the smallest singular value of the centred A-side points: it
        goes to zero when the touches lie on a line or a plane, and that is
        precisely when the fit stops determining the transform.
        """
        if len(self.points) < MIN_POINTS:
            return 0.0
        P = np.array([p.p_a for p in self.points])
        return float(np.linalg.svd(P - P.mean(axis=0), compute_uv=False)[-1])

    def solve(self):
        """Return (T_baseA_baseB, report).

        Points are matched the other way round from what reads naturally:
        a point sits at p_a in A's frame and p_b in B's frame, so the
        transform that carries B's readings into A's frame is the one that
        maps p_b onto p_a — and that is exactly A-base -> B-base.
        """
        if len(self.points) < MIN_POINTS:
            raise CalibrationError(
                "%d points recorded, need at least %d (%d recommended)"
                % (len(self.points), MIN_POINTS, RECOMMENDED_POINTS))

        P = np.array([p.p_b for p in self.points])
        Q = np.array([p.p_a for p in self.points])
        T, _ = kabsch(P, Q)

        residuals = np.linalg.norm((T[:3, :3] @ P.T).T + T[:3, 3] - Q, axis=1)
        spread = self.spread()
        report = {
            "points": len(self.points),
            "rms_mm": float(np.sqrt((residuals ** 2).mean()) * 1000),
            "max_mm": float(residuals.max() * 1000),
            "per_point_mm": [float(r * 1000) for r in residuals],
            "spread_mm": spread * 1000,
            "warnings": [],
        }
        if spread < MIN_SPREAD:
            report["warnings"].append(
                "touch points span only %.0f mm out of plane — move them "
                "further apart in all three directions, or the rotation is "
                "guesswork" % (spread * 1000))
        if len(self.points) < RECOMMENDED_POINTS:
            report["warnings"].append(
                "%d points is the bare minimum; with no redundancy a single "
                "bad touch cannot be detected" % len(self.points))
        if report["max_mm"] > 2.0:
            report["warnings"].append(
                "worst point is off by %.1f mm — suspect that touch"
                % report["max_mm"])
        return T, report

    def apply_to_config(self, config):
        """Write the measured geometry into arm B, keeping arm A as the
        reference. The cell stops being a preset and becomes measured."""
        T_a_b = self.solve()[0]
        T_world_b = config.arms["A"].base_matrix() @ T_a_b
        config.arms["B"].set_base_matrix(T_world_b)
        config.set_custom_mount()
        config.calibrated = True
        return T_a_b

    # -- io ----------------------------------------------------------------
    def to_dict(self):
        return {"points": [p.to_dict() for p in self.points]}

    def load_dict(self, d):
        self.points = [TouchPoint.from_dict(p) for p in (d.get("points") or [])]
        return self


def describe(T):
    """Human-readable transform, for the wizard's result line."""
    xyz, rpy = mat_to_xyz_rpy(T)
    return ("xyz %.1f %.1f %.1f mm   rpy %.2f %.2f %.2f deg"
            % (xyz[0] * 1000, xyz[1] * 1000, xyz[2] * 1000,
               np.degrees(rpy[0]), np.degrees(rpy[1]), np.degrees(rpy[2])))


# ── pairing the two flanges ───────────────────────────────────────────────
# The measurement anyone can take without a jig: drive the two arms until the
# axis-6 flanges face each other, lay a gauge across them, and write down the
# gap. What that reading is worth is worth being exact about, because it looks
# like it says more than it does.
#
# One reading constrains three of the six numbers between the two bases:
#
#   parallel faces   fixes the direction of B's flange axis in A's flange
#                    frame — two angles. The third, the *spin* of one flange
#                    about that common axis, is invisible: turn J6 on either
#                    arm and the faces stay just as parallel and just as far
#                    apart.
#   the gap          fixes the distance along that common axis — one number.
#                    Where the two flange centres sit sideways of each other,
#                    the other two, it says nothing about.
#
# So a single pairing cannot replace a touch-off. It can do two other things.
# It can *correct* a geometry that is already roughly right, in exactly the
# three directions it measures and no others — `refine_from_flange_pair`. And
# several of them, taken in postures that point the flange axis in genuinely
# different directions, over-determine the whole transform — which is what
# `FlangePairCalibration` fits, and what config/flange_log.json has been
# collecting samples for.
#
# What limits it either way is not the gauge. The flange poses come from
# forward kinematics, and on these two arms the published DH table is about
# 3 mm out — the same size as the answer. Samples taken through
# `ur5dual-flange-log --direct`, which reads each controller's own factory
# table, are worth more than the gauge resolution suggests; samples computed
# from the published table are worth about 3 mm however carefully measured.

# A UR5 tool flange is 63 mm across. Half of that turns a parallelism error
# into the millimetres it puts on the rim, so an angle and a gap can be
# weighed against each other in one least-squares problem.
FLANGE_RADIUS = 0.0315      # m

MIN_PAIR_SAMPLES = 3
# Each pairing constrains the translation along one direction only, so the
# count that matters is how many directions the samples cover, not how many
# readings were taken. Eight well-spread ones leave every direction with
# something to say about it and a little redundancy on top.
RECOMMENDED_PAIR_SAMPLES = 8
# smallest singular value of the measurement directions, normalised so a
# perfectly isotropic set scores 0.58 whatever the sample count
MIN_PAIR_SPREAD = 0.25
# Worst residual a fit may leave and still be called a calibration. Past this
# the transform is written but the cell is only marked translation-calibrated:
# rotating a held object multiplies a base-distance error by the angle turned,
# and max_pair_drift stops the arms at 4 mm.
PAIR_TRUST_MM = 2.0

PAIR_MAX_ITER = 60
PAIR_MAX_STEP = 0.05        # m and rad per Gauss-Newton step


def facing_flanges(gap, spin=0.0, offset=(0.0, 0.0)):
    """The transform from one flange to another one facing it. 4x4.

    UR's tool0 frame has +Z out of the mounting face, so two flanges facing
    each other across `gap` have their Z axes opposed: half a turn about X
    carries one onto the other, and `spin` then turns the far flange about
    the axis they share. `offset` is how far the far flange sits sideways of
    the near one's axis, in the near one's flange plane.

    `spin` and `offset` are exactly what a gauge reading cannot see. They are
    arguments rather than assumptions so that a caller who has aligned the
    flanges by eye, or keyed them on a fixture, can say so — and so that a
    caller who has not is at least stating what they are taking on trust.
    """
    T = np.eye(4)
    c, s = math.cos(spin), math.sin(spin)
    # Rx(pi) @ Rz(spin), written out: the far flange's Z opposes this one's
    T[:3, :3] = np.array([[c, -s, 0.0],
                          [-s, -c, 0.0],
                          [0.0, 0.0, -1.0]])
    T[:3, 3] = [float(offset[0]), float(offset[1]), float(gap)]
    return T


def flange_pair_transform(F_a, F_b, gap, spin=0.0, offset=(0.0, 0.0)):
    """A-base -> B-base from one pairing, taking `spin` and `offset` as given.

    `F_a` and `F_b` are the two flange poses in their own arms' base frames —
    `ur_kinematics.fk(q, dh)`. The chain is the whole measurement:

        A base -> A flange -> B flange -> B base

    Exact, and only as true as the three numbers a gauge cannot measure. Use
    `refine_from_flange_pair` instead when there is a configured geometry to
    correct: it takes those three from the config rather than from a guess.
    """
    return np.asarray(F_a, float) @ facing_flanges(gap, spin, offset) \
        @ inv(np.asarray(F_b, float))


def refine_from_flange_pair(T_ab, F_a, F_b, gap, concentric=False):
    """The smallest change to a base-to-base transform that fits one pairing.

    The configured transform is taken as the answer for everything the gauge
    is blind to — the spin of the flanges about their common axis, and how far
    one sits sideways of the other — and is overruled only where the
    measurement actually speaks: the faces are made parallel, and the distance
    between them is set to what was measured.

    `concentric=True` adds the one assumption an operator can often make good
    on: that the two flange axes are lined up, not merely parallel. Worth
    saying when the flanges were set square to each other on a fixture, and
    worth leaving alone when they were eyeballed — it moves arm B sideways by
    whatever the config's lateral offset was.

    Returns the corrected A-base -> B-base transform.
    """
    F_a = np.asarray(F_a, dtype=float)
    F_b = np.asarray(F_b, dtype=float)
    G = inv(F_a) @ np.asarray(T_ab, dtype=float) @ F_b   # A flange -> B flange

    # 1. make the faces parallel: the far flange's Z must oppose this one's,
    #    by the smallest rotation that does it — so the two angles the gauge
    #    fixes are corrected and the spin it cannot see is left alone
    R_fix = _rotation_between(G[:3, 2], [0.0, 0.0, -1.0])

    # 2. the same rotation carries the far flange's origin with it, keeping
    #    the correction a rigid one; then the measured gap replaces the
    #    distance along the axis, and only that
    t = R_fix @ G[:3, 3]
    t[2] = float(gap)
    if concentric:
        t[:2] = 0.0

    G_fixed = np.eye(4)
    G_fixed[:3, :3] = R_fix @ G[:3, :3]
    G_fixed[:3, 3] = t
    return F_a @ G_fixed @ inv(F_b)


class FlangePair:
    """One gauge reading, and where both flanges were when it was taken."""

    def __init__(self, F_a, F_b, gap, name=""):
        self.F_a = np.asarray(F_a, dtype=float)
        self.F_b = np.asarray(F_b, dtype=float)
        self.gap = float(gap)
        self.name = name


class FlangePairCalibration:
    """Fit A-base -> B-base to a set of measured flange pairings.

    One pairing leaves three of the six numbers unsaid; several, taken with
    the flange axis pointing in genuinely different directions, do not — each
    reading constrains the geometry along its own axis, and enough axes cover
    the space. That is the same argument the touch-off makes with points, made
    here with distances, and it needs no fixture: the gauge and the two arms
    are the whole apparatus.

    Two ways to read a gauge, and they are not interchangeable:

      separation   the reading is the distance between the two flange centres.
                   What `tools/flange_log.py` prints the configured figure for,
                   so it is the model that matches a log recorded with it, and
                   it asks nothing about how the faces were oriented.
      facing       the reading is the perpendicular distance between two
                   parallel faces. Says more per sample — the parallelism is
                   two constraints of its own — and is only true of samples
                   actually taken with the faces parallel.

    The solve is Gauss-Newton from a seed, and the seed matters: distances
    alone have mirrored solutions that fit every reading and put arm B on the
    wrong side of the cell. Seeding with the configured geometry is what keeps
    the answer on the right one, which is also why this is a way to improve a
    cell that is roughly right rather than to measure one from nothing.
    """

    def __init__(self, model="separation"):
        if model not in ("separation", "facing"):
            raise CalibrationError("model is separation or facing, not %r"
                                   % model)
        self.model = model
        self.samples = []

    # -- collecting --------------------------------------------------------
    def add(self, F_a, F_b, gap, name=""):
        """One pairing, from two flange poses in their own base frames."""
        self.samples.append(FlangePair(F_a, F_b, gap,
                                       name or "S%d" % (len(self.samples) + 1)))
        return self.samples[-1]

    def add_joints(self, q_a, q_b, gap, dh_a=None, dh_b=None, name=""):
        """One pairing, from joint angles — the form a log holds them in.

        The DH table is the caller's choice on purpose. Each controller holds
        the corrections measured for its own arm and they are worth about 3 mm
        here, which is the size of the answer; the published table is what
        there is when the robots are switched off.
        """
        return self.add(fk(np.asarray(q_a, float), dh_a or UR5_DH),
                        fk(np.asarray(q_b, float), dh_b or UR5_DH), gap, name)

    @classmethod
    def from_log(cls, data, model="separation", dh=None):
        """Load the samples in a config/flange_log.json.

        The log records joint angles in degrees and the gap in millimetres,
        which is what a person reads; everything here is radians and metres.
        """
        cal = cls(model)
        dh = dh or {}
        for i, s in enumerate(data.get("samples") or []):
            cal.add_joints(np.radians(s["A"]["joints_deg"]),
                           np.radians(s["B"]["joints_deg"]),
                           float(s["gap_mm"]) / 1000.0,
                           dh.get("A"), dh.get("B"),
                           name=s.get("t", "S%d" % (i + 1)))
        return cal

    def clear(self):
        self.samples = []

    # -- the fit -----------------------------------------------------------
    def residuals(self, T_ab):
        """How far this transform is from explaining every reading, in metres.

        One number per sample for `separation`, three for `facing` — where the
        two extra are the parallelism error written as the millimetres it puts
        on the rim of the flange, so that angles and gaps can be summed.
        """
        T_ab = np.asarray(T_ab, dtype=float)
        out = []
        for s in self.samples:
            far = T_ab @ s.F_b                      # B's flange, in A's base
            if self.model == "separation":
                out.append(float(np.linalg.norm(far[:3, 3] - s.F_a[:3, 3]))
                           - s.gap)
                continue
            G = inv(s.F_a) @ far                    # A's flange -> B's flange
            out.extend([FLANGE_RADIUS * G[0, 2], FLANGE_RADIUS * G[1, 2],
                        G[2, 3] - s.gap])
        return np.array(out, dtype=float)

    def _directions(self, T_ab):
        """The unit vector each reading constrains the geometry along.

        For a separation this is the line between the two flange centres; for
        a facing pair it is the flange axis. Either way it is the one
        direction that sample says anything about, which is what makes the
        spread of them the thing to check rather than the count.
        """
        T_ab = np.asarray(T_ab, dtype=float)
        out = []
        for s in self.samples:
            if self.model == "separation":
                v = (T_ab @ s.F_b)[:3, 3] - s.F_a[:3, 3]
            else:
                v = s.F_a[:3, 2]
            n = float(np.linalg.norm(v))
            if n > 1e-9:
                out.append(v / n)
        return np.array(out) if out else np.zeros((0, 3))

    def spread(self, T_ab):
        """How much of the space the measurement directions cover, 0 to 0.58.

        The smallest singular value of those unit vectors, divided by the root
        of the sample count so that taking the same reading twice does not
        flatter it. Zero means every gauge reading was taken along one line
        and the geometry square to it is whatever the seed said it was.
        """
        U = self._directions(T_ab)
        if len(U) < 2:
            return 0.0
        return float(np.linalg.svd(U, compute_uv=False)[-1] / math.sqrt(len(U)))

    def solve(self, seed):
        """Return (T_baseA_baseB, report), fitted from `seed`.

        `seed` is the geometry as currently configured — `config.a_to_b()`.
        """
        if len(self.samples) < MIN_PAIR_SAMPLES:
            raise CalibrationError(
                "%d pairings recorded, need at least %d (%d recommended)"
                % (len(self.samples), MIN_PAIR_SAMPLES,
                   RECOMMENDED_PAIR_SAMPLES))

        T = np.asarray(seed, dtype=float).copy()
        for _ in range(PAIR_MAX_ITER):
            r = self.residuals(T)
            J = self._jacobian(T, r)
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
            longest = float(np.max(np.abs(step)))
            if longest > PAIR_MAX_STEP:
                step *= PAIR_MAX_STEP / longest
            T = _perturb(T, step)
            if longest < 1e-12:
                break
        return T, self._report(T)

    def _jacobian(self, T, r0, eps=1e-7):
        """Numeric, and deliberately so: six columns of a handful of samples
        is nothing to compute, and an analytic derivative of a norm is one
        more place for a sign to hide."""
        J = np.zeros((len(r0), 6))
        for i in range(6):
            xi = np.zeros(6)
            xi[i] = eps
            J[:, i] = (self.residuals(_perturb(T, xi)) - r0) / eps
        return J

    def _report(self, T):
        r = self.residuals(T)
        if self.model == "separation":
            gap_err = r
            tilt = None
        else:
            gap_err = r[2::3]
            tilt = np.degrees(np.arcsin(np.clip(
                np.hypot(r[0::3], r[1::3]) / FLANGE_RADIUS, -1.0, 1.0)))
        spread = self.spread(T)
        flipped = [s.name for s in self.samples
                   if (inv(s.F_a) @ T @ s.F_b)[2, 2] > 0.0]
        report = {
            "samples": len(self.samples),
            "model": self.model,
            "rms_mm": float(np.sqrt((gap_err ** 2).mean()) * 1000),
            "max_mm": float(np.abs(gap_err).max() * 1000),
            "per_sample_mm": [float(v * 1000) for v in gap_err],
            "tilt_rms_deg": None if tilt is None else float(
                np.sqrt((tilt ** 2).mean())),
            "spread": spread,
            "flipped": flipped,
            "warnings": [],
        }
        if spread < MIN_PAIR_SPREAD:
            report["warnings"].append(
                "the readings were all taken along much the same direction "
                "(spread %.2f) — the geometry square to it is still whatever "
                "the seed said, so pose the arms differently and measure "
                "again" % spread)
        if len(self.samples) < RECOMMENDED_PAIR_SAMPLES:
            report["warnings"].append(
                "%d pairings is enough to solve and not enough to catch a bad "
                "one; %d is where a single mis-read gauge stops mattering"
                % (len(self.samples), RECOMMENDED_PAIR_SAMPLES))
        if report["max_mm"] > PAIR_TRUST_MM:
            report["warnings"].append(
                "the worst reading is %.1f mm from what this transform "
                "predicts, against %.1f mm rms — either that gauge reading is "
                "wrong, or the samples were not all taken the way this model "
                "assumes" % (report["max_mm"], report["rms_mm"]))
        if flipped:
            report["warnings"].append(
                "%d sample%s put the two flanges back to back rather than "
                "face to face (%s) — those were not the measurement this "
                "solves for" % (len(flipped), "" if len(flipped) == 1 else "s",
                                ", ".join(flipped[:3])))
        return report

    def apply_to_config(self, config, force=False):
        """Write the fitted geometry into arm B, arm A staying the reference.

        What gets *claimed* about the cell afterwards is deliberately graded.
        `translation_calibrated` is set whenever this is written, because
        carrying an object in a straight line needs only the two bases'
        relative orientation and this fit determines it. `calibrated` — which
        also unlocks turning a held object through any angle — is set only
        when the readings agree to better than PAIR_TRUST_MM, because the
        error a base distance carries into a rotation grows with the angle,
        and the drift guard stops the arms at 4 mm.

        Returns (T_baseA_baseB, report). The report says which was set.
        """
        T_ab, report = self.solve(config.a_to_b())
        blocking = [w for w in report["warnings"]
                    if w.startswith("the readings were all")]
        if blocking and not force:
            raise CalibrationError(blocking[0])

        config.arms["B"].set_base_matrix(config.arms["A"].base_matrix() @ T_ab)
        config.set_custom_mount()
        config.translation_calibrated = True
        report["marked_calibrated"] = report["max_mm"] <= PAIR_TRUST_MM
        if report["marked_calibrated"]:
            config.calibrated = True
        return T_ab, report


def _perturb(T, xi):
    """`T` moved by a small twist in the frame it is expressed in.

    Left-multiplied: the six numbers are a rotation and a translation of arm
    B's base as seen from arm A's, which is the thing being solved for and
    the thing an operator would recognise if it were printed.
    """
    xi = np.asarray(xi, dtype=float)
    D = np.eye(4)
    D[:3, :3] = rotvec_to_mat(xi[3:])
    D[:3, 3] = xi[:3]
    return D @ np.asarray(T, dtype=float)


# ── teaching the directions by hand ───────────────────────────────────────
MIN_MOVE = 0.010            # m; shorter samples are mostly encoder noise
MIN_DIRECTION_SPREAD = 0.25  # smallest singular value of the direction cloud
SLIP_RESIDUAL_MM = 3.0       # worst-motion residual that means a grip moved
LINK_TOLERANCE = 0.25        # how unequal two tips' travel may be


class DirectionCalibration:
    """Solve the two bases' relative *rotation* by hand-guiding the pair.

    Translating a held object needs less information than the full geometry.
    Both TCPs have to move along the same physical vector, and

        delta_B = R_BA · delta_A

    so only the relative rotation appears — where the bases sit, how far apart
    they are, and which way gravity points all cancel out of a pure
    translation. That matters because the rotation can be taught in a minute
    with no jig: put both grippers on the workpiece, switch both arms to
    freedrive, and push the thing around. One physical motion, seen from two
    base frames, is one equation.

    What it does not give is the translation between the bases. Motion alone
    cannot separate that from the unknown offset between the two grips — they
    only ever appear added together. Rotating the object needs it, so rotation
    stays with the touch-off wizard.
    """

    def __init__(self):
        self.pairs = []          # (delta in A's base, delta in B's base)
        self._origin = None

    def start(self, cell):
        """Mark where this push begins.

        Deliberately does not clear what came before: one press of the teach
        button is one push, and three pushes in three directions are what the
        solve needs. Wiping here would mean the count could never climb past
        one. Use reset() to actually start over.
        """
        self._origin = {a: cell.arms[a].tcp_pose()[:3].copy() for a in ("A", "B")}
        return self

    def reset(self):
        self.pairs = []
        self._origin = None
        return self

    def displacement(self, cell):
        """How far each TCP has moved since start(), in its own base frame."""
        if self._origin is None:
            raise CalibrationError("call start() before measuring")
        return (cell.arms["A"].tcp_pose()[:3] - self._origin["A"],
                cell.arms["B"].tcp_pose()[:3] - self._origin["B"])

    def commit(self, cell):
        """Turn the motion since start() into one sample, or explain why not.

        Returns (accepted, message). The refusals matter more than the
        acceptance here: a push that moves one arm and not the other is the
        signature of grippers that are not both on the same rigid thing, and
        silently dropping it leaves the operator pushing at nothing and
        wondering why the count never rises.
        """
        d_a, d_b = self.displacement(cell)
        len_a, len_b = float(np.linalg.norm(d_a)), float(np.linalg.norm(d_b))

        if max(len_a, len_b) < MIN_MOVE:
            return False, ("nothing moved (%.0f / %.0f mm) — push the object "
                           "at least %.0f mm"
                           % (len_a * 1000, len_b * 1000, MIN_MOVE * 1000))
        if min(len_a, len_b) < MIN_MOVE:
            near, far = ("A", "B") if len_a < len_b else ("B", "A")
            return False, ("arm %s moved %.0f mm but arm %s only %.0f mm — the "
                           "two grippers are not both holding the same rigid "
                           "object, so this push says nothing about how they "
                           "relate"
                           % (far, max(len_a, len_b) * 1000,
                              near, min(len_a, len_b) * 1000))
        # the pair must travel the same distance; a rigid link guarantees it
        if abs(len_a - len_b) > LINK_TOLERANCE * max(len_a, len_b):
            return False, ("the tips travelled %.0f mm and %.0f mm — for one "
                           "rigid object those must match, so something "
                           "slipped or flexed"
                           % (len_a * 1000, len_b * 1000))

        self.pairs.append((d_a.copy(), d_b.copy()))
        return True, ("recorded a %.0f mm push (%d so far, spread %.2f)"
                      % (len_a * 1000, len(self.pairs), self.spread()))

    def spread(self):
        """How many directions the samples actually cover.

        Pushing the object back and forth along one line gives any number of
        samples and still leaves the rotation about that line unknown.
        """
        if len(self.pairs) < 2:
            return 0.0
        A = np.array([d / np.linalg.norm(d) for d, _ in self.pairs])
        return float(np.linalg.svd(A, compute_uv=False)[-1])

    def solve(self):
        """Return (R_BA, report): the rotation carrying a displacement in A's
        base frame into B's."""
        if len(self.pairs) < 3:
            raise CalibrationError(
                "%d usable motions recorded, need at least 3 in different "
                "directions" % len(self.pairs))

        A = np.array([d for d, _ in self.pairs])
        B = np.array([d for _, d in self.pairs])
        # orthogonal Procrustes on vectors — no centroids, these are already
        # displacements rather than positions
        H = A.T @ B
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

        residuals = np.linalg.norm((R @ A.T).T - B, axis=1)
        lengths = np.linalg.norm(A, axis=1)
        spread = self.spread()
        report = {
            "motions": len(self.pairs),
            "rms_mm": float(np.sqrt((residuals ** 2).mean()) * 1000),
            "max_mm": float(residuals.max() * 1000),
            "median_move_mm": float(np.median(lengths) * 1000),
            "spread": spread,
            "warnings": [],
        }
        if spread < MIN_DIRECTION_SPREAD:
            report["warnings"].append(
                "the motions barely leave one plane — push the object along "
                "all three directions, not just back and forth")
        # Measured against 0.5 mm of hand-guiding slop on 80 mm pushes: a
        # sound grip puts the worst residual around 1.6 mm and rarely past 2,
        # while a single 6 mm slip pushes it to 5. Three separates them.
        if report["max_mm"] > SLIP_RESIDUAL_MM:
            report["warnings"].append(
                "the motions disagree by as much as %.1f mm (%.1f mm rms) — "
                "the workpiece rotated, flexed, or a gripper let go of the workpiece; "
                "redo the measurement"
                % (report["max_mm"], report["rms_mm"]))
        return R, report

    def apply_to_config(self, config):
        """Re-orient arm B's base so a world translation reaches both arms.

        Only the orientation is touched. Arm B's position stays whatever it
        was, because these samples say nothing about it — and the flag that is
        set says exactly that much.

        Note what this does *not* fix: arm A. Everything here is relative, so
        the pair ends up agreeing with each other while both stay hung on
        whatever orientation arm A was assumed to have. If that assumption is
        wrong the two arms carry the object in a perfectly straight line, in
        perfect agreement, in the wrong direction — see `level_report`.
        """
        R_ba = self.solve()[0]
        R_a = config.arms["A"].base_matrix()[:3, :3]
        T_b = config.arms["B"].base_matrix().copy()
        T_b[:3, :3] = R_a @ R_ba.T
        config.arms["B"].set_base_matrix(T_b)
        config.set_custom_mount()
        config.translation_calibrated = True
        return R_ba


# ── the whole geometry, from a jointly held object ────────────────────────
# `DirectionCalibration` above throws away everything except where each tip
# ended up, and pays for it twice. It cannot see where the bases are, because
# a pure translation leaves the base offset and the unknown gap between the two
# grips added together and never apart. And it has to reject any push that was
# not a pure translation, which by hand means most of them: two degrees of
# unintended turn on a 300 mm workpiece already puts 10 mm of disagreement into
# a solve whose slip threshold is 3.
#
# Both problems have the same cause — the orientation each controller reports
# is being thrown away — and the same fix. Keep the whole pose, and one
# placement of the workpiece is one equation:
#
#     M_i · X = Z · N_i
#
#   M_i   arm A's TCP in arm A's base, as its controller reports it
#   N_i   arm B's TCP in arm B's base
#   X     arm B's TCP frame expressed in arm A's — constant while both
#         grippers keep their bite on the one object, whatever it is doing
#   Z     arm B's base expressed in arm A's — the answer, `config.a_to_b()`
#
# This is hand-eye calibration in its AX=ZB form, and turning the workpiece
# stops being the thing that ruins the measurement and becomes the thing that
# makes it work: rotation, and only rotation, is what separates X from Z. Two
# turns about non-parallel axes fix all six numbers, which is why the guard
# here counts the *second* smallest singular value of the turn axes where the
# translation teach counted the third.
#
# What that buys, plainly: the base positions as well as their orientations,
# out of the same hand-guiding, with no touch-off fixture.

MIN_HELD_POSES = 3
# Six unknowns in Z and six in X against six equations a placement, so three
# placements can solve it and cannot check it. Eight leaves enough redundancy
# that one gripper losing its bite shows up as one bad residual rather than as
# a plausible wrong answer.
RECOMMENDED_HELD_POSES = 8
# how far the workpiece has to have moved or turned before another placement
# is telling us anything new
MIN_POSE_CHANGE = 0.010     # m
MIN_POSE_TURN = 5.0         # deg
# a relative turn smaller than this is noise, not an axis worth counting
MIN_TURN_DEG = 10.0
# How much this set of placements multiplies pose noise on its way into the
# base *position*, straight off the fit's own Jacobian. Dimensionless: half a
# millimetre of slop in the reported TCP poses comes out as this many times
# half a millimetre of error in where arm B's base is put. A well-spread
# session scores about 3.
#
# Eight is not a round number. Hand-guiding leaves about half a millimetre of
# slop, `max_pair_drift` stops the arms at 4 mm, and 8 x 0.5 mm is 4 mm — so
# this is the point at which the geometry stops being good enough to carry an
# object with, expressed in the only units that matter. Past it the answer is
# refused rather than merely flagged.
MAX_POSITION_AMPLIFICATION = 8.0
# Below this fraction of the largest singular value, a direction of the fit is
# not determined by the placements at all — the geometry can slide along it
# and no reading changes. Distinct from the bar above, and not a substitute
# for it: a session that never turned the workpiece is exactly singular only
# while the poses are exact, and the moment there is noise in them it becomes
# merely ill-conditioned, which is the same fault wearing a disguise.
RANK_TOLERANCE = 1e-8
# A slipped grip is an outlier, not a raised noise floor, so it is caught by
# how far the worst placement stands out from the rest rather than by its size
# alone — both tests have to fire. Measured over 150 sessions of eight
# placements at 0.5 mm and 0.1 deg of pose slop: a sound grip puts the worst
# placement at 1.7 times the rest and reaches 2.2 at the 95th percentile, a
# 6 mm slip sits at 2.9 and a 10 mm slip at 3.6. Two and a half catches a
# 10 mm slip every time and a 6 mm one usually, at about two false accusations
# in a hundred. A 3 mm slip scores 1.9 and is simply not separable from the
# noise — which is worth knowing rather than pretending otherwise.
HELD_SLIP_RATIO = 2.5
# Worst residual a fit may leave and still unlock rotating a held object — the
# same bar `FlangePairCalibration` applies and for the same reason: the drift
# guard stops the arms at 4 mm.
HELD_TRUST_MM = 2.0
# An orientation error in the fit is felt as the millimetres it puts at arm's
# length from the grip, which is what makes angles and distances addable in one
# least-squares problem. Measured against 0.5 mm and 0.1 deg of pose slop, the
# base position lands about 2.5 mm out at a lever of 50 mm and 2.0 mm out at
# 300 and beyond — the length of a workpiece two grippers can hold between
# them, which is not a coincidence.
HELD_ANGLE_LEVER = 0.300    # m
HELD_MAX_ITER = 60
HELD_MAX_DAMPING = 12       # damping retries within one Levenberg step


def _nearest_rotation(Y):
    """The rotation matrix closest to `Y`, which is not quite one."""
    U, _, Vt = np.linalg.svd(np.asarray(Y, dtype=float))
    return U @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))]) @ Vt


class HeldPose:
    """One placement of the workpiece, as each arm saw it in its own base."""

    def __init__(self, M, N, name=""):
        self.M = np.asarray(M, dtype=float)
        self.N = np.asarray(N, dtype=float)
        self.name = name


class HeldObjectCalibration:
    """Solve the full base-to-base transform from a jointly held workpiece.

    Both grippers take one rigid object, both arms go into freedrive, and
    somebody moves the thing to a handful of placements — sliding it *and*
    turning it, the more variety the better. Every placement is recorded whole,
    both poses, and every placement constrains the answer.

    What comes out is `Z`, arm B's base in arm A's frame: position and
    orientation, the same six numbers the touch-off wizard measures with a
    fixture. `X` — how the two grips relate — falls out alongside it and is
    worth reading, because a tape measure can check it and nothing else in this
    file offers that.

    The one thing that must hold: neither gripper may shift its bite on the
    object between the first placement and the last. That is what makes X
    constant, and it is what the residual tests.
    """

    def __init__(self):
        self.samples = []

    # -- collecting --------------------------------------------------------
    def add(self, M, N, name=""):
        self.samples.append(HeldPose(M, N,
                                     name or "P%d" % (len(self.samples) + 1)))
        return self.samples[-1]

    def add_from_cell(self, cell, name=""):
        """Record where both arms currently think their TCPs are.

        Both grippers must be on the object and the object must be still. No
        origin, no start and stop — a placement stands on its own, which is
        what lets every one of them into the solve.
        """
        for a in ("A", "B"):
            if not cell.arms[a].connected:
                raise CalibrationError("arm %s is not connected" % a)
        return self.add(pose_to_mat(cell.arms["A"].tcp_pose()),
                        pose_to_mat(cell.arms["B"].tcp_pose()), name)

    def commit(self, cell):
        """Record a placement, or say why it is not worth recording.

        Returns (accepted, message). The only refusal is a placement too close
        to one already taken: unlike the translation teach there is nothing
        here to reject a motion *for*, because turning the object has stopped
        being a fault to catch and become the measurement itself.
        """
        M = pose_to_mat(cell.arms["A"].tcp_pose())
        N = pose_to_mat(cell.arms["B"].tcp_pose())
        for s in self.samples:
            moved = float(np.linalg.norm(M[:3, 3] - s.M[:3, 3]))
            turned = math.degrees(float(np.linalg.norm(
                mat_to_rotvec(s.M[:3, :3].T @ M[:3, :3]))))
            if moved < MIN_POSE_CHANGE and turned < MIN_POSE_TURN:
                return False, ("this is %.0f mm and %.1f deg from placement %s "
                               "— too close to it to say anything new; move or "
                               "turn the workpiece further"
                               % (moved * 1000, turned, s.name))
        s = self.add(M, N)
        return True, ("recorded placement %s (%d so far, widest turn %.0f deg, "
                      "axis spread %.2f)"
                      % (s.name, len(self.samples), self.largest_turn_deg(),
                         self.turn_spread()))

    def clear(self):
        self.samples = []

    # -- how well conditioned the set is -----------------------------------
    def _relative_turn(self, i, j):
        """The rotation from placement i to placement j, as an axis-angle."""
        return mat_to_rotvec(self.samples[i].M[:3, :3].T
                             @ self.samples[j].M[:3, :3])

    def _turn_axes(self):
        """Unit axes of every relative turn big enough to count."""
        axes = []
        for i in range(len(self.samples)):
            for j in range(i + 1, len(self.samples)):
                rv = self._relative_turn(i, j)
                angle = float(np.linalg.norm(rv))
                if math.degrees(angle) >= MIN_TURN_DEG:
                    axes.append(rv / angle)
        return np.array(axes) if axes else np.zeros((0, 3))

    def largest_turn_deg(self):
        """The biggest angle between any two placements, in degrees."""
        best = 0.0
        for i in range(len(self.samples)):
            for j in range(i + 1, len(self.samples)):
                best = max(best, math.degrees(float(np.linalg.norm(
                    self._relative_turn(i, j)))))
        return best

    def turn_spread(self):
        """How many different axes the workpiece was turned about, 0 to 0.71.

        The *second* smallest singular value, and the difference from
        `DirectionCalibration.spread` is the whole point: a translation teach
        needs motion in all three dimensions and so watches the third singular
        value, while two turns about non-parallel axes already determine all
        six numbers here. Demanding a third axis would be asking the operator
        for work the arithmetic does not need.

        Reported rather than enforced. What actually decides whether a session
        determined anything is `_conditioning`, which asks the fit's own
        Jacobian instead of a proxy for it; this number is here because "turn
        it about another axis" is advice somebody can act on and a singular
        value is not.
        """
        U = self._turn_axes()
        if len(U) < 2:
            return 0.0
        return float(np.linalg.svd(U, compute_uv=False)[1] / math.sqrt(len(U)))

    # -- the fit -----------------------------------------------------------
    def residuals(self, X, Z):
        """How far this pair of transforms is from explaining every placement.

        Six numbers a placement: where the prediction lands, in metres, then
        how far it is turned, written as the metres that turn puts at
        HELD_ANGLE_LEVER from the grip so the two can be summed.
        """
        out = []
        for s in self.samples:
            E = inv(s.M @ X) @ (Z @ s.N)      # identity when the fit is exact
            out.extend(E[:3, 3])
            out.extend(mat_to_rotvec(E[:3, :3]) * HELD_ANGLE_LEVER)
        return np.array(out, dtype=float)

    def _seed(self):
        """Closed-form (X, Z), with no starting guess needed.

        The rotation halves of M_i X = Z N_i are linear in the entries of R_X
        and R_Z once vectorised — vec(A Y B) = (B^T kron A) vec(Y) — so
        stacking the placements gives a homogeneous system whose null vector
        holds both matrices. They come out scaled by a common unknown factor,
        whose sign the determinant gives back, and only approximately
        orthogonal, which the SVD projection fixes.

        Solving it in closed form rather than starting from the configured
        geometry matters: seeding a fit with the answer it is meant to check is
        how a wrong cell.yaml survives its own calibration.
        """
        rows = []
        for s in self.samples:
            rows.append(np.hstack([np.kron(np.eye(3), s.M[:3, :3]),
                                   -np.kron(s.N[:3, :3].T, np.eye(3))]))
        Vt = np.linalg.svd(np.vstack(rows))[2]
        v = Vt[-1]
        # column-major, because that is the convention the Kronecker identity
        # above is written in
        Y_x = v[:9].reshape(3, 3, order="F")
        Y_z = v[9:].reshape(3, 3, order="F")
        # both carry the same scale factor s, and det(s R) = s^3, so its sign
        # is recoverable — and has to be applied to the pair together
        if np.linalg.det(Y_z) < 0.0:
            Y_x, Y_z = -Y_x, -Y_z
        R_x, R_z = _nearest_rotation(Y_x), _nearest_rotation(Y_z)

        # the translations are then linear too: R_M t_X - t_Z = R_Z t_N - t_M
        A = np.vstack([np.hstack([s.M[:3, :3], -np.eye(3)])
                       for s in self.samples])
        b = np.concatenate([R_z @ s.N[:3, 3] - s.M[:3, 3]
                            for s in self.samples])
        t = np.linalg.lstsq(A, b, rcond=None)[0]

        X, Z = np.eye(4), np.eye(4)
        X[:3, :3], X[:3, 3] = R_x, t[:3]
        Z[:3, :3], Z[:3, 3] = R_z, t[3:]
        return X, Z

    def _jacobian(self, X, Z, r0, eps=1e-7):
        """Numeric, twelve columns: a twist of X, then a twist of Z."""
        J = np.zeros((len(r0), 12))
        for i in range(12):
            xi = np.zeros(12)
            xi[i] = eps
            J[:, i] = (self.residuals(_perturb(X, xi[:6]),
                                      _perturb(Z, xi[6:])) - r0) / eps
        return J

    def _refine(self, X, Z):
        """Levenberg-Marquardt from (X, Z), and damped for a reason.

        Plain Gauss-Newton is what a well-posed version of this wants, and it
        is exactly wrong here: a session that never turned the workpiece about
        two axes leaves the Jacobian rank-deficient, the normal equations pick
        an arbitrary step along the direction nothing constrains, and the
        solver walks half a metre away from an answer that fitted perfectly.
        Damping turns that unconstrained direction into one it simply does not
        move along, which is the honest response to being told nothing about
        it.
        """
        r = self.residuals(X, Z)
        cost = float(r @ r)
        lam = 1e-6
        for _ in range(HELD_MAX_ITER):
            J = self._jacobian(X, Z, r)
            JtJ, Jtr = J.T @ J, J.T @ r
            scale = np.diag(np.maximum(np.diag(JtJ), 1e-12))
            for _ in range(HELD_MAX_DAMPING):
                try:
                    step = np.linalg.solve(JtJ + lam * scale, -Jtr)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
                longest = float(np.max(np.abs(step)))
                if longest > PAIR_MAX_STEP:
                    step *= PAIR_MAX_STEP / longest
                X_try = _perturb(X, step[:6])
                Z_try = _perturb(Z, step[6:])
                r_try = self.residuals(X_try, Z_try)
                if float(r_try @ r_try) < cost:
                    X, Z, r = X_try, Z_try, r_try
                    cost = float(r @ r)
                    lam = max(lam * 0.3, 1e-12)
                    break
                lam *= 10.0
            else:
                break          # no damping found a way downhill: converged
            if longest < 1e-10:
                break
        return X, Z

    def solve(self, seed=None):
        """Return (Z, report): arm B's base expressed in arm A's base frame.

        `report["grip"]` holds X, the transform between the two gripper frames.

        `seed` is an optional starting guess — `config.a_to_b()` is the one
        worth passing. It is a candidate, not an assumption: the closed-form
        answer is tried alongside it and whichever ends up fitting the
        placements better is the one returned. That distinction is what keeps
        a wrong cell.yaml from surviving its own calibration while still
        letting a roughly-right one rescue a session whose closed-form seed
        came out of a badly conditioned set.
        """
        if len(self.samples) < MIN_HELD_POSES:
            raise CalibrationError(
                "%d placements recorded, need at least %d (%d recommended)"
                % (len(self.samples), MIN_HELD_POSES, RECOMMENDED_HELD_POSES))

        starts = [self._seed()]
        if seed is not None:
            Z0 = np.asarray(seed, dtype=float)
            # M X = Z N at the first placement gives the grip this seed implies
            starts.append((inv(self.samples[0].M) @ Z0 @ self.samples[0].N, Z0))

        best = None
        for X, Z in starts:
            X, Z = self._refine(X, Z)
            r = self.residuals(X, Z)
            cost = float(r @ r)
            if best is None or cost < best[0]:
                best = (cost, X, Z)
        return best[2], self._report(best[1], best[2])

    def _conditioning(self, X, Z):
        """How hard this set of placements has to lean on the pose readings.

        Returns (amplification, determined). The Jacobian of the fit already
        holds the answer, so this asks it rather than guessing from a proxy:
        invert the normal equations, take the block belonging to arm B's base
        position, and the result is dimensionless — the factor by which slop in
        the reported TCP poses is multiplied on its way into the geometry.

        A direction the placements say nothing about shows up as a singular
        value at zero, and no amount of averaging recovers it, so that case
        comes back as infinity rather than as a large number.
        """
        J = self._jacobian(X, Z, self.residuals(X, Z))
        sv = np.linalg.svd(J, compute_uv=False)
        if sv[-1] <= sv[0] * RANK_TOLERANCE:
            return float("inf"), False
        cov = np.linalg.inv(J.T @ J)
        return math.sqrt(float(np.trace(cov[6:9, 6:9]))), True

    def _report(self, X, Z):
        r = self.residuals(X, Z).reshape(-1, 6)
        pos = np.linalg.norm(r[:, :3], axis=1)
        ang = np.degrees(np.linalg.norm(r[:, 3:], axis=1) / HELD_ANGLE_LEVER)
        amplification, full_rank = self._conditioning(X, Z)
        # One bar, two ways of failing it. Exact rank deficiency is what a
        # noiseless session that never turned the workpiece looks like; the
        # same session with real readings in it is merely ill-conditioned, and
        # would otherwise sail through the rank test and produce a base
        # position metres away with every appearance of confidence.
        determined = full_rank and amplification <= MAX_POSITION_AMPLIFICATION
        spread, turn = self.turn_spread(), self.largest_turn_deg()
        worst = int(np.argmax(pos))
        rest = np.delete(pos, worst)
        rest_rms = float(np.sqrt((rest ** 2).mean())) if len(rest) else 0.0
        report = {
            "placements": len(self.samples),
            "grip": X,
            "grip_mm": float(np.linalg.norm(X[:3, 3])) * 1000,
            "rms_mm": float(np.sqrt((pos ** 2).mean()) * 1000),
            "max_mm": float(pos.max() * 1000),
            "per_placement_mm": [float(v * 1000) for v in pos],
            "tilt_rms_deg": float(np.sqrt((ang ** 2).mean())),
            "turn_deg": turn,
            "spread": spread,
            "amplification": amplification,
            "determined": determined,
            "warnings": [],
        }
        if not full_rank:
            report["warnings"].append(
                "these placements do not determine the geometry — the "
                "workpiece was never turned about two genuinely different axes "
                "(axis spread %.2f, widest turn %.0f deg), so arm B's base can "
                "slide without any reading changing. Turn it about a second "
                "axis and record a few more placements" % (spread, turn))
        elif not determined:
            report["warnings"].append(
                "these placements do not pin the base positions down: they "
                "multiply pose slop by %.0f on the way in, where a well-spread "
                "session manages about 3 and %.0f is where the error reaches "
                "the drift guard. The widest turn was %.0f deg about an axis "
                "spread of %.2f — turn it further, and about more axes"
                % (amplification, MAX_POSITION_AMPLIFICATION, turn, spread))
        if len(self.samples) < RECOMMENDED_HELD_POSES:
            report["warnings"].append(
                "%d placements is enough to solve and not enough to catch a "
                "bad one; %d is where a single slipped grip stops passing for "
                "an answer" % (len(self.samples), RECOMMENDED_HELD_POSES))
        # A slip is one placement standing out from the others, not a raised
        # floor under all of them, so being large is not on its own enough to
        # be accused of it — otherwise an honest session with a soft grip gets
        # told a gripper let go every time the noise happens to peak.
        if (report["max_mm"] > HELD_TRUST_MM
                and pos[worst] > HELD_SLIP_RATIO * max(rest_rms, 1e-12)):
            report["warnings"].append(
                "placement %s is %.1f mm from what this geometry predicts "
                "while the rest agree to %.1f mm — a gripper shifted its bite "
                "on the workpiece there, or the workpiece flexed"
                % (self.samples[worst].name, report["max_mm"],
                   rest_rms * 1000))
        return report

    def apply_to_config(self, config, force=False, seed=None):
        """Write the measured geometry into arm B, arm A staying the reference.

        The position as well as the orientation, which is what separates this
        from the translation teach — and why it can set `calibrated` rather
        than only `translation_calibrated`. It still only does so when the
        placements agree to better than HELD_TRUST_MM, because turning a held
        object multiplies a base-distance error by the angle turned.

        Returns (T_baseA_baseB, report). The report says which flag was set.
        """
        Z, report = self.solve(seed=seed)
        if not report["determined"] and not force:
            raise CalibrationError(report["warnings"][0])

        config.arms["B"].set_base_matrix(config.arms["A"].base_matrix() @ Z)
        config.set_custom_mount()
        config.translation_calibrated = True
        report["marked_calibrated"] = (report["max_mm"] <= HELD_TRUST_MM
                                       and report["determined"])
        if report["marked_calibrated"]:
            config.calibrated = True
        return Z, report

    def as_direction_calibration(self):
        """The orientation-only fallback, for a workpiece that cannot be turned.

        Consecutive placements that differ by no appreciable turn are pure
        translations of the object, which is exactly what
        `DirectionCalibration` wants. So a session where somebody slid the
        workpiece about and never rotated it is not wasted — it still fixes the
        two bases' relative orientation, which is the part that makes a world
        jog reach both arms. It cannot fix the base positions, and the flag it
        sets says so.
        """
        cal = DirectionCalibration()
        for i in range(len(self.samples) - 1):
            first, second = self.samples[i], self.samples[i + 1]
            if math.degrees(float(np.linalg.norm(
                    self._relative_turn(i, i + 1)))) >= MIN_TURN_DEG:
                continue                # not a translation; the pair would lie
            d_a = second.M[:3, 3] - first.M[:3, 3]
            d_b = second.N[:3, 3] - first.N[:3, 3]
            if min(float(np.linalg.norm(d_a)),
                   float(np.linalg.norm(d_b))) >= MIN_MOVE:
                cal.pairs.append((d_a, d_b))
        return cal


# ── which way is up ───────────────────────────────────────────────────────
# Nothing above measures this. The touch-off wizard relates arm B to arm A,
# the direction teach does the same with less information, and both leave arm
# A sitting wherever the mounting numbers said it sits. So "world +Z" means
# whatever `mount.tilt_deg` claimed, and a world jog goes off by exactly the
# angle that claim is wrong by — straight, coordinated, and not where the
# operator pointed.
#
# The flange accelerometers settle it. Each arm feels gravity in its own base
# frame, which fixes two of the world frame's three angles outright, and the
# two readings can be checked against each other, which grades the relative
# orientation the teach produced without anyone pushing anything.

# how far apart the two arms' idea of "up" may be before the transform
# between them is the thing at fault
UP_AGREEMENT_DEG = 3.0


def _rotation_between(a, b):
    """Smallest rotation carrying unit vector a onto unit vector b.

    Smallest matters: it is the correction that changes the world frame by no
    more than gravity actually demands, leaving the heading of the horizontal
    axes as close to what the operator is used to as the fix allows.
    """
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(a @ b)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # exactly opposite: half a turn about any axis square to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def level_report(cell):
    """Measure the world frame against gravity and say what is wrong with it.

    Returns a dict, or raises CalibrationError when the arms cannot answer —
    they are moving, or the firmware does not report the accelerometer.

      tilt_deg[arm]      measured angle between that base's Z and the vertical,
                         which is the same number the pendant shows under
                         Installation -> General -> Mounting
      config_tilt_deg    what the config currently claims for each
      pair_error_deg     how far apart the two arms' measured "up" lands once
                         each is carried into the world frame by the config.
                         This grades the A-to-B transform *only* — it is blind
                         to how the pair as a whole is oriented, which is
                         exactly what makes it a clean test of the teach.

                         It has a blind spot of its own, and it is not a small
                         one: gravity says nothing about rotation *about* the
                         vertical. Two arms whose relative transform is out by
                         a pure yaw both report the same up and this number
                         stays at zero. So a clean pair error clears two of
                         the three angles between the arms, not all three —
                         it can condemn the teach but cannot acquit it.
      world_error_deg    how far the configured world +Z is from real up
      correction         3x3 world-side rotation that fixes it
      bracket_fit[arm]   (tilt_deg, rotate_deg) — the same reading told the
                         other way round, see below

    One reading, two stories, and gravity cannot choose between them. An arm's
    accelerometer fixes two of the three angles its base has; `correction`
    spends that on one rotation shared by both arms, which is the right answer
    when the whole cell hangs off plumb and the wrong one when it does not.
    The same numbers are equally well explained by each arm sitting where the
    config says with its bracket turned — a different `tilt_deg` and a spin on
    its own mounting flange — which is what `bracket_fit` reports, exactly the
    pair `mount.tilt_deg_<arm>` and `mount.rotate_deg_<arm>` exist to hold.

    A spirit level on the mast decides it, and nothing here can. Plumb mast =
    the world is already right and the brackets are not; out of plumb = apply
    the correction. Note which one keeps what: the correction turns both arms
    together and so preserves the A-to-B transform the touch-off measured,
    while per-arm bracket numbers rewrite it.
    """
    ups, missing = {}, []
    for a in ("A", "B"):
        if not cell.arms[a].connected:
            continue
        up = cell.arms[a].up_in_base()
        if up is None:
            missing.append(a)
        else:
            ups[a] = up
    if not ups:
        raise CalibrationError(
            "no arm could report gravity (%s) — hold the arms still, and check "
            "that the controller is new enough to send the tool accelerometer"
            % (", ".join(missing) or "none connected"))

    world_up = {a: cell.arms[a].base_matrix()[:3, :3] @ up
                for a, up in ups.items()}
    report = {
        "arms": sorted(ups),
        "up_in_base": ups,
        "tilt_deg": {}, "config_tilt_deg": {}, "bracket_fit": {},
        "pair_error_deg": None,
        "warnings": [],
    }
    for a, up in ups.items():
        base_z = np.array([0.0, 0.0, 1.0])
        report["tilt_deg"][a] = math.degrees(
            math.acos(max(-1.0, min(1.0, float(up @ base_z)))))
        report["config_tilt_deg"][a] = cell.config.tilt_of(a)
        # the plumb-mast reading of the same two numbers. The preset builds
        # this base as Rx(-side*tilt) then Rz(spin), so with world +Z really
        # up, gravity in the base frame is Rz(-spin) @ [0, -side*sin t, cos t]
        # — one tilt and one spin, read straight back off the vector.
        side = 1.0 if a == "A" else -1.0
        report["bracket_fit"][a] = (
            report["tilt_deg"][a],
            math.degrees(math.atan2(-side * up[0], -side * up[1])))

    if len(world_up) == 2:
        a, b = world_up["A"], world_up["B"]
        report["pair_error_deg"] = math.degrees(
            math.acos(max(-1.0, min(1.0, float(a @ b)))))
        if report["pair_error_deg"] > UP_AGREEMENT_DEG:
            report["warnings"].append(
                "the two arms disagree about which way is up by %.1f deg once "
                "the configured geometry is applied, so the transform between "
                "them is wrong — a world move sends them in different "
                "directions and they fight. Re-run the touch-off calibration, "
                "or teach the directions again with wider pushes"
                % report["pair_error_deg"])

    # one vector for the pair: their average, which is the best estimate of
    # real up in the current world frame
    consensus = sum(world_up.values())
    consensus = consensus / float(np.linalg.norm(consensus))
    report["measured_up_world"] = consensus
    report["world_error_deg"] = math.degrees(
        math.acos(max(-1.0, min(1.0, float(consensus @ [0.0, 0.0, 1.0])))))
    report["correction"] = _rotation_between(consensus, [0.0, 0.0, 1.0])
    if report["world_error_deg"] > 60.0:
        report["warnings"].append(
            "real up is %.0f deg off the configured world +Z. That is far more "
            "than a bracket tolerance — check the mounting numbers against the "
            "pendant before applying this"
            % report["world_error_deg"])
    return report


def apply_level(config, report, tipped=False):
    """Turn both arms until world +Z is really up.

    The same rotation goes on both arms either way, so the transform *between*
    them — the only thing the touch-off and the direction teach ever measured
    — comes through untouched. What changes is where that rotation is centred,
    and that is a question about the building, not about the arithmetic:

      tipped=False   the column is plumb and the arms are not square to it:
                     a crossbar welded with a twist in it, pads machined off
                     angle, an arm bolted round a few degrees on its flange.
                     Only the orientations turn. The flanges stay exactly
                     where they are, which is the whole point — they are on a
                     column that has not moved.

      tipped=True    the column itself is out of plumb, so the flanges really
                     are somewhere else. Everything turns about the foot,
                     positions included, and the drawn structure leans with
                     the arms.

    Getting this backwards is not subtle for long but it is subtle at first:
    it moves each flange by column_height times the angle — 110 mm on this
    cell at 5 deg — and every taught point in the cell frame moves with it.
    Put a level on the column before choosing. Gravity cannot tell you: both
    hypotheses fit the same accelerometer reading exactly.
    """
    R_c = np.asarray(report["correction"], dtype=float)
    for arm_id in ("A", "B"):
        T = config.arms[arm_id].base_matrix()
        if tipped:
            T = np.vstack([np.hstack([R_c, np.zeros((3, 1))]),
                           [0.0, 0.0, 0.0, 1.0]]) @ T
        else:
            T[:3, :3] = R_c @ T[:3, :3]
        config.arms[arm_id].set_base_matrix(T)
    config.set_custom_mount()
    return R_c
