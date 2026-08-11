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

from .kinematics import inv, mat_to_xyz_rpy, pose_to_mat

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
                "one motion is %.1f mm out while the rest agree to %.1f mm — "
                "a gripper let go of the workpiece part way through; redo it"
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
        "tilt_deg": {}, "config_tilt_deg": {},
        "pair_error_deg": None,
        "warnings": [],
    }
    for a, up in ups.items():
        base_z = np.array([0.0, 0.0, 1.0])
        report["tilt_deg"][a] = math.degrees(
            math.acos(max(-1.0, min(1.0, float(up @ base_z)))))
        report["config_tilt_deg"][a] = cell.config.tilt_of(a)

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


def apply_level(config, report):
    """Turn the whole cell until world +Z is really up.

    The same rotation goes on both arms, so the transform *between* them —
    the only thing the touch-off and the direction teach ever measured — comes
    through untouched. What changes is the frame the pair is described in, and
    with it every world jog, which is the point.
    """
    R_c = np.asarray(report["correction"], dtype=float)
    T_c = np.eye(4)
    T_c[:3, :3] = R_c
    for arm_id in ("A", "B"):
        config.arms[arm_id].set_base_matrix(T_c @ config.arms[arm_id].base_matrix())
    config.set_custom_mount()
    return R_c
