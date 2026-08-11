"""
UR5 forward kinematics, Jacobian, and inverse kinematics — in Python.

Nothing else in this project needed these. Cartesian targets went over the
wire and the controller solved for joints on its own, which is fine for one
arm going somewhere. It stops being fine the moment two arms carry one box:
each controller picks its own IK branch, and the elbow one of them flips
halfway through a rotation is a metre of unplanned travel while the other arm
is still holding the other side.

So the solving moves here. Both arms' joints come out of one function, on
branches this side chose, and the whole path can be walked and checked before
anything is sent.

Frames: `fk` returns the *flange* (what UR calls tool0), in the arm's own base
frame. The pose the controller reports on 30003 is the TCP, which is the
flange times whatever tool offset is set on the pendant — `arm.tcp_offset`.
Mixing the two up is a quiet error of exactly the tool length, so every
function here says which one it means.
"""

import numpy as np

from .kinematics import inv, mat_to_pose, mat_to_rotvec, pose_to_mat

# Universal Robots' published Denavit-Hartenberg parameters for the UR5
# (CB3 series — the e-series numbers are different, and this cell runs
# PolyScope 3.7.2, so CB3 it is).
#
# These describe the design. They do not describe either of these two robots
# to better than a few millimetres, because every arm is measured at the
# factory and the corrections stored in its own controller. `with_corrections`
# below folds those in; `theta` exists only to carry them, and is zero here.
UR5_DH = {
    "a":     np.array([0.0, -0.425, -0.39225, 0.0, 0.0, 0.0]),
    "d":     np.array([0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823]),
    "alpha": np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]),
    "theta": np.zeros(6),
}

DH_FIELDS = ("a", "d", "alpha", "theta")


def with_corrections(delta, base=UR5_DH):
    """The published table plus one robot's measured corrections.

    Not what the two controllers in this cell send — they send the finished
    table, and `as_dh` is what reads it. Kept as a candidate because which of
    the two a firmware means is decided by measurement in `choose_dh` and not
    by belief, and a reading that is wrong for this cell simply loses.
    """
    return {k: (np.asarray(base.get(k, np.zeros(6)), dtype=float)
                + np.asarray(delta.get(k, np.zeros(6)), dtype=float))
            for k in DH_FIELDS}


def as_dh(values):
    """A DH table straight from a controller, with nothing added.

    This is what PolyScope 3.7 and 3.15 both send, and what reproduces both of
    these arms exactly. Two things about the numbers are worth expecting.

    An *uncalibrated* controller (calibration status 0, arm A here) echoes the
    published table back verbatim, so agreeing with the textbook is not
    evidence of anything — it is the absence of a measurement.

    A *calibrated* one (status 1, arm B) sends a linearized table in which d2
    and d3 come back as roughly -54.6 m and +54.5 m. That is not corruption. A
    DH chain cannot express a small change in the angle between two parallel
    joints, which is exactly what joints 2 and 3 are, so the fit escapes into
    enormous offsets that cancel — the classic parallel-joint degeneracy that
    the Hayati parameters exist to avoid. The forward kinematics and the
    Jacobian are both exact anyway, because they are derived from the frames as
    given: on this cell it moves the tool 3.18 mm to where the controller says
    it is, and leaves manipulability at 0.2392 against the published table's
    0.2394. Anything that sanity-checks these values by their size will throw
    away the only correct table available.
    """
    return {k: np.asarray(values.get(k, np.zeros(6)), dtype=float)
            for k in DH_FIELDS}

# UR5 joints turn +-360 degrees. The pendant can be set tighter, and a cell
# usually is; treat this as the outermost bound and let callers narrow it.
JOINT_LIMITS = np.array([[-2 * np.pi, 2 * np.pi]] * 6)

# below this smallest singular value the Jacobian is losing a direction, and a
# joint velocity that reaches it grows without bound. 0.01 is roughly where a
# UR5 wrist starts to feel it; the solver damps rather than refuses.
SIGMA_MIN_WARN = 0.01


class KinematicsError(RuntimeError):
    pass


def _link(theta, a, d, alpha):
    """One standard-DH link transform: Rz(theta) Tz(d) Tx(a) Rx(alpha)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def link_frames(q, dh=UR5_DH):
    """Every joint frame from the base outward: [T_0_0, T_0_1, ... T_0_6].

    Seven matrices for six joints — the first is the identity, the base
    itself. The Jacobian needs each joint's axis and origin, so it wants the
    whole chain rather than only the end of it.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (6,):
        raise KinematicsError("a UR5 pose is six joint angles, got %r" % (q.shape,))
    theta = dh.get("theta")
    frames = [np.eye(4)]
    T = np.eye(4)
    for i in range(6):
        angle = q[i] if theta is None else q[i] + theta[i]
        T = T @ _link(angle, dh["a"][i], dh["d"][i], dh["alpha"][i])
        frames.append(T.copy())
    return frames


def fk(q, dh=UR5_DH):
    """Joint angles -> 4x4 flange pose in the arm's own base frame."""
    return link_frames(q, dh)[-1]


def fk_tcp(q, tcp_offset_mat=None, dh=UR5_DH):
    """Joint angles -> 4x4 *TCP* pose, the thing 30003 reports.

    Pass the tool offset as a matrix (`pose_to_mat(arm.tcp_offset)`); with no
    offset the TCP is the flange, which is what an arm with no tool configured
    actually reports.
    """
    T = fk(q, dh)
    return T if tcp_offset_mat is None else T @ tcp_offset_mat


def tcp_disagreement(q, tcp_pose, tcp_offset_mat=None, dh=UR5_DH):
    """(metres, radians) between our TCP for these joints and the robot's.

    The controller reports both the angles it is sitting at and the pose they
    produce, so this needs nothing but a single read to say whether the table
    below describes the machine in front of you.
    """
    mine = fk_tcp(q, tcp_offset_mat, dh)
    theirs = pose_to_mat(np.asarray(tcp_pose, dtype=float))
    return (float(np.linalg.norm(mine[:3, 3] - theirs[:3, 3])),
            float(np.linalg.norm(mat_to_pose(mine @ inv(theirs))[3:])))


def choose_dh(q, tcp_pose, tcp_offset_mat=None, calibration=None):
    """Which DH table actually describes this robot.

    Candidates are the published table, that table plus the controller's own
    corrections, and those corrections read as a finished table. The referee is
    the robot: whichever reproduces the pose it reports for the joints it is
    sitting at wins. Which convention a given firmware uses is then not
    something this has to know or guess — a wrong reading simply loses.

    Returns (dh, what it is, position error in metres).
    """
    candidates = [(UR5_DH, "the published UR5 table")]
    if calibration:
        # A controller with nothing measured (status 0) sends the published
        # table back verbatim, and that table then wins on an exact tie. Naming
        # it "this robot's own calibration" would report the absence of a
        # measurement as a successful one — the reading that flatters the arm
        # nobody has calibrated.
        own = ("the published UR5 table, which is all this controller holds"
               if calibration.get("status") == 0
               else "this robot's own calibration")
        # first, because it is what both controllers here actually send
        candidates.insert(0, (as_dh(calibration), own))
        candidates.append((with_corrections(calibration),
                           "this robot's calibration read as corrections"))
    scored = [(tcp_disagreement(q, tcp_pose, tcp_offset_mat, dh)[0], name, dh)
              for dh, name in candidates]
    scored.sort(key=lambda row: row[0])
    error, name, dh = scored[0]
    return dh, name, error


def jacobian(q, tcp_offset_mat=None, dh=UR5_DH):
    """Geometric Jacobian at the TCP, expressed in the arm's base frame.

    Rows are [vx vy vz wx wy wz], columns are the six joints, so
    `J @ qd` is the TCP twist. Revolute joints only, so each column is the
    axis and the moment arm from that joint's origin to the tool point.
    """
    frames = link_frames(q, dh)
    T_end = frames[-1] if tcp_offset_mat is None else frames[-1] @ tcp_offset_mat
    p_end = T_end[:3, 3]

    J = np.zeros((6, 6))
    for i in range(6):
        z = frames[i][:3, 2]          # joint i+1 turns about frame i's z
        p = frames[i][:3, 3]
        J[:3, i] = np.cross(z, p_end - p)
        J[3:, i] = z
    return J


def manipulability(q, tcp_offset_mat=None, dh=UR5_DH):
    """Smallest singular value of the Jacobian.

    One number for "how close is this pose to losing a direction". Zero is a
    singularity exactly; anything under SIGMA_MIN_WARN means a small tool
    motion is buying a large joint motion somewhere.
    """
    return float(np.linalg.svd(jacobian(q, tcp_offset_mat, dh),
                               compute_uv=False)[-1])


def pose_error(T_current, T_target):
    """6-vector from where the tool is to where it should be.

    Translation straight out of the difference in origins; rotation as the
    axis-angle of the residual rotation, which is the only form that both
    behaves near zero and composes with a Jacobian.
    """
    e = np.empty(6)
    e[:3] = T_target[:3, 3] - T_current[:3, 3]
    e[3:] = mat_to_rotvec(T_target[:3, :3] @ T_current[:3, :3].T)
    return e


def ik(T_target, q_seed, tcp_offset_mat=None, dh=UR5_DH,
       tol_p=1e-6, tol_r=1e-6, max_iter=100, damping=0.02, max_step=0.4):
    """Damped-least-squares IK for the TCP, starting from `q_seed`.

    Numerical rather than closed-form, and that is the point rather than a
    shortcut. UR5 has eight analytic solutions and picking between them is
    exactly the branch-flip problem this module exists to prevent; a solver
    seeded with the previous answer cannot leave the branch it started on,
    because it only ever walks downhill from there. In the servo loop the
    seed is one cycle old and 0.4 mm away, so it converges in two or three
    iterations.

    The damping is what keeps it sane near a singularity: instead of the
    unbounded joint velocity a pseudo-inverse would ask for, it trades a
    little tracking error for a bounded step. `max_step` then caps how far any
    one joint may move per iteration, so a bad seed cannot throw the solution
    across the workspace on the first pass.

    Returns (q, info). `info["converged"]` is the only thing a caller must
    check — an unconverged answer is a pose this arm cannot reach from here.
    """
    q = np.array(q_seed, dtype=float).copy()
    if q.shape != (6,):
        raise KinematicsError("seed must be six joint angles")

    lam2 = float(damping) ** 2
    I6 = np.eye(6)
    err = np.full(6, np.inf)

    for iteration in range(int(max_iter)):
        T = fk_tcp(q, tcp_offset_mat, dh)
        err = pose_error(T, T_target)
        e_p = float(np.linalg.norm(err[:3]))
        e_r = float(np.linalg.norm(err[3:]))
        if e_p < tol_p and e_r < tol_r:
            return q, {"converged": True, "iterations": iteration,
                       "pos_error": e_p, "rot_error": e_r,
                       "sigma_min": manipulability(q, tcp_offset_mat, dh)}

        J = jacobian(q, tcp_offset_mat, dh)
        # J^T (J J^T + lam^2 I)^-1 e — the damped right pseudo-inverse. Solve
        # rather than invert: the matrix is 6x6 and near-singular by design.
        dq = J.T @ np.linalg.solve(J @ J.T + lam2 * I6, err)
        np.clip(dq, -max_step, max_step, out=dq)
        q += dq

    return q, {"converged": False, "iterations": int(max_iter),
               "pos_error": float(np.linalg.norm(err[:3])),
               "rot_error": float(np.linalg.norm(err[3:])),
               "sigma_min": manipulability(q, tcp_offset_mat, dh)}


def within_limits(q, limits=JOINT_LIMITS):
    """[(joint index, angle, low, high)] for every joint outside its range."""
    q = np.asarray(q, dtype=float)
    out = []
    for i in range(6):
        low, high = limits[i]
        if not (low <= q[i] <= high):
            out.append((i, float(q[i]), float(low), float(high)))
    return out


def limit_margin(q, limits=JOINT_LIMITS):
    """Radians from the nearest joint limit — small means about to run out.

    Used to score IK branches against each other: between two ways of holding
    the same box, the one with more room left is the one that survives the
    rotation that comes next.
    """
    q = np.asarray(q, dtype=float)
    return float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))


def nearest_turn(q, q_reference):
    """Slide each joint by whole turns to sit as close to a reference as it can.

    A UR5 joint at +350 degrees and the same joint at -10 degrees put the tool
    in the same place, but a servo loop told to go from one to the other winds
    a full turn to get there. Any solution handed to the robot passes through
    here first.
    """
    q = np.asarray(q, dtype=float).copy()
    q_ref = np.asarray(q_reference, dtype=float)
    return q - 2 * np.pi * np.round((q - q_ref) / (2 * np.pi))
