"""
Rigid-body transforms in the one convention this project uses everywhere.

A *pose* is the 6-vector Universal Robots speaks natively:

    [x, y, z, rx, ry, rz]     metres, and an axis-angle rotation in radians

A *transform* is the 4x4 homogeneous matrix that pose stands for. Poses are
what cross the wire and what a user reads; matrices are what you compose. The
two conversions below are the only place that distinction is handled, so
nothing else in the codebase has to think about it.

Frames used by the cell (see cell.py):

    world           the fixed cell frame every arm is measured against
    <arm>_base      an arm's own base, where its controller reports poses
    object          the frame attached to the workpiece both arms are holding
"""

import math

import numpy as np


# ── axis-angle <-> rotation matrix ────────────────────────────────────────
def rotvec_to_mat(rv):
    """Axis-angle -> 3x3 rotation matrix (Rodrigues)."""
    rv = np.asarray(rv, dtype=float)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def mat_to_rotvec(R):
    """3x3 rotation matrix -> axis-angle.

    The near-180 degree case is handled separately: there sin(theta) goes to
    zero and the usual antisymmetric-part formula loses all its precision.
    """
    R = np.asarray(R, dtype=float)
    cos_t = (np.trace(R) - 1.0) / 2.0
    theta = math.acos(max(-1.0, min(1.0, cos_t)))
    if theta < 1e-9:
        return np.zeros(3)
    if theta > math.pi - 1e-6:
        # k k^T = (R + I) / 2 — take its best-conditioned column
        A = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        k = A[:, i] / math.sqrt(max(A[i, i], 1e-12))
        k = k / np.linalg.norm(k)
        return k * theta
    k = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2.0 * math.sin(theta))
    return k * theta


def rpy_to_mat(rpy):
    """Fixed-axis roll-pitch-yaw (the URDF convention) -> rotation matrix."""
    r, p, y = (float(v) for v in rpy)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def mat_to_rpy(R):
    """Rotation matrix -> fixed-axis roll-pitch-yaw, for writing URDF/YAML."""
    R = np.asarray(R, dtype=float)
    sy = -R[2, 0]
    if abs(sy) > 1.0 - 1e-9:                 # gimbal lock: pitch = +-90 deg
        pitch = math.copysign(math.pi / 2, sy)
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    else:
        pitch = math.asin(max(-1.0, min(1.0, sy)))
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


# ── pose <-> transform ────────────────────────────────────────────────────
def pose_to_mat(pose):
    """[x,y,z,rx,ry,rz] -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = rotvec_to_mat(pose[3:6])
    T[:3, 3] = np.asarray(pose[:3], dtype=float)
    return T


def mat_to_pose(T):
    """4x4 -> [x,y,z,rx,ry,rz]."""
    return np.concatenate([np.asarray(T)[:3, 3], mat_to_rotvec(np.asarray(T)[:3, :3])])


def xyz_rpy_to_mat(xyz, rpy):
    """The form mounting geometry is written in (YAML, URDF) -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = rpy_to_mat(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def mat_to_xyz_rpy(T):
    T = np.asarray(T)
    return T[:3, 3].copy(), mat_to_rpy(T[:3, :3])


def inv(T):
    """Inverse of a homogeneous transform, without a general matrix inverse."""
    T = np.asarray(T, dtype=float)
    R, p = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


# ── comparisons and interpolation ─────────────────────────────────────────
def pose_distance(a, b):
    """(translation in m, rotation in rad) between two poses.

    Returned as two numbers on purpose: metres and radians do not add up to
    anything meaningful, and every threshold in this project wants them apart.
    """
    Ta, Tb = pose_to_mat(a), pose_to_mat(b)
    dt = float(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3]))
    dR = Ta[:3, :3].T @ Tb[:3, :3]
    dr = float(np.linalg.norm(mat_to_rotvec(dR)))
    return dt, dr


def interp_pose(a, b, s):
    """Straight-line position, shortest-arc rotation, s in [0, 1]."""
    Ta, Tb = pose_to_mat(a), pose_to_mat(b)
    T = np.eye(4)
    T[:3, 3] = Ta[:3, 3] + s * (Tb[:3, 3] - Ta[:3, 3])
    dr = mat_to_rotvec(Ta[:3, :3].T @ Tb[:3, :3])
    T[:3, :3] = Ta[:3, :3] @ rotvec_to_mat(dr * s)
    return mat_to_pose(T)


def _axis_vector(axis):
    """An axis of rotation, however it was named.

    "z" and 2 both mean the third axis of whichever frame is being turned, and
    a taught program says it that way. A jog button says it as a vector,
    because the axis it means may be one of the *object's* resolved into the
    world, which no letter names. All three arrive here and leave as a unit
    vector; a vector's length is not a magnitude, only the angle is.
    """
    if isinstance(axis, str):
        k = np.zeros(3)
        k["xyz".index(axis)] = 1.0
        return k
    a = np.asarray(axis, dtype=float)
    if a.ndim == 0:
        k = np.zeros(3)
        k[int(a)] = 1.0
        return k
    norm = float(np.linalg.norm(a))
    if norm < 1e-12:
        raise ValueError("an axis of rotation cannot be the zero vector")
    return a / norm


def rotate_about_own_axis(pose, axis, angle):
    """Rotate a frame about one of *its own* axes, keeping its origin fixed.

    This is what "turn the drum 45 degrees" means — the object spins in place
    rather than swinging around some other point.
    """
    T = pose_to_mat(pose)
    T[:3, :3] = T[:3, :3] @ rotvec_to_mat(_axis_vector(axis) * angle)
    return mat_to_pose(T)


def rotate_about_world_axis(pose, axis, angle, pivot=None):
    """Rotate a frame about one of the *cell's* axes.

    The whole difference from the function above is which side the rotation
    multiplies on. Post-multiplying turns the frame about an axis that travels
    with it — tip the box over and its "up" tips too. Pre-multiplying turns it
    about an axis fixed in the cell, so RZ is the vertical of the room however
    the box has been turned since it was picked up. That is what an operator
    means by "yaw it left", and what a pendant's base-frame RX/RY/RZ does for
    one arm.

    `pivot` is the point the rotation happens around, in world coordinates.
    None keeps the frame's own origin fixed, so the box spins in place; for a
    two-arm grasp captured at the midpoint that origin sits between the
    grippers, which is the one point that costs both arms the same travel.
    Passing a pivot instead swings the box around a place in the cell — the
    equivalent of rotating about a taught work object.
    """
    T = pose_to_mat(pose)
    R = rotvec_to_mat(_axis_vector(axis) * angle)
    T[:3, :3] = R @ T[:3, :3]
    if pivot is not None:
        p = np.asarray(pivot, dtype=float)
        T[:3, 3] = R @ (T[:3, 3] - p) + p
    return mat_to_pose(T)
