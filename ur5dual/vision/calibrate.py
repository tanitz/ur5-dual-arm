"""
Putting the camera in the cell.

The detector answers in the camera's own frame. Everything a program does —
the taught pick, the reach check, the keep-out — is in world. The transform
between them is the one number this module produces, and until it exists
`vision.camera_to_world` is the identity and every detection is wrong by
wherever the lens happens to be.

Two ways in, because they suit different rigs:

  points     the arm carries something the camera can locate the *middle* of,
             and is driven to several places. Three correspondences are the
             minimum and four well-spread ones are a real answer. No marker
             orientation is needed, so anything the detector can find works —
             including the box already being picked.

  hand-eye   the arm carries a marker whose *pose* the camera can read. Fewer
             samples for the same rotation accuracy, at the price of needing
             something with a readable orientation on the flange.

Both report what they could not explain. A calibration that fits perfectly is
usually a calibration with too few samples to be wrong, so the residual is
returned beside the answer rather than left for a caller to compute.
"""

import numpy as np

from ..geometry.kinematics import inv, mat_to_xyz_rpy


class CalibrationError(RuntimeError):
    pass


def rigid_from_pairs(source, target):
    """The rotation and translation carrying `source` points onto `target`.

    Kabsch: centre both clouds, take the SVD of their covariance, and fix the
    sign so the answer is a rotation rather than a reflection. Reflections fit
    mirrored data better than rotations do, and a mirrored calibration puts an
    arm exactly as far the wrong side of the cell as it should have been the
    right side.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.shape[0] < 3:
        raise CalibrationError(
            "three matched points are the fewest that fix a frame; got %d"
            % len(source))

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _s, vt = np.linalg.svd(covariance)
    sign = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(vt.T @ u.T)))])
    rotation = vt.T @ sign @ u.T

    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = target_mean - rotation @ source_mean
    return out


def point_residuals(transform, source, target):
    """How far each point lands from where it should, in metres."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    moved = (np.asarray(transform, dtype=float)[:3, :3] @ source.T).T \
        + np.asarray(transform, dtype=float)[:3, 3]
    return np.linalg.norm(moved - target, axis=1)


def solve_from_points(camera_points, world_points):
    """Where the camera is, from places seen twice.

    `camera_points` are what the detector reported, `world_points` are where
    the arm says the same physical thing was. Returns the transform, the worst
    point and the RMS — a calibration is judged by the points it *cannot*
    explain, so both are returned rather than the fit alone.
    """
    transform = rigid_from_pairs(camera_points, world_points)
    errors = point_residuals(transform, camera_points, world_points)
    return transform, float(np.sqrt(np.mean(errors ** 2))), float(errors.max())


def spread_of(points):
    """How much of a volume the samples cover, as the three principal spans.

    Points along a line fix a rotation about that line not at all, and a
    calibration solved from them is confident and wrong. This is what a
    caller checks before believing an answer.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return np.zeros(3)
    centred = points - points.mean(axis=0)
    _u, singular, _vt = np.linalg.svd(centred, full_matrices=False)
    return singular / np.sqrt(len(points))


def solve_hand_eye(flange_mats, marker_in_camera):
    """Where the camera is, and where the marker sits on the flange.

    For a camera bolted to the cell and a marker bolted to the arm, every
    sample says the same thing twice:

        T_world_camera · T_camera_marker = T_world_flange · T_flange_marker

    Two unknowns, both constant. Eliminating the marker between a pair of
    samples turns it into the classical AX = XB, which OpenCV solves; doing
    the elimination here rather than handing OpenCV the raw poses is what
    keeps the answer's meaning under this module's control instead of under a
    naming convention that has changed between releases.
    """
    try:
        import cv2
    except ImportError as e:
        raise CalibrationError("hand-eye needs OpenCV: %s" % e)

    flange_mats = [np.asarray(m, dtype=float) for m in flange_mats]
    marker_in_camera = [np.asarray(m, dtype=float) for m in marker_in_camera]
    if len(flange_mats) != len(marker_in_camera):
        raise CalibrationError("every arm pose needs the reading taken at it")
    if len(flange_mats) < 3:
        raise CalibrationError(
            "three poses are the fewest that fix a hand-eye transform; got %d"
            % len(flange_mats))

    # A_k X = X B_k, with A from consecutive arm poses and B from the readings
    a_rot, a_pos, b_rot, b_pos = [], [], [], []
    for i in range(len(flange_mats) - 1):
        a = inv(flange_mats[i + 1]) @ flange_mats[i]
        b = marker_in_camera[i + 1] @ inv(marker_in_camera[i])
        a_rot.append(a[:3, :3])
        a_pos.append(a[:3, 3])
        b_rot.append(b[:3, :3])
        b_pos.append(b[:3, 3])

    rotation, translation = cv2.calibrateHandEye(
        a_rot, a_pos, b_rot, b_pos, method=cv2.CALIB_HAND_EYE_PARK)
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = np.asarray(rotation, dtype=float)
    camera_to_world[:3, 3] = np.asarray(translation, dtype=float).ravel()

    # the marker's place on the flange falls out of any one sample, and
    # averaging over all of them is the first thing that says whether the
    # answer is consistent
    on_flange = [inv(f) @ camera_to_world @ m
                 for f, m in zip(flange_mats, marker_in_camera)]
    spread = float(np.max([np.linalg.norm(on_flange[0][:3, 3] - m[:3, 3])
                           for m in on_flange]))
    return camera_to_world, on_flange[0], spread


def as_config(transform):
    """The xyz + rpy `vision.camera_to_world` is written in — the same shape
    an arm base uses, so one convention covers everything placed in the cell."""
    xyz, rpy = mat_to_xyz_rpy(np.asarray(transform, dtype=float))
    return {"xyz": [float(v) for v in xyz], "rpy": [float(v) for v in rpy]}
