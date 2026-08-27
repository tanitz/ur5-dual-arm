"""ArUco pose readings for robot/camera hand-eye calibration."""

import cv2
import numpy as np


DICTIONARY_ID = cv2.aruco.DICT_4X4_50
DEFAULT_MARKER_ID = 42
DEFAULT_MARKER_SIZE = 0.050


class MarkerError(RuntimeError):
    pass


class MarkerPose:
    def __init__(self, marker_id, corners, matrix):
        self.id = int(marker_id)
        self.corners = np.asarray(corners, dtype=float).reshape(4, 2)
        self.matrix = np.asarray(matrix, dtype=float).reshape(4, 4)

    @property
    def centre(self):
        return self.corners.mean(axis=0)


def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx, 0.0, intrinsics.cx],
                     [0.0, intrinsics.fy, intrinsics.cy],
                     [0.0, 0.0, 1.0]], dtype=float)


def find_marker(frame, marker_id=DEFAULT_MARKER_ID,
                marker_size=DEFAULT_MARKER_SIZE):
    """The requested marker pose as camera<-marker, in metres."""
    if frame is None or frame.color is None:
        raise MarkerError("camera did not provide a colour image")
    dictionary = cv2.aruco.Dictionary_get(DICTIONARY_ID)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _rejected = cv2.aruco.detectMarkers(
        np.asarray(frame.color), dictionary, parameters=parameters)
    if ids is None:
        raise MarkerError("ArUco ID %d is not visible" % int(marker_id))
    matches = [i for i, value in enumerate(ids.ravel())
               if int(value) == int(marker_id)]
    if not matches:
        seen = ", ".join(str(int(value)) for value in ids.ravel())
        raise MarkerError("ArUco ID %d is not visible (saw %s)"
                          % (int(marker_id), seen))
    # A duplicate print in the background should not make the answer depend
    # on detector order. The larger quadrilateral is the calibration target.
    index = max(matches, key=lambda i: abs(cv2.contourArea(
        np.asarray(corners[i], dtype=np.float32).reshape(4, 2))))
    chosen = np.asarray(corners[index], dtype=np.float32).reshape(1, 4, 2)
    rotations, translations, _objects = cv2.aruco.estimatePoseSingleMarkers(
        chosen, float(marker_size), camera_matrix(frame.intrinsics),
        np.zeros(5, dtype=float))
    rotation, _ = cv2.Rodrigues(rotations[0, 0])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translations[0, 0]
    return MarkerPose(marker_id, chosen[0], matrix)
