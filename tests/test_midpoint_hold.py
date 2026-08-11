"""Pure geometry tests for the independent midpoint hold path."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.geometry import kinematics as K
from ur5dual.config import CellConfig
from ur5dual.tools.midpoint_hold import (
    MidpointHoldError, MidpointObject, MidpointServoSession,
)


class FakeArm:
    def __init__(self, cfg, tcp_world):
        self.cfg = cfg
        self._tcp_world = tcp_world

    def tcp_matrix_world(self):
        return self._tcp_world

    def tcp_pose(self):
        return K.mat_to_pose(K.inv(self.cfg.base_matrix()) @ self._tcp_world)

    def world_to_base(self, pose_world):
        return K.mat_to_pose(K.inv(self.cfg.base_matrix())
                             @ K.pose_to_mat(pose_world))


class FakeCell:
    def __init__(self):
        cfg = CellConfig()
        cfg.apply_mount_preset()
        self.arms = {
            "A": FakeArm(cfg.arms["A"], K.pose_to_mat([0.4, 0.1, 0.8, 0, 0, 0])),
            "B": FakeArm(cfg.arms["B"], K.pose_to_mat([0.4, -0.1, 0.8, 0, 0, 0])),
        }

    @property
    def connected_ids(self):
        return ["A", "B"]


cell = FakeCell()
obj = MidpointObject().capture(cell)
assert np.allclose(obj.pose_world[:3], [0.4, 0.0, 0.8])
assert max(obj.reconstruction_errors(cell).values()) < 1e-9
for arm_id in ("A", "B"):
    regenerated = (cell.arms[arm_id].cfg.base_matrix()
                   @ K.pose_to_mat(obj.targets_base[arm_id]))
    assert np.allclose(regenerated, cell.arms[arm_id].tcp_matrix_world())

session = MidpointServoSession(cell, obj)
session.state = "holding"
session.samples = session.READY_SAMPLES
before = session._target_pose.copy()
after = session.step_world(2, 0.0005)
assert abs(after[2] - before[2] - 0.0005) < 1e-12
try:
    session.step_world(0, 0.0021)
    raise AssertionError("an oversized commissioning step was accepted")
except MidpointHoldError:
    pass
print("midpoint HeldObject geometry: PASS")
