"""
Object-centric coordination, checked without a robot.

A fake arm reports whatever TCP pose the test plants in it, so the whole
capture -> carry -> arm-target chain can be exercised on the geometry alone.
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.geometry import kinematics as K
from ur5dual.geometry import ur_kinematics as UK
from ur5dual.cell import Cell
from ur5dual.config import CellConfig
from ur5dual.coupling import (
    Coordinator, CouplingError, HeldObject, JOG_WATCHDOG, _Jog as C_JOG,
)

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


class FakeArm:
    """Just enough Arm to satisfy capture() and targets()."""

    def __init__(self, arm_cfg, tcp_world):
        self.cfg = arm_cfg
        self.id = arm_cfg.id
        self.connected = True
        self._tcp_world = np.asarray(tcp_world, dtype=float)

    def base_matrix(self):
        return self.cfg.base_matrix()

    def tcp_matrix_world(self):
        return self._tcp_world

    def world_to_base(self, pose_world):
        return K.mat_to_pose(K.inv(self.base_matrix()) @ K.pose_to_mat(pose_world))


def make_cell(tcp_a, tcp_b):
    cfg = CellConfig()
    cfg.apply_mount_preset()
    cfg.arms["B"].enabled = True
    cell = Cell(cfg)
    cell.arms["A"] = FakeArm(cfg.arms["A"], tcp_a)
    cell.arms["B"] = FakeArm(cfg.arms["B"], tcp_b)
    return cell


# Two grippers facing each other across a 200 mm object, both at z = 1.0
tcp_a = K.xyz_rpy_to_mat([0.40, 0.10, 1.00], [0, math.pi / 2, 0])
tcp_b = K.xyz_rpy_to_mat([0.40, -0.10, 1.00], [0, math.pi / 2, math.pi])
cell = make_cell(tcp_a, tcp_b)

print("capture")
obj = HeldObject("bottle").capture(cell, ("A", "B"), origin="midpoint")
check("object frame lands between the grippers",
      np.allclose(obj.pose_world[:3], [0.40, 0.0, 1.00], atol=1e-12),
      str(np.round(obj.pose_world[:3], 4)))
check("span is the gripper separation", abs(obj.span() - 0.20) < 1e-12,
      "%.4f" % obj.span())
check("both grasps captured", obj.arm_ids == ["A", "B"])
check("HeldObject exposes the virtual box pose",
      np.allclose(obj.virtual_pose_world, obj.pose_world, atol=1e-12))
check("virtual box remembers its midpoint origin", obj.frame_origin == "midpoint")
check("virtual box matrix is its pose as a transform",
      np.allclose(obj.virtual_matrix_world(), K.pose_to_mat(obj.pose_world),
                  atol=1e-12))

# The explicit virtual-pose API returns a copy.  Mutating a readout must not
# move the object behind the coordinator's back.
readout = obj.virtual_pose_world
readout[0] += 1.0
check("virtual pose readout cannot mutate the held object",
      not np.allclose(readout, obj.pose_world, atol=1e-12))

# regenerating the TCPs from the object pose must give back what we captured
for a, T in (("A", tcp_a), ("B", tcp_b)):
    check("grasp %s reproduces the TCP" % a,
          np.allclose(obj.tcp_world(a), T, atol=1e-12))

print("translate the object")
coord = Coordinator(cell, simulate=True)
start_pose = obj.pose_world.copy()
target = start_pose.copy()
target[0] += 0.10                      # 100 mm in +X
dur = coord.move_object(obj, target, lin_speed=0.05)
check("duration = distance / speed", abs(dur - 2.0) < 1e-9, "%.3f s" % dur)
check("trace was recorded", len(coord.trace) > 200, "%d samples" % len(coord.trace))

first, last = coord.trace[0][1], coord.trace[-1][1]
# each arm's own base-frame target, converted back to world, must equal the
# grasp carried along with the object
for a, T0 in (("A", tcp_a), ("B", tcp_b)):
    p_end_world = cell.arms[a].base_matrix() @ K.pose_to_mat(last[a])
    expect = T0.copy()
    expect[0, 3] += 0.10
    check("arm %s ends 100 mm along +X" % a,
          np.allclose(p_end_world, expect, atol=1e-9),
          str(np.round(p_end_world[:3, 3], 4)))

# the constraint that matters: the two TCPs never move relative to each other
worst = 0.0
for _t, tg in coord.trace:
    Ta = cell.arms["A"].base_matrix() @ K.pose_to_mat(tg["A"])
    Tb = cell.arms["B"].base_matrix() @ K.pose_to_mat(tg["B"])
    worst = max(worst, abs(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3]) - 0.20))
check("grip separation held over the whole path", worst < 1e-9,
      "worst %.3e m" % worst)

print("rotate the object about its own axis")
obj2 = HeldObject("drum").capture(cell, ("A", "B"), origin="midpoint")
centre = obj2.pose_world[:3].copy()
dur = coord.rotate_object(obj2, "z", math.radians(90), ang_speed=0.30)
check("duration = angle / speed", abs(dur - math.radians(90) / 0.30) < 1e-9,
      "%.3f s" % dur)
check("object centre stayed put",
      np.allclose(obj2.pose_world[:3], centre, atol=1e-12))

last = coord.trace[-1][1]
Ta = cell.arms["A"].base_matrix() @ K.pose_to_mat(last["A"])
Tb = cell.arms["B"].base_matrix() @ K.pose_to_mat(last["B"])
check("grip separation survived the spin",
      abs(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3]) - 0.20) < 1e-9)
# "about its own z" is not "about world z": the object frame here inherits
# arm A's orientation, so its z points along world +x. The gripper offset
# must swing about *that* axis.
R_obj = K.rotvec_to_mat(obj2.pose_world[3:])
R_delta_world = R_obj @ K.rotvec_to_mat([0, 0, math.radians(90)]) @ R_obj.T
expected_offset = R_delta_world @ (tcp_a[:3, 3] - centre)
check("arm A swung a quarter turn about the object's own z",
      np.allclose(Ta[:3, 3] - centre, expected_offset, atol=1e-9),
      "%s vs %s" % (np.round(Ta[:3, 3] - centre, 4), np.round(expected_offset, 4)))
check("the swing was not about world z",
      not np.allclose(expected_offset, [-0.10, 0.0, 0.0], atol=1e-6))

print("leader-follower falls out of the same machinery")
obj3 = HeldObject("bottle").capture(cell, ("A", "B"), origin="A")
check("object frame sits on arm A's TCP",
      np.allclose(K.pose_to_mat(obj3.pose_world), tcp_a, atol=1e-12))
check("arm A's grasp is the identity",
      np.allclose(obj3.grasps["A"], np.eye(4), atol=1e-12))

print("round trip through a dict")
back = HeldObject.from_dict(obj.to_dict())
check("grasps survive save/load",
      all(np.allclose(back.grasps[a], obj.grasps[a], atol=1e-9) for a in "AB"))
check("virtual frame origin survives save/load",
      back.frame_origin == obj.frame_origin)

# ── driving the object by hand ────────────────────────────────────────────
# The feed thread needs a robot, but the two things it does every cycle do
# not: work out where the object is now, and advance it. Both are called
# here directly, on a fake clock, which is the only way to test a jog that
# would otherwise take a real second of real motion to observe.
print("a move in flight is where the next command starts from")
live = Coordinator(cell)
live.object = obj4 = HeldObject("crate").capture(cell, ("A", "B"), "midpoint")
home = obj4.pose_world.copy()
away = home.copy()
away[0] += 0.100
live._hold = home.copy()
live._plan = (lambda s: K.interp_pose(home, away, s), 10.0,
              time.monotonic() - 5.0)          # planted half way along
mid = live.current_pose(obj4)
check("current_pose follows the path, not the last finished move",
      abs(mid[0] - (home[0] + 0.050)) < 2e-3, "%.4f m" % (mid[0] - home[0]))
check("obj.pose_world alone would have said the move never happened",
      abs(obj4.pose_world[0] - home[0]) < 1e-12)

print("a held jog, one cycle at a time")
live._plan = None
live._hold = home.copy()
dt = live._dt
speed = 0.050
t = time.monotonic()
live._jog = jog = C_JOG(np.array([0.0, 0.0, 1.0]), speed, t)
pose = home.copy()
poses = [pose]
for i in range(int(1.0 / dt)):                  # one second, button held
    t += dt
    jog.refreshed = t                           # the button says it is still down
    pose = live._advance_jog(jog, pose, t)
    poses.append(pose)
held_travel = pose[2] - home[2]
check("a second at 50 mm/s travels ~50 mm", abs(held_travel - 0.050) < 0.002,
      "%.1f mm" % (held_travel * 1000))
check("the ramp got the object up to speed",
      abs(jog.v - speed) < 1e-9, "%.3f m/s" % jog.v)
check("nothing moved on the axes that were not pressed",
      np.allclose(pose[:2], home[:2], atol=1e-12)
      and np.allclose(pose[3:], home[3:], atol=1e-12))

live.stop_jog()
for i in range(int(0.5 / dt)):                  # let go; it must coast down
    t += dt
    pose = live._advance_jog(jog, pose, t)
    poses.append(pose)
check("letting go brings it to a stop", live._jog is None and jog.v == 0.0)
check("the stop is a ramp, not a cut — held 1 s ends up ~50 mm on",
      abs((pose[2] - home[2]) - 0.050) < 0.001,
      "%.2f mm" % ((pose[2] - home[2]) * 1000))

# the point of the whole exercise: both arms carried it, and neither wandered
worst_gap, worst_follow = 0.0, 0.0
for p in poses:
    tg = obj4.targets(cell, p)
    Ta = cell.arms["A"].base_matrix() @ K.pose_to_mat(tg["A"])
    Tb = cell.arms["B"].base_matrix() @ K.pose_to_mat(tg["B"])
    worst_gap = max(worst_gap, abs(np.linalg.norm(Tb[:3, 3] - Ta[:3, 3]) - 0.20))
    # each TCP must have travelled exactly what the object travelled
    for T, T0 in ((Ta, tcp_a), (Tb, tcp_b)):
        worst_follow = max(worst_follow,
                           float(np.linalg.norm((T[:3, 3] - T0[:3, 3])
                                                - (p[:3] - home[:3]))))
check("the grippers stayed the same distance apart all through the jog",
      worst_gap < 1e-12, "worst %.3e m" % worst_gap)
check("both TCPs moved exactly as far as the object did",
      worst_follow < 1e-12, "worst %.3e m" % worst_follow)

print("a jog nobody is holding any more stops itself")
t = time.monotonic()
live._jog = jog = C_JOG(np.array([1.0, 0.0, 0.0]), speed, t)
pose = home.copy()
for i in range(int(3.0 / dt)):                  # refreshed is never updated
    t += dt
    pose = live._advance_jog(jog, pose, t)
check("the watchdog ended it", live._jog is None)
check("and it ended within a hair of the watchdog period",
      abs((pose[0] - home[0]) - speed * JOG_WATCHDOG) < 0.002,
      "%.1f mm vs %.1f mm" % ((pose[0] - home[0]) * 1000,
                              speed * JOG_WATCHDOG * 1000))

print("geometry that describes a different cell is refused, not calibrated")
check("the preset cell is not complained about",
      cell.geometry_complaint() is None, str(cell.geometry_complaint()))
flipped = make_cell(tcp_a, tcp_b)
T_b = flipped.config.arms["B"].base_matrix()
# half a turn about the base's own X, which is what tips its Z back over the
# mast. A turn about its own Z would not: that spins the arm on its flange and
# leaves it reaching the same way, which is a different (and legal) thing.
T_b[:3, :3] = T_b[:3, :3] @ K.rotvec_to_mat([math.pi, 0, 0])
flipped.config.arms["B"].set_base_matrix(T_b)
complaint = flipped.geometry_complaint()
check("an arm reaching back at the mast is caught", complaint is not None)
check("and the arm is named", complaint and "arm B" in complaint,
      (complaint or "")[:60])
check("the fix named is the one that corrects it",
      complaint and "config/cell.yaml" in complaint)
flipped.config.mount["enforce_outward_guard"] = False
check("a verified custom cell can disable only the outward heuristic",
      flipped.geometry_complaint() is None)

print("simulation says so rather than sitting still")
try:
    Coordinator(cell, simulate=True).command_jog(obj4, [0, 0, 1])
    check("a simulated jog is refused", False, "it was accepted")
except CouplingError as e:
    check("a simulated jog is refused", "simulation" in str(e))

print("joint control that cannot prove itself steps aside")
# The state this replaces: the panel refused to start the servo loop at all
# because our forward kinematics missed one arm by 3 mm — the size of that
# robot's own factory calibration. The operator was left with a workpiece
# gripped by two live arms and a jog grid gone grey. Cartesian targets are
# derived from poses the controller itself reported and solved by that same
# controller, so the error disqualifying joint targets is one they cannot
# carry: falling back is safe, and being stranded is not.
seed_q = np.array([0.2, -1.2, 1.4, -1.0, -1.5, 0.3])


class KinematicArm(FakeArm):
    def __init__(self, arm_cfg, tcp_world, q, error=0.0):
        super().__init__(arm_cfg, tcp_world)
        self.tcp_offset = None
        pose = K.mat_to_pose(UK.fk_tcp(q))
        pose[0] += error                       # where we and it disagree
        self._state = {"q_actual": q, "tcp_pose": pose}

    def state(self):
        return self._state


def solver_cell(error_b):
    c = make_cell(tcp_a, tcp_b)
    c.arms["A"] = KinematicArm(c.config.arms["A"], tcp_a, seed_q)
    c.arms["B"] = KinematicArm(c.config.arms["B"], tcp_b, seed_q, error_b)
    return c

agreeing = solver_cell(0.0)
held = HeldObject("crate").capture(agreeing, ("A", "B"), "midpoint")
coord = Coordinator(agreeing, drive_robots=True)
check("with the kinematics matching, the solver runs",
      coord._start_solver(held) is True and coord.solver is not None)

off = solver_cell(0.003)          # 3 mm, as arm B actually reported
held_off = HeldObject("crate").capture(off, ("A", "B"), "midpoint")
said = []
off.listeners.append(said.append)
coord_off = Coordinator(off, drive_robots=True)
stood_down = None
try:
    stood_down = coord_off._start_solver(held_off)
except CouplingError as e:
    check("a 3 mm disagreement no longer strands the operator", False, str(e)[:70])
if stood_down is not None:
    check("a 3 mm disagreement no longer strands the operator",
          stood_down is False)
    check("and the solver is left off rather than half up",
          coord_off.solver is None)
    check("the panel is told which arm and by how much",
          any("arm B" in m and "3.0 mm" in m for m in said),
          (said[-1] if said else "")[:70])
    check("and told what it is running instead",
          any("Cartesian targets" in m for m in said))

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
