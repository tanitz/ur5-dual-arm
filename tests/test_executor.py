"""
Running a line, against a simulated cell.

test_program.py checks what a line *says*; this checks what it *does*. The
two halves that only appear at run time are here: an offset resolved against
where the arm actually is, and a line that sends both arms and does not end
until both have arrived.

Nothing here needs a robot. The simulated arms answer the same calls the real
ones do, so every path except the servo loop itself runs exactly as it will on
the cell -- which is the point, because the alternative is finding out with two
arms holding something.
"""

import math
import os
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.cell import Cell
from ur5dual.config import CellConfig
from ur5dual.program.executor import Executor, ProgramError
from ur5dual.program.steps import PointLibrary, Program, Step, make_target

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


def move(link="solo", **slots):
    return Step("MOVE", link=link, **slots)


def build(calibrated=True):
    """A simulated cell, off the singularity, with an executor on it."""
    cfg = CellConfig.load()
    cfg.arms["B"].enabled = True
    # Both flags are set either way. The file on disk carries whatever this
    # cell was last measured to, and a test that inherits it passes or fails
    # depending on the bench rather than on the code.
    cfg.calibrated = False
    cfg.translation_calibrated = bool(calibrated)
    cell = Cell(cfg, simulated=True)
    cell.sim_ready_pose()
    points = PointLibrary()
    ex = Executor(cell, points)
    ex.on_log = lambda text: logged.append(text)
    return cell, points, ex


logged = []
cell, points, ex = build()
here_a = cell.arms["A"].tcp_pose_world()
here_b = cell.arms["B"].tcp_pose_world()
print("both arms start where sim_ready_pose put them")
print("   A %s" % np.round(here_a[:3] * 1000, 1))
print("   B %s" % np.round(here_b[:3] * 1000, 1))

# somewhere each arm can certainly reach: 60 mm above where it already is
points.set("above_A", here_a + np.array([0, 0, 0.06, 0, 0, 0]))
points.set("above_B", here_b + np.array([0, 0, 0.06, 0, 0, 0]))


def pos(arm_id):
    return cell.arms[arm_id].tcp_pose_world()[:3]


# ── one arm ───────────────────────────────────────────────────────────────
print("\na solo line moves the column that is filled, and only that one")
before_b = pos("B")
ex._execute(move("solo", a=make_target(point="above_A", motion="movel")))
check("arm A arrived", np.allclose(pos("A"), points.get("above_A")[:3], atol=2e-3),
      str(np.round(pos("A") * 1000, 1)))
check("arm B did not move", np.allclose(pos("B"), before_b, atol=1e-9))

print("\nan offset with no place in it is read against where the arm is now")
start = pos("A").copy()
step = move("solo", a=make_target(offset=[0, 0, -0.02, 0, 0, 0], motion="movel"))
ex._execute(step)
check("the first run drops 20 mm",
      abs((pos("A") - start)[2] + 0.02) < 2e-3,
      "%.1f mm" % ((pos("A") - start)[2] * 1000))
ex._execute(step)
check("the same line run again drops 20 mm more — it is relative, not absolute",
      abs((pos("A") - start)[2] + 0.04) < 2e-3,
      "%.1f mm" % ((pos("A") - start)[2] * 1000))

print("\na place plus an offset needs no second taught point")
ex._execute(move("solo", a=make_target(point="above_A",
                                       offset=[0, 0, 0.03, 0, 0, 0],
                                       motion="movel")))
check("it lands one offset above the place",
      abs(pos("A")[2] - (points.get("above_A")[2] + 0.03)) < 2e-3,
      "%.1f mm" % (pos("A")[2] * 1000))

print("\na tool-frame offset follows the arm's own axes, not the cell's")
T = cell.arms["A"].tcp_matrix_world()
want = T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, 0.03])
ex._execute(move("solo", a=make_target(offset=[0, 0, 0.03, 0, 0, 0],
                                       frame="tool", motion="movel")))
check("it moved along the tool's Z", np.allclose(pos("A"), want, atol=2e-3),
      "%s vs %s" % (np.round(pos("A") * 1000, 1), np.round(want * 1000, 1)))

print("\nan unreachable target is refused before anything is sent")
points.set("mars", [40.0, 0.0, 1.0, 0, 0, 0])
was = pos("A").copy()
try:
    ex._execute(move("solo", a=make_target(point="mars")))
    check("a target 40 m away is refused", False, "it was accepted")
except ProgramError as e:
    check("a target 40 m away is refused", "reach" in str(e).lower(), str(e))
check("and the arm never moved", np.allclose(pos("A"), was, atol=1e-9))

# ── both arms on one line ─────────────────────────────────────────────────
print("\n'together' sends both columns and waits for both")
ex._execute(move("solo", a=make_target(point="above_A", motion="movel")))
a_before, b_before = pos("A").copy(), pos("B").copy()
t0 = time.monotonic()
ex._execute(move("together",
                 a=make_target(offset=[0, 0, -0.03, 0, 0, 0], motion="movel"),
                 b=make_target(offset=[0, 0, -0.03, 0, 0, 0], motion="movel")))
elapsed = time.monotonic() - t0
check("arm A moved", abs((pos("A") - a_before)[2] + 0.03) < 2e-3)
check("arm B moved too", abs((pos("B") - b_before)[2] + 0.03) < 2e-3)
check("the line did not return until both had settled", elapsed >= 0.15,
      "%.2f s" % elapsed)

print("\na second column that cannot be reached stops the first from moving")
a_before, b_before = pos("A").copy(), pos("B").copy()
try:
    ex._execute(move("together", a=make_target(point="above_A"),
                     b=make_target(point="mars")))
    check("the line is refused", False, "it was accepted")
except ProgramError:
    check("the line is refused", True)
check("arm A was never sent", np.allclose(pos("A"), a_before, atol=1e-9))
check("arm B was never sent", np.allclose(pos("B"), b_before, atol=1e-9))

# ── the pair ──────────────────────────────────────────────────────────────
print("\na pair line gives one world delta to both arms")
a_before, b_before = pos("A").copy(), pos("B").copy()
gap_before = cell.tcp_separation()
ex._execute(move("pair", pair=make_target(offset=[0, 0, 0.04, 0, 0, 0])))
check("arm A rose 40 mm", abs((pos("A") - a_before)[2] - 0.04) < 2e-3,
      "%.1f mm" % ((pos("A") - a_before)[2] * 1000))
check("arm B rose 40 mm", abs((pos("B") - b_before)[2] - 0.04) < 2e-3,
      "%.1f mm" % ((pos("B") - b_before)[2] * 1000))
check("and the gap between them is unchanged — that is the whole point",
      abs(cell.tcp_separation() - gap_before) < 1e-3,
      "%.2f -> %.2f mm" % (gap_before * 1000, cell.tcp_separation() * 1000))

print("\nwhat a pair line refuses at run time, not only on paper")
for name, target, want in (
        ("a rotation", make_target(offset=[0, 0, 0, 0, 0, 0.1]), "cannot rotate"),
        ("a tool frame", make_target(offset=[0.01, 0, 0, 0, 0, 0], frame="tool"),
         "world frame only")):
    try:
        ex._execute(move("pair", pair=target))
        check("%s is refused" % name, False, "it was accepted")
    except ProgramError as e:
        check("%s is refused" % name, want in str(e), str(e))

print("\nand it refuses to run at all before the directions are measured")
_c2, _p2, ex2 = build(calibrated=False)
try:
    ex2._execute(move("pair", pair=make_target(offset=[0, 0, 0.01, 0, 0, 0])))
    check("an uncalibrated pair move is refused", False, "it was accepted")
except ProgramError as e:
    check("an uncalibrated pair move is refused",
          "check_directions_online" in str(e), str(e))

# ── the rest of the vocabulary ────────────────────────────────────────────
print("\none OUT line drives both controllers")
ex._execute(Step("OUT", arm="both", output=0, state=True, settle=0.0))
check("arm A output 0 is on", cell.arms["A"].digital_out.get(0) is True)
check("arm B output 0 is on too", cell.arms["B"].digital_out.get(0) is True)
ex._execute(Step("OUT", arm="A", output=0, state=False, settle=0.0))
check("naming one arm leaves the other alone",
      cell.arms["A"].digital_out.get(0) is False
      and cell.arms["B"].digital_out.get(0) is True)
ex._execute(Step("OUT", arm="B", output=5, state=True, settle=0.0))
check("any of the eight outputs can be driven",
      cell.arms["B"].digital_out.get(5) is True)

print("\nWAIT_IN holds until the input reads as asked")
cell.arms["A"].motion.set_digital_in(2, False)
waited = []


def _flip_after(seconds, arm_id, number, state):
    def run():
        time.sleep(seconds)
        cell.arms[arm_id].motion.set_digital_in(number, state)
        waited.append(True)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


_flip_after(0.25, "A", 2, True)
t0 = time.monotonic()
ex._execute(Step("WAIT_IN", arm="A", input=2, state=True, timeout=10))
elapsed = time.monotonic() - t0
check("it waited for the signal rather than running straight through",
      elapsed >= 0.2, "%.2f s" % elapsed)
check("and stopped waiting once it arrived", elapsed < 5, "%.2f s" % elapsed)

cell.arms["A"].motion.set_digital_in(3, False)
try:
    ex._execute(Step("WAIT_IN", arm="A", input=3, state=True, timeout=0.3))
    check("a signal that never comes fails the program", False, "it carried on")
except ProgramError as e:
    check("a signal that never comes fails the program",
          "never went ON" in str(e), str(e))

print("\nvariables and branching")
ex.vars = {}
ex._execute(Step("SET_VAR", name="n", op="=", value=2))
check("a variable can be set", ex.vars["n"] == 2.0, str(ex.vars))
ex._execute(Step("SET_VAR", name="n", op="+=", value=3))
check("and counted up", ex.vars["n"] == 5.0, str(ex.vars))
ex._execute(Step("SET_VAR", name="n", op="-=", value=1))
check("and down", ex.vars["n"] == 4.0, str(ex.vars))

labels = {"yes": 7, "no": 9}
check("a true test jumps to its label",
      ex._execute(Step("IF", source="var", name="n", compare=">=", value=4,
                       target="yes", otherwise="no"), labels) == 7)
check("a false one takes the other",
      ex._execute(Step("IF", source="var", name="n", compare=">", value=99,
                       target="yes", otherwise="no"), labels) == 9)
check("and with no else it carries on to the next line",
      ex._execute(Step("IF", source="var", name="n", compare=">", value=99,
                       target="yes"), labels) is None)
cell.arms["A"].motion.set_digital_in(4, True)
check("a test on an input reads the controller",
      ex._execute(Step("IF", source="input", arm="A", input=4, state=True,
                       target="yes"), labels) == 7)
check("JUMP goes there unconditionally",
      ex._execute(Step("JUMP", target="no"), labels) == 9)
try:
    ex._execute(Step("JUMP", target="gone"), labels)
    check("a jump to a label that is not there stops the program", False)
except ProgramError as e:
    check("a jump to a label that is not there stops the program",
          "no label called" in str(e), str(e))

print("\na counted loop runs the right number of times")
counted = Program("counted")
counted.add(Step("LABEL", name="top"))
counted.add(Step("SET_VAR", name="i", op="+=", value=1))
counted.add(Step("OUT", arm="A", output=6, state=True, settle=0.0))
counted.add(Step("IF", source="var", name="i", compare="<", value=4,
                 target="top"))
ex.start(counted)
deadline = time.monotonic() + 30
while ex.running and time.monotonic() < deadline:
    time.sleep(0.02)
check("it went round until the count said stop", ex.vars.get("i") == 4.0,
      str(ex.vars))

print("\na loop of nothing but jumps is stopped rather than left spinning")
spin = Program("spin")
spin.add(Step("LABEL", name="here"))
spin.add(Step("JUMP", target="here"))
finished = []
ex.on_finished = lambda ok, msg: finished.append((ok, msg))
ex.start(spin)
deadline = time.monotonic() + 60
while ex.running and time.monotonic() < deadline:
    time.sleep(0.05)
check("it gave up and said why",
      finished and not finished[0][0] and "looping on itself" in finished[0][1],
      str(finished))
ex.on_finished = None

print("\nWHERE reads the cell out and moves nothing")
a_before, b_before = pos("A").copy(), pos("B").copy()
logged.clear()
ex._execute(Step("WHERE"))
check("both arms are reported", any(" A " in m for m in logged)
      and any(" B " in m for m in logged), str(len(logged)) + " lines")
check("the gap is reported", any("gap" in m for m in logged))
check("nothing moved", np.allclose(pos("A"), a_before, atol=1e-9)
      and np.allclose(pos("B"), b_before, atol=1e-9))

print("\na coupled line with nothing attached is refused by the executor too")
try:
    ex._execute(move("coupled", obj=make_target(point="above_A")))
    check("refused", False, "it was accepted")
except ProgramError as e:
    check("refused", "nothing is attached" in str(e), str(e))

# ── a whole program, on its own thread ────────────────────────────────────
print("\na program runs, reports each line, and finishes")
prog = Program("smoke")
prog.add(move("together", a=make_target(point="above_A", motion="movel"),
              b=make_target(point="above_B", motion="movel")))
prog.add(Step("GRIP", a={"output": 0, "state": True},
              b={"output": 0, "state": True}, settle=0.0))
prog.add(move("pair", pair=make_target(offset=[0, 0, 0.02, 0, 0, 0])))
prog.add(Step("WHERE"))
prog.add(move("solo", a=make_target(offset=[0, 0, -0.02, 0, 0, 0], motion="movel")))

seen, done = [], []
ex.on_step = lambda i, s: seen.append(i)
ex.on_finished = lambda ok, msg: done.append((ok, msg))
ex.start(prog)
deadline = time.monotonic() + 30
while ex.running and time.monotonic() < deadline:
    time.sleep(0.02)
check("it finished", bool(done) and done[0][0], str(done))
check("every line was reported", seen == [0, 1, 2, 3, 4], str(seen))
check("both arms ended up where the program left them",
      abs(pos("A")[2] - (points.get("above_A")[2] + 0.02 - 0.02)) < 3e-3,
      "%.1f mm" % (pos("A")[2] * 1000))

print("\n▷ To runs one line and stops")
ex._execute(move("solo", a=make_target(point="above_A", motion="movel")))
picked = Program("pick one")
picked.add(move("solo", a=make_target(offset=[0, 0, -0.03, 0, 0, 0],
                                      motion="movel")))
picked.add(move("solo", a=make_target(offset=[0, 0, -0.03, 0, 0, 0],
                                      motion="movel")))
picked.add(move("solo", a=make_target(offset=[0, 0, -0.03, 0, 0, 0],
                                      motion="movel")))
seen = []
ex.on_step = lambda i, s: seen.append(i)
ex.on_finished = None
start = pos("A").copy()
ex.start(picked, only=1)
deadline = time.monotonic() + 30
while ex.running and time.monotonic() < deadline:
    time.sleep(0.02)
check("only the chosen line ran", seen == [1], str(seen))
check("and it moved once, not three times",
      abs((pos("A") - start)[2] + 0.03) < 2e-3,
      "%.1f mm" % ((pos("A") - start)[2] * 1000))
try:
    ex.start(picked, only=9)
    check("a row that is not there is refused", False, "it started")
except ProgramError as e:
    check("a row that is not there is refused", "no step selected" in str(e))

print("\nCALL is expanded before the thread starts")
sub = Program("sub")
sub.add(Step("WHERE"))
sub.add(Step("DELAY", seconds=0.01))
outer = Program("outer")
outer.add(Step("CALL", program="sub", repeat=2))
seen = []
ex.on_step = lambda i, s: seen.append(i)
ex.start(outer, load=lambda name: sub)
deadline = time.monotonic() + 30
while ex.running and time.monotonic() < deadline:
    time.sleep(0.02)
check("the called program ran, twice", len(seen) == 4, str(seen))
check("and every line reported the CALL's own row",
      set(seen) == {0}, str(seen))

bad = Program("bad call")
bad.add(Step("CALL", program="missing"))
try:
    ex.start(bad, load=lambda name: (_ for _ in ()).throw(KeyError(name)))
    check("a call that cannot be loaded stops it before the thread", False)
except ProgramError as e:
    check("a call that cannot be loaded stops it before the thread",
          "cannot load" in str(e), str(e))
check("and nothing is running", not ex.running)

try:
    ex.start(bad)
    check("with no library at all it says so", False, "it started")
except ProgramError as e:
    check("with no library at all it says so", "no program library" in str(e),
          str(e))

print("\nFIND corrects a taught pick by where the box actually is")
from ur5dual.geometry.kinematics import mat_to_pose as _mat_to_pose   # noqa: E402
from ur5dual.program.steps import resolve_target                      # noqa: E402
from ur5dual.vision.planar import (                                   # noqa: E402
    PlaneFile, PlaneMap, box_on_plane, rim_corners,
)
from ur5dual.vision.service import VisionService                      # noqa: E402

# The simulated source, and none of the operator's own camera settings: this
# check is about what FIND does with a detection, and a RealSense plugged into
# the bench would answer it with the room — or refuse to open at all, being
# already held by the panel that is using it.
# The opening is given too: the size on file is whatever crate the cell was
# last pointed at, and the simulated source draws that size at 1.2 m whether
# or not anything can be found in it. A 600 x 400 box is what this check is
# about — a detection arriving at all — not what the bench is set to today.
BOX = (0.60, 0.40, 0.20)
work = tempfile.mkdtemp(prefix="openbox-executor-")
vision = VisionService(dict(cell.config.vision, source="sim",
                            box_size=list(BOX)))
vision.start()
ex.vision = vision
try:
    # the camera sits at the world origin for this check, so what it sees is
    # already in world and the transform under test is the correction alone.
    # `calibrated` says that on purpose: a cell that has never been told where
    # its lens is refuses FIND rather than measuring against the identity.
    cell.config.vision["camera_to_world"] = {"xyz": [0, 0, 0], "rpy": [0, 0, 0]}
    cell.config.vision["calibrated"] = True
    home = vision.fresh(timeout=5.0)
    check("the simulated camera sees its box", home.found, str(home.error))
    points.set("box_home", _mat_to_pose(home.detection.matrix()))
    taught = cell.arms["A"].tcp_pose_world()
    points.set("pick_here", taught)

    ex._execute(Step("FIND", into="part", reference="box_home", timeout=5))
    check("a box that has not moved needs no correction",
          np.allclose(ex.poses["part"], np.eye(4), atol=1e-6))
    check("and says it found something", ex.vars.get("part_found") == 1.0)

    vision.camera.place(centre=(0.04, -0.025), yaw=math.radians(18))
    ex._execute(Step("FIND", into="part", reference="box_home", timeout=5))
    shift = _mat_to_pose(ex.poses["part"])
    check("a box that moved 40 by -25 reports that",
          abs(shift[0] - 0.040) < 5e-3 and abs(shift[1] + 0.025) < 5e-3,
          str(np.round(shift[:3] * 1000, 1)))
    check("and the 18 degrees it turned",
          abs(np.degrees(np.linalg.norm(shift[3:])) - 18.0) < 0.5,
          "%.2f deg" % np.degrees(np.linalg.norm(shift[3:])))

    plain = resolve_target(make_target(point="pick_here"), points)
    moved = resolve_target(make_target(point="pick_here", correct_by="part"),
                           points, correction=ex.poses["part"])
    check("the taught pick is carried with the box",
          not np.allclose(plain[:3], moved[:3], atol=1e-4),
          "%s -> %s" % (np.round(plain[:3] * 1000, 1),
                        np.round(moved[:3] * 1000, 1)))
    check("and the wrist turns with it rather than spinning on the spot",
          not np.allclose(plain[3:], moved[3:], atol=1e-4))

    print("\nwhat a camera is not allowed to do")
    vision.camera.place(centre=(0.30, 0.20), yaw=0.0)
    try:
        ex._execute(Step("FIND", into="part", reference="box_home", timeout=5))
        check("a correction bigger than the limit is refused", False,
              "it was accepted")
    except ProgramError as e:
        check("a correction bigger than the limit is refused",
              "past the" in str(e), str(e)[:80])

    ex.poses.clear()
    try:
        ex._execute(move("solo", a=make_target(point="pick_here",
                                               correct_by="part")))
        check("a line corrected by nothing found is refused", False,
              "it ran uncorrected")
    except ProgramError as e:
        check("a line corrected by nothing found is refused",
              "nothing has found it" in str(e), str(e)[:80])

    vision.camera.place(centre=(0.0, 0.0), yaw=0.0)
    vision.detector.tracker.max_error = -1.0    # reject every candidate
    vision.detector.tracker.hold_frames = 0
    ex._execute(Step("FIND", into="part", reference="box_home", timeout=5))
    check("finding nothing is not an error — it is a flag to branch on",
          ex.vars.get("part_found") == 0.0 and "part" not in ex.poses,
          str(ex.vars.get("part_found")))
    vision.detector.tracker.max_error = 4.0
    vision.detector.tracker.hold_frames = 15

    print("\nand a cell told neither where its lens is nor where its "
          "surface is")
    cell.config.vision["calibrated"] = False
    try:
        ex._execute(Step("FIND", into="part", reference="box_home", timeout=5))
        check("refuses to turn a detection into a place in itself", False,
              "it answered anyway")
    except ProgramError as e:
        check("refuses to turn a detection into a place in itself",
              "no plane map" in str(e), str(e)[:60])
    cell.config.vision["calibrated"] = True

    # -- the same FIND, read off the surface instead of the lens -----------
    print("\nthe same FIND, three numbers off a surface instead of six")
    # The plane the rim lies on is one box height nearer than the table, and
    # the simulated lens looks straight down its own +Z. A right-handed plane
    # frame there has +Y turned away from the lens, which is -Y in camera
    # axes; handing it the camera's own Y is the reflection `from_samples`
    # refuses, and this is the only place in the tests that has to know it.
    def seen_at(x, y, yaw=0.0):
        vision.camera.place(centre=(x, -y), yaw=-yaw)
        reading = vision.fresh(timeout=5.0)
        assert reading.found, reading.why_not()
        return reading.detection.corners

    fit_at = [(-0.15, -0.10), (0.15, -0.10), (0.15, 0.10), (-0.15, 0.10)]
    plane_map = PlaneMap.from_samples(
        np.vstack([seen_at(x, y) for x, y in fit_at]),
        np.vstack([rim_corners(x, y, 0.0, BOX) for x, y in fit_at]),
        height=BOX[2])
    check("a map fits the simulated surface to under five millimetres",
          plane_map.rms < 0.005, plane_map.description)

    ex.surface = PlaneFile(plane_map, box_size=BOX,
                           path=os.path.join(work, "plane.json"))
    ex.surface.teach("box_home",
                     box_on_plane(seen_at(0.0, 0.0), plane_map, BOX))
    check("FIND's reference now resolves against the surface",
          ex.taught_on_surface() == {"box_home"})

    ex._execute(Step("FIND", into="part", reference="box_home", timeout=5))
    check("a box that has not moved still needs no correction",
          np.allclose(ex.poses["part"], np.eye(4), atol=5e-3),
          str(np.round(_mat_to_pose(ex.poses["part"])[:3] * 1000, 1)))

    seen_at(0.060, -0.035, math.radians(17))
    ex._execute(Step("FIND", into="part", reference="box_home", timeout=5))
    shift = _mat_to_pose(ex.poses["part"])
    check("a box moved 60 by -35 on the surface reports that",
          abs(shift[0] - 0.060) < 6e-3 and abs(shift[1] + 0.035) < 6e-3,
          str(np.round(shift[:3] * 1000, 1)))
    check("and the 17 degrees it turned, about the surface's normal alone",
          abs(np.degrees(np.linalg.norm(shift[3:])) - 17.0) < 1.0
          and abs(shift[2]) < 1e-9,
          "%.2f deg, z %+.4f mm" % (np.degrees(np.linalg.norm(shift[3:])),
                                    shift[2] * 1000))

    ex.surface.forget("box_home")
    try:
        ex._execute(Step("FIND", into="part", reference="box_home", timeout=5))
        check("a surface with nothing taught on it does not fall back", False,
              "it used the lens instead")
    except ProgramError as e:
        check("a surface with nothing taught on it does not fall back",
              "nothing was taught" in str(e), str(e)[:56])
    ex.surface = None
finally:
    vision.stop()
    ex.vision = None

print("\na program that cannot start says why, and starts nothing")
bad = Program("bad")
bad.add(move("coupled", obj=make_target(point="above_A")))
try:
    ex.start(bad)
    check("it is refused before the thread starts", False, "it started")
except ProgramError as e:
    check("it is refused before the thread starts", "ATTACH" in str(e), str(e))
check("and nothing is running", not ex.running)

print("\nwarnings are logged rather than refused")
logged.clear()
warn = Program("warn")
warn.add(move("solo", b=make_target(offset=[0, 0, -0.01, 0, 0, 0], motion="movel")))
ex.start(warn)
deadline = time.monotonic() + 30
while ex.running and time.monotonic() < deadline:
    time.sleep(0.02)
check("the offset-from-an-unknown-start warning reached the log",
      any("wherever it was left" in m for m in logged),
      str([m for m in logged if "warning" in m]))

# ── pause and resume ──────────────────────────────────────────────────────
print("\npause holds the program where it is, and resume carries on from there")
ran = []
paused_program = Program("pausing")
paused_program.add(Step("DELAY", seconds=0.2))
paused_program.add(move("solo", a=make_target(offset=[0, 0, 0.01, 0, 0, 0],
                                              motion="movel")))
paused_program.add(Step("DELAY", seconds=0.2))
paused_program.add(move("solo", a=make_target(offset=[0, 0, -0.01, 0, 0, 0],
                                              motion="movel")))
ex.on_step = lambda row, step: ran.append(row)
finished = []
ex.on_finished = lambda ok, message: finished.append((ok, message))

ex.start(paused_program)
deadline = time.monotonic() + 5
while not ran and time.monotonic() < deadline:
    time.sleep(0.01)
ex.pause()
time.sleep(0.5)          # long enough for every remaining step to have run
held_at = list(ran)
check("it is still running rather than aborted", ex.running)
check("and it reports itself paused", ex.paused)
check("no further line started while paused", ran == held_at,
      "%s then %s" % (held_at, ran))
check("nothing was reported finished", finished == [], str(finished))

ex.resume()
deadline = time.monotonic() + 15
while ex.running and time.monotonic() < deadline:
    time.sleep(0.02)
check("resume carries on rather than restarting",
      ran[:len(held_at)] == held_at, "%s" % (ran,))
check("every line ran, exactly once", ran == [0, 1, 2, 3], str(ran))
check("and the program finished cleanly",
      finished and finished[0][0], str(finished))

print("\npause in the middle of a move stops the arms and re-sends on resume")
ex._pause.clear()
ex._stop.clear()
halts = []
real_halt = cell.halt
cell.halt = lambda: (halts.append(time.monotonic()), real_halt())[1]
sent = []
outcome = []
# 50 mm above where arm A is, which nothing in sim will carry it to — so the
# wait stays in the wait, which is the state this is about
never = {"A": cell.arms["A"].tcp_pose_world() + np.array([0, 0, 0.05, 0, 0, 0])}


def _resend():
    sent.append(time.monotonic())


def _arrive():
    try:
        ex._wait_until_arrived(never, timeout=0.5, resend=_resend)
        outcome.append("arrived")
    except ProgramError as e:
        outcome.append(str(e))


ex.pause()
began = time.monotonic()
waiter = threading.Thread(target=_arrive, daemon=True)
waiter.start()
time.sleep(0.4)
check("the arms are halted rather than left running", len(halts) >= 1,
      "%d halts" % len(halts))
check("and nothing is re-sent while it is held", sent == [], str(sent))
check("the wait has not given up", outcome == [], str(outcome))

time.sleep(1.1)          # well past the 0.5 s arrival limit, all of it held
check("a hold does not count against the arrival limit", outcome == [],
      str(outcome))

ex.resume()
waiter.join(timeout=10)
elapsed = time.monotonic() - began
check("the move is put back on the wire exactly once", len(sent) == 1,
      str(len(sent)))
check("and only after the hold ended", sent and sent[0] - began >= 1.4,
      "%.2f s" % ((sent[0] - began) if sent else -1))
check("the limit then runs from the resume, not from before the hold",
      elapsed >= 1.5 + 0.4, "%.2f s" % elapsed)
check("and an arm that truly never arrives still says so",
      outcome and "did not arrive" in outcome[0], str(outcome))

print("\nStop while held part way through aborts rather than resuming")
sent.clear()
outcome.clear()
ex._pause.clear()
ex._stop.clear()
ex.pause()
waiter = threading.Thread(target=_arrive, daemon=True)
waiter.start()
time.sleep(0.3)
ex.stop()
waiter.join(timeout=10)
check("it comes out stopped", outcome == ["stopped"], str(outcome))
check("and the move is never re-sent", sent == [], str(sent))
cell.halt = real_halt
ex._pause.clear()
ex._stop.clear()

print("\na pause holds the delay clock instead of running it down")
ex._pause.clear()
ex._stop.clear()
slept = []


def _timed_sleep():
    began = time.monotonic()
    try:
        ex._sleep(0.3)
    except ProgramError:
        pass
    slept.append(time.monotonic() - began)


ex.pause()
worker = threading.Thread(target=_timed_sleep, daemon=True)
worker.start()
time.sleep(0.6)
check("the delay has not returned while paused", slept == [], str(slept))
ex.resume()
worker.join(timeout=5)
check("and the whole delay is still owed after resume",
      slept and slept[0] >= 0.6 + 0.3 - 0.05, str(slept))

ex.on_step = None
ex.on_finished = None

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
