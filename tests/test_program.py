"""
Program validation: the mistakes that must be caught before anything moves.

A two-arm program has failure modes a one-arm program does not. Carrying an
object the arms never took hold of, driving one arm while both are gripping
the same drum, or asking two arms to *turn* together without attaching first —
none of these are runtime hiccups, they are the arms tearing at each other.
All of them are visible in the step list, so all of them get rejected on paper.

The other half of this file is the target grammar: four ways to fill a column,
three frames to read an offset in, and the rule that decides which side a
rotation multiplies on. That is arithmetic, so it is checked here rather than
against two robots.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.geometry.kinematics import pose_to_mat
from ur5dual.program.steps import (
    PointLibrary, Program, Step, apply_offset, describe_target, format_offset,
    make_target, resolve_target, target_kind, target_turns,
)

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


def close(a, b, tol=1e-9):
    return bool(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))) < tol)


class FakeConfig:
    """Only the one flag the pair rule reads."""

    def __init__(self, translation_calibrated):
        self.translation_calibrated = translation_calibrated


pts = PointLibrary()
for n in ("home_A", "home_B", "pick", "place"):
    pts.set(n, [0.4, 0.0, 1.0, 0, 0, 0])


def _disabled_label():
    step = Step("LABEL", name="off")
    step.enabled = False
    return step


def prog(*steps):
    p = Program("t")
    for s in steps:
        p.add(s)
    return p


def move(link="solo", **slots):
    return Step("MOVE", link=link, **slots)


# ── the target grammar ────────────────────────────────────────────────────
print("four ways to fill a column")
check("a place", target_kind({"point": "pick"}) == "point")
check("a place, shifted",
      target_kind({"point": "pick", "offset": [0] * 6}) == "point_offset")
check("an offset on its own", target_kind({"offset": [0] * 6}) == "offset")
check("a captured pose", target_kind({"pose": [0] * 6}) == "pose")
check("an empty column has no target", target_kind({}) is None
      and target_kind(None) is None)

print("a place, resolved")
check("a bare place is the place",
      close(resolve_target({"point": "pick"}, pts), pts.get("pick")))
check("a place plus 50 mm up is 50 mm above it",
      close(resolve_target(make_target(point="pick", offset=[0, 0, 0.05, 0, 0, 0]),
                           pts)[:3], [0.4, 0.0, 1.05]))
check("a captured pose needs nothing else",
      close(resolve_target({"pose": [1, 2, 3, 0, 0, 0]}, pts)[:3], [1, 2, 3]))

print("an offset with no place in it starts from where the arm is")
here = [0.5, 0.1, 0.9, 0, 0, 0]
check("world Z-50 drops 50 mm from here",
      close(resolve_target(make_target(offset=[0, 0, -0.05, 0, 0, 0]),
                           pts, current=here)[:3], [0.5, 0.1, 0.85]))
try:
    resolve_target(make_target(offset=[0] * 6), pts)
    check("an offset with nothing to offset from is refused", False)
except ValueError as e:
    check("an offset with nothing to offset from is refused",
          "where the arm is now" in str(e))

print("which frame the six numbers are read in")
turned = pose_to_mat([0, 0, 0, 0, 0, math.pi / 2])      # yawed 90 deg about Z
check("tool X+100 on a frame yawed 90 deg is world Y+100",
      close(apply_offset(turned, [0.1, 0, 0, 0, 0, 0], "tool")[:3, 3],
            [0, 0.1, 0]))
check("world X+100 on the same frame is still world X+100",
      close(apply_offset(turned, [0.1, 0, 0, 0, 0, 0], "world")[:3, 3],
            [0.1, 0, 0]))
check("base X+100 follows that arm's base axes",
      close(apply_offset(np.eye(4), [0.1, 0, 0, 0, 0, 0], "base",
                         base=turned)[:3, 3], [0, 0.1, 0]))
try:
    apply_offset(np.eye(4), [0.1] + [0] * 5, "base")
    check("a base offset without a base matrix is refused", False)
except ValueError as e:
    check("a base offset without a base matrix is refused", "base matrix" in str(e))

print("which side a rotation multiplies on")
at_x = pose_to_mat([1, 0, 0, 0, 0, 0])
spun = apply_offset(at_x, [0, 0, 0, 0, 0, math.pi / 2], "world")
check("a world rotation about its own origin leaves the origin alone",
      close(spun[:3, 3], [1, 0, 0]))
swung = apply_offset(at_x, [0, 0, 0, 0, 0, math.pi / 2], "world",
                     pivot=[0, 0, 0])
check("a pivot swings it round a place in the cell",
      close(swung[:3, 3], [0, 1, 0], tol=1e-12) or close(swung[:3, 3], [0, 1, 0], tol=1e-9))
own = apply_offset(turned, [0, 0, 0, math.pi / 2, 0, 0], "object")
world = apply_offset(turned, [0, 0, 0, math.pi / 2, 0, 0], "world")
check("object RX and world RX are not the same rotation",
      not close(own[:3, :3], world[:3, :3], tol=1e-6))
check("world RX pre-multiplies",
      close(world[:3, :3],
            pose_to_mat([0, 0, 0, math.pi / 2, 0, 0])[:3, :3] @ turned[:3, :3]))
check("object RX post-multiplies",
      close(own[:3, :3],
            turned[:3, :3] @ pose_to_mat([0, 0, 0, math.pi / 2, 0, 0])[:3, :3]))

print("an offset that turns is not an offset that only shifts")
check("a pure shift does not turn", not target_turns(make_target(offset=[0, 0, 0.1, 0, 0, 0])))
check("any rotation counts", target_turns(make_target(offset=[0, 0, 0, 0, 0, 0.01])))
check("an empty column does not turn", not target_turns(None))

# ── what is already on disk ───────────────────────────────────────────────
print("the old spellings still read")
s = Step("MOVE_ARM", arm="A", point="pick", motion="movel")
check("MOVE_ARM becomes a solo line with one column",
      s.kind == "MOVE" and s.link == "solo"
      and s.slot("a") == {"point": "pick", "motion": "movel"}
      and s.slot("b") is None, str(s.fields))
s = Step("MOVE_OBJ", point="place")
check("MOVE_OBJ becomes a coupled line",
      s.kind == "MOVE" and s.link == "coupled"
      and s.slot("obj").get("point") == "place", str(s.fields))
s = Step("ROTATE_OBJ", axis="z", angle_deg=45)
check("ROTATE_OBJ becomes a coupled offset that turns",
      s.link == "coupled" and target_turns(s.slot("obj"))
      and close(s.slot("obj")["offset"][5], math.radians(45)), str(s.fields))
check("a rotation written before world rotation existed still means its own axis",
      s.slot("obj")["frame"] == "object")
s = Step("GRIP", arm="both", output=0, state=True)
check("GRIP becomes OUT — a gripper is one digital output among eight",
      s.kind == "OUT" and s.fields == {"arm": "both", "output": 0,
                                       "state": True}, str(s.fields))
s = Step("GRIP", arm="A", output=1, state=False)
check("and keeps which arm it was for",
      s.kind == "OUT" and s.get("arm") == "A" and s.get("output") == 1
      and s.get("state") is False, str(s.fields))
s = Step("GRIP", a={"output": 2, "state": True}, b={"output": 2, "state": True})
check("a two-column GRIP from a saved file becomes OUT on both",
      s.kind == "OUT" and s.get("arm") == "both" and s.get("output") == 2,
      str(s.fields))

print("the program on disk still loads and still passes")
stored = Program.load(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "config", "programs",
    "six_axis_dry_check.json"))
check("every step migrated", all(s.kind in ("MOVE", "ATTACH", "DETACH")
                                 for s in stored.steps), str(len(stored.steps)))
check("it still validates", stored.validate(pts) == [], str(stored.validate(pts)))

# ── a well-formed program ─────────────────────────────────────────────────
print("a well-formed pick with two arms")
good = prog(
    move("together", a=make_target(point="home_A", motion="movej"),
         b=make_target(point="home_B", motion="movej")),
    move("together", a=make_target(point="pick", offset=[0, 0, 0.05, 0, 0, 0],
                                   motion="movel"),
         b=make_target(point="pick", offset=[0, 0, 0.05, 0, 0, 0], motion="movel")),
    move("together", a=make_target(offset=[0, 0, -0.05, 0, 0, 0], motion="movel"),
         b=make_target(offset=[0, 0, -0.05, 0, 0, 0], motion="movel")),
    Step("OUT", arm="both", output=0, state=True),
    move("pair", pair=make_target(offset=[0, 0, 0.10, 0, 0, 0], speed=0.05)),
    Step("ATTACH", object="bottle", origin="midpoint"),
    move("coupled", obj=make_target(point="place")),
    move("coupled", obj=make_target(offset=[0, 0, 0, 0, 0, math.radians(45)],
                                    frame="object")),
    Step("DETACH"),
    Step("WHERE"),
    move("solo", a=make_target(point="home_A", motion="movej")),
)
problems, warnings = good.check(pts, FakeConfig(True))
check("passes validation", problems == [], str(problems))
check("and warns about nothing", warnings == [], str(warnings))

# ── the rules ─────────────────────────────────────────────────────────────
print("coupling")
p = prog(move("coupled", obj=make_target(point="place")))
check("a coupled move without ATTACH is rejected",
      any("needs an ATTACH" in m for m in p.validate(pts)), str(p.validate(pts)))
p = prog(Step("ATTACH", object="drum"),
         move("solo", a=make_target(point="place")),
         Step("DETACH"))
check("a solo move inside a grasp is rejected",
      any("tear it out" in m for m in p.validate(pts)), str(p.validate(pts)))
p = prog(Step("ATTACH"),
         move("together", a=make_target(point="pick"), b=make_target(point="pick")),
         Step("DETACH"))
check("'together' inside a grasp is rejected",
      any("tear it out" in m for m in p.validate(pts)))
p = prog(Step("ATTACH"),
         move("pair", pair=make_target(offset=[0, 0, 0.05, 0, 0, 0])),
         Step("DETACH"))
check("'pair' inside a grasp is rejected — that is what coupled is for",
      any("tear it out" in m for m in p.validate(pts, FakeConfig(True))))

print("unbalanced attach/detach")
p = prog(Step("ATTACH", object="drum"), move("coupled", obj=make_target(point="place")))
check("ending while still holding is rejected",
      any("still holding" in m for m in p.validate(pts)))
p = prog(Step("ATTACH"), Step("ATTACH"), Step("DETACH"))
check("double ATTACH is rejected",
      any("already holding" in m for m in p.validate(pts)))
p = prog(Step("DETACH"))
check("DETACH with nothing held is rejected",
      any("nothing is held" in m for m in p.validate(pts)))

print("how many arms a link speaks for")
p = prog(move("solo", a=make_target(point="pick"), b=make_target(point="place")))
check("a solo move with both columns filled is rejected",
      any("drives one arm" in m for m in p.validate(pts)), str(p.validate(pts)))
p = prog(move("together", a=make_target(point="pick")))
check("'together' with one column is rejected",
      any("both arms" in m for m in p.validate(pts)))
p = prog(move("solo"))
check("a move with no column at all is rejected",
      any("no arm to move" in m for m in p.validate(pts)))
p = prog(Step("MOVE_ARM", arm="C", point="pick"))
check("bad arm id is rejected",
      any("needs arm A or B" in m for m in p.validate(pts)), str(p.validate(pts)))
p = prog(Step("ATTACH"),
         move("coupled", obj=make_target(point="place"), a=make_target(point="pick")),
         Step("DETACH"))
check("a coupled move with an arm column is rejected",
      any("leave the arm columns empty" in m for m in p.validate(pts)))

print("what a pair move may and may not ask for")
def pair_problems(target, config=FakeConfig(True)):
    return prog(move("pair", pair=target)).validate(pts, config)

check("a straight world shift is allowed",
      pair_problems(make_target(offset=[0, 0, 0.05, 0, 0, 0])) == [],
      str(pair_problems(make_target(offset=[0, 0, 0.05, 0, 0, 0]))))
check("any rotation is rejected",
      any("cannot rotate" in m for m in
          pair_problems(make_target(offset=[0, 0, 0, 0, 0, 0.1]))))
check("a tool-frame offset is rejected",
      any("world frame only" in m for m in
          pair_problems(make_target(offset=[0.05, 0, 0, 0, 0, 0], frame="tool"))))
check("a named place is rejected — two arms cannot stand on one point",
      any("not a place" in m for m in pair_problems(make_target(point="pick"))))
check("an empty pair column is rejected",
      any("needs a world offset" in m for m in pair_problems({})))
check("an uncalibrated cell is rejected",
      any("check_directions_online" in m for m in
          pair_problems(make_target(offset=[0, 0, 0.05, 0, 0, 0]), FakeConfig(False))),
      str(pair_problems(make_target(offset=[0, 0, 0.05, 0, 0, 0]), FakeConfig(False))))
check("with no config the calibration gate is skipped rather than guessed",
      pair_problems(make_target(offset=[0, 0, 0.05, 0, 0, 0]), None) == [])

print("what a coupled move may ask for")
p = prog(Step("ATTACH"),
         move("coupled", obj=make_target(offset=[0.05, 0, 0, 0, 0, 0], frame="tool")),
         Step("DETACH"))
check("a tool-frame object offset is rejected",
      any("world frame or the object" in m for m in p.validate(pts)),
      str(p.validate(pts)))
p = prog(Step("ATTACH"), move("coupled"), Step("DETACH"))
check("an empty object column is rejected",
      any("a place or an offset" in m for m in p.validate(pts)))

print("point names and motions")
p = prog(move("solo", a=make_target(point="nowhere")))
check("unknown point is rejected",
      any("no point named" in m for m in p.validate(pts)))
p = prog(move("solo", a=make_target(point="pick", motion="movec")))
check("an unknown motion is rejected",
      any("is not a motion" in m for m in p.validate(pts)))

print("an offset from a pose nothing in the program set")
p = prog(move("solo", a=make_target(offset=[0, 0, -0.05, 0, 0, 0])))
problems, warnings = p.check(pts)
check("warns rather than refuses", problems == [] and len(warnings) == 1,
      str(problems) + str(warnings))
check("and says which arm", "arm A" in warnings[0], warnings[0])
p = prog(move("solo", a=make_target(point="pick")),
         move("solo", a=make_target(offset=[0, 0, -0.05, 0, 0, 0])))
check("no warning once the arm has been sent somewhere",
      p.check(pts)[1] == [], str(p.check(pts)[1]))
p = prog(move("solo", a=make_target(offset=[0, 0, -0.05, 0, 0, 0])),
         move("solo", a=make_target(offset=[0, 0, -0.05, 0, 0, 0])))
check("and only the first time", len(p.check(pts)[1]) == 1)

# ── calling another program ───────────────────────────────────────────────
print("a CALL is flattened before anything runs")
library = {
    "approach": prog(move("solo", a=make_target(point="pick", motion="movej")),
                     Step("BARRIER")),
    "grip": prog(Step("OUT", arm="both", output=0, state=True)),
}
library["pick_cycle"] = prog(Step("CALL", program="approach"),
                             Step("CALL", program="grip"))


def load(name):
    if name not in library:
        raise KeyError("no program named %r" % name)
    return library[name]


p = prog(Step("CALL", program="approach", repeat=2), Step("DETACH"))
plan = p.expand(load)
check("the called program's steps take its caller's place",
      [s.kind for s, _row in plan] == ["MOVE", "BARRIER", "MOVE", "BARRIER",
                                       "DETACH"],
      str([s.kind for s, _r in plan]))
check("repeat runs it that many times",
      sum(1 for s, _r in plan if s.kind == "BARRIER") == 2)
check("every called step points back at the CALL's own row",
      [row for _s, row in plan] == [0, 0, 0, 0, 1],
      str([row for _s, row in plan]))

check("a call inside a call is expanded too",
      [s.kind for s, _r in prog(Step("CALL", program="pick_cycle")).expand(load)]
      == ["MOVE", "BARRIER", "OUT"])

print("what a CALL is not allowed to do")
library["loop_a"] = prog(Step("CALL", program="loop_b"))
library["loop_b"] = prog(Step("CALL", program="loop_a"))
try:
    prog(Step("CALL", program="loop_a")).expand(load)
    check("a call that comes back to itself is refused", False, "it expanded")
except ValueError as e:
    check("a call that comes back to itself is refused", "calls itself" in str(e),
          str(e))

library["deep"] = prog(Step("CALL", program="deep2"))
for i in range(2, 9):
    library["deep%d" % i] = prog(Step("CALL", program="deep%d" % (i + 1)))
try:
    prog(Step("CALL", program="deep")).expand(load)
    check("a call nested too deep is refused", False, "it expanded")
except ValueError as e:
    check("a call nested too deep is refused", "nested deeper" in str(e), str(e))

try:
    prog(Step("CALL", program="nowhere")).expand(load)
    check("a call to a program that is not there is refused", False)
except ValueError as e:
    check("a call to a program that is not there is refused",
          "cannot load" in str(e), str(e))

check("a CALL with no program named is a problem on paper",
      any("names no program" in m
          for m in prog(Step("CALL", program="")).validate(pts)))

disabled = Step("CALL", program="approach")
disabled.enabled = False
check("a disabled CALL drags nothing in",
      prog(disabled, Step("DETACH")).expand(load) and
      [s.kind for s, _r in prog(disabled, Step("DETACH")).expand(load)]
      == ["DETACH"])

print("validation can start from a cell that is already holding something")
p = prog(move("coupled", obj=make_target(point="place")), Step("DETACH"))
check("a coupled line alone is a problem from a free cell",
      any("needs an ATTACH" in m for m in p.validate(pts)))
check("and not from a cell that is already holding",
      p.validate(pts, None, holding=True) == [],
      str(p.validate(pts, None, holding=True)))

print("disabled steps are ignored")
s = move("coupled", obj=make_target(point="place"))
s.enabled = False
check("a disabled bad step does not fail validation", prog(s).validate(pts) == [])

print("editing")
p = prog(Step("DELAY", seconds=1), Step("BARRIER"), Step("DETACH"))
p.move(0, 1)
check("move swaps rows", p.steps[0].kind == "BARRIER")
p.remove(2)
check("remove drops a row", len(p.steps) == 2)

print("save and load")
import json
import tempfile

path = os.path.join(tempfile.mkdtemp(), "p.json")
good.save(path)
back = Program.load(path)
check("round trip keeps every step", len(back.steps) == len(good.steps))
check("round trip keeps the columns",
      back.steps[0].slot("a") == good.steps[0].slot("a")
      and back.steps[4].slot("pair") == good.steps[4].slot("pair"))
check("round trip keeps the links",
      [s.link for s in back.steps] == [s.link for s in good.steps])
check("reloaded program still validates",
      back.validate(pts, FakeConfig(True)) == [],
      str(back.validate(pts, FakeConfig(True))))
check("what is written is JSON, not numpy",
      isinstance(json.load(open(path))["steps"][4]["pair"]["offset"][2], float))

print("a line reads as two columns")
span, a, b, link = good.steps[0].render()
check("an ordinary move has no span and two cells",
      span is None and a == "home_A  movej" and b == "home_B  movej",
      "%r %r %r" % (span, a, b))
check("and names its link", link == "⇉ together", link)
span, a, b, link = good.steps[4].render()
check("a pair move spans both columns", a is None and b is None
      and span == "world Z+100.0", str(span))
span, a, b, link = good.steps[7].render()
check("a coupled turn spans both columns and says which frame",
      span == "⟨obj⟩ object RZ+45.0", str(span))
span, a, b, link = good.steps[3].render()
check("an output line spans both columns — it is one instruction, not two",
      span == "both arms  DO0 ON" and a is None and b is None,
      "%r %r %r" % (span, a, b))
check("an empty column is a dot",
      good.steps[10].render()[2] == "·", str(good.steps[10].render()))

print("offsets read in the units they were typed in")
check("millimetres and degrees",
      format_offset([0, 0, 0.05, 0, 0, math.radians(45)]) == "Z+50.0 RZ+45.0",
      format_offset([0, 0, 0.05, 0, 0, math.radians(45)]))
check("an offset of nothing says so", format_offset([0] * 6) == "0")
check("a place plus a shift reads as both",
      describe_target(make_target(point="pick", offset=[0, 0, 0.05, 0, 0, 0]))
      == "pick + world Z+50.0",
      describe_target(make_target(point="pick", offset=[0, 0, 0.05, 0, 0, 0])))

print("descriptions read like instructions")
for s, want in ((good.steps[7], "⟨obj⟩ object RZ+45.0"),
                (good.steps[5], "take hold of 'bottle' (origin midpoint)"),
                (good.steps[10], "arm A  home_A  movej"),
                (good.steps[9], "read where both arms are")):
    check("%s reads well" % s.kind, s.describe() == want, s.describe())

print("outputs and inputs")
close_ = Step("OUT", arm="both", output=0, state=True)
open_ = Step("OUT", arm="both", output=0, state=False)
check("an output can be driven on", close_.describe().endswith("ON"),
      close_.describe())
check("and off", open_.describe().endswith("OFF"), open_.describe())
check("an output line speaks to both controllers or one",
      Step("OUT", arm="A", output=1, state=True).describe().startswith("arm A"))
check("a bad output number is refused",
      any("must be 0 to 7" in m
          for m in prog(Step("OUT", arm="A", output=9, state=True)).validate(pts)))
check("waiting watches one arm's inputs",
      any("pick A or B" in m
          for m in prog(Step("WAIT_IN", arm="both", input=0,
                             state=True)).validate(pts)))
check("and a well-formed wait passes",
      prog(Step("WAIT_IN", arm="B", input=3, state=False,
                timeout=5)).validate(pts) == [])

print("labels, jumps and conditions")
loop = prog(Step("LABEL", name="top"),
            Step("SET_VAR", name="n", op="+=", value=1),
            Step("IF", source="var", name="n", compare="<", value=3,
                 target="top"),
            Step("JUMP", target="done"),
            Step("LABEL", name="done"))
check("a counted loop passes on paper", loop.validate(pts) == [],
      str(loop.validate(pts)))
check("a jump to a label that is not there is refused",
      any("no label called" in m
          for m in prog(Step("JUMP", target="nowhere")).validate(pts)))
check("two labels with one name is refused — a jump would have two places to go",
      any("already a label" in m
          for m in prog(Step("LABEL", name="x"),
                        Step("LABEL", name="x")).validate(pts)))
check("a label with no name is refused",
      any("has no name" in m for m in prog(Step("LABEL", name="")).validate(pts)))
check("an IF with no landing place is refused",
      any("nowhere to jump" in m
          for m in prog(Step("IF", source="var", name="n", compare="==",
                             value=1, target="")).validate(pts)))
check("an IF on an input still needs a real input number",
      any("must be 0 to 7" in m
          for m in prog(Step("LABEL", name="go"),
                        Step("IF", source="input", arm="A", input=12,
                             state=True, target="go")).validate(pts)))
check("the else branch has to exist too",
      any("no label called 'missing'" in m
          for m in prog(Step("LABEL", name="go"),
                        Step("IF", source="var", name="n", compare="==",
                             value=1, target="go",
                             otherwise="missing")).validate(pts)))
check("a disabled label is not a place to land",
      any("no label called" in m for m in prog(_disabled_label(),
                                               Step("JUMP", target="off")).validate(pts)))

print("a called program may not carry labels")
library["with_label"] = prog(Step("LABEL", name="inner"), Step("BARRIER"))
try:
    prog(Step("CALL", program="with_label")).expand(load)
    check("expanding it is refused", False, "it expanded")
except ValueError as e:
    check("expanding it is refused", "contains a label" in str(e), str(e))

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
