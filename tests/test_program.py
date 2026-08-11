"""
Program validation: the mistakes that must be caught before anything moves.

A two-arm program has failure modes a one-arm program does not. Carrying an
object the arms never took hold of, or driving one arm while both are gripping
the same drum, are not runtime hiccups — they are the arms tearing at each
other. All of them are visible in the step list, so all of them get rejected
on paper.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.program.steps import PointLibrary, Program, Step

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


pts = PointLibrary()
for n in ("home_A", "home_B", "pick", "place"):
    pts.set(n, [0.4, 0.0, 1.0, 0, 0, 0])


def prog(*steps):
    p = Program("t")
    for s in steps:
        p.add(s)
    return p


print("a well-formed pick with two arms")
good = prog(
    Step("MOVE_ARM", arm="A", point="home_A", motion="movej"),
    Step("MOVE_ARM", arm="B", point="home_B", motion="movej"),
    Step("BARRIER"),
    Step("GRIP", arm="both", output=0, state=True),
    Step("ATTACH", object="bottle", origin="midpoint"),
    Step("MOVE_OBJ", point="place"),
    Step("ROTATE_OBJ", axis="z", angle_deg=45),
    Step("DETACH"),
    Step("GRIP", arm="both", output=0, state=False),
)
check("passes validation", good.validate(pts) == [], str(good.validate(pts)))

print("carrying with nothing held")
p = prog(Step("MOVE_OBJ", point="place"))
check("MOVE_OBJ without ATTACH is rejected",
      any("needs an ATTACH" in m for m in p.validate(pts)), str(p.validate(pts)))

print("moving one arm while both hold the object")
p = prog(Step("ATTACH", object="drum"),
         Step("MOVE_ARM", arm="A", point="place"),
         Step("DETACH"))
msgs = p.validate(pts)
check("single-arm move inside a grasp is rejected",
      any("tear it out" in m for m in msgs), str(msgs))

print("unbalanced attach/detach")
p = prog(Step("ATTACH", object="drum"), Step("MOVE_OBJ", point="place"))
check("ending while still holding is rejected",
      any("still holding" in m for m in p.validate(pts)))
p = prog(Step("ATTACH"), Step("ATTACH"), Step("DETACH"))
check("double ATTACH is rejected",
      any("already holding" in m for m in p.validate(pts)))
p = prog(Step("DETACH"))
check("DETACH with nothing held is rejected",
      any("nothing is held" in m for m in p.validate(pts)))

print("point names")
p = prog(Step("MOVE_ARM", arm="A", point="nowhere"))
check("unknown point is rejected",
      any("no point named" in m for m in p.validate(pts)))
p = prog(Step("MOVE_ARM", arm="C", point="pick"))
check("bad arm id is rejected",
      any("needs arm A or B" in m for m in p.validate(pts)))

print("disabled steps are ignored")
s = Step("MOVE_OBJ", point="place")
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
check("round trip keeps the fields",
      back.steps[4].get("object") == "bottle"
      and back.steps[6].get("angle_deg") == 45)
check("reloaded program still validates", back.validate(pts) == [])

print("descriptions read like instructions")
for s, want in ((good.steps[6], "spin object +45.0 deg about its own z"),
                (good.steps[4], "take hold of 'bottle' (origin midpoint)"),
                (good.steps[0], "arm A  movej -> home_A")):
    check("%s reads well" % s.kind, s.describe() == want, s.describe())

print("gripper close and open are both program steps")
close = Step("GRIP", arm="both", output=0, state=True)
open_ = Step("GRIP", arm="both", output=0, state=False)
check("close drives the output on", close.get("state") is True
      and close.describe().endswith("ON"))
check("open drives the output off", open_.get("state") is False
      and open_.describe().endswith("OFF"))

print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
