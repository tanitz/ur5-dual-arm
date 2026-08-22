"""
The program tab, driven the way a finger drives it.

test_program.py checks the grammar and test_executor.py checks what a line
does to the arms. This checks the third thing, which is the one an operator
actually touches: that pressing these buttons in this order writes the line
the operator meant.

The panel records rather than asks. `● MOVEJ` and `↗ MOVEL` take where the
arms already are, because on a pendant the arm is the input device; the insert
list covers the lines that have nothing to record, and the editor dialog is
where anything a recorded step cannot say gets said.
"""

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                        # noqa: E402
from PyQt5.QtWidgets import QApplication                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ur5dual.cell import Cell                             # noqa: E402
from ur5dual.config import CellConfig                     # noqa: E402
from ur5dual.gui.app import MainWindow                    # noqa: E402
from ur5dual.gui.panels.step_edit import StepEditDialog   # noqa: E402
from ur5dual.program.steps import (                       # noqa: E402
    Program, Step, make_target,
)

fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + detail if detail else ""))
    if not ok:
        fail += 1


app = QApplication.instance() or QApplication([])
cfg = CellConfig.load()
cfg.arms["B"].enabled = True
cfg.calibrated = False
cfg.translation_calibrated = True
window = MainWindow(cell=Cell(cfg, simulated=True), connect_on_start=False)
window.resize(1280, 800)
window.show()
app.processEvents()
window.cell.sim_ready_pose()

panel = window.panels["program"]
window.points.set("pick", [0.4, 0.0, 1.0, 0, 0, 0])
window.points.set("place", [0.3, 0.0, 1.0, 0, 0, 0])
panel.refresh()
panel.tick()
app.processEvents()


def edit(step):
    return StepEditDialog(step, window.points, cell=window.cell,
                          held_object=window.executor.object)


def _legacy_still_runs():
    """Nothing was deleted, only unlisted: a program written before the grasp
    left the panel must still open, read and validate."""
    saved = Program("legacy")
    saved.add(Step("ATTACH", object="box", origin="midpoint"))
    saved.add(Step("MOVE_OBJ", point="place"))
    saved.add(Step("ROTATE_OBJ", axis="z", angle_deg=45))
    saved.add(Step("DETACH"))
    if saved.validate(window.points) != []:
        return False
    dialog = edit(saved.steps[1])
    return (dialog.link_combo.currentText() == "coupled"
            and dialog.get_step().link == "coupled")


print("the table is two columns of targets, and it has the room for them")
check("five columns", panel.table.columnCount() == 5)
check("headed by the two arms and the link",
      [panel.table.horizontalHeaderItem(i).text() for i in range(5)]
      == ["#", "step", "Arm A", "Arm B", "link"])
window.rail.buttons["jog"].click()          # close the sidebar
app.processEvents()
arm_w = (panel.table.width() - panel.table.columnWidth(0)
         - panel.table.columnWidth(1) - panel.table.columnWidth(4)) / 2
check("each arm gets a real column with the sidebar shut", arm_w > 400,
      "%d px" % arm_w)

print("\nwhere the arms are, above the program being written")
panel.tick()
lines = panel.now_lbl.text().split("\n")
check("a header names all six numbers, not just the position",
      lines[0].split()[:6] == ["x", "y", "z", "rx", "ry", "rz"], lines[0])
check("and carries the gap between the arms", "gap" in lines[0], lines[0])
check("one line per arm", lines[1].startswith("A") and lines[2].startswith("B"),
      str(lines[1:3]))
check("each with six numbers under the header",
      len(lines[1].split()) == 7 and len(lines[2].split()) == 7,
      str(len(lines[1].split())))


def _column(line, index):
    """The nth number on a readout line, by the header's own spacing."""
    return float(line.split()[1 + index])


live = window.cell.arms["A"].tcp_pose_world()
check("the position is millimetres",
      abs(_column(lines[1], 0) - live[0] * 1000) < 0.05,
      "%s vs %.1f" % (_column(lines[1], 0), live[0] * 1000))
check("and the rotation is degrees, not radians",
      abs(_column(lines[1], 3) - np.degrees(live[3])) < 0.05,
      "%s vs %.1f" % (_column(lines[1], 3), np.degrees(live[3])))
check("the columns line up under the header",
      lines[1].index("-") > 0 and len(lines[1]) == len(lines[2]),
      "%d vs %d" % (len(lines[1]), len(lines[2])))

print("\nrecording takes where the arms already are")
panel._select_column("mirror")
panel._record("movej")
step = panel.program.steps[-1]
check("⧉ A+B records one line with both columns",
      step.slot("a") is not None and step.slot("b") is not None, str(step.fields))
check("and the line is together, because two arms were recorded",
      step.link == "together", step.link)
check("each column carries a pose, not a name",
      step.slot("a").get("pose") is not None
      and step.slot("a").get("point") is None, str(step.slot("a")))
check("and it is where that arm actually is",
      np.allclose(step.slot("a")["pose"], window.cell.arms["A"].tcp_pose_world(),
                  atol=1e-9))
check("MOVEJ records a movej", step.slot("a")["motion"] == "movej")

panel._select_column("b")
panel._record("movel")
step = panel.program.steps[-1]
check("one arm records a solo line", step.link == "solo", step.link)
check("into the column that was chosen",
      step.slot("b") is not None and step.slot("a") is None, str(step.fields))
check("MOVEL records a movel", step.slot("b")["motion"] == "movel")

panel.speed.setValue(80)
panel._select_column("a")
panel._record("movel")
check("the speed box is stamped onto what is recorded",
      abs(panel.program.steps[-1].slot("a")["speed"] - 0.080) < 1e-9,
      str(panel.program.steps[-1].slot("a")))

print("\nthe steps with nothing to record go straight in")
before = len(panel.program.steps)
offered = [panel.insert_combo.itemText(i)
           for i in range(panel.insert_combo.count())]
panel.insert_combo.setCurrentText("IN POSE")
check("IN POSE is on the list at all",
      panel.insert_combo.currentText() == "IN POSE",
      panel.insert_combo.currentText())
panel._insert()
check("IN POSE needs no dialog, and is still a BARRIER underneath — the "
      "rename is what the operator reads, not what the file holds",
      [s.kind for s in panel.program.steps[before:]] == ["BARRIER"],
      str([s.kind for s in panel.program.steps[before:]]))

print("\nthe two-arm grasp is not on the list any more")
for gone in ("ATTACH", "DETACH", "carry object", "WHERE", "BARRIER"):
    check("%s cannot be inserted" % gone, gone not in offered, str(offered))
check("but SYNC OFFSET still is — it carries a workpiece without one",
      "SYNC OFFSET" in offered, str(offered))
check("and a WHERE that a saved program already holds still renders",
      Step("WHERE").describe() == "read where both arms are")
check("and a saved program that uses them still loads and still validates",
      _legacy_still_runs(), "see below")

print("\nthe editor shows exactly the columns the link speaks for")
dialog = edit(Step("MOVE", link="together",
                   a=make_target(point="pick", motion="movej"),
                   b=make_target(point="pick", motion="movej")))
check("two arm columns for an arm move", sorted(dialog.columns) == ["a", "b"])
dialog.link_combo.setCurrentText("SYNC OFFSET")
app.processEvents()
check("one column for a SYNC OFFSET", sorted(dialog.columns) == ["pair"])
check("and the step it writes still says pair", dialog.get_step().link == "pair")
check("and no rotation to type into it",
      # isVisibleTo, not isVisible: the dialog was never shown, so every child
      # reports hidden and the check would pass without testing anything
      not any(dialog.columns["pair"].offset_rows[a].isVisibleTo(dialog)
              for a in ("RX", "RY", "RZ")))
check("world frame only",
      [dialog.columns["pair"].frame_combo.itemText(i)
       for i in range(dialog.columns["pair"].frame_combo.count())] == ["world"])
check("coupled is not offered for a new line — the ATTACH it needs cannot be "
      "inserted any more",
      [dialog.link_combo.itemText(i)
       for i in range(dialog.link_combo.count())] == ["arms", "SYNC OFFSET"],
      str([dialog.link_combo.itemText(i)
           for i in range(dialog.link_combo.count())]))

# ...but a line that already is coupled opens as one, rather than silently
# turning into an arm move the moment it is looked at
old_line = edit(Step("MOVE", link="coupled",
                     obj=make_target(offset=[0, 0, 0.05, 0, 0, 0],
                                     frame="object")))
check("a saved coupled line opens as coupled",
      old_line.link_combo.currentText() == "coupled"
      and sorted(old_line.columns) == ["obj"], str(sorted(old_line.columns)))
check("its object column may rotate",
      all(old_line.columns["obj"].offset_rows[a].isVisibleTo(old_line)
          for a in ("RX", "RY", "RZ")))
check("in the world frame or its own",
      [old_line.columns["obj"].frame_combo.itemText(i)
       for i in range(old_line.columns["obj"].frame_combo.count())]
      == ["world", "object"])

print("\nthe link follows from how many columns were filled")
dialog = edit(Step("MOVE", link="solo"))
dialog.columns["a"].target_combo.setCurrentText("place")
dialog.columns["a"]._target_changed()
dialog.columns["a"].point_combo.setCurrentText("pick")
step = dialog.get_step()
check("one column is one arm on its own", step.link == "solo", step.link)
check("and the other column is left out", step.slot("b") is None)
dialog.columns["b"].target_combo.setCurrentText("place")
dialog.columns["b"]._target_changed()
dialog.columns["b"].point_combo.setCurrentText("place")
step = dialog.get_step()
check("filling the second makes it one gesture", step.link == "together",
      step.link)
check("with each arm's own place",
      step.slot("a")["point"] == "pick" and step.slot("b")["point"] == "place",
      str(step.fields))

print("\noffsets, in the dialog")
dialog = edit(Step("MOVE", link="solo"))
column = dialog.columns["a"]
column.target_combo.setCurrentText("place + offset")
column._target_changed()
column.point_combo.setCurrentText("pick")
column.frame_combo.setCurrentText("tool")
column.offset_spins["Z"].setValue(50.0)
column.offset_spins["RZ"].setValue(45.0)
step = dialog.get_step()
target = step.slot("a")
check("millimetres go in as metres", abs(target["offset"][2] - 0.050) < 1e-9,
      str(target["offset"]))
check("degrees go in as radians",
      abs(target["offset"][5] - np.radians(45.0)) < 1e-9)
check("and the frame is what was chosen", target["frame"] == "tool")

print("\nHere writes the live pose into a column")
dialog = edit(Step("MOVE", link="solo"))
column = dialog.columns["a"]
column.target_combo.setCurrentText("here")
column._target_changed()
check("the Here button is offered", column.here_btn.isVisibleTo(dialog))
column.here_btn.click()
step = dialog.get_step()
check("the column carries the pose arm A is at",
      np.allclose(step.slot("a")["pose"],
                  window.cell.arms["A"].tcp_pose_world(), atol=1e-9),
      str(np.round(np.array(step.slot("a")["pose"])[:3] * 1000, 1)))

print("\nediting an existing line reads it back into the dialog")
original = Step("MOVE", link="coupled",
                obj=make_target(offset=[0, 0, 0.05, 0, 0, np.radians(30)],
                                frame="object"))
dialog = edit(original)
column = dialog.columns["obj"]
check("the offset comes back in the units it was typed in",
      abs(column.offset_spins["Z"].value() - 50.0) < 1e-6
      and abs(column.offset_spins["RZ"].value() - 30.0) < 1e-6,
      "%s %s" % (column.offset_spins["Z"].value(),
                 column.offset_spins["RZ"].value()))
check("and so does the frame", column.frame_combo.currentText() == "object")
check("round tripping it changes nothing",
      dialog.get_step().describe() == original.describe(),
      dialog.get_step().describe())

print("\nduplicating a step")
panel.program.steps.clear()
panel._select_column("a")
panel._record("movej")
panel.table.selectRow(0)
panel._duplicate()
check("the copy lands below the original", len(panel.program.steps) == 2)
check("and is a copy, not the same object",
      panel.program.steps[0] is not panel.program.steps[1]
      and panel.program.steps[0].describe() == panel.program.steps[1].describe())
panel.program.steps[1].fields["link"] = "together"
check("editing the copy leaves the original alone",
      panel.program.steps[0].link == "solo")

print("\nwhat the table shows")
panel.program.steps.clear()
panel._select_column("mirror")
panel._record("movej")
panel.insert_combo.setCurrentText("SYNC OFFSET")
panel.program.add(Step("MOVE", link="pair",
                       pair=make_target(offset=[0, 0, 0.1, 0, 0, 0],
                                        frame="world")))
panel.program.add(Step("OUT", arm="both", output=0, state=True))
panel.refresh()
check("an arm line fills both arm cells",
      panel.table.item(0, 2).text().startswith("here")
      and panel.table.item(0, 3).text().startswith("here"),
      "%r %r" % (panel.table.item(0, 2).text(), panel.table.item(0, 3).text()))
check("and names its link", panel.table.item(0, 4).text() == "⇉ together")
check("a pair line spans both arm columns",
      panel.table.columnSpan(1, 2) == 2
      and panel.table.item(1, 2).text() == "world Z+100.0",
      panel.table.item(1, 2).text())
check("the arms are tinted apart",
      panel.table.item(0, 2).background().color().name()
      != panel.table.item(0, 3).background().color().name())
check("the kind is coloured so the list can be scanned",
      panel.table.item(0, 1).foreground().color().name() == "#1a5fa8",
      panel.table.item(0, 1).foreground().color().name())
check("a pair line is named SYNC in the link column",
      panel.table.item(1, 4).text() == "⇉⇉ SYNC", panel.table.item(1, 4).text())

# the rename is a label, so it has to appear where the operator reads and
# nowhere the file is written
panel.program.steps.clear()
panel.program.add(Step("BARRIER"))
panel.refresh()
check("a BARRIER reads as IN POSE in the table",
      panel.table.item(0, 1).text() == "IN POSE", panel.table.item(0, 1).text())
check("while the step it saves is still a BARRIER",
      panel.program.steps[0].to_dict()["kind"] == "BARRIER")

print("\nthe arm pickers name the arms, and still store the ids")
for kind, key in (("OUT", "output"), ("WAIT_IN", "input")):
    dialog = edit(Step(kind, arm="B", **{key: 0, "state": True}))
    shown = [dialog.arm_combo.itemText(i)
             for i in range(dialog.arm_combo.count())]
    check("%s offers named arms" % kind, shown[:2] == ["Arm A", "Arm B"],
          str(shown))
    check("%s opens on the arm the step names" % kind,
          dialog.arm_combo.currentText() == "Arm B", dialog.arm_combo.currentText())
    check("%s writes back the id, not the label" % kind,
          dialog.get_step().get("arm") == "B", dialog.get_step().get("arm"))
dialog = edit(Step("OUT", arm="both", output=0, state=True))
check("OUT keeps 'both' as a choice, under a readable name",
      dialog.arm_combo.currentText() == "Both arms"
      and dialog.get_step().get("arm") == "both",
      dialog.arm_combo.currentText())

print("\nproblems and warnings both reach the operator")
panel.program.steps.clear()
panel.program.add(Step("MOVE", link="coupled", obj=make_target(point="place")))
panel.refresh()
check("a coupled line with no ATTACH is a problem",
      "ATTACH" in panel.problems.text(), panel.problems.text())
panel.program.steps.clear()
panel.program.add(Step("MOVE", link="solo",
                       a=make_target(offset=[0, 0, -0.01, 0, 0, 0],
                                     frame="world", motion="movel")))
panel.refresh()
check("an offset from an unknown start is a warning, not a problem",
      panel.problems.text() == ""
      and "wherever it was left" in panel.warnings.text(),
      "%r / %r" % (panel.problems.text(), panel.warnings.text()))

print("\nthe function grid is laid out on one unit")
buttons = {"Run": panel.run_btn, "Pause": panel.pause_btn,
           "Stop": panel.stop_btn, "MOVEJ": panel.movej_btn,
           "MOVEL": panel.movel_btn, "A": panel.column_btns["a"],
           "B": panel.column_btns["b"], "A+B": panel.column_btns["mirror"],
           "insert": panel.insert_combo, "speed": panel.speed,
           "＋": panel.add_btn}


def box(widget):
    top_left = widget.mapTo(panel, widget.rect().topLeft())
    return top_left.x(), top_left.y(), widget.width(), widget.height()


check("every control is the same height", 
      len({box(w)[3] for w in buttons.values()}) == 1,
      str(sorted({box(w)[3] for w in buttons.values()})))
check("and none of them is a small target",
      min(box(w)[2] for w in buttons.values()) >= 90,
      "%d px is the narrowest" % min(box(w)[2] for w in buttons.values()))

program_col = [panel.run_btn, panel.pause_btn, panel.stop_btn]
check("Run, Pause and Stop stand in one column",
      len({box(w)[0] for w in program_col}) == 1
      and len({box(w)[2] for w in program_col}) == 1,
      str([box(w)[:1] + box(w)[2:3] for w in program_col]))
check("on four different lines", len({box(w)[1] for w in program_col}) == 3,
      str(sorted({box(w)[1] for w in program_col})))

a, b, both = box(panel.column_btns["a"]), box(panel.column_btns["b"]), \
    box(panel.column_btns["mirror"])
check("⧉ A+B starts where A does", both[0] == a[0], "%d vs %d" % (both[0], a[0]))
check("and reaches the end of B — the layout says what the button means",
      abs((both[0] + both[2]) - (b[0] + b[2])) <= 2,
      "%d vs %d" % (both[0] + both[2], b[0] + b[2]))
check("with A and B on the line below it", a[1] > both[1] and a[1] == b[1])

check("insert has the room to show a step kind, ＋ does not need it",
      box(panel.insert_combo)[2] > box(panel.run_btn)[2] * 2)

print("\nwhat is no longer on the panel")
check("Dry run is gone", not hasattr(panel, "dry_btn"))
check("On/off is gone", not hasattr(panel, "_toggle"))
check("and Teach obj went with the grasp",
      sorted(window.panels["points"].teach_btns) == ["A", "B"],
      str(sorted(window.panels["points"].teach_btns)))
panel.program.steps.clear()
panel._select_column("a")
panel._record("movej")
panel.program.steps[0].enabled = False
panel.refresh()
check("and the list marks it rather than hiding it",
      panel.table.item(0, 0).text() == "·", panel.table.item(0, 0).text())
check("a Run is always a real one now",
      "self.app.executor.simulate = False" in
      open(os.path.join(os.path.dirname(os.path.dirname(
          os.path.abspath(__file__))), "ur5dual", "gui", "panels",
          "program.py")).read())

print("\nthe rail carries only the panels that are used")
check("Vars and Object are gone",
      sorted(window.rail.buttons) == ["camera", "jog", "points"],
      str(sorted(window.rail.buttons)))
check("and nothing else tried to build them",
      sorted(window.panels) == ["camera", "jog", "points", "program"],
      str(sorted(window.panels)))

print("\n▷ To runs the selected line by itself")
panel.program.steps.clear()
panel._select_column("a")
panel._record("movej")
panel._record("movej")
panel.table.clearSelection()
panel._move_to()
check("with nothing selected it asks rather than guessing",
      not window.executor.running)
panel.table.selectRow(1)
panel._move_to()
deadline = time.monotonic() + 30
while window.executor.running and time.monotonic() < deadline:
    app.processEvents()
    time.sleep(0.02)
check("the selected line ran on its own", not window.executor.running)

print("\nCALL runs another saved program")
sub = Program("sub_for_test")
sub.add(Step("WHERE"))
sub_path = os.path.join(window.programs_dir, "sub_for_test.json")
sub.save(sub_path)
try:
    check("the library offers it", "sub_for_test" in panel.program_names(),
          str(panel.program_names()))
    check("but never the program being edited",
          panel.program.name not in panel.program_names())
    panel.program.steps.clear()
    panel.program.add(Step("CALL", program="sub_for_test", repeat=2))
    panel.refresh()
    check("the row spans both arm columns — a call is not one arm's business",
          panel.table.columnSpan(0, 2) == 2)
    check("and says what it runs",
          "sub_for_test" in panel.table.item(0, 2).text(),
          panel.table.item(0, 2).text())
    plan = panel.program.expand(panel.load_named)
    check("it expands to the called steps, twice",
          [s.kind for s, _r in plan] == ["WHERE", "WHERE"],
          str([s.kind for s, _r in plan]))
    check("no problems while the call resolves", panel.problems.text() == "",
          panel.problems.text())

    # a call to a program that stops existing must show up without pressing Run
    os.remove(sub_path)
    panel.refresh()
    check("a call that cannot be loaded is shown before Run",
          "cannot load" in panel.problems.text(), panel.problems.text())
finally:
    if os.path.exists(sub_path):
        os.remove(sub_path)

print("\nthe record buttons say when they cannot record")
panel._select_column("mirror")
panel.tick()
check("both arms answer in SIM, so recording is offered",
      panel.movej_btn.isEnabled() and panel.movel_btn.isEnabled())
window.executor.running = True
panel.tick()
check("and is taken away while a program is running",
      not panel.movej_btn.isEnabled())
window.executor.running = False
panel.tick()
check("and given back when it stops", panel.movej_btn.isEnabled())

window.close()
print()
print("FAILURES: %d" % fail)
sys.exit(1 if fail else 0)
