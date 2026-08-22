"""
Program tab: the step list, and the buttons that build and run it.

The table is the program, and a row of it is a *gesture* rather than a command
to one robot: Arm A and Arm B are columns, and a `link` field says how tightly
the two are tied. A step that addresses the pair as one body — a SYNC OFFSET, a
coupled carry — spans both columns, which is the visual difference between
"two arms" and "one object".

A step is *recorded*, not typed. `● MOVEJ` and `↗ MOVEL` take where the arms
already are and write it into a line, which is the one input device a pendant
actually has; the insert list covers the lines that have nothing to record,
and both open the step editor when there is more to say. Anything already in
the table is edited by double-tapping it.

That is why the function buttons sit under the table instead of beside it. A
252 px editor column costs each arm the difference between 509 px and 196, and
it is wanted for a few seconds against a table that is wanted the rest of the
time.

Validation runs before every start and the problems are shown as text, not as
a refusal: "line 4: a coupled move needs an ATTACH first" tells you what to
fix. Warnings sit under them in a quieter colour — an offset from a pose
nothing in the program set is usually meant, and always worth knowing.
"""

import os

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QGridLayout, QHeaderView, QLabel, QSizePolicy, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...program.executor import ProgramError
from ...program.steps import KIND_LABEL, Program, Step, make_target
from .. import style as S
from .step_edit import StepEditDialog

# What the insert list can add, and the line each one starts as. Recording
# covers the two that have a position to record; these are the rest.
#
# ATTACH, DETACH and the coupled carry are not offered. They are the two-arm
# grasp — one object frame driving both arms through the 125 Hz servo loop —
# and this cell's work does not turn a workpiece while both arms hold it, which
# is the only thing `SYNC OFFSET` cannot do instead. WHERE is off the list for a
# duller reason: the live position row above the table says the same thing
# without a line in the program. Nothing was deleted: steps.py still knows the
# kinds, the executor still runs them, coupling.py is untouched, and a saved
# program that uses them still loads, still validates and still runs.
# Putting any of them back is one line here.
INSERTS = (
    ("OFFSET",       lambda: Step("MOVE", link="solo")),
    ("SYNC OFFSET",  lambda: Step("MOVE", link="pair")),
    ("OUT",          lambda: Step("OUT", arm="both", output=0, state=True)),
    ("WAIT IN",      lambda: Step("WAIT_IN", arm="A", input=0, state=True,
                                  timeout=0.0)),
    ("DELAY",        lambda: Step("DELAY", seconds=1.0)),
    ("IN POSE",      lambda: Step("BARRIER")),
    ("LABEL",        lambda: Step("LABEL", name="top")),
    ("JUMP",         lambda: Step("JUMP", target="")),
    ("IF",           lambda: Step("IF", source="input", arm="A", input=0,
                                  state=True, name="count", compare=">=",
                                  value=1, target="", otherwise="")),
    ("SET VAR",      lambda: Step("SET_VAR", name="count", op="+=", value=1)),
    ("FIND",         lambda: Step("FIND", into="part", reference="",
                                  timeout=5.0)),
    ("CALL",         lambda: Step("CALL", program="", repeat=1)),
)

# the colour a step kind is named in, so the list is scannable at a glance
# The function grid. Twelve units rather than eight columns: the rows carry
# four, five and six buttons, and 12 is what divides evenly by all of them, so
# every row lines up with the ones above and below it.
UNITS = 12
BUTTON_H = 40

KIND_COLOR = {"MOVE": "#1a5fa8", "OUT": "#2a7050", "WAIT_IN": "#2a7050",
              "ATTACH": "#5a3a8a", "DETACH": "#5a3a8a", "DELAY": "#8a5020",
              "BARRIER": "#8a5020", "WHERE": "#666666", "CALL": "#2a5a8a",
              "LABEL": "#8a5020", "JUMP": "#8a5020", "IF": "#8a5020",
              "SET_VAR": "#8a5020", "FIND": "#1a5fa8"}


class ProgramPanel(QWidget):
    # same rule as the log: these fire on the executor's thread, so they are
    # signals rather than direct widget calls
    _step = pyqtSignal(int)
    _done = pyqtSignal(bool, str)

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.program = Program("untitled")
        self.column = "mirror"

        body = QVBoxLayout(self)
        body.setContentsMargins(S.sx(6), S.sx(6), S.sx(6), S.sx(6))
        body.setSpacing(S.sx(6))
        body.addWidget(self._build_table(), 1)
        body.addWidget(self._build_functions(), 0)

        self._step.connect(self._show_step)
        self._done.connect(self._show_finished)
        app.executor.on_step = lambda i, _s: self._step.emit(i)
        app.executor.on_finished = lambda ok, msg: self._done.emit(ok, msg)

    # ---- table -----------------------------------------------------------
    def _build_table(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(5))
        v.addWidget(S.strip("Program — teach & run"))

        # Where the arms are, above the program being written. Teaching used to
        # mean leaving for the Jog tab's Details window to read a number; the
        # number belongs where the step that needs it is written.
        self.now_lbl = QLabel("—")
        self.now_lbl.setStyleSheet(
            f"font-size:{S.fpx(11)}px;font-family:monospace;color:#333333;"
            f"background:#f7f9fb;border:1px solid #dcdcdc;"
            f"padding:{S.sx(3)}px {S.sx(6)}px;")
        v.addWidget(self.now_lbl)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "step", "Arm A", "Arm B",
                                              "link"])
        self.table.verticalHeader().hide()
        self.table.setStyleSheet(f"font-size:{S.fpx(12)}px;")
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.itemDoubleClicked.connect(lambda _i: self._edit())
        v.addWidget(self.table, 1)

        self.problems = QLabel("")
        self.problems.setWordWrap(True)
        self.problems.setStyleSheet(f"font-size:{S.fpx(11)}px;color:{S.RED};")
        v.addWidget(self.problems)
        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        self.warnings.setStyleSheet(f"font-size:{S.fpx(11)}px;color:{S.AMBER};")
        v.addWidget(self.warnings)
        return page

    # ---- the function row ------------------------------------------------
    def _build_functions(self):
        """Run, save, record, insert, and the step operations.

        Four rows on a twelve-unit grid, and the unit is what keeps every
        target big rather than every button identical: `insert` needs the room
        to show a step kind and `＋` does not, so they get five units and one
        instead of the same eight-column cell each.

        The columns are the grouping. Run, Pause, Stop and Load stand in the
        left one, so the four things that act on the program as a whole are
        under the same thumb; the record keys and who they record for are the
        middle two rows; and `⧉ A+B` sits directly over `A` and `B`, which is
        the layout saying what the button means.
        """
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(S.sx(4))
        grid.setVerticalSpacing(S.sx(4))
        for column in range(UNITS):
            grid.setColumnStretch(column, 1)

        def cell(widget, row, column, span):
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            widget.setMinimumHeight(S.sx(BUTTON_H))
            grid.addWidget(widget, row, column, 1, span)
            return widget

        def button(label, tip, callback, row, column, span, colour=None,
                   checkable=False):
            b = S.touch_button(label, colour, height=BUTTON_H, font_px=13,
                               checkable=checkable)
            b.setToolTip(tip)
            b.clicked.connect(callback)
            return cell(b, row, column, span)

        # -- the program, down the left
        self.run_btn = button("▶ Run", "run the whole program", self._run,
                              0, 0, 2, S.GREEN)
        self.pause_btn = button("⏸ Pause", "hold where the program is, part "
                                "way through a move if that is where it is; "
                                "press again to carry on from there",
                                self._pause, 1, 0, 2, S.AMBER)
        self.stop_btn = button("■ Stop", "stop immediately", self._stop,
                               2, 0, 2, S.RED)
        button("📂 Load", "load a program from JSON", self._load_program,
               3, 0, 2, S.PURPLE)

        # -- inserting a step, and who a recorded one is for
        self.insert_combo = QComboBox()
        self.insert_combo.addItems([name for name, _mk in INSERTS])
        self.insert_combo.setStyleSheet(S.combo())
        self.insert_combo.setToolTip(
            "OFFSET       shift from wherever the arm is when the line runs\n"
            "SYNC OFFSET  one world shift to both arms, no ATTACH needed\n"
            "OUT          drive a digital output — a gripper is one of these\n"
            "WAIT IN      hold here until an input reads as asked\n"
            "IN POSE      wait for both arms to settle before going on\n"
            "FIND         look for the box, and correct the lines below it\n"
            "CALL         run another saved program here, then carry on")
        cell(self.insert_combo, 0, 2, 5)
        self.add_btn = button("＋", "insert the selected kind of step",
                              self._insert, 0, 7, 1, S.GREEN)

        self.column_btns = {}
        self.column_btns["mirror"] = button(
            "⧉ A+B", "record for both arms at once",
            lambda _c=False: self._select_column("mirror"), 0, 8, 4,
            checkable=True)
        self.movej_btn = button("●  MOVEJ", "record where the arm is now, "
                                "reached by movej", lambda: self._record("movej"),
                                1, 2, 3, S.BLUE)
        self.movel_btn = button("↗  MOVEL", "record where the arm is now, "
                                "reached in a straight line",
                                lambda: self._record("movel"), 1, 5, 3, S.BLUE)
        for key, column in (("a", 8), ("b", 10)):
            self.column_btns[key] = button(
                key.upper(), "record for arm %s" % key.upper(),
                lambda _c=False, k=key: self._select_column(k), 1, column, 2,
                checkable=True)

        # -- the step under the finger
        for i, (label, tip, callback) in enumerate((
                ("▷ To", "run just the selected step — drive to that taught "
                         "position without running the rest", self._move_to),
                ("✏ Edit", "edit the step (or double-tap it)", self._edit),
                ("⎘ Dup", "duplicate the step, below itself", self._duplicate),
                ("✕ Del", "delete the step", self._delete))):
            button(label, tip, callback, 2, 2 + i * 2, 2)
        button("↑", "move the step up", lambda: self._move(-1), 2, 10, 2)
        button("↓", "move the step down", lambda: self._move(+1), 3, 10, 2)

        # -- the file, and what is stamped on what is recorded
        button("💾 Save", "save this program to JSON", self._save, 3, 2, 2,
               S.PURPLE)
        self.loop_btn = button("🔁 Loop", "run it again when it reaches the end",
                               self._set_loop, 3, 4, 2, checkable=True)
        self.speed = QDoubleSpinBox()
        self.speed.setRange(0, 500)
        self.speed.setDecimals(0)
        self.speed.setValue(120)
        self.speed.setSuffix(" mm/s")
        self.speed.setStyleSheet(S.field())
        self.speed.setToolTip("speed stamped onto newly recorded steps")
        cell(self.speed, 3, 6, 4)

        self._select_column("mirror")
        return page

    # ---- recording and inserting -----------------------------------------
    def _select_column(self, key):
        self.column = key
        for k, button in self.column_btns.items():
            button.setChecked(k == key)

    def _arm_ids(self):
        return ("A", "B") if self.column == "mirror" else (self.column.upper(),)

    def _record(self, motion):
        """Write where the arms are now into a line.

        The pose is baked in rather than named. That is what recording means on
        a pendant — it replays exactly what was taught and does not follow a
        point re-taught later, which is the trade the Points tab exists to
        offer the other side of.
        """
        speed = self.speed.value() / 1000.0 if self.speed.value() > 0 else None
        fields, missing = {}, []
        for arm_id in self._arm_ids():
            arm = self.app.cell.arms[arm_id]
            if not arm.connected:
                missing.append(arm_id)
                continue
            fields[arm_id.lower()] = make_target(
                pose=arm.tcp_pose_world(), motion=motion, speed=speed)
        if missing:
            self.app.log("arm %s is not connected — nothing recorded"
                         % " and ".join(missing))
            return
        fields["link"] = "together" if len(fields) > 1 else "solo"
        self._push(Step("MOVE", **fields))
        self.app.log("recorded %s for arm %s"
                     % (motion, " and ".join(self._arm_ids())))

    def _insert(self):
        """Everything that has nothing to record opens the editor first."""
        name = self.insert_combo.currentText()
        make = dict(INSERTS)[name]
        step = make()
        if name == "OFFSET":
            step = Step("MOVE", link="solo",
                        a=make_target(offset=[0.0] * 6, frame="world",
                                      motion="movel"))
        elif name == "SYNC OFFSET":
            step = Step("MOVE", link="pair",
                        pair=make_target(offset=[0.0] * 6, frame="world"))
        if name == "CALL":
            others = self.program_names()
            if not others:
                self.app.log("no other programs saved yet — save one first")
                return
            step = Step("CALL", program=others[0], repeat=1)
        if step.kind in ("DETACH", "BARRIER"):
            self._push(step)
            return
        edited = self._open_editor(step)
        if edited is not None:
            self._push(edited)

    def program_names(self):
        """Every other saved program, for CALL to choose from.

        The one being edited is left out: a program that calls itself is
        refused when it is expanded, and offering it here would be offering a
        refusal.
        """
        try:
            names = sorted(os.path.splitext(f)[0]
                           for f in os.listdir(self.app.programs_dir)
                           if f.endswith(".json"))
        except OSError:
            return []
        return [n for n in names if n != self.program.name]

    def label_names(self):
        """Every label in this program, for a jump to choose from."""
        return [(s.get("name") or "").strip() for s in self.program.steps
                if s.kind == "LABEL" and (s.get("name") or "").strip()]

    def correction_names(self):
        """What a FIND in this program has looked for, for a target to be
        carried by."""
        return [(s.get("into") or "").strip() for s in self.program.steps
                if s.kind == "FIND" and (s.get("into") or "").strip()]

    def load_named(self, name):
        """Load a saved program by name, for CALL to expand."""
        return Program.load(os.path.join(self.app.programs_dir, name + ".json"))

    def _open_editor(self, step):
        dialog = StepEditDialog(step, self.app.points, cell=self.app.cell,
                                held_object=self.app.executor.object,
                                programs=self.program_names(),
                                labels=self.label_names(),
                                corrections=self.correction_names(),
                                parent=self)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.get_step()
        return None

    def _push(self, step):
        index = self.table.currentRow()
        self.program.add(step, None if index < 0 else index + 1)
        self.refresh()
        self.table.selectRow(len(self.program.steps) - 1
                             if index < 0 else index + 1)

    def _selected_row(self):
        """The row the operator actually chose, or None.

        `currentRow` is not that: Qt keeps a current cell after the selection
        is cleared, so a button reading it acts on a row nothing is
        highlighting. ▷ To drives an arm, which makes the difference between
        those two the difference between a move and a surprise.
        """
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        row = rows[0].row()
        return row if 0 <= row < len(self.program.steps) else None

    def _edit(self):
        row = self._selected_row()
        if row is None:
            return
        edited = self._open_editor(self.program.steps[row])
        if edited is not None:
            self.program.steps[row] = edited
            self.refresh()
            self.table.selectRow(row)

    def _duplicate(self):
        row = self._selected_row()
        if row is not None:
            copy = Step.from_dict(self.program.steps[row].to_dict())
            self.program.add(copy, row + 1)
            self.refresh()
            self.table.selectRow(row + 1)

    # ---- table edits -----------------------------------------------------
    def _move(self, delta):
        row = self._selected_row()
        if row is None:
            return
        new = self.program.move(row, delta)
        self.refresh()
        self.table.selectRow(new)

    def _delete(self):
        row = self._selected_row()
        if row is None:
            return
        self.program.remove(row)
        self.refresh()

    def _set_loop(self):
        self.program.loop = self.loop_btn.isChecked()

    # ---- running ---------------------------------------------------------
    def _run(self):
        self._start()

    def _move_to(self):
        """Run the selected line by itself."""
        row = self._selected_row()
        if row is None:
            self.app.log("select a step first")
            return
        self._start(only=row)

    def _start(self, only=None):
        # Dry run left the panel with On/off. The executor still walks a
        # program without commanding either arm -- the tests run on it -- but
        # nothing on screen turns it on, so a Run is always a real one.
        self.app.executor.simulate = False
        try:
            self.app.executor.start(self.program, only=only,
                                    load=self.load_named)
        except ProgramError as e:
            self.problems.setText(str(e))
            return
        self.problems.setText("")
        self.app.log("%s started"
                     % ("line %d" % (only + 1) if only is not None
                        else "program"))

    def _pause(self):
        executor = self.app.executor
        executor.resume() if executor.paused else executor.pause()

    def _stop(self):
        self.app.executor.stop()

    def _show_step(self, index):
        self.table.selectRow(index)

    def _show_finished(self, ok, message):
        if not ok:
            self.problems.setText(message)

    # ---- files -----------------------------------------------------------
    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save program", os.path.join(self.app.programs_dir,
                                               self.program.name + ".json"),
            "Programs (*.json)")
        if path:
            self.program.name = os.path.splitext(os.path.basename(path))[0]
            self.program.save(path)
            self.app.log("saved program to %s" % path)

    def _load_program(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load program", self.app.programs_dir, "Programs (*.json)")
        if path:
            self.program = Program.load(path)
            self.loop_btn.setChecked(self.program.loop)
            self.refresh()
            self.app.log("loaded program %s" % path)

    # ---- lifecycle -------------------------------------------------------
    def refresh(self):
        self.table.clearSpans()
        self.table.setRowCount(len(self.program.steps))
        for row, step in enumerate(self.program.steps):
            span, a, b, link = step.render()
            # a step can still arrive disabled from a saved program; the
            # panel shows it as one and no longer offers to switch it back
            index = str(row + 1) if step.enabled else "·"
            self._set_cell(row, 0, index)
            self._set_cell(row, 1, KIND_LABEL.get(step.kind, step.kind),
                           ink=KIND_COLOR.get(step.kind))
            if span is not None:
                # one body, one cell: a pair or coupled line has nothing
                # different to say to each arm
                self._set_cell(row, 2, span, tint="#f0f0ee")
                self.table.setSpan(row, 2, 1, 2)
            else:
                self._set_cell(row, 2, a, tint=S.ARM_TINT["A"] if a != "·" else None)
                self._set_cell(row, 3, b, tint=S.ARM_TINT["B"] if b != "·" else None)
            self._set_cell(row, 4, link)

        surfaces = self.app.executor.taught_on_surface()
        problems, warnings = self.program.check(self.app.points,
                                                self.app.cell.config,
                                                surfaces=surfaces)
        if not problems and self.program.calls():
            # what a CALL drags in is only visible once it is expanded, and an
            # operator should not have to press Run to find out
            try:
                called = Program(self.program.name)
                called.steps = [step for step, _row
                                in self.program.expand(self.load_named)]
                problems, warnings = called.check(self.app.points,
                                                  self.app.cell.config,
                                                  surfaces=surfaces)
            except ValueError as e:
                problems = [str(e)]
        self.problems.setText("\n".join(problems[:3]))
        self.warnings.setText("\n".join(warnings[:2]))

    def _set_cell(self, row, col, text, tint=None, ink=None):
        item = QTableWidgetItem(text or "")
        if tint:
            item.setBackground(QColor(tint))
        if ink:
            item.setForeground(QColor(ink))
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        self.table.setItem(row, col, item)

    def tick(self):
        running = self.app.executor.running
        connected = all(self.app.cell.arms[a].connected for a in self._arm_ids())
        for button in (self.movej_btn, self.movel_btn):
            button.setEnabled(connected and not running)
        self.run_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
        self.pause_btn.setText("▶ Resume" if self.app.executor.paused
                               else "⏸ Pause")

        self._show_now()

    def _show_now(self):
        """All six numbers per arm, above the program being written.

        Position alone was half a readout. A place is a pose, and orientation
        is the half of it that a `here` capture carries and that nothing else
        on this page shows — so a step recorded from a wrist that is turned
        the wrong way looked exactly like one that was not.

        One line per thing, under one header, so the columns line up and a
        difference between the two arms is a difference in one column rather
        than something to be read for.
        """
        lines = ["%-4s%8s%8s%8s%8s%8s%8s"
                 % ("", "x", "y", "z", "rx", "ry", "rz")]
        gap = self.app.cell.tcp_separation()
        if gap is not None:
            lines[0] += "     gap %.1f" % (gap * 1000)

        for arm_id in ("A", "B"):
            arm = self.app.cell.arms[arm_id]
            if not arm.connected:
                lines.append("%-4s  %s" % (arm_id, "—"))
                continue
            lines.append("%-4s%s" % (arm_id, self._pose_row(arm.tcp_pose_world())))

        obj = self.app.executor.object
        if obj.held:
            lines.append("%-4s%s" % ("obj", self._pose_row(obj.pose_world)))
        self.now_lbl.setText("\n".join(lines))

    @staticmethod
    def _pose_row(pose):
        """mm and degrees, in the column widths the header is spaced to."""
        return ("%8.1f%8.1f%8.1f%8.1f%8.1f%8.1f"
                % (pose[0] * 1000, pose[1] * 1000, pose[2] * 1000,
                   *np.degrees(pose[3:])))
