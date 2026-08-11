"""
Program tab: the step list, and the buttons that run it.

The table is the program. Adding a step asks only for what that step needs —
an arm move needs an arm and a place, an object spin needs an axis and an
angle — so the same table holds both kinds of step without either one
growing columns it does not use.

Validation runs before every start and the problems are shown as text, not as
a refusal: "line 4: MOVE_OBJ needs an ATTACH first" tells you what to fix.
"""

import os

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...program.executor import ProgramError
from ...program.steps import Program, Step
from .. import style as S

STEP_CHOICES = ("MOVE_ARM", "MOVE_OBJ", "ROTATE_OBJ", "ATTACH", "DETACH",
                "GRIP", "BARRIER", "DELAY")


class ProgramPanel(QWidget):
    # same rule as the log: these fire on the executor's thread, so they are
    # signals rather than direct widget calls
    _step = pyqtSignal(int)
    _done = pyqtSignal(bool, str)

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.program = Program("untitled")

        body = QHBoxLayout(self)
        body.setContentsMargins(S.sx(6), S.sx(6), S.sx(6), S.sx(6))
        body.setSpacing(S.sx(8))
        body.addWidget(self._build_table(), 3)
        body.addWidget(self._build_editor(), 2)

        self._step.connect(self._show_step)
        self._done.connect(self._show_finished)
        app.executor.on_step = lambda i, _s: self._step.emit(i)
        app.executor.on_finished = lambda ok, msg: self._done.emit(ok, msg)

    # ---- table -----------------------------------------------------------
    def _build_table(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(6))
        v.addWidget(S.strip("Program — teach & run"))

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["#", "step", "detail"])
        self.table.verticalHeader().hide()
        self.table.setStyleSheet(f"font-size:{S.fpx(13)}px;")
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        v.addWidget(self.table, 1)

        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        for label, callback in (("▲", lambda: self._move(-1)),
                                ("▼", lambda: self._move(+1)),
                                ("Delete", self._delete),
                                ("On/off", self._toggle)):
            button = S.touch_button(label, height=42, font_px=13)
            button.clicked.connect(callback)
            row.addWidget(button, 1)
        v.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        self.run_btn = S.touch_button("▶ Run", S.GREEN, height=54, font_px=16)
        self.run_btn.clicked.connect(self._run)
        self.pause_btn = S.touch_button("⏸ Pause", S.AMBER, height=54, font_px=16)
        self.pause_btn.clicked.connect(self._pause)
        self.stop_btn = S.touch_button("■ Stop", S.RED, height=54, font_px=16)
        self.stop_btn.clicked.connect(self._stop)
        for button in (self.run_btn, self.pause_btn, self.stop_btn):
            row.addWidget(button, 1)
        v.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        self.loop_btn = S.touch_button("🔁 Loop", height=42, font_px=13,
                                       checkable=True)
        self.loop_btn.toggled.connect(self._set_loop)
        self.dry_btn = S.touch_button("Dry run", height=42, font_px=13,
                                      checkable=True)
        self.dry_btn.setToolTip("walk the steps without commanding either arm")
        for label, callback in (("💾 Save", self._save), ("📂 Load", self._load)):
            button = S.touch_button(label, S.PURPLE, height=42, font_px=13)
            button.clicked.connect(callback)
            row.addWidget(button, 1)
        row.addWidget(self.loop_btn, 1)
        row.addWidget(self.dry_btn, 1)
        v.addLayout(row)

        self.problems = QLabel("")
        self.problems.setWordWrap(True)
        self.problems.setStyleSheet(f"font-size:{S.fpx(12)}px;color:{S.RED};")
        v.addWidget(self.problems)
        return page

    # ---- editor ----------------------------------------------------------
    def _build_editor(self):
        page = QWidget()
        page.setMaximumWidth(S.sx(250))
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(6))
        v.addWidget(S.strip("Add a step"))

        self.kind_combo = QComboBox()
        self.kind_combo.addItems(STEP_CHOICES)
        self.kind_combo.setMinimumHeight(S.sx(40))
        self.kind_combo.setStyleSheet(S.combo())
        self.kind_combo.currentIndexChanged.connect(self._kind_changed)
        v.addWidget(self.kind_combo)

        self.arm_combo = QComboBox()
        self.arm_combo.addItems(["A", "B", "both"])
        self.point_combo = QComboBox()
        self.motion_combo = QComboBox()
        self.motion_combo.addItems(["movej", "movel"])
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["x", "y", "z"])
        self.frame_combo = QComboBox()
        self.frame_combo.addItems(["object", "world"])
        self.origin_combo = QComboBox()
        self.origin_combo.addItems(["midpoint", "A", "B"])
        self.grip_state_combo = QComboBox()
        self.grip_state_combo.addItems(["close (output ON)",
                                        "open (output OFF)"])
        for widget in (self.arm_combo, self.point_combo, self.motion_combo,
                       self.axis_combo, self.frame_combo, self.origin_combo,
                       self.grip_state_combo):
            widget.setMinimumHeight(S.sx(36))
            widget.setStyleSheet(S.combo())

        self.number = QDoubleSpinBox()
        self.number.setRange(-360.0, 360.0)
        self.number.setDecimals(2)
        self.number.setMinimumHeight(S.sx(36))
        self.number.setStyleSheet(S.field())

        self.rows = {}
        for key, label, widget in (("arm", "arm", self.arm_combo),
                                   ("point", "place", self.point_combo),
                                   ("motion", "motion", self.motion_combo),
                                   ("axis", "axis", self.axis_combo),
                                   ("frame", "about", self.frame_combo),
                                   ("origin", "origin", self.origin_combo),
                                   ("grip_state", "action", self.grip_state_combo),
                                   ("number", "value", self.number)):
            row = QHBoxLayout()
            caption = S.caption(label)
            caption.setFixedWidth(S.sx(56))
            row.addWidget(caption)
            row.addWidget(widget, 1)
            holder = QWidget()
            holder.setLayout(row)
            v.addWidget(holder)
            self.rows[key] = holder

        self.add_btn = S.touch_button("＋  Add step", S.GREEN, height=50)
        self.add_btn.clicked.connect(self._add)
        v.addWidget(self.add_btn)

        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"font-size:{S.fpx(12)}px;color:#444444;")
        v.addWidget(self.hint)
        v.addStretch(1)
        self._kind_changed()
        return page

    def _kind_changed(self):
        kind = self.kind_combo.currentText()
        show = {
            "MOVE_ARM": ("arm", "point", "motion"),
            "MOVE_OBJ": ("point",),
            "ROTATE_OBJ": ("axis", "frame", "number"),
            "ATTACH": ("origin",),
            "GRIP": ("arm", "grip_state", "number"),
            "DELAY": ("number",),
        }.get(kind, ())
        for key, holder in self.rows.items():
            holder.setVisible(key in show)
        hints = {
            "MOVE_ARM": "one arm on its own — only valid while nothing is held",
            "MOVE_OBJ": "carry the held object to a place, both arms together",
            "ROTATE_OBJ": ("value is degrees. 'object' turns the box about its "
                           "own axis, 'world' about the cell's — needs a "
                           "touch-off either way"),
            "ATTACH": "freeze the current grip; run it with the grippers closed",
            "GRIP": "value is the digital output number",
            "DELAY": "value is seconds",
            "BARRIER": "wait until both arms have stopped moving",
            "DETACH": "give the arms back their independence",
        }
        self.hint.setText(hints.get(kind, ""))
        if kind == "GRIP":
            self.number.setValue(float(self.app.grip_output))

    def _add(self):
        kind = self.kind_combo.currentText()
        fields = {}
        if kind == "MOVE_ARM":
            fields = {"arm": self.arm_combo.currentText(),
                      "point": self.point_combo.currentText(),
                      "motion": self.motion_combo.currentText()}
        elif kind == "MOVE_OBJ":
            fields = {"point": self.point_combo.currentText()}
        elif kind == "ROTATE_OBJ":
            fields = {"axis": self.axis_combo.currentText(),
                      "frame": self.frame_combo.currentText(),
                      "angle_deg": self.number.value()}
        elif kind == "ATTACH":
            fields = {"object": "object", "origin": self.origin_combo.currentText()}
        elif kind == "GRIP":
            fields = {"arm": self.arm_combo.currentText(),
                      "output": int(self.number.value()),
                      "state": self.grip_state_combo.currentIndex() == 0}
        elif kind == "DELAY":
            fields = {"seconds": self.number.value()}

        index = self.table.currentRow()
        self.program.add(Step(kind, **fields),
                         None if index < 0 else index + 1)
        self.refresh()

    # ---- table edits -----------------------------------------------------
    def _move(self, delta):
        row = self.table.currentRow()
        new = self.program.move(row, delta)
        self.refresh()
        self.table.selectRow(new)

    def _delete(self):
        self.program.remove(self.table.currentRow())
        self.refresh()

    def _toggle(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.program.steps):
            step = self.program.steps[row]
            step.enabled = not step.enabled
            self.refresh()
            self.table.selectRow(row)

    def _set_loop(self, on):
        self.program.loop = on

    # ---- running ---------------------------------------------------------
    def _run(self):
        self.app.executor.simulate = self.dry_btn.isChecked()
        try:
            self.app.executor.start(self.program)
        except ProgramError as e:
            self.problems.setText(str(e))
            return
        self.problems.setText("")
        self.app.log("program started%s"
                     % (" (dry run)" if self.dry_btn.isChecked() else ""))

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

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load program", self.app.programs_dir, "Programs (*.json)")
        if path:
            self.program = Program.load(path)
            self.loop_btn.setChecked(self.program.loop)
            self.refresh()
            self.app.log("loaded program %s" % path)

    # ---- lifecycle -------------------------------------------------------
    def refresh(self):
        names = self.app.points.names()
        current = self.point_combo.currentText()
        self.point_combo.clear()
        self.point_combo.addItems(names)
        if current in names:
            self.point_combo.setCurrentText(current)

        self.table.setRowCount(len(self.program.steps))
        for row, step in enumerate(self.program.steps):
            for col, text in enumerate((str(row + 1), step.kind, step.describe())):
                item = QTableWidgetItem(text)
                if not step.enabled:
                    item.setText("· " + text if col == 2 else text)
                self.table.setItem(row, col, item)

        problems = self.program.validate(self.app.points)
        self.problems.setText("\n".join(problems[:4]))

    def tick(self):
        running = self.app.executor.running
        self.run_btn.setEnabled(not running)
        self.dry_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
        self.pause_btn.setText("▶ Resume" if self.app.executor.paused
                               else "⏸ Pause")
