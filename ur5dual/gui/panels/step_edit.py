"""
Editing one line, in a dialog.

The panel's function row records a step from where the arms already are —
`● MOVEJ` and `↗ MOVEL` are one press each, because on a pendant the arm is
the input device and typing coordinates is what you do when the teaching
failed. Everything a recorded step cannot say is said here instead.

A dialog rather than a column beside the table, for one measurable reason: the
step list is two columns of targets now, and a 252 px editor standing next to
it costs each arm the difference between 509 px and 196. The editing surface
is wanted for a few seconds and the table is wanted the rest of the time, so
the table keeps the width and the editor borrows the screen.

`ColumnEditor` is one arm's half of a line, and the dialog composes one or two
of them depending on what the link speaks for — which is the same rule the
validator applies, arrived at from the other side.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget,
)

from ...program.steps import (
    ARM_FRAMES, COMPARES, IF_SOURCES, IO_RANGE, MOTIONS, OBJECT_FRAMES,
    PAIR_FRAMES, VAR_OPS, Step, make_target, target_kind,
)
from .. import style as S

AXES = ("X", "Y", "Z", "RX", "RY", "RZ")

# What the dialog calls the links: one entry for the two that differ only by
# how many arm columns are filled. `coupled` is not offered — the ATTACH it
# needs is no longer insertable — but it is added back for a step that already
# is one, so opening a saved program's coupled line shows it instead of
# quietly turning it into an arm move.
# Each is (what the step stores, what the operator is shown). Renaming a
# link on screen must not rename it in a saved program, so the two are kept
# apart and the combo is read back by data rather than by text.
LINK_CHOICES = (("arms", "arms"), ("pair", "SYNC OFFSET"))
LINK_LABELS = dict(LINK_CHOICES + (("coupled", "coupled"),))

# Same rule for the arm pickers: the step still holds "A", "B" or "both".
ARM_LABEL = {"A": "Arm A", "B": "Arm B", "both": "Both arms"}
# "—" is a column that is not used, and it has to be sayable: a MOVE line may
# drive one arm, and without an empty option every column reads as filled with
# whatever the pickers happened to default to.
UNUSED = "—"
TARGET_KINDS = (UNUSED, "place", "place + offset", "offset", "here")
_TARGET_FROM_KIND = {"point": "place", "point_offset": "place + offset",
                     "offset": "offset", "pose": "here"}


class ColumnEditor(QGroupBox):
    """One column of a line: what this arm, pair or object is asked for."""

    def __init__(self, title, points, frames=ARM_FRAMES, with_motion=True,
                 tint=None, capture=None, allow_rotation=True,
                 corrections=None):
        super().__init__(title)
        self.points = points
        self.capture = capture              # () -> pose, for the Here button
        self.allow_rotation = allow_rotation
        if tint:
            self.setStyleSheet(
                f"QGroupBox{{background:{tint};border:1px solid #c4c4c4;"
                f"border-radius:{S.sx(4)}px;margin-top:{S.sx(8)}px;"
                f"font-size:{S.fpx(13)}px;font-weight:bold;}}"
                f"QGroupBox::title{{subcontrol-origin:margin;left:{S.sx(8)}px;"
                f"padding:0 {S.sx(4)}px;}}")

        v = QVBoxLayout(self)
        v.setContentsMargins(S.sx(8), S.sx(14), S.sx(8), S.sx(8))
        v.setSpacing(S.sx(4))

        self.target_combo = self._combo(TARGET_KINDS)
        self.target_combo.currentIndexChanged.connect(self._target_changed)
        self.point_combo = self._combo(points.names())
        self.frame_combo = self._combo(frames)
        self.motion_combo = self._combo(MOTIONS)
        self.speed = self._spin(0.0, 500.0, 0.0, decimals=0)
        # what a FIND above this line has looked for. Carrying the target by
        # one of those is how a camera moves a taught pick.
        self.correct_combo = self._combo([""] + list(corrections or []))

        self.rows = {
            "target": self._row(v, "target", self.target_combo),
            "point": self._row(v, "place", self.point_combo),
            "frame": self._row(v, "frame", self.frame_combo),
        }
        self.offset_spins, self.offset_rows = {}, {}
        for axis in AXES:
            spin = self._spin(-2000.0, 2000.0, 0.0)
            self.offset_spins[axis] = spin
            self.offset_rows[axis] = self._row(
                v, "%s  %s" % (axis, "mm" if len(axis) == 1 else "°"), spin)
        if with_motion:
            self.rows["motion"] = self._row(v, "motion", self.motion_combo)
        self.rows["speed"] = self._row(v, "mm/s", self.speed)
        self.rows["correct"] = self._row(v, "moved by", self.correct_combo)

        self.here_btn = S.touch_button("⌖  Here", S.BLUE, height=34, font_px=12)
        self.here_btn.setToolTip("write where this is now into the step")
        self.here_btn.clicked.connect(self._capture)
        v.addWidget(self.here_btn)
        v.addStretch(1)

        self._pose = None
        self._target_changed()

    # -- builders ----------------------------------------------------------
    def _combo(self, items):
        combo = QComboBox()
        combo.addItems(list(items))
        combo.setMinimumHeight(S.sx(30))
        combo.setStyleSheet(S.combo())
        return combo

    def _spin(self, low, high, value, decimals=1):
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setMinimumHeight(S.sx(28))
        spin.setStyleSheet(S.field())
        return spin

    def _row(self, layout, label, widget):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(S.sx(4))
        caption = S.caption(label)
        caption.setFixedWidth(S.sx(52))
        row.addWidget(caption)
        row.addWidget(widget, 1)
        holder = QWidget()
        holder.setLayout(row)
        layout.addWidget(holder)
        return holder

    # -- behaviour ---------------------------------------------------------
    def _target_changed(self):
        choice = self.target_combo.currentText()
        used = choice != UNUSED
        wants_offset = choice in ("place + offset", "offset")
        self.rows["point"].setVisible(choice in ("place", "place + offset"))
        self.rows["frame"].setVisible(wants_offset)
        for axis in AXES:
            turning = axis.startswith("R")
            self.offset_rows[axis].setVisible(
                wants_offset and (self.allow_rotation or not turning))
        for key in ("motion", "speed", "correct"):
            if key in self.rows:
                self.rows[key].setVisible(used)
        self.here_btn.setVisible(choice == "here" and self.capture is not None)

    def _capture(self):
        pose = self.capture() if self.capture else None
        if pose is not None:
            self._pose = [float(v) for v in pose]

    # -- the target it describes -------------------------------------------
    def set_target(self, target):
        target = target or {}
        kind = target_kind(target)
        self.target_combo.setCurrentText(
            _TARGET_FROM_KIND[kind] if kind else UNUSED)
        if target.get("point"):
            self.point_combo.setCurrentText(target["point"])
        if target.get("frame"):
            self.frame_combo.setCurrentText(target["frame"])
        if target.get("offset") is not None:
            for i, axis in enumerate(AXES):
                value = float(target["offset"][i])
                self.offset_spins[axis].setValue(
                    value * 1000.0 if i < 3 else np.degrees(value))
        if target.get("motion"):
            self.motion_combo.setCurrentText(target["motion"])
        if target.get("correct_by"):
            self.correct_combo.setCurrentText(target["correct_by"])
        if target.get("speed"):
            self.speed.setValue(float(target["speed"]) * 1000.0)
        self._pose = target.get("pose")
        self._target_changed()

    def get_target(self):
        choice = self.target_combo.currentText()
        if choice == UNUSED:
            return {}
        offset = None
        if choice in ("place + offset", "offset"):
            offset = [self.offset_spins[a].value() / 1000.0 for a in AXES[:3]]
            offset += [np.radians(self.offset_spins[a].value()) for a in AXES[3:]]
        point = (self.point_combo.currentText()
                 if choice in ("place", "place + offset") else None)
        return make_target(
            point=point or None,
            pose=self._pose if choice == "here" else None,
            offset=offset, frame=self.frame_combo.currentText(),
            motion=(self.motion_combo.currentText()
                    if "motion" in self.rows else None),
            correct_by=self.correct_combo.currentText() or None,
            speed=self.speed.value() / 1000.0 if self.speed.value() > 0 else None)


class StepEditDialog(QDialog):
    """Edit one step. Returns a new Step, or nothing if it was cancelled."""

    def __init__(self, step, points, cell=None, held_object=None,
                 programs=None, labels=None, corrections=None, parent=None):
        super().__init__(parent)
        self.step = step
        self.points = points
        self.cell = cell
        self.held_object = held_object
        self.programs = list(programs or [])
        self.labels = list(labels or [])
        self.corrections = list(corrections or [])
        self.setWindowTitle("Edit  %s" % step.kind)
        self.setMinimumWidth(S.sx(560))

        self.columns = {}
        v = QVBoxLayout(self)
        v.setContentsMargins(S.sx(10), S.sx(10), S.sx(10), S.sx(10))
        v.setSpacing(S.sx(8))

        # The dialog offers three, not the model's four. `solo` and `together`
        # differ only in how many arm columns are filled, so asking for both
        # the link and the columns lets the operator state a contradiction the
        # validator would then reject -- one arm on a `together` line, or two
        # on a `solo` one. Filling the columns says it once.
        self.link_combo = QComboBox()
        for value, label in LINK_CHOICES:
            self.link_combo.addItem(label, value)
        if step.link == "coupled":
            self.link_combo.addItem(LINK_LABELS["coupled"], "coupled")
        self.link_combo.setMinimumHeight(S.sx(34))
        self.link_combo.setStyleSheet(S.combo())
        self.link_combo.setCurrentIndex(self.link_combo.findData(
            "arms" if (step.link or "solo") in ("solo", "together") else step.link))
        self.link_combo.currentIndexChanged.connect(self._rebuild)

        if step.kind == "MOVE":
            row = QHBoxLayout()
            row.addWidget(S.caption("link"), 0)
            row.addWidget(self.link_combo, 1)
            self.link_hint = S.caption("")
            self.link_hint.setWordWrap(True)
            v.addLayout(row)
            v.addWidget(self.link_hint)

        self.body = QHBoxLayout()
        self.body.setSpacing(S.sx(8))
        v.addLayout(self.body, 1)

        self.simple = self._build_simple(step)
        if self.simple is not None:
            v.addWidget(self.simple)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        for button in (buttons.button(QDialogButtonBox.Ok),
                       buttons.button(QDialogButtonBox.Cancel)):
            button.setMinimumHeight(S.sx(42))
            button.setMinimumWidth(S.sx(110))
        v.addWidget(buttons)

        if step.kind == "MOVE":
            self._rebuild()

    # -- the simple kinds --------------------------------------------------
    def _build_simple(self, step):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(6))
        self.origin_combo = self.grip_state = self.number = None
        self.program_combo = self.repeat = None
        self.arm_combo = self.io_number = self.io_state = self.timeout = None
        self.point_pick = None
        self.name_edit = self.op_combo = self.value_spin = None
        self.source_combo = self.compare_combo = None
        self.target_combo = self.else_combo = None
        return self._fill_simple(page, v, step)

    def _pick(self, items, current=None, height=34):
        combo = QComboBox()
        combo.addItems([str(i) for i in items])
        if current is not None and str(current) in [str(i) for i in items]:
            combo.setCurrentText(str(current))
        combo.setMinimumHeight(S.sx(height))
        combo.setStyleSheet(S.combo())
        return combo

    def _pick_arm(self, values, current):
        """Which arm a line speaks to. Read back with `currentData`, because
        what it shows is a name and what a step holds is an id."""
        combo = self._pick([ARM_LABEL[v] for v in values])
        for i, value in enumerate(values):
            combo.setItemData(i, value)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        return combo

    def _num(self, low, high, value, decimals=0):
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setValue(float(value))
        spin.setMinimumHeight(S.sx(34))
        spin.setStyleSheet(S.field())
        return spin

    def _label_pick(self, current):
        """Where a jump may land. Only labels that exist are offered, so a
        jump into nowhere cannot be written in the first place."""
        combo = self._pick([""] + self.labels, current)
        if not self.labels:
            combo.setToolTip("no labels in this program yet — insert a LABEL "
                             "first")
        return combo

    def _fill_simple(self, page, v, step):

        if step.kind == "OUT":
            v.addWidget(QLabel("which controller, which output, and what it does"))
            self.arm_combo = self._pick_arm(("A", "B", "both"),
                                            step.get("arm", "both"))
            self.io_number = self._num(0, IO_RANGE - 1, step.get("output", 0))
            self.io_state = self._pick(["ON", "OFF"],
                                       "ON" if step.get("state", True) else "OFF")
            for widget in (self.arm_combo, self.io_number, self.io_state):
                v.addWidget(widget)
        elif step.kind == "WAIT_IN":
            v.addWidget(QLabel("hold here until this input reads as asked"))
            self.arm_combo = self._pick_arm(("A", "B"), step.get("arm", "A"))
            self.io_number = self._num(0, IO_RANGE - 1, step.get("input", 0))
            self.io_state = self._pick(["ON", "OFF"],
                                       "ON" if step.get("state", True) else "OFF")
            self.timeout = self._num(0, 3600, step.get("timeout", 0) or 0,
                                     decimals=1)
            self.timeout.setSuffix(" s   (0 = wait as long as it takes)")
            for widget in (self.arm_combo, self.io_number, self.io_state,
                           self.timeout):
                v.addWidget(widget)
        elif step.kind == "LABEL":
            v.addWidget(QLabel("a name a jump can land on"))
            self.name_edit = QLineEdit(step.get("name", ""))
            self.name_edit.setMinimumHeight(S.sx(34))
            self.name_edit.setStyleSheet(S.field())
            v.addWidget(self.name_edit)
        elif step.kind == "JUMP":
            v.addWidget(QLabel("carry on from this label instead"))
            self.target_combo = self._label_pick(step.get("target", ""))
            v.addWidget(self.target_combo)
        elif step.kind == "SET_VAR":
            v.addWidget(QLabel("set or count a value the program can test"))
            self.name_edit = QLineEdit(step.get("name", "count"))
            self.name_edit.setMinimumHeight(S.sx(34))
            self.name_edit.setStyleSheet(S.field())
            self.op_combo = self._pick(VAR_OPS, step.get("op", "="))
            self.value_spin = self._num(-1e6, 1e6, step.get("value", 0),
                                        decimals=3)
            for widget in (self.name_edit, self.op_combo, self.value_spin):
                v.addWidget(widget)
        elif step.kind == "IF":
            v.addWidget(QLabel("test an input or a variable"))
            self.source_combo = self._pick(IF_SOURCES,
                                           step.get("source", "input"))
            self.arm_combo = self._pick_arm(("A", "B"), step.get("arm", "A"))
            self.io_number = self._num(0, IO_RANGE - 1, step.get("input", 0))
            self.io_state = self._pick(["ON", "OFF"],
                                       "ON" if step.get("state", True) else "OFF")
            self.name_edit = QLineEdit(step.get("name", "count"))
            self.name_edit.setMinimumHeight(S.sx(34))
            self.name_edit.setStyleSheet(S.field())
            self.compare_combo = self._pick(COMPARES, step.get("compare", ">="))
            self.value_spin = self._num(-1e6, 1e6, step.get("value", 0),
                                        decimals=3)
            for widget in (self.source_combo, self.arm_combo, self.io_number,
                           self.io_state, self.name_edit, self.compare_combo,
                           self.value_spin):
                v.addWidget(widget)
            v.addWidget(QLabel("if it holds, jump here"))
            self.target_combo = self._label_pick(step.get("target", ""))
            v.addWidget(self.target_combo)
            v.addWidget(QLabel("otherwise jump here (leave empty to carry on "
                               "with the next line)"))
            self.else_combo = self._label_pick(step.get("otherwise", ""))
            v.addWidget(self.else_combo)
            self.source_combo.currentIndexChanged.connect(self._if_source_changed)
            self._if_source_changed()
        elif step.kind == "FIND":
            v.addWidget(QLabel("look for the box, and remember how far it has "
                               "moved since the pick was taught"))
            self.name_edit = QLineEdit(step.get("into", "part"))
            self.name_edit.setMinimumHeight(S.sx(34))
            self.name_edit.setStyleSheet(S.field())
            self.name_edit.setToolTip("the name a later line is corrected by; "
                                      "a variable called <name>_found says "
                                      "whether there was anything")
            v.addWidget(QLabel("call the correction"))
            v.addWidget(self.name_edit)
            v.addWidget(QLabel("where the box was when the pick was taught"))
            self.point_pick = self._pick([""] + self.points.names(),
                                         step.get("reference", ""))
            v.addWidget(self.point_pick)
            self.timeout = self._num(0, 120, step.get("timeout", 5.0) or 5.0,
                                     decimals=1)
            self.timeout.setSuffix(" s to wait for a reading")
            v.addWidget(self.timeout)
        elif step.kind == "CALL":
            self.program_combo = QComboBox()
            self.program_combo.addItems(self.programs)
            if step.get("program") in self.programs:
                self.program_combo.setCurrentText(step.get("program"))
            self.program_combo.setMinimumHeight(S.sx(34))
            self.program_combo.setStyleSheet(S.combo())
            self.repeat = QDoubleSpinBox()
            self.repeat.setRange(1, 999)
            self.repeat.setDecimals(0)
            self.repeat.setValue(float(step.get("repeat", 1) or 1))
            self.repeat.setMinimumHeight(S.sx(34))
            self.repeat.setStyleSheet(S.field())
            v.addWidget(QLabel("run this program here, then carry on below"))
            v.addWidget(self.program_combo)
            v.addWidget(QLabel("how many times"))
            v.addWidget(self.repeat)
            if not self.programs:
                warn = QLabel("no other programs are saved yet")
                warn.setStyleSheet("color:%s;" % S.RED)
                v.addWidget(warn)
        elif step.kind == "ATTACH":
            self.origin_combo = QComboBox()
            self.origin_combo.addItems(["midpoint", "A", "B"])
            self.origin_combo.setCurrentText(step.get("origin", "midpoint"))
            self.origin_combo.setMinimumHeight(S.sx(34))
            self.origin_combo.setStyleSheet(S.combo())
            v.addWidget(QLabel("origin the object frame is captured at"))
            v.addWidget(self.origin_combo)
        elif step.kind == "DELAY":
            self.number = QDoubleSpinBox()
            self.number.setRange(0, 600)
            self.number.setDecimals(2)
            self.number.setValue(float(step.get("seconds", 1.0)))
            self.number.setMinimumHeight(S.sx(34))
            self.number.setStyleSheet(S.field())
            v.addWidget(QLabel("seconds to wait"))
            v.addWidget(self.number)
        else:
            return None
        return page

    def _if_source_changed(self):
        """An IF tests one thing or the other, so it shows one or the other."""
        on_input = self.source_combo.currentText() == "input"
        for widget in (self.arm_combo, self.io_number, self.io_state):
            widget.setVisible(on_input)
        for widget in (self.name_edit, self.compare_combo, self.value_spin):
            widget.setVisible(not on_input)

    # -- MOVE, one or two columns -----------------------------------------
    def _clear_body(self):
        while self.body.count():
            item = self.body.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self.columns = {}

    def _capture_arm(self, arm_id):
        arm = self.cell.arms[arm_id] if self.cell else None
        return arm.tcp_pose_world() if arm is not None and arm.connected else None

    def _capture_object(self):
        obj = self.held_object
        return obj.pose_world if obj is not None and obj.held else None

    def _rebuild(self):
        """Show exactly the columns this link speaks for.

        The same rule the validator applies, arrived at from the other side: a
        pair line drives both arms and has one thing to say, so offering two
        columns would be offering a line that cannot be run.
        """
        link = self.link_combo.currentData()
        self._clear_body()
        self.link_hint.setText({
            "arms": "fill one column for one arm, or both to send them together "
                    "— the line then ends when both have arrived",
            "pair": "SYNC OFFSET: one world shift to both arms at matched "
                    "speed — no ATTACH needed, and translation only",
            "coupled": "the object frame drives both arms; needs an ATTACH above",
        }.get(link, ""))

        if link == "arms":
            for arm_id, key in (("A", "a"), ("B", "b")):
                editor = ColumnEditor(
                    "Arm %s" % arm_id, self.points, frames=ARM_FRAMES,
                    tint=S.ARM_TINT[arm_id], corrections=self.corrections,
                    capture=lambda a=arm_id: self._capture_arm(a))
                editor.set_target(self.step.slot(key))
                self.body.addWidget(editor, 1)
                self.columns[key] = editor
        elif link == "pair":
            editor = ColumnEditor(
                "SYNC OFFSET — one world shift to both", self.points,
                frames=PAIR_FRAMES,
                with_motion=False, tint="#f4f0e0", allow_rotation=False)
            editor.set_target(self.step.slot("pair")
                              or make_target(offset=[0.0] * 6, frame="world"))
            self.body.addWidget(editor, 1)
            self.columns["pair"] = editor
        else:
            editor = ColumnEditor(
                "The object", self.points, frames=OBJECT_FRAMES,
                with_motion=False, tint="#f0f0ee", capture=self._capture_object)
            editor.set_target(self.step.slot("obj"))
            self.body.addWidget(editor, 1)
            self.columns["obj"] = editor

    # -- the step it describes ---------------------------------------------
    def get_step(self):
        kind = self.step.kind
        fields = {}
        if kind == "MOVE":
            link = self.link_combo.currentData()
            for key, editor in self.columns.items():
                target = editor.get_target()
                if target_kind(target):
                    fields[key] = target
            if link == "arms":
                # one column is one arm on its own; two is one gesture
                link = "together" if len(fields) > 1 else "solo"
            fields["link"] = link
        elif kind == "OUT":
            fields = {"arm": self.arm_combo.currentData(),
                      "output": int(self.io_number.value()),
                      "state": self.io_state.currentText() == "ON"}
        elif kind == "WAIT_IN":
            fields = {"arm": self.arm_combo.currentData(),
                      "input": int(self.io_number.value()),
                      "state": self.io_state.currentText() == "ON",
                      "timeout": float(self.timeout.value())}
        elif kind == "LABEL":
            fields = {"name": self.name_edit.text().strip()}
        elif kind == "JUMP":
            fields = {"target": self.target_combo.currentText()}
        elif kind == "SET_VAR":
            fields = {"name": self.name_edit.text().strip(),
                      "op": self.op_combo.currentText(),
                      "value": float(self.value_spin.value())}
        elif kind == "IF":
            fields = {"source": self.source_combo.currentText(),
                      "arm": self.arm_combo.currentData(),
                      "input": int(self.io_number.value()),
                      "state": self.io_state.currentText() == "ON",
                      "name": self.name_edit.text().strip(),
                      "compare": self.compare_combo.currentText(),
                      "value": float(self.value_spin.value()),
                      "target": self.target_combo.currentText(),
                      "otherwise": self.else_combo.currentText()}
        elif kind == "ATTACH":
            fields = {"object": self.step.get("object", "object"),
                      "origin": self.origin_combo.currentText()}
        elif kind == "DELAY":
            fields = {"seconds": self.number.value()}
        elif kind == "FIND":
            fields = {"into": self.name_edit.text().strip(),
                      "reference": self.point_pick.currentText(),
                      "timeout": float(self.timeout.value())}
        elif kind == "CALL":
            fields = {"program": (self.program_combo.currentText()
                                  if self.program_combo.count() else ""),
                      "repeat": int(self.repeat.value())}
        step = Step(kind, **fields)
        step.enabled = self.step.enabled
        return step
