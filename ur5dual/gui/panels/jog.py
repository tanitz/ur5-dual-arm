"""
Jog tab: one grid, three things it can drive.

Arm A and arm B jog exactly as a single-arm pendant does. The third target is
both of them at once: press +Z and A and B each travel along the cell's +Z,
same direction, same speed, for as long as the button is held. Two arms
holding one workpiece between them therefore carry it, because neither end of
it moves relative to the other.

That third target used to drive a *held object* through the coordinated servo
loop, which meant it could not be used until something had been attached, the
loop had come up, and the two bases' relative orientation had been measured —
three prerequisites deep, and none of them visible from the button being
pressed. It is now the same world-frame jog arm A and arm B already do,
issued to both arms in the same breath, so it works exactly when they do and
needs nothing that they do not.

What it is not: a rigid-body move of the pair. Each arm is sent the same world
twist about its own TCP, so X/Y/Z carries a gripped workpiece without
straining it, while RX/RY/RZ turns each wrist on the spot about the world axis
— the same direction at both ends, but not one object turning about one pivot.
For that, attach an object and use the coordinated path on the Object tab.
"""

import math

import numpy as np
from PyQt5.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from ...geometry.kinematics import mat_to_pose, rotvec_to_mat
from ...robot import motion as M
from .. import style as S
from ..widgets.jog_grid import JogGrid

# World first: it is the one reference that means the same thing at both arms,
# so the default selection matches the cell the operator is looking at.
ARM_FRAMES = ("world", "base", "tool", "joint")

# the two arms the pair target drives, in the order the commands go out
PAIR_IDS = ("A", "B")

# On this mast the work area is on world -X.  The operator-facing X+ button
# means "out from the mast", so only that displayed axis is reversed.
WORLD_AXIS_SIGN = (-1.0, 1.0, 1.0)


class JogPanel(QWidget):
    """The jog tab: cell state down the left, the grid filling the right.

    The status column is built by the main window and handed in, rather than
    owned here — it is the same widgets whatever tab is open, and this panel
    is only where they are put on the screen.
    """

    def __init__(self, app, status_column=None):
        super().__init__()
        self.app = app
        self.target = "A"
        self.frame = ARM_FRAMES[0]
        self.hold_mode = True
        self.preset = 1

        body = QHBoxLayout(self)
        body.setContentsMargins(S.sx(6), S.sx(6), S.sx(6), S.sx(6))
        body.setSpacing(S.sx(8))
        if status_column is not None:
            body.addWidget(status_column, 0)

        # right side: what is being driven, then the grid that drives it. The
        # four choices are one strip across the top so the six rows below get
        # the whole remaining height and grow into real touch targets.
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(6))
        v.addWidget(self._build_controls())
        self.grid = JogGrid()
        self.grid.pressed_axis.connect(self._pressed)
        self.grid.ticked.connect(self._ticked)
        self.grid.released_axis.connect(self._released)
        v.addWidget(self.grid, 1)
        v.addLayout(self._build_footer())
        body.addWidget(page, 1)
        self._retarget()

    # ---- controls --------------------------------------------------------
    def _build_controls(self):
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(S.sx(6))

        self.target_combo = QComboBox()
        self.target_combo.addItems(["Arm A", "Arm B", "Synchronized A+B"])
        self.target_combo.currentIndexChanged.connect(self._target_changed)

        self.frame_combo = QComboBox()
        self.frame_combo.currentIndexChanged.connect(self._frame_changed)

        self.motion_combo = QComboBox()
        self.motion_combo.addItems(["Hold to move", "Step per press"])
        self.motion_combo.currentIndexChanged.connect(self._motion_changed)

        for col, (title, combo) in enumerate((("Drive", self.target_combo),
                                              ("Frame", self.frame_combo),
                                              ("Motion", self.motion_combo))):
            combo.setMinimumHeight(S.sx(42))
            combo.setStyleSheet(S.combo())
            grid.addWidget(S.strip(title), 0, col)
            grid.addWidget(combo, 1, col)
            grid.setColumnStretch(col, 2)

        sizes = QHBoxLayout()
        sizes.setSpacing(S.sx(4))
        self.preset_btns = []
        for i in range(4):
            button = S.touch_button(str(i + 1), height=42)
            button.clicked.connect(lambda _c=False, n=i: self._set_preset(n))
            sizes.addWidget(button, 1)
            self.preset_btns.append(button)
        grid.addWidget(S.strip("Step size"), 0, 3)
        grid.addLayout(sizes, 1, 3)
        grid.setColumnStretch(3, 3)
        return page

    def _build_footer(self):
        """Under the grid: how big a press is, and why one was refused."""
        row = QHBoxLayout()
        row.setSpacing(S.sx(6))

        self.size_lbl = QLabel("")
        self.size_lbl.setStyleSheet(f"font-size:{S.fpx(14)}px;color:{S.INK};"
                                    f"font-weight:bold;")
        row.addWidget(self.size_lbl, 0)

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"font-size:{S.fpx(12)}px;color:{S.RED};")
        row.addWidget(self.note, 1)

        # An attached object belongs to the coordinated loop, which writes both
        # arms at servo rate — so while one exists, jogging the arms directly
        # is refused. Letting go is offered here rather than only on the Object
        # tab, because this is where the refusal is read.
        self.release_btn = S.touch_button("Let go of the object", S.RED,
                                          height=42, font_px=13)
        self.release_btn.clicked.connect(self.app.detach)
        self.release_btn.hide()
        row.addWidget(self.release_btn, 0)
        return row

    # ---- target / frame --------------------------------------------------
    def _target_changed(self, index):
        self.grid.release()
        target = ("A", "B", "AB")[index]
        if self.app.cell.simulated and self.app.executor.object.held:
            # The coordinated loop writes both simulated arms at servo rate.
            # Leaving it alive would immediately overwrite a jog issued from
            # this panel and make an enabled button appear broken.  In SIM the
            # grip is virtual, so dropping it on an explicit retarget is safe.
            self.app.detach()
            self.app.log("SIM: let go of the object so the jog grid can drive "
                         "%s directly"
                         % ("both arms" if target == "AB" else "arm %s" % target))
        self.target = target
        self._retarget()

    def _pair_frames(self):
        # One frame, and it is the cell's own. Two arms only agree about a
        # direction in a frame that is outside both of them; "base" and "tool"
        # name a different direction at each arm, which is the one thing this
        # target must never do.
        return ("world",)

    def _frames(self):
        return self._pair_frames() if self.target == "AB" else ARM_FRAMES

    def _retarget(self):
        frames = self._frames()
        labels = {"base": "Cartesian — base", "tool": "Cartesian — tool",
                  "joint": "Joint J1-J6", "world": "Cartesian — world"}
        self.frame_combo.blockSignals(True)
        self.frame_combo.clear()
        self.frame_combo.addItems([labels[f] for f in frames])
        self.frame_combo.blockSignals(False)
        self.frame = frames[0]

        self.grid.set_tint("#f4f0e0" if self.target == "AB"
                           else S.ARM_TINT[self.target])
        self.grid.set_axis_count(6)
        self.motion_combo.setEnabled(True)
        self._relabel()
        self._set_preset(self.preset)

    def _frame_changed(self, index):
        self.grid.release()
        frames = self._frames()
        if 0 <= index < len(frames):
            self.frame = frames[index]
        self._relabel()

    def _relabel(self):
        if self.frame == "joint":
            self.grid.set_labels(M.JOINT_NAMES)
        else:
            self.grid.set_labels(M.AXIS_NAMES)

    def _motion_changed(self, index):
        self.grid.release()
        self.hold_mode = index == 0
        self._set_preset(self.preset)

    def _set_preset(self, n):
        self.preset = n
        for i, button in enumerate(self.preset_btns):
            button.setStyleSheet(S.solid(S.GREEN, 16) if i == n else S.pill(16))

        # The pair uses the arm presets unchanged, because it *is* the arm jog
        # sent twice: two arms carrying one thing between them travel at the
        # speed each of them travels at, and quoting a second set of numbers
        # for the same motion is how an operator ends up mistrusting both.
        if self.frame == "joint":
            text = ("%.1f deg / press" % math.degrees(M.STEP_ANG[n])
                    if not self.hold_mode
                    else "%.1f deg/s" % math.degrees(M.CONT_ANG[n]))
            self.grid.set_interval(M.REFRESH if self.hold_mode
                                   else max(0.10, M.STEP_ANG[n] / M.STEP_JVEL + 0.03))
        elif self.hold_mode:
            text = "%.0f mm/s   %.1f deg/s" % (M.CONT_LIN[n] * 1000,
                                               math.degrees(M.CONT_ANG[n]))
            self.grid.set_interval(M.REFRESH)
        else:
            text = "%.1f mm   %.1f deg  / press" % (M.STEP_LIN[n] * 1000,
                                                    math.degrees(M.STEP_ANG[n]))
            self.grid.set_interval(max(0.10, M.STEP_LIN[n] / M.STEP_VEL + 0.03))
        self.size_lbl.setText(text)

    # ---- pressing --------------------------------------------------------
    def _blocked_reason(self):
        """Why this press cannot go out — the same two questions for one arm
        as for both, asked of every arm the target drives.

        Everything else that used to stand in front of the pair target is
        gone: nothing has to be attached, no servo loop has to be up, and no
        calibration has to have been run. What is left is what a single-arm
        jog has always needed — an arm on the other end of the socket, powered
        and out of any stop.
        """
        cell = self.app.cell
        # An attached object is driven by the coordinated loop at servo rate.
        # Two writers on one controller is the one state neither can recover
        # from, so the direct jog stands aside rather than racing it.
        if self.app.executor.object.held:
            return ("an object is attached and the coordinated loop owns both "
                    "arms — press 'Let go of the object' to jog them directly")
        for arm_id in (PAIR_IDS if self.target == "AB" else (self.target,)):
            arm = cell.arms[arm_id]
            if not arm.connected:
                return "arm %s is not connected" % arm_id
            if not arm.ready():
                return "arm %s: %s / %s" % (arm_id, arm.robot_mode(),
                                            arm.safety_mode())
        return None

    def _pressed(self, row, sign):
        reason = self._blocked_reason()
        if reason:
            self.note.setText(reason)
            self.grid.release()
            return
        self.note.setText("")
        self._command(row, sign)

    def _ticked(self, row, sign):
        if self._blocked_reason():
            self.grid.release()
            return
        self._command(row, sign)

    def _command(self, row, sign):
        try:
            if self.target == "AB":
                self._jog_pair_world(row, sign)
            elif self.frame == "world":
                self._jog_arm_world(self.target, row, sign)
            elif self.hold_mode:
                cmd = np.zeros(6)
                cmd[row] = sign
                self.app.cell.arms[self.target].motion.speed(
                    self.frame, cmd, self.preset)
            else:
                self.app.cell.arms[self.target].motion.step(
                    self.frame, row, sign, self.preset)
        except OSError as e:
            self.note.setText("send failed: %s" % e)
            self.grid.release()

    def _jog_pair_world(self, row, sign):
        """The same world-axis jog, to both arms, off one press.

        Each arm gets exactly what it would get if it were selected on its
        own — one direction, named in the cell frame, resolved through that
        arm's own mounting transform. Two arms given the same world direction
        travel the same way at the same speed, so anything gripped between
        them is carried rather than pulled: the pairing is in the geometry,
        not in any attempt here to keep two commands in step.

        The two sends are microseconds apart and each ramps at the same
        acceleration on its own controller, so they start together closely
        enough for a workpiece with any give in it at all. One send failing
        after the other has gone out is the one outcome worth spending code
        on, because that is one arm moving and one arm not.
        """
        for i, arm_id in enumerate(PAIR_IDS):
            try:
                self._jog_arm_world(arm_id, row, sign)
            except OSError as e:
                if i:
                    self._halt_pair()
                raise OSError("arm %s did not take the command (%s) — both "
                              "arms stopped" % (arm_id, e))

    def _halt_pair(self):
        """Stop both arms, whatever either of them has to say about it."""
        for arm_id in PAIR_IDS:
            arm = self.app.cell.arms[arm_id]
            if not arm.connected:
                continue
            try:
                arm.motion.halt("base")
            except OSError as e:
                self.app.log("arm %s did not stop: %s" % (arm_id, e))

    def _jog_arm_world(self, arm_id, row, sign):
        """Jog one arm along the cell's fixed world axes.

        A UR controller accepts twists and poses in its own base frame.  The
        GUI keeps the operator-facing direction in world, then resolves it
        through this arm's mounting transform immediately before sending.
        """
        arm = self.app.cell.arms[arm_id]
        direction = np.zeros(3)
        direction[row % 3] = float(sign)
        if row < 3:
            direction[row] *= WORLD_AXIS_SIGN[row]

        if self.hold_mode:
            # Rotate a world vector into this arm's base. Linear and angular
            # parts transform by the same rotation because the jog is about
            # the TCP itself, so there is no translation cross-term.
            direction_base = arm.base_matrix()[:3, :3].T @ direction
            cmd = np.zeros(6)
            if row < 3:
                cmd[:3] = direction_base
            else:
                cmd[3:] = direction_base
            arm.motion.speed("base", cmd, self.preset)
            return

        target = arm.tcp_matrix_world().copy()
        if row < 3:
            target[:3, 3] += direction * M.STEP_LIN[self.preset]
        else:
            target[:3, :3] = rotvec_to_mat(
                direction * M.STEP_ANG[self.preset]) @ target[:3, :3]
        arm.motion.movel_pose(
            arm.world_to_base(mat_to_pose(target)),
            vel=M.STEP_VEL, acc=M.STEP_ACC)

    def _released(self):
        if not self.hold_mode:
            return
        if self.target == "AB":
            # Both, always, whether or not both took the press: an arm that
            # was sent a speed and not a stop keeps going until its watchdog
            # expires, and the watchdog is not the operator's finger.
            self._halt_pair()
            return
        arm = self.app.cell.arms[self.target]
        if arm.connected:
            try:
                arm.motion.halt("base" if self.frame == "world" else self.frame)
            except OSError as e:
                self.note.setText("stop failed: %s" % e)

    # ---- lifecycle -------------------------------------------------------
    def tick(self):
        blocked = self._blocked_reason()
        self.grid.set_enabled(blocked is None)
        if blocked and not self.grid.held:
            self.note.setText(blocked)
        elif not blocked:
            self.note.setText("")

        # The one way out of the refusal above, offered where it is read.
        self.release_btn.setVisible(self.app.executor.object.held)

    def release(self):
        self.grid.release()

    def mode_changed(self):
        """Re-read the target after SIM/REAL is switched.

        Only the pair, and only to refresh it: an arm target keeps whatever
        frame the operator had chosen, because switching between the drawing
        and the robots is not a reason to move their finger.
        """
        if self.target == "AB":
            self._retarget()
