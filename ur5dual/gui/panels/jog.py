"""Jog page for arm A, synchronized A+B, and arm B.

This page used to keep all three targets on screen at once, on the argument
that a selector can leave an operator pressing a key for the arm they are not
looking at. That argument has not gone away — it has been paid for instead.
The page is a 320 px sidebar now, which cannot hold three columns of
finger-sized keys, so one target shows at a time behind three large buttons.

What stops the selector being the hazard it was:

  * the chosen target's button is filled, not merely ticked
  * the band above the keys is the arm's own colour and names it
  * the keys themselves carry that tint
  * switching target releases whatever key is held, so a change of target can
    never inherit a press meant for the other arm

The A+B column is still deliberately world-frame only. "Base" and "tool" name
a different direction at each robot, so presenting either as one shared motion
would be a dangerously ambiguous control.
"""

import math

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from ...geometry.kinematics import mat_to_pose, rotvec_to_mat
from ...robot import motion as M
from .. import style as S
from ..widgets.jog_grid import JogGrid
from ..widgets.monitor import ArmMonitor, CompactArmMonitor, PairMonitor

ARM_FRAMES = ("world", "base", "tool", "joint")
PAIR_IDS = ("A", "B")

# On this mast the work area is on world -X. The operator-facing X+ button
# means "out from the mast", so only that displayed axis is reversed.
WORLD_AXIS_SIGN = (-1.0, 1.0, 1.0)


class JogPanel(QWidget):
    """One jog target at a time, in the sidebar's width."""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.frame = ARM_FRAMES[0]
        self.hold_mode = True
        self.preset = 1
        # which target the keys drive. Only one is on screen at a time.
        self.target = "AB"

        body = QVBoxLayout(self)
        body.setContentsMargins(S.sx(6), S.sx(6), S.sx(6), S.sx(6))
        body.setSpacing(S.sx(5))
        body.addWidget(self._build_selector(), 0)
        body.addWidget(self._build_controls(), 0)
        body.addWidget(self._build_targets(), 1)
        body.addWidget(self._build_compact_status(), 0)
        body.addLayout(self._build_footer())

        self._relabel()
        self._set_preset(self.preset)
        self._select_target(self.target)

    # ---- layout ---------------------------------------------------------
    def _build_selector(self):
        """Which arm the keys drive, as three large buttons.

        Buttons rather than a drop-down: a closed combo shows one line of text
        and needs a press to reveal what else it could have been, while three
        buttons show every choice and which one is live without being touched.
        On a page that moves robots that difference is the whole point.
        """
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(S.sx(4))
        self.target_btns = {}
        for target, label in (("A", "Arm A"), ("AB", "A+B"), ("B", "Arm B")):
            button = S.touch_button(label, height=38, font_px=14,
                                    checkable=True)
            button.clicked.connect(
                lambda _c=False, t=target: self._select_target(t))
            row.addWidget(button, 1)
            self.target_btns[target] = button
        return page

    def _select_target(self, target):
        """Show one target's keys, and let go of anything held first.

        Releasing is not tidiness. A key held while the target changes would
        otherwise carry on driving the arm it was pressed for, from a grid the
        operator is no longer looking at.
        """
        self.release()
        self.target = target
        for key, column in self.target_columns.items():
            column.setVisible(key == target)
        for key, button in self.target_btns.items():
            chosen = key == target
            button.setChecked(chosen)
            button.setStyleSheet(
                S.solid(S.ARM_COLOR.get(key, S.GREEN), 14) if chosen
                else S.pill(14))
        self._refresh_enabled()

    def _build_controls(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(S.sx(4))
        # two lines rather than one: the combos and the four step presets do
        # not both fit across a sidebar, and a control squeezed to nothing is
        # worse than a control on the next line
        combos = QHBoxLayout()
        combos.setSpacing(S.sx(4))
        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        outer.addLayout(combos)
        outer.addLayout(row)

        self.motion_combo = QComboBox()
        self.motion_combo.addItems(["Hold to move", "Step per press"])
        self.motion_combo.currentIndexChanged.connect(self._motion_changed)
        self.frame_combo = QComboBox()
        self.frame_combo.addItems([
            "Cartesian — world", "Cartesian — base",
            "Cartesian — tool", "Joint J1-J6",
        ])
        self.frame_combo.currentIndexChanged.connect(self._frame_changed)
        for combo in (self.motion_combo, self.frame_combo):
            combo.setMinimumHeight(S.sx(38))
            combo.setStyleSheet(S.combo())
            combos.addWidget(combo, 1)

        # No "Step" caption and no extra weight on the readout: across a
        # 320 px sidebar those two took the four preset keys down to 29 px,
        # which is a number on a screen rather than something a finger hits.
        # The readout says "10 mm" and labels itself.
        self.preset_btns = []
        for i in range(4):
            button = S.touch_button(str(i + 1), height=38, font_px=13)
            button.clicked.connect(lambda _c=False, n=i: self._set_preset(n))
            row.addWidget(button, 1)
            self.preset_btns.append(button)
        # The readout gets its own line. Sharing the presets' row it was
        # either 62 px and clipped or wide enough to take the four keys down
        # to 37, and "30 mm/s · 8.6 deg/s" is not a caption -- it is what the
        # next press will do.
        self.size_lbl = QLabel("")
        self.size_lbl.setAlignment(Qt.AlignCenter)
        self.size_lbl.setStyleSheet(
            f"font-size:{S.fpx(12)}px;color:{S.INK};font-weight:bold;")
        outer.addWidget(self.size_lbl)
        return page

    def _build_targets(self):
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(S.sx(5))
        self.grids = {}
        # every target is built, and all but one is hidden. They occupy the
        # same place, so the keys never move under a finger when the target
        # changes -- only their colour and their heading do.
        self.target_columns = {}

        for target, title, tint in (
                ("A", "ARM A", S.ARM_TINT["A"]),
                ("AB", "A+B — synchronized", "#f4f0e0"),
                ("B", "ARM B", S.ARM_TINT["B"])):
            column = QWidget()
            v = QVBoxLayout(column)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(S.sx(4))
            header = S.strip(title, tint)
            header.setStyleSheet(
                f"background:{tint};border-radius:{S.sx(4)}px;"
                f"padding:{S.sx(4)}px {S.sx(6)}px;"
                f"font-size:{S.fpx(14)}px;font-weight:bold;"
                f"color:{S.ARM_COLOR.get(target, S.GREEN)};")
            v.addWidget(header, 0)

            grid = JogGrid()
            grid.set_tint(tint)
            grid.pressed_axis.connect(
                lambda axis, sign, t=target: self._pressed(t, axis, sign))
            grid.ticked.connect(
                lambda axis, sign, t=target: self._ticked(t, axis, sign))
            grid.released_axis.connect(
                lambda t=target: self._released(t))
            v.addWidget(grid, 1)
            self.grids[target] = grid
            self.target_columns[target] = column
            row.addWidget(column, 1)
        return page

    def _build_compact_status(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(3))

        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        self.compact_monitors = {}
        for arm_id in PAIR_IDS:
            monitor = CompactArmMonitor(arm_id)
            self.compact_monitors[arm_id] = monitor
            row.addWidget(monitor, 1)
        v.addLayout(row)

        pair_row = QHBoxLayout()
        pair_row.setSpacing(S.sx(4))
        self.pair = PairMonitor()
        pair_row.addWidget(self.pair, 1)
        self.details_btn = S.touch_button("Details", height=34, font_px=12)
        self.details_btn.clicked.connect(self._show_details)
        pair_row.addWidget(self.details_btn, 0)
        v.addLayout(pair_row)

        # Full joint readouts live in a separate window. Keeping them inline
        # would shrink six always-visible jog rows below a safe touch height.
        self.detail_dialog = QDialog(self)
        self.detail_dialog.setWindowTitle("Dual arm details")
        detail_v = QVBoxLayout(self.detail_dialog)
        detail_v.setContentsMargins(S.sx(8), S.sx(8), S.sx(8), S.sx(8))
        detail_v.setSpacing(S.sx(4))
        detail_row = QHBoxLayout()
        detail_row.setSpacing(S.sx(6))
        self.detail_monitors = {}
        for arm_id in PAIR_IDS:
            holder = QWidget()
            holder_v = QVBoxLayout(holder)
            holder_v.setContentsMargins(0, 0, 0, 0)
            holder_v.setSpacing(0)
            holder_v.addWidget(S.arm_strip(arm_id, "Arm %s details" % arm_id))
            monitor = ArmMonitor(arm_id)
            holder_v.addWidget(monitor)
            self.detail_monitors[arm_id] = monitor
            detail_row.addWidget(holder, 1)
        detail_v.addLayout(detail_row)
        pair_title = S.strip("Both arms")
        pair_title.setFixedHeight(S.sx(28))
        detail_v.addWidget(pair_title)
        self.detail_pair = PairMonitor()
        detail_v.addWidget(self.detail_pair)
        self.detail_dialog.resize(S.sx(800), S.sx(230))
        return page

    def _build_footer(self):
        row = QHBoxLayout()
        row.setSpacing(S.sx(6))
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet(f"font-size:{S.fpx(12)}px;color:{S.RED};")
        row.addWidget(self.note, 1)
        self.release_btn = S.touch_button(
            "Let go of the object", S.RED, height=38, font_px=12)
        self.release_btn.clicked.connect(self.app.detach)
        self.release_btn.hide()
        row.addWidget(self.release_btn, 0)
        return row

    def _show_details(self):
        self.release()
        self.detail_dialog.show()
        self.detail_dialog.raise_()
        self.detail_dialog.activateWindow()

    # ---- display controls ----------------------------------------------
    def _relabel(self):
        self.release()
        labels = M.JOINT_NAMES if self.frame == "joint" else M.AXIS_NAMES
        for grid in self.grids.values():
            grid.set_labels(labels)
            grid.set_axis_count(6)
        self._refresh_enabled()

    def _frame_changed(self, index):
        self.release()
        self.frame = ARM_FRAMES[index]
        self._relabel()
        self._set_preset(self.preset)
        self._select_target(self.target)

    def _motion_changed(self, index):
        self.release()
        self.hold_mode = index == 0
        self._set_preset(self.preset)

    def _set_preset(self, n):
        self.preset = n
        for i, button in enumerate(self.preset_btns):
            button.setStyleSheet(S.solid(S.GREEN, 16) if i == n else S.pill(16))

        if self.frame == "joint":
            text = ("%.1f deg / press" % math.degrees(M.STEP_ANG[n])
                    if not self.hold_mode else
                    "%.1f deg/s" % math.degrees(M.CONT_ANG[n]))
            interval = (M.REFRESH if self.hold_mode else
                        max(0.10, M.STEP_ANG[n] / M.STEP_JVEL + 0.03))
        elif self.hold_mode:
            text = "%.0f mm/s · %.1f deg/s" % (
                M.CONT_LIN[n] * 1000, math.degrees(M.CONT_ANG[n]))
            interval = M.REFRESH
        else:
            text = "%.1f mm · %.1f deg / press" % (
                M.STEP_LIN[n] * 1000, math.degrees(M.STEP_ANG[n]))
            interval = max(0.10, M.STEP_LIN[n] / M.STEP_VEL + 0.03)
        self.size_lbl.setText(text)
        for grid in self.grids.values():
            grid.set_interval(interval)

    # ---- command routing ------------------------------------------------
    def _blocked_reason(self, target):
        if self.app.executor.object.held:
            return ("an object is attached and the coordinated loop owns both "
                    "arms — let go before direct jogging")
        if target == "AB" and self.frame != "world":
            return "A+B is available only in Cartesian — world"
        if (target == "AB" and not self.app.cell.simulated
                and not self.app.cell.config.translation_calibrated):
            return ("A+B needs direction calibration: run "
                    "python3 tests/check_directions_online.py --apply, then "
                    "restart the panel")
        for arm_id in (PAIR_IDS if target == "AB" else (target,)):
            arm = self.app.cell.arms[arm_id]
            if not arm.connected:
                return "arm %s is not connected" % arm_id
            if not arm.ready():
                return "arm %s: %s / %s" % (
                    arm_id, arm.robot_mode(), arm.safety_mode())
        return None

    def _pressed(self, target, row, sign):
        # A and B may be held independently. The pair target shares both
        # controllers, so it must never overlap either individual target.
        if target == "AB":
            self.grids["A"].release()
            self.grids["B"].release()
        else:
            self.grids["AB"].release()

        reason = self._blocked_reason(target)
        if reason:
            self.note.setText(reason)
            self.grids[target].release()
            return
        self.note.setText("")
        self._command(target, row, sign)

    def _ticked(self, target, row, sign):
        if self._blocked_reason(target):
            self.grids[target].release()
            return
        self._command(target, row, sign)

    def _command(self, target, row, sign):
        try:
            if target == "AB":
                self._jog_pair_world(row, sign)
            elif self.frame == "world":
                self._jog_arm_world(target, row, sign)
            elif self.hold_mode:
                cmd = np.zeros(6)
                cmd[row] = sign
                self.app.cell.arms[target].motion.speed(
                    self.frame, cmd, self.preset)
            else:
                self.app.cell.arms[target].motion.step(
                    self.frame, row, sign, self.preset)
        except OSError as exc:
            self.note.setText("send failed: %s" % exc)
            self.grids[target].release()

    def _jog_pair_world(self, row, sign):
        for i, arm_id in enumerate(PAIR_IDS):
            try:
                self._jog_arm_world(arm_id, row, sign)
            except OSError as exc:
                if i:
                    self._halt_pair()
                raise OSError("arm %s did not take the command (%s) — both "
                              "arms stopped" % (arm_id, exc))

    def _halt_pair(self):
        for arm_id in PAIR_IDS:
            arm = self.app.cell.arms[arm_id]
            if not arm.connected:
                continue
            try:
                arm.motion.halt("base")
            except OSError as exc:
                self.app.log("arm %s did not stop: %s" % (arm_id, exc))

    def _jog_arm_world(self, arm_id, row, sign):
        arm = self.app.cell.arms[arm_id]
        direction = np.zeros(3)
        direction[row % 3] = float(sign)
        if row < 3:
            direction[row] *= WORLD_AXIS_SIGN[row]

        if self.hold_mode:
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

    def _released(self, target):
        if not self.hold_mode:
            return
        if target == "AB":
            self._halt_pair()
            return
        arm = self.app.cell.arms[target]
        if arm.connected:
            try:
                arm.motion.halt("base" if self.frame == "world" else self.frame)
            except OSError as exc:
                self.note.setText("stop failed: %s" % exc)

    # ---- lifecycle ------------------------------------------------------
    def _refresh_enabled(self):
        reasons = {target: self._blocked_reason(target)
                   for target in self.grids}
        for target, grid in self.grids.items():
            grid.set_enabled(reasons[target] is None)
        held = self.app.executor.object.held
        self.release_btn.setVisible(held)
        # a target that cannot be driven should say so on its own button, so
        # the refusal is where the finger is going rather than only at the
        # bottom of the page
        for key, button in self.target_btns.items():
            button.setEnabled(reasons[key] is None or key == self.target)
        if held:
            self.note.setText(reasons["A"] or "")
        elif self.target == "AB" and self.frame != "world":
            self.note.setText("A+B disabled: synchronized jog uses world frame")
        elif reasons.get(self.target):
            self.note.setText(reasons[self.target])
        elif self.frame != "world":
            self.note.setText("A+B disabled: synchronized jog uses world frame")
        elif any(reasons.values()):
            # A disconnected arm must not hide the fact that the other arm is
            # still usable; this is informational, not a page-wide refusal.
            self.note.setText(" · ".join(sorted(set(
                reason for reason in reasons.values() if reason))))
        else:
            self.note.setText("")

    def tick(self):
        limit = float(self.app.cell.config.motion["max_tcp_force"])
        for arm_id in PAIR_IDS:
            arm = self.app.cell.arms[arm_id]
            self.compact_monitors[arm_id].update_from(arm, limit)
            self.detail_monitors[arm_id].update_from(arm, limit)
        self.pair.update_from(
            self.app.cell, self.app.executor.object,
            float(self.app.cell.config.motion["max_pair_drift"]))
        self.detail_pair.update_from(
            self.app.cell, self.app.executor.object,
            float(self.app.cell.config.motion["max_pair_drift"]))
        self._refresh_enabled()

    def release(self):
        for grid in self.grids.values():
            grid.release()

    def mode_changed(self):
        self.release()
        self._refresh_enabled()
