"""
Object tab: taking hold, and moving what is held.

Attaching is the moment a pair of arms becomes one carried object. It is
deliberately a button an operator presses with the grippers already closed,
not something a program does implicitly, because whatever the arms are
holding at that instant is what every later object move is computed from.

Freedrive lives here too: hand-guiding both arms onto the workpiece is how
the grasp gets set up in the first place.

Two halves, in the order the work happens: the left one takes hold, the right
one drives what is now held. The named places the object goes are on the
Points tab, because they are taught from the arms as often as from a grip.
"""

import math

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QSizePolicy,
    QVBoxLayout, QWidget,
)

from .. import style as S
from .jog import WORLD_AXIS_SIGN


class ObjectPanel(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self._sim_jog = None
        self._sim_jog_timer = QTimer(self)
        self._sim_jog_timer.setInterval(80)
        self._sim_jog_timer.timeout.connect(self._sim_held_jog_refresh)

        body = QHBoxLayout(self)
        body.setContentsMargins(S.sx(6), S.sx(6), S.sx(6), S.sx(6))
        body.setSpacing(S.sx(8))
        body.addWidget(self._build_grip(), 1)
        body.addWidget(self._build_held(), 1)

    # ---- grip ------------------------------------------------------------
    def _build_grip(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(6))

        v.addWidget(S.strip("Hand-guide onto the workpiece"))
        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        self.fd_btns = {}
        for arm_id in ("A", "B"):
            button = S.touch_button("✋ Freedrive %s" % arm_id, S.AMBER, height=52)
            button.pressed.connect(lambda a=arm_id: self._freedrive(a, True))
            button.released.connect(lambda a=arm_id: self._freedrive(a, False))
            row.addWidget(button, 1)
            self.fd_btns[arm_id] = button
        v.addLayout(row)

        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        for arm_id in ("A", "B"):
            for state, label, colour in ((True, "close", S.BLUE),
                                         (False, "open", None)):
                button = S.touch_button("%s %s" % (label, arm_id), colour,
                                        height=42, font_px=13)
                button.clicked.connect(
                    lambda _c=False, a=arm_id, s=state: self._grip(a, s))
                row.addWidget(button, 1)
        v.addLayout(row)

        v.addWidget(S.strip("Take hold — create Virtual Box"))
        self.origin_combo = QComboBox()
        self.origin_combo.addItems([
            "Object frame midway between the grippers",
            "Object frame on arm A's TCP (A leads)",
            "Object frame on arm B's TCP (B leads)",
        ])
        self.origin_combo.setMinimumHeight(S.sx(40))
        self.origin_combo.setStyleSheet(S.combo())
        v.addWidget(self.origin_combo)

        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        self.attach_btn = S.touch_button("ATTACH", S.GREEN, height=56, font_px=17)
        self.attach_btn.clicked.connect(self._attach)
        self.detach_btn = S.touch_button("DETACH", S.RED, height=56, font_px=17)
        self.detach_btn.clicked.connect(self._detach)
        row.addWidget(self.attach_btn, 1)
        row.addWidget(self.detach_btn, 1)
        v.addLayout(row)

        self.rename_btn = S.touch_button("Rename object", height=40, font_px=13)
        self.rename_btn.clicked.connect(self._rename)
        v.addWidget(self.rename_btn)

        self.grip_info = QLabel("nothing attached")
        self.grip_info.setWordWrap(True)
        self.grip_info.setStyleSheet(f"font-size:{S.fpx(12)}px;"
                                     f"font-family:monospace;color:#444444;")
        v.addWidget(self.grip_info)

        v.addStretch(1)
        return page

    # ---- driving what is held --------------------------------------------
    def _build_held(self):
        """The same world-frame Held Object controls for SIM and REAL."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(S.sx(6))

        layout.addWidget(S.strip("Held Object — world X/Y/Z/RX/RY/RZ"))
        row = QHBoxLayout()
        self.sim_start_btn = S.touch_button("Ready + take hold", S.GREEN,
                                            height=44, font_px=13)
        self.sim_start_btn.clicked.connect(self._sim_held_start)
        self.sim_stop_btn = S.touch_button("Detach", S.RED, height=44,
                                           font_px=13)
        self.sim_stop_btn.clicked.connect(self._sim_held_stop)
        row.addWidget(self.sim_start_btn, 2)
        row.addWidget(self.sim_stop_btn, 1)
        layout.addLayout(row)

        self.sim_step_mm = QComboBox()
        self.sim_step_mm.addItems(["1 mm/s", "5 mm/s", "10 mm/s", "25 mm/s"])
        self.sim_step_mm.setCurrentIndex(1)
        self.sim_step_mm.setMinimumHeight(S.sx(36))
        self.sim_step_mm.setStyleSheet(S.combo())
        layout.addWidget(self.sim_step_mm)

        self.sim_step_deg = QComboBox()
        self.sim_step_deg.addItems(["Rotation 0.25 deg/s", "Rotation 1 deg/s",
                                    "Rotation 2 deg/s", "Rotation 5 deg/s"])
        self.sim_step_deg.setCurrentIndex(1)
        self.sim_step_deg.setMinimumHeight(S.sx(36))
        self.sim_step_deg.setStyleSheet(S.combo())
        layout.addWidget(self.sim_step_deg)

        grid = QGridLayout()
        grid.setSpacing(S.sx(4))
        self.sim_step_btns = []
        for row_i, (name, axis) in enumerate(
                (("X", 0), ("Y", 1), ("Z", 2),
                 ("RX", 3), ("RY", 4), ("RZ", 5))):
            for col, (sign, suffix) in enumerate(((-1, "−"), (+1, "+"))):
                button = S.touch_button("%s%s" % (name, suffix), height=38,
                                        font_px=15)
                # the tab is theirs now that the points table has moved out,
                # so the six rows grow into the height instead of sitting in a
                # block of small keys under everything else
                button.setSizePolicy(QSizePolicy.Expanding,
                                     QSizePolicy.Expanding)
                button.pressed.connect(
                    lambda a=axis, s=sign: self._sim_held_jog_start(a, s))
                button.released.connect(self._sim_held_jog_stop)
                grid.addWidget(button, row_i, col)
                self.sim_step_btns.append(button)
        layout.addLayout(grid, 1)
        self.sim_status = QLabel(
            "Moves both arms from one shared world-frame box pose")
        self.sim_status.setWordWrap(True)
        self.sim_status.setStyleSheet(
            f"font-size:{S.fpx(12)}px;color:#444444;")
        layout.addWidget(self.sim_status)
        return page

    def _sim_held_start(self):
        """Capture both tools and start the shared SIM/REAL coordinator."""
        if self.app.executor.object.held:
            self.app.log("Held Object: DETACH the current object first")
            return
        if not self.app.attach("midpoint"):
            return
        # Driven from here, not from the Jog tab: an attached object belongs to
        # the coordinated loop, and the Jog tab's A+B target writes both arms
        # directly, so it stands aside for as long as this holds anything.
        self.app.log("Held Object ready — use the world X/Y/Z/RX/RY/RZ grid "
                     "on this tab")

    def _sim_held_stop(self):
        self._sim_held_jog_stop()
        if self.app.executor.object.held:
            self.app.detach()

    def _sim_held_jog_start(self, axis, sign):
        """Start a pure world-axis jog shared by both captured TCPs."""
        self._sim_jog = (int(axis), int(sign))
        if self._sim_held_jog_refresh():
            self._sim_jog_timer.start()

    def _sim_held_jog_refresh(self):
        coordinator = self.app.executor.coordinator
        obj = self.app.executor.object
        if (self._sim_jog is None or coordinator is None
                or not coordinator.alive or not obj.held):
            self._sim_held_jog_stop()
            self.app.log("Held Object: take hold before jogging")
            return False
        axis, sign = self._sim_jog
        direction_world = np.zeros(3)
        direction_world[axis % 3] = float(sign)
        if axis < 3:
            direction_world[axis] *= WORLD_AXIS_SIGN[axis]
        try:
            if axis >= 3:
                speed = math.radians((0.25, 1.0, 2.0, 5.0)[
                    self.sim_step_deg.currentIndex()])
                coordinator.command_jog(obj, direction_world, speed, "ang")
            else:
                speed = (1.0, 5.0, 10.0, 25.0)[
                    self.sim_step_mm.currentIndex()] / 1000.0
                coordinator.command_jog(obj, direction_world, speed, "lin")
            return True
        except Exception as exc:
            self.app.log("Held Object jog refused: %s" % exc)
            self._sim_held_jog_stop()
            return False

    def _sim_held_jog_stop(self):
        self._sim_jog_timer.stop()
        self._sim_jog = None
        coordinator = self.app.executor.coordinator
        if coordinator is not None:
            coordinator.stop_jog()

    def _freedrive(self, arm_id, on):
        arm = self.app.cell.arms[arm_id]
        if not arm.connected:
            return
        try:
            arm.motion.freedrive_on() if on else arm.motion.freedrive_off()
            self.app.log("arm %s freedrive %s" % (arm_id, "on" if on else "off"))
        except OSError as e:
            self.app.log("arm %s freedrive failed: %s" % (arm_id, e))

    def _grip(self, arm_id, closed):
        arm = self.app.cell.arms[arm_id]
        if not arm.connected:
            return
        try:
            arm.motion.set_digital_out(self.app.grip_output, closed)
            self.app.log("arm %s DO%d %s" % (arm_id, self.app.grip_output,
                                             "ON" if closed else "OFF"))
        except OSError as e:
            self.app.log("arm %s gripper failed: %s" % (arm_id, e))

    def _attach(self):
        """Freeze the current grip.

        No name prompt: the operator presses this with both grippers already
        closed on the workpiece and often no keyboard in front of them, and a
        modal dialog at that moment is one more thing to dismiss — cancel it
        and nothing happens, with nothing to say why. The object gets a name
        automatically and Rename is there for anyone who wants a better one.
        """
        origin = ("midpoint", "A", "B")[self.origin_combo.currentIndex()]
        self.app.attach(origin)

    def _rename(self):
        obj = self.app.executor.object
        if not obj.held:
            self.app.log("nothing is attached to rename")
            return
        name, ok = QInputDialog.getText(self, "Rename", "Object name:",
                                        text=obj.name)
        if ok and name.strip():
            obj.name = name.strip()
            self.refresh()

    def _detach(self):
        self.app.detach()

    # ---- lifecycle -------------------------------------------------------
    def _refresh_virtual_box(self):
        """Update the live HeldObject frame without rebuilding the point table."""
        obj = self.app.executor.object
        if obj.held:
            # The live coordinator is the authority while a plan is moving;
            # pose_world follows it at servo rate, but current_pose also makes
            # the display correct between command issue and the next feed tick.
            coordinator = self.app.executor.coordinator
            pose = (coordinator.current_pose(obj) if coordinator is not None
                    else obj.virtual_pose_world)
            lines = ["Virtual Box '%s' held by %s" %
                     (obj.name, "+".join(obj.arm_ids)),
                     "frame origin  %s" % (obj.frame_origin or "saved")]
            if obj.span() is not None:
                lines.append("grip span   %.1f mm" % (obj.span() * 1000))
            lines.append("X Y Z       %7.1f %7.1f %7.1f mm"
                         % tuple(v * 1000 for v in pose[:3]))
            lines.append("RX RY RZ    %7.2f %7.2f %7.2f deg (rotvec)"
                         % tuple(np.degrees(pose[3:])))
            lines.append("6-DOF       %s" %
                         ("world XYZ + RX/RY/RZ (SIM)" if self.app.cell.simulated
                          else "XYZ + RX/RY/RZ enabled" if
                          self.app.cell.config.calibrated else
                          "XYZ only — full calibration required for rotation"))
            self.grip_info.setText("\n".join(lines))
        else:
            self.grip_info.setText("nothing attached")

    def refresh(self):
        self._refresh_virtual_box()

    def tick(self):
        self._refresh_virtual_box()
        held = self.app.executor.object.held
        coordinator = self.app.executor.coordinator
        sim = self.app.cell.simulated
        connected = self.app.cell.connected_ids
        drive_ready = bool(held and coordinator is not None
                           and coordinator.alive)
        if not drive_ready and self._sim_jog is not None:
            self._sim_held_jog_stop()
        self.sim_start_btn.setText(
            "Ready + take hold" if sim else "Take hold at current pose")
        self.sim_start_btn.setEnabled(
            not held and (sim or len(connected) >= 2))
        self.sim_stop_btn.setEnabled(held)
        for button in self.sim_step_btns:
            button.setEnabled(drive_ready and not coordinator.busy())
        if drive_ready:
            pose = coordinator.current_pose(self.app.executor.object)
            status = (
                "%s holding — world XYZ %.1f %.1f %.1f mm, "
                "rotvec RX/RY/RZ %.2f %.2f %.2f deg"
                % ("SIM" if sim else "REAL",
                   pose[0] * 1000.0, pose[1] * 1000.0, pose[2] * 1000.0,
                   *np.degrees(pose[3:])))
            if coordinator.uncalibrated_real:
                status += ("\nUNCALIBRATED OVERRIDE — hold-jog only, "
                           "capped at 1 mm/s and 0.25 deg/s; stops at 1 mm drift")
            self.sim_status.setText(status)
        elif held:
            detail = self.app.coordinator_start_error or \
                "the coordinated servo loop is stopped"
            self.sim_status.setText("held, but motion is unavailable — %s" % detail)
        elif sim:
            self.sim_status.setText(
                "Ready + take hold sets both arms, captures the midpoint box, "
                "and starts the SIM solver")
        elif len(connected) < 2:
            self.sim_status.setText(
                "REAL — connect both arms A and B before taking hold")
        else:
            self.sim_status.setText(
                "REAL — place both tools on the box, close both grippers, then "
                "Take hold at current pose")
        # a greyed-out button that never says why is the same as a broken one
        if held:
            reason = "already holding '%s' — DETACH first" % self.app.executor.object.name
        elif not connected:
            reason = "no arm is connected"
        else:
            reason = ""
        self.attach_btn.setEnabled(not reason)
        self.attach_btn.setToolTip(reason)
        self.detach_btn.setEnabled(held)
        self.rename_btn.setEnabled(held)
        for arm_id, button in self.fd_btns.items():
            button.setEnabled(self.app.cell.arms[arm_id].connected and not held)

    def release(self):
        self._sim_held_jog_stop()
