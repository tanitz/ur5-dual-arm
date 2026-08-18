"""
Live readouts.

ArmMonitor shows one arm the way a pendant does. PairMonitor shows the two
numbers that only exist because there are two arms — how far apart the TCPs
are, and whether that distance is holding still. While an object is gripped
by both, a moving separation is the earliest visible sign that the arms have
started fighting, well before either one trips on force.
"""

import math

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QGroupBox, QHBoxLayout, QLabel

from .. import style as S


class ArmMonitor(QGroupBox):
    def __init__(self, arm_id):
        super().__init__("")
        self.arm_id = arm_id
        self.setStyleSheet("QGroupBox{border:none;margin-top:0;}")
        g = QGridLayout(self)
        g.setContentsMargins(S.sx(4), 0, S.sx(4), 0)
        g.setVerticalSpacing(S.sx(5))
        g.setHorizontalSpacing(S.sx(4))

        colour = S.ARM_COLOR[arm_id]
        self.pose_lbls, self.joint_lbls = [], []
        rows = ((["X", "Y", "Z"], self.pose_lbls, "mm"),
                (["Rx", "Ry", "Rz"], self.pose_lbls, "deg"),
                (["J1", "J2", "J3"], self.joint_lbls, "deg"),
                (["J4", "J5", "J6"], self.joint_lbls, "deg"))
        for row, (names, store, unit) in enumerate(rows):
            for col, name in enumerate(names):
                head = S.caption(name)
                head.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                g.addWidget(head, row, col * 2)
                value = S.value_label("—", colour)
                # the widest thing written here is "%7.2f" — seven monospace
                # characters, and three of these columns set the width of the
                # whole status column
                value.setMinimumWidth(S.sx(60))
                g.addWidget(value, row, col * 2 + 1)
                store.append(value)
            g.addWidget(S.caption(unit), row, 6)

        self.force_lbl = S.value_label("—", colour)
        self.mode_lbl = QLabel("—")
        g.addWidget(S.caption("force"), 4, 0)
        g.addWidget(self.force_lbl, 4, 1, 1, 2)
        g.addWidget(self.mode_lbl, 4, 3, 1, 4)
        g.setColumnStretch(6, 1)

    def update_from(self, arm, force_limit):
        if not arm.connected:
            return self.clear("not connected")
        try:
            state = arm.state()
            pose = arm.tcp_pose_world()
        except (OSError, ConnectionError):
            return self.clear("stream lost")

        for i in range(3):
            self.pose_lbls[i].setText("%7.1f" % (pose[i] * 1000.0))
            self.pose_lbls[3 + i].setText("%7.2f" % math.degrees(pose[3 + i]))
        for i, q in enumerate(state["q_actual"]):
            self.joint_lbls[i].setText("%7.2f" % math.degrees(q))

        force = arm.tcp_force_magnitude()
        hot = force > force_limit
        self.force_lbl.setText("%.0f N" % force)
        self.force_lbl.setStyleSheet(
            f"font-size:{S.fpx(14)}px;font-family:monospace;"
            f"color:{S.RED if hot else S.ARM_COLOR[self.arm_id]};"
            + ("font-weight:bold;" if hot else ""))

        mode, safety = arm.robot_mode(), arm.safety_mode()
        ok = arm.ready()
        # a feed that has gone quiet must never look like an arm standing still
        age = arm.stream.age if arm.stream is not None else 0.0
        stale = age > 0.25
        # Details must name both controller states even when they are healthy;
        # otherwise SAFETY_NORMAL disappears precisely when it is most useful
        # as the baseline an operator compares a fault against.
        text = "%s / %s" % (mode, safety)
        if stale:
            text += "   FEED %.1fs OLD" % age
        self.mode_lbl.setText(text)
        self.mode_lbl.setStyleSheet(
            f"font-size:{S.fpx(13)}px;font-weight:bold;"
            f"color:{S.GREEN if ok and not stale else S.RED};")

    def clear(self, note=""):
        for label in self.pose_lbls + self.joint_lbls:
            label.setText("—")
        self.force_lbl.setText("—")
        self.mode_lbl.setText(note)
        self.mode_lbl.setStyleSheet(f"font-size:{S.fpx(13)}px;color:#888888;")


class CompactArmMonitor(QGroupBox):
    """One-line arm state for the bottom of the full-width Jog page."""

    def __init__(self, arm_id):
        super().__init__("")
        self.arm_id = arm_id
        self.setStyleSheet(
            "QGroupBox{border:1px solid #c8c8c8;border-radius:%dpx;"
            "background:%s;}" % (S.sx(5), S.ARM_TINT[arm_id]))
        row = QHBoxLayout(self)
        row.setContentsMargins(S.sx(6), S.sx(3), S.sx(6), S.sx(3))
        row.setSpacing(S.sx(6))

        name = QLabel("ARM %s" % arm_id)
        name.setStyleSheet(
            f"font-size:{S.fpx(13)}px;font-weight:bold;"
            f"color:{S.ARM_COLOR[arm_id]};")
        row.addWidget(name, 0)
        self.pose_lbl = S.value_label("XYZ —", S.ARM_COLOR[arm_id], 12)
        row.addWidget(self.pose_lbl, 1)
        self.force_lbl = S.value_label("— N", S.ARM_COLOR[arm_id], 12)
        row.addWidget(self.force_lbl, 0)
        self.mode_lbl = QLabel("—")
        self.mode_lbl.setMinimumWidth(S.sx(66))
        self.mode_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.mode_lbl, 0)

    def update_from(self, arm, force_limit):
        if not arm.connected:
            return self.clear("not connected")
        try:
            pose = arm.tcp_pose_world()
        except (OSError, ConnectionError):
            return self.clear("stream lost")

        self.pose_lbl.setText("XYZ %.0f  %.0f  %.0f mm" %
                              tuple(v * 1000.0 for v in pose[:3]))
        force = arm.tcp_force_magnitude()
        hot = force > force_limit
        self.force_lbl.setText("%.0f N" % force)
        self.force_lbl.setStyleSheet(
            f"font-size:{S.fpx(12)}px;font-family:monospace;"
            f"color:{S.RED if hot else S.ARM_COLOR[self.arm_id]};"
            + ("font-weight:bold;" if hot else ""))

        ready = arm.ready()
        text = arm.robot_mode() if ready else "%s/%s" % (
            arm.robot_mode(), arm.safety_mode())
        age = arm.stream.age if arm.stream is not None else 0.0
        stale = age > 0.25
        if stale:
            text = "FEED %.1fs" % age
        self.mode_lbl.setText(text)
        self.mode_lbl.setStyleSheet(
            f"font-size:{S.fpx(11)}px;font-weight:bold;"
            f"color:{S.GREEN if ready and not stale else S.RED};")

    def clear(self, note=""):
        self.pose_lbl.setText("XYZ —")
        self.force_lbl.setText("— N")
        self.mode_lbl.setText(note)
        self.mode_lbl.setStyleSheet(f"font-size:{S.fpx(11)}px;color:#888888;")


class PairMonitor(QGroupBox):
    """The two-arm numbers: TCP separation, and how much it has wandered
    since the grip was captured."""

    def __init__(self):
        super().__init__("")
        self.setStyleSheet("QGroupBox{border:none;margin-top:0;}")
        g = QGridLayout(self)
        g.setContentsMargins(S.sx(4), 0, S.sx(4), 0)
        g.setHorizontalSpacing(S.sx(6))

        self.sep_lbl = S.value_label("—", font_px=15)
        self.drift_lbl = S.value_label("—", font_px=15)
        self.sep_lbl.setMinimumWidth(S.sx(80))
        self.drift_lbl.setMinimumWidth(S.sx(72))
        self.state_lbl = QLabel("not holding")
        self.state_lbl.setStyleSheet(f"font-size:{S.fpx(13)}px;color:#444444;")

        g.addWidget(S.caption("TCP gap"), 0, 0)
        g.addWidget(self.sep_lbl, 0, 1)
        g.addWidget(S.caption("drift"), 0, 2)
        g.addWidget(self.drift_lbl, 0, 3)
        g.addWidget(self.state_lbl, 0, 4)
        g.setColumnStretch(4, 1)

    def update_from(self, cell, held_object, drift_limit):
        separation = cell.tcp_separation()
        self.sep_lbl.setText("—" if separation is None
                             else "%.1f mm" % (separation * 1000))

        if held_object is None or not held_object.held \
                or held_object.captured_relative is None:
            self.drift_lbl.setText("—")
            self.drift_lbl.setStyleSheet(
                f"font-size:{S.fpx(15)}px;font-family:monospace;color:{S.INK};")
            self.state_lbl.setText("not holding")
            return

        now = cell.relative_transform()
        if now is None:
            return
        drift = float(((now[:3, 3] - held_object.captured_relative[:3, 3]) ** 2)
                      .sum() ** 0.5)
        hot = drift > drift_limit
        self.drift_lbl.setText("%.2f mm" % (drift * 1000))
        self.drift_lbl.setStyleSheet(
            f"font-size:{S.fpx(15)}px;font-family:monospace;"
            f"color:{S.RED if hot else S.GREEN};"
            + ("font-weight:bold;" if hot else ""))
        self.state_lbl.setText("holding '%s'" % held_object.name)
