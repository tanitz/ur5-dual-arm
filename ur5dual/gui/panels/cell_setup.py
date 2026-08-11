"""
Cell tab: where the arms are, and how to find out for real.

Two halves, because there are two different questions. The left half is the
mounting geometry as designed — four numbers off the drawing, which give a
model good enough to look at in RViz and to catch gross reach errors. The
right half is the geometry as built, measured by touching both TCPs to the
same physical point several times.

The second one is what two-arm grasping actually runs on. Drawing numbers are
good to a few millimetres, and a few millimetres of error between the bases
becomes a permanent fight between two arms holding one bottle.
"""

import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QDoubleSpinBox, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...geometry import calibration as C
from ...config import ARM_IDS
from .. import style as S


def _compass(v):
    """A unit vector as something an operator can check against the cell."""
    names = (("+X", (1, 0, 0)), ("-X", (-1, 0, 0)),
             ("+Y", (0, 1, 0)), ("-Y", (0, -1, 0)),
             ("up", (0, 0, 1)), ("down", (0, 0, -1)))
    best = max(names, key=lambda n: float(np.dot(v, n[1])))
    return "%s %s" % (best[0], np.round(v, 2))


class CellSetupPanel(QWidget):
    geometry_changed = pyqtSignal()

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.cal = C.BaseCalibration()

        body = QHBoxLayout(self)
        body.setContentsMargins(S.sx(6), S.sx(6), S.sx(6), S.sx(6))
        body.setSpacing(S.sx(8))
        body.addWidget(self._build_mount(), 1)
        body.addWidget(self._build_calibration(), 1)

    # ---- mounting geometry ----------------------------------------------
    def _build_mount(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(6))
        v.addWidget(S.strip("Mounting — as drawn"))

        grid = QGridLayout()
        grid.setSpacing(S.sx(4))
        self.spins = {}
        fields = (
            ("column_height", "column height", "m", 0.0, 3.0, 0.005),
            ("spacing", "shoulder spacing", "m", 0.0, 2.0, 0.005),
            # 0-180, because that is the range the number means: 0 stands the
            # arm up off a table, 90 lays the flange vertical, 180 hangs it
            # upside down. Clamping this to 90 silently truncated the 120 this
            # cell actually uses — the box showed 90, and the next edit of any
            # mount field wrote that 90 back over the real value, tipping the
            # world frame 30 deg without anyone touching it.
            ("tilt_deg", "outward tilt", "deg", 0.0, 180.0, 1.0),
            ("yaw_deg", "pair yaw", "deg", -180.0, 180.0, 1.0),
            # Which way each arm faces on its own flange — the number the
            # pendant calls "rotate" under Installation -> General -> Mounting.
            # apply_mount_preset has always honoured these, but nothing could
            # set them, so they stayed at zero and both arms were drawn facing
            # the same way whatever the cell actually looked like. An arm
            # bolted on a third of a turn round cannot be described without
            # them, and the drawing being wrong here is not cosmetic: every
            # world-frame target for that arm is computed through this
            # transform.
            ("rotate_deg_A", "arm A rotate", "deg", -180.0, 180.0, 5.0),
            ("rotate_deg_B", "arm B rotate", "deg", -180.0, 180.0, 5.0),
        )
        for row, (key, label, unit, lo, hi, step) in enumerate(fields):
            grid.addWidget(S.caption(label), row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setDecimals(3)
            spin.setMinimumHeight(S.sx(34))
            spin.setStyleSheet(S.field())
            spin.valueChanged.connect(self._mount_edited)
            grid.addWidget(spin, row, 1)
            grid.addWidget(S.caption(unit), row, 2)
            self.spins[key] = spin
        grid.setColumnStretch(1, 1)
        v.addLayout(grid)

        self.ip_edits = {}
        v.addWidget(S.strip("Controllers"))
        ips = QGridLayout()
        ips.setSpacing(S.sx(4))
        for row, arm_id in enumerate(ARM_IDS):
            ips.addWidget(S.caption("arm %s" % arm_id), row, 0)
            edit = QLineEdit()
            edit.setMinimumHeight(S.sx(34))
            edit.setStyleSheet(S.field())
            ips.addWidget(edit, row, 1)
            self.ip_edits[arm_id] = edit
            enable = S.touch_button("enabled", height=34, font_px=12,
                                    checkable=True)
            ips.addWidget(enable, row, 2)
            self.ip_edits[arm_id + "_enabled"] = enable
        ips.setColumnStretch(1, 1)
        v.addLayout(ips)

        self.derived = QLabel("")
        self.derived.setWordWrap(True)
        self.derived.setStyleSheet(f"font-size:{S.fpx(12)}px;color:#444444;"
                                   f"font-family:monospace;")
        v.addWidget(self.derived)

        buttons = QHBoxLayout()
        buttons.setSpacing(S.sx(4))
        self.apply_btn = S.touch_button("Apply + save", S.GREEN, height=48)
        self.apply_btn.clicked.connect(self._apply_mount)
        self.reload_btn = S.touch_button("Reload", height=48)
        self.reload_btn.clicked.connect(self.refresh)
        buttons.addWidget(self.apply_btn, 2)
        buttons.addWidget(self.reload_btn, 1)
        v.addLayout(buttons)
        v.addStretch(1)
        return page

    def _mount_edited(self):
        cfg = self.app.cell.config
        for key, spin in self.spins.items():
            cfg.mount[key] = float(spin.value())
        if cfg.mount["style"] != "pedestal":
            self.derived.setText(
                "geometry is measured (custom) — editing these numbers has no "
                "effect until you press Apply, which overwrites the measurement")
            return
        cfg.apply_mount_preset()
        self._show_derived()

    def _show_derived(self):
        cfg = self.app.cell.config
        lines = []
        for arm_id in ARM_IDS:
            arm = cfg.arms[arm_id]
            lines.append("%s  xyz %7.3f %7.3f %7.3f   rpy %7.2f %7.2f %7.2f deg"
                         % (arm_id, arm.xyz[0], arm.xyz[1], arm.xyz[2],
                            *np.degrees(arm.rpy)))
        lines.append("base separation %.1f mm   [%s]"
                     % (cfg.base_separation() * 1000, cfg.mount["style"]))

        # Which way each arm faces, in words. Three rpy angles do not tell
        # anyone standing at the cell whether the drawing matches it, and
        # "both arms are drawn reaching the same way" is a thing you can check
        # against the real robots in a second — where reading a roll of
        # -120 degrees off a line of numbers is not.
        facing = {}
        for arm_id in ARM_IDS:
            facing[arm_id] = cfg.arms[arm_id].base_matrix()[:3, :3] @ [1.0, 0, 0]
        for arm_id in ARM_IDS:
            lines.append("%s  reaches towards %s" % (arm_id, _compass(facing[arm_id])))
        agree = float(np.dot(facing[ARM_IDS[0]], facing[ARM_IDS[1]]))
        if agree > 0.9:
            lines.append("both arms face the SAME way — for a facing pair, set "
                         "one arm's rotate to 180")
        self.derived.setText("\n".join(lines))

    def _apply_mount(self):
        cfg = self.app.cell.config
        for key, spin in self.spins.items():
            cfg.mount[key] = float(spin.value())
        if cfg.mount["style"] == "custom":
            cfg.mount["style"] = "pedestal"      # Apply means "use the drawing"
        cfg.apply_mount_preset()
        for arm_id in ARM_IDS:
            cfg.arms[arm_id].ip = self.ip_edits[arm_id].text().strip()
            cfg.arms[arm_id].enabled = self.ip_edits[arm_id + "_enabled"].isChecked()
        cfg.save()
        self.app.cell.reload_config(cfg)
        self._show_derived()
        self.app.log("cell geometry saved to %s" % cfg.path)
        self.geometry_changed.emit()

    # ---- calibration -----------------------------------------------------
    def _build_calibration(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(6))
        v.addWidget(S.strip("Base calibration — as built"))

        how = QLabel(
            "Touch both TCPs to the same physical point, then Record. Repeat "
            "at %d well-spread points — far apart, and at different heights, "
            "or the rotation is guesswork." % C.RECOMMENDED_POINTS)
        how.setWordWrap(True)
        how.setStyleSheet(f"font-size:{S.fpx(12)}px;color:#444444;")
        v.addWidget(how)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["point", "arm A (mm)", "arm B (mm)"])
        self.table.verticalHeader().hide()
        self.table.setStyleSheet(f"font-size:{S.fpx(12)}px;")
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        v.addWidget(self.table, 1)

        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        self.record_btn = S.touch_button("Record point", S.BLUE, height=46)
        self.record_btn.clicked.connect(self._record)
        self.drop_btn = S.touch_button("Delete", S.RED, height=46)
        self.drop_btn.clicked.connect(self._drop)
        row.addWidget(self.record_btn, 2)
        row.addWidget(self.drop_btn, 1)
        v.addLayout(row)

        self.result = QLabel("no solution yet")
        self.result.setWordWrap(True)
        self.result.setStyleSheet(f"font-size:{S.fpx(12)}px;"
                                  f"font-family:monospace;color:#444444;")
        v.addWidget(self.result)

        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        self.solve_btn = S.touch_button("Solve", height=46)
        self.solve_btn.clicked.connect(self._solve)
        self.use_btn = S.touch_button("Use this geometry", S.GREEN, height=46)
        self.use_btn.clicked.connect(self._apply_calibration)
        row.addWidget(self.solve_btn, 1)
        row.addWidget(self.use_btn, 2)
        v.addLayout(row)
        return page

    def _record(self):
        try:
            point = self.cal.add_from_cell(self.app.cell)
        except C.CalibrationError as e:
            self.app.log("calibration: %s" % e)
            return
        self._refresh_table()
        self.app.log("recorded %s" % point.name)

    def _drop(self):
        self.cal.remove(self.table.currentRow())
        self._refresh_table()

    def _refresh_table(self):
        self.table.setRowCount(len(self.cal.points))
        for row, point in enumerate(self.cal.points):
            cells = (point.name,
                     " ".join("%7.1f" % (v * 1000) for v in point.p_a),
                     " ".join("%7.1f" % (v * 1000) for v in point.p_b))
            for col, text in enumerate(cells):
                self.table.setItem(row, col, QTableWidgetItem(text))

    def _solve(self):
        try:
            transform, report = self.cal.solve()
        except C.CalibrationError as e:
            self.result.setText(str(e))
            return None
        lines = ["A base -> B base:  " + C.describe(transform),
                 "%d points   rms %.2f mm   worst %.2f mm   spread %.0f mm"
                 % (report["points"], report["rms_mm"], report["max_mm"],
                    report["spread_mm"])]
        lines += ["! " + w for w in report["warnings"]]
        self.result.setText("\n".join(lines))
        self.result.setStyleSheet(
            f"font-size:{S.fpx(12)}px;font-family:monospace;"
            f"color:{S.RED if report['warnings'] else S.GREEN};")
        return transform

    def _apply_calibration(self):
        if self._solve() is None:
            return
        cfg = self.app.cell.config
        self.cal.apply_to_config(cfg)
        cfg.save()
        self.app.cell.reload_config(cfg)
        self._show_derived()
        self.app.log("arm B geometry replaced by the measurement — "
                     "mount style is now 'custom'")
        self.geometry_changed.emit()

    # ---- lifecycle -------------------------------------------------------
    def refresh(self):
        cfg = self.app.cell.config
        for key, spin in self.spins.items():
            # the per-arm overrides are allowed to be null in cell.yaml, which
            # means "use the shared figure" — a spin box has no way to show
            # that, and float(None) is a crash on opening the tab
            value = cfg.mount.get(key)
            spin.blockSignals(True)
            spin.setValue(float(0.0 if value is None else value))
            spin.blockSignals(False)
        for arm_id in ARM_IDS:
            self.ip_edits[arm_id].setText(cfg.arms[arm_id].ip)
            self.ip_edits[arm_id + "_enabled"].setChecked(cfg.arms[arm_id].enabled)
        self._show_derived()

    def tick(self):
        both = len(self.app.cell.connected_ids) >= 2
        self.record_btn.setEnabled(both)
        if not both:
            self.record_btn.setToolTip("both arms must be connected to touch "
                                       "off the same point")
