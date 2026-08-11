"""
Points tab: the named places, and the buttons that teach them.

A point is taught by driving there and pressing a button — the arm's own
reading is the record, so a place is exactly where the arm could reach, not
where a drawing said it was. Object points are taught the same way from the
frame of whatever the pair is carrying.

The table is the whole left side because it is the thing being read; teaching
and deleting are four buttons down the right, where a finger lands without
covering the numbers.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QInputDialog, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from .. import style as S


class PointsPanel(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app

        body = QHBoxLayout(self)
        body.setContentsMargins(S.sx(6), S.sx(6), S.sx(6), S.sx(6))
        body.setSpacing(S.sx(8))
        body.addWidget(self._build_table(), 1)
        body.addWidget(self._build_actions(), 0)

    # ---- table -----------------------------------------------------------
    def _build_table(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(6))
        v.addWidget(S.strip("Named places"))

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(
            ["name", "x y z (mm)   rx ry rz (deg)"])
        self.table.verticalHeader().hide()
        self.table.setStyleSheet(f"font-size:{S.fpx(13)}px;")
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        v.addWidget(self.table, 1)
        return page

    # ---- actions ---------------------------------------------------------
    def _build_actions(self):
        page = QWidget()
        page.setFixedWidth(S.sx(250))
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(6))

        v.addWidget(S.strip("Teach where the arms are now"))
        self.teach_btns = {}
        for key, label, callback, colour in (
                ("A", "Teach A", lambda: self._teach_arm("A"), S.BLUE),
                ("B", "Teach B", lambda: self._teach_arm("B"), S.BLUE),
                ("object", "Teach object", self._teach_object, S.PURPLE)):
            button = S.touch_button(label, colour, height=50, font_px=14)
            button.clicked.connect(callback)
            v.addWidget(button)
            self.teach_btns[key] = button

        hint = S.caption("An arm point is that arm's TCP in world. An object "
                         "point is the carried frame, so both arms go back to "
                         "the same grip.")
        hint.setWordWrap(True)
        v.addWidget(hint)

        v.addWidget(S.strip("Library"))
        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        for label, callback, colour in (("Delete", self._delete, S.RED),
                                        ("Save", self._save, S.PURPLE)):
            button = S.touch_button(label, colour, height=46, font_px=13)
            button.clicked.connect(callback)
            row.addWidget(button, 1)
        v.addLayout(row)

        self.count_lbl = S.caption("")
        v.addWidget(self.count_lbl)
        v.addStretch(1)
        return page

    # ---- teaching --------------------------------------------------------
    def _ask_name(self, default):
        name, ok = QInputDialog.getText(self, "Teach", "Point name:",
                                        text=default)
        return name.strip() if ok and name.strip() else None

    def _teach_arm(self, arm_id):
        if not self.app.cell.arms[arm_id].connected:
            self.app.log("arm %s is not connected" % arm_id)
            return
        name = self._ask_name("%s_%d" % (arm_id, len(self.app.points.points) + 1))
        if name:
            self.app.points.teach_arm(self.app.cell, arm_id, name)
            self.app.log("taught %s from arm %s" % (name, arm_id))
            self._refresh_everywhere()

    def _teach_object(self):
        obj = self.app.executor.object
        if not obj.held:
            self.app.log("nothing is attached")
            return
        name = self._ask_name("%s_%d" % (obj.name, len(self.app.points.points) + 1))
        if name:
            self.app.points.teach_object(obj, name)
            self.app.log("taught %s from the object frame" % name)
            self._refresh_everywhere()

    def _delete(self):
        row = self.table.currentRow()
        names = self.app.points.names()
        if 0 <= row < len(names):
            self.app.points.remove(names[row])
            self._refresh_everywhere()

    def _save(self):
        path = self.app.save_points()
        self.app.log("saved %d points to %s"
                     % (len(self.app.points.points), path))

    def _refresh_everywhere(self):
        # the program's place picker is filled from this library, so a point
        # taught here has to reach it without the operator changing tab first
        self.refresh()
        self.app.panels["program"].refresh()

    # ---- lifecycle -------------------------------------------------------
    def refresh(self):
        names = self.app.points.names()
        self.table.setRowCount(len(names))
        for row, name in enumerate(names):
            pose = self.app.points.get(name)
            text = ("%7.1f %7.1f %7.1f    %6.1f %6.1f %6.1f"
                    % (pose[0] * 1000, pose[1] * 1000, pose[2] * 1000,
                       *np.degrees(pose[3:])))
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(text))
        self.count_lbl.setText("%d point%s" % (len(names),
                                               "" if len(names) == 1 else "s"))

    def tick(self):
        # a button that cannot do anything says so by being out, not by
        # failing after the name has been typed
        for arm_id in ("A", "B"):
            self.teach_btns[arm_id].setEnabled(
                self.app.cell.arms[arm_id].connected)
        self.teach_btns["object"].setEnabled(self.app.executor.object.held)
