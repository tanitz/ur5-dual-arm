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
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QInputDialog, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from .. import style as S


class PointsPanel(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app

        body = QVBoxLayout(self)
        body.setContentsMargins(S.sx(6), S.sx(6), S.sx(6), S.sx(6))
        body.setSpacing(S.sx(6))
        body.addWidget(self._build_table(), 1)
        body.addWidget(self._build_actions(), 0)

    # ---- table -----------------------------------------------------------
    def _build_table(self):
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(6))
        v.addWidget(S.strip("Named places"))

        # Only the position is on the row. All six numbers need about 45
        # characters and the sidebar is 320 px wide, so the three that a
        # taught place is picked by are shown and the rotation is on the
        # row's tooltip -- clipped numbers would be worse than named ones.
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["name", "x y z (mm)"])
        self.table.verticalHeader().hide()
        self.table.setStyleSheet(f"font-size:{S.fpx(12)}px;")
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        v.addWidget(self.table, 1)
        return page

    # ---- actions ---------------------------------------------------------
    def _build_actions(self):
        """Teaching and tidying, under the list.

        The sidebar is the width the jog keys need, and the program must not
        move when the two swap places -- so this page has the same 320 px and
        a column of buttons beside the table would leave the table 50 px. They
        go underneath, in two rows of three.
        """
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(4))

        top = QHBoxLayout()
        top.setSpacing(S.sx(4))
        self.teach_btns = {}
        # No "Teach obj": an object point is the frame of something both arms
        # are holding, and nothing on this panel can take hold any more.
        for key, label, callback, colour in (
                ("A", "Teach A", lambda: self._teach_arm("A"), S.BLUE),
                ("B", "Teach B", lambda: self._teach_arm("B"), S.BLUE)):
            button = S.touch_button(label, colour, height=44, font_px=13)
            button.setToolTip(
                "record where %s is now as a named place"
                % ("the carried object" if key == "object" else "arm " + key))
            button.clicked.connect(callback)
            top.addWidget(button, 1)
            self.teach_btns[key] = button
        v.addLayout(top)

        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        for label, callback, colour in (("✕ Delete", self._delete, S.RED),
                                        ("💾 Save", self._save, S.PURPLE)):
            button = S.touch_button(label, colour, height=40, font_px=13)
            button.clicked.connect(callback)
            row.addWidget(button, 1)
        self.count_lbl = S.caption("")
        self.count_lbl.setAlignment(Qt.AlignCenter)
        row.addWidget(self.count_lbl, 1)
        v.addLayout(row)
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
            text = ("%7.1f %7.1f %7.1f"
                    % (pose[0] * 1000, pose[1] * 1000, pose[2] * 1000))
            full = ("x y z   %7.1f %7.1f %7.1f  mm\n"
                    "rx ry rz %6.1f %6.1f %6.1f  deg"
                    % (pose[0] * 1000, pose[1] * 1000, pose[2] * 1000,
                       *np.degrees(pose[3:])))
            for column, value in ((0, name), (1, text)):
                item = QTableWidgetItem(value)
                item.setToolTip(full)
                self.table.setItem(row, column, item)
        self.count_lbl.setText("%d point%s" % (len(names),
                                               "" if len(names) == 1 else "s"))

    def tick(self):
        # a button that cannot do anything says so by being out, not by
        # failing after the name has been typed
        for arm_id in ("A", "B"):
            self.teach_btns[arm_id].setEnabled(
                self.app.cell.arms[arm_id].connected)
