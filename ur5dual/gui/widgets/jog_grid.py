"""
The six-row -/+ grid, shared by every jog target.

Switching between arm A, arm B and the held object swaps the row labels and
where the press is routed. The geometry never moves, so the finger that found
"+Z" last time finds it again.

Buttons give real press and release events, which is why this can hold a
speed command while pressed and stop the instant it is let go — the terminal
jog has to guess from key auto-repeat.
"""

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QFrame, QGridLayout, QLabel, QSizePolicy

from .. import style as S


class JogGrid(QFrame):
    """Emits pressed(row, sign) then repeatedly ticked(row, sign) while held,
    and released() when let go."""

    pressed_axis = pyqtSignal(int, int)
    ticked = pyqtSignal(int, int)
    released_axis = pyqtSignal()

    def __init__(self, labels=("X", "Y", "Z", "RX", "RY", "RZ")):
        super().__init__()
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(S.sx(4))
        grid.setHorizontalSpacing(S.sx(6))

        self.axis_labels, self.buttons, self.rows = [], [], []
        for row in range(6):
            label = QLabel(labels[row])
            label.setStyleSheet(f"font-size:{S.fpx(19)}px;font-weight:bold;")
            label.setFixedWidth(S.sx(52))
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(label, row, 0)
            self.axis_labels.append(label)
            row_widgets = [label]

            for col, (sign, text) in enumerate(((-1, "−"), (+1, "+"))):
                from PyQt5.QtWidgets import QPushButton
                button = QPushButton(text)
                button.setAutoRepeat(False)
                button.setMinimumHeight(S.sx(46))
                button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                button.setStyleSheet(S.jog_button())
                button.pressed.connect(lambda r=row, s=sign: self._press(r, s))
                button.released.connect(self.release)
                grid.addWidget(button, row, col + 1)
                self.buttons.append(button)
                row_widgets.append(button)

            self.rows.append(row_widgets)

        self.held = None
        self.axis_count = len(self.rows)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    # -- appearance --------------------------------------------------------
    def set_labels(self, labels):
        for label, text in zip(self.axis_labels, labels):
            label.setText(text)

    def set_axis_count(self, n):
        """Show only the first n axes.

        An axis a target cannot move is worse than absent: a greyed-out RX
        invites a press, and then has to explain itself. A held object shows
        three until a touch-off has measured the distance between the two
        bases, and six afterwards — so this is called again when that lands,
        and the extra rows simply appear.
        """
        self.release()
        self.axis_count = int(n)
        for i, widgets in enumerate(self.rows):
            for w in widgets:
                w.setVisible(i < n)

    def set_visible_axes(self, axes):
        """Show an arbitrary subset of the six rows."""
        self.release()
        visible = {int(axis) for axis in axes}
        self.axis_count = len(visible)
        for i, widgets in enumerate(self.rows):
            for w in widgets:
                w.setVisible(i in visible)

    def set_enabled(self, enabled):
        for button in self.buttons:
            button.setEnabled(enabled)
        if not enabled:
            self.release()

    def set_tint(self, colour):
        css = S.jog_button().replace("background:#ffffff",
                                     "background:%s" % colour)
        for button in self.buttons:
            button.setStyleSheet(css)

    # -- press / hold / release -------------------------------------------
    def set_interval(self, seconds):
        self._interval_ms = max(20, int(seconds * 1000))

    def _press(self, row, sign):
        if self.held is not None:
            return
        self.held = (row, sign)
        self.pressed_axis.emit(row, sign)
        self.timer.start(getattr(self, "_interval_ms", 80))

    def _tick(self):
        if self.held is None:
            return
        self.ticked.emit(*self.held)

    def release(self):
        if self.held is None:
            return
        self.timer.stop()
        self.held = None
        self.released_axis.emit()
