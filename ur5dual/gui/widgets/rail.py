"""
The icon rail, and the sidebar it opens.

The panel used to be two fixed halves: the program on the left and a tab
column on the right. Two columns of targets do not fit that left half — five
columns share 350 px and each arm gets 86, which is an ellipsis rather than a
column — so the tab column becomes a sidebar the operator can close, and the
program takes whatever is left.

The rail is what makes that safe to do. It is one finger wide, it is never
hidden, and it carries the four panels as icons: the way back is never behind
the thing that took it away, which is the one real flaw in a bare "wide"
button. The lit icon is the open panel; pressing it again closes the sidebar
and leaves the rail.

Below the icons sit the two window buttons: ⬚ fills the screen or gives the
frame back, and ⇄ moves the rail to the other edge. Fullscreen hides the
title bar, so the button that undoes it has to live inside the window.

Either edge will do. Jogging is a two-handed job on a pendant and which hand
reaches the keys is the operator's business, so `ui.sidebar_side` moves the
rail and its panel to the other side and stays moved.

One rule is not cosmetic: closing or moving the sidebar has to release
whatever jog button is held, exactly as leaving the tab did. A held key whose
button has left the screen must not go on driving an arm.
"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from .. import style as S

RAIL_W = 56          # one finger, and the same 56 the drawings are dimensioned to
ICON_H = 52

# panel id, glyph, and the label under it. Order is the order they were tabs in.
#
# Vars and Object were dropped from the rail: the cell's geometry is measured
# with the command-line tools rather than typed at the pendant, and taking hold
# is an ATTACH step in a program rather than a button. Both panels still exist
# in gui/panels/ and neither is imported -- putting either back is one line
# here and one in app._build.
RAIL_ITEMS = (
    ("points", "◎", "Pts"),
    ("camera", "◉", "Cam"),
    ("jog", "✥", "Jog"),
)


class IconRail(QWidget):
    """The always-visible strip of panel icons, a toggle, and a side swap."""

    # (panel id) — asked to show this panel
    selected = pyqtSignal(str)
    # asked to open or close whatever is selected
    toggled = pyqtSignal()
    # asked to move to the other edge
    swapped = pyqtSignal()
    # asked to fill the screen, or to give the window frame back
    fullscreen = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFixedWidth(S.sx(RAIL_W))
        v = QVBoxLayout(self)
        v.setContentsMargins(S.sx(4), S.sx(4), S.sx(4), S.sx(4))
        v.setSpacing(S.sx(4))

        self.toggle_btn = S.touch_button("◀", height=34, font_px=15)
        self.toggle_btn.setToolTip("open or close the panel beside the rail")
        self.toggle_btn.clicked.connect(self.toggled.emit)
        v.addWidget(self.toggle_btn)

        self.buttons = {}
        for panel_id, glyph, label in RAIL_ITEMS:
            button = S.touch_button("%s\n%s" % (glyph, label), height=ICON_H,
                                    font_px=13)
            button.setToolTip(label)
            button.clicked.connect(
                lambda _checked=False, p=panel_id: self.selected.emit(p))
            v.addWidget(button)
            self.buttons[panel_id] = button

        v.addStretch(1)
        # The way out of fullscreen has to be on the screen: fullscreen takes
        # the title bar with it, and this panel is a touchscreen with no
        # keyboard in reach of it.
        self.full_btn = S.touch_button("⬚", height=34, font_px=15)
        self.full_btn.setToolTip("fill the screen, or show the window frame")
        self.full_btn.clicked.connect(self.fullscreen.emit)
        v.addWidget(self.full_btn)

        self.swap_btn = S.touch_button("⇄", height=34, font_px=15)
        self.swap_btn.setToolTip("move the rail and its panel to the other side")
        self.swap_btn.clicked.connect(self.swapped.emit)
        v.addWidget(self.swap_btn)

    def show_fullscreen(self, is_full):
        """Say what the button would do, not what the window is doing."""
        self.full_btn.setText("❐" if is_full else "⬚")
        self.full_btn.setToolTip("show the window frame" if is_full
                                 else "fill the screen")

    def show_state(self, panel_id, is_open, side):
        """Light the open panel, and point the toggle the way it would move.

        A closed sidebar still shows which panel *would* open, in outline
        rather than filled — pressing the same icon twice should be a toggle
        the operator can predict, not a guess.
        """
        for pid, button in self.buttons.items():
            chosen = pid == panel_id
            if chosen and is_open:
                button.setStyleSheet(S.rail_icon(S.INK, filled=True))
            elif chosen:
                button.setStyleSheet(S.rail_icon(S.INK, filled=False))
            else:
                button.setStyleSheet(S.rail_icon(None))
        # the arrow says where the edge of the panel goes, not where it is
        closing = "◀" if side == "right" else "▶"
        opening = "▶" if side == "right" else "◀"
        self.toggle_btn.setText(closing if is_open else opening)
