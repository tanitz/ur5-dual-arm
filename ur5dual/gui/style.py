"""
One place for every pixel size and colour in the GUI.

The layout is designed at 1280x800 — the panel it runs on — and scaled from
there, so `sx(48)` is 48 px on the panel and grows or shrinks with
UR5DUAL_UI_SCALE everywhere else. Nothing in the panels hard-codes a pixel.
"""

import os

from PyQt5.QtWidgets import QLabel, QPushButton

UI_SCALE = float(os.environ.get("UR5DUAL_UI_SCALE", 1.0))

# arm identity carries a colour the whole app agrees on, so a number on the
# monitor and a button in the jog grid are obviously the same arm
ARM_COLOR = {"A": "#1a5fa8", "B": "#8a4a1a"}
ARM_TINT = {"A": "#e6eef8", "B": "#f8ece4"}

GREEN = "#1a7a3a"
RED = "#aa2222"
AMBER = "#cc7700"
PURPLE = "#5a3a8a"
BLUE = "#2a5a8a"
INK = "#0a58a8"


def set_scale(f):
    global UI_SCALE
    UI_SCALE = max(0.4, min(2.0, float(f)))
    return UI_SCALE


def sx(n):
    return max(1, int(round(n * UI_SCALE)))


def fpx(n):
    return max(7, int(round(n * UI_SCALE)))


def solid(bg, font_px=None, radius=6):
    return (f"QPushButton{{background:{bg};color:white;font-weight:bold;"
            f"font-size:{fpx(font_px or 15)}px;border-radius:{sx(radius)}px;}}"
            f"QPushButton:disabled{{background:#999;color:#ddd;}}")


def pill(font_px=13):
    return (f"QPushButton{{background:#ffffff;border:1px solid #b0b0b0;"
            f"border-radius:{sx(10)}px;font-size:{fpx(font_px)}px;"
            f"font-weight:bold;padding:0 {sx(6)}px;}}"
            f"QPushButton:pressed{{background:#dbe9f7;}}"
            f"QPushButton:checked{{background:{GREEN};color:white;}}"
            f"QPushButton:disabled{{color:#aaaaaa;background:#fbfbfb;}}")


def jog_button():
    return (f"QPushButton{{font-size:{fpx(22)}px;font-weight:bold;"
            f"background:#ffffff;border:1px solid #9a9a9a;"
            f"border-radius:{sx(6)}px;}}"
            f"QPushButton:pressed{{background:#c8dcf0;}}"
            f"QPushButton:disabled{{background:#f0f0f0;color:#c0c0c0;"
            f"border:1px solid #d0d0d0;}}")


def combo():
    return (f"QComboBox{{font-size:{fpx(15)}px;background:#ffffff;color:#000000;"
            f"border:1px solid #b0b0b0;border-radius:{sx(4)}px;"
            f"padding-left:{sx(8)}px;}}"
            f"QComboBox QAbstractItemView{{background:#ffffff;color:#000000;"
            f"selection-background-color:#dbe9f7;selection-color:#000000;}}")


def field():
    return (f"QLineEdit,QDoubleSpinBox,QSpinBox{{font-size:{fpx(14)}px;"
            f"background:#ffffff;border:1px solid #b0b0b0;"
            f"border-radius:{sx(4)}px;padding:{sx(2)}px {sx(6)}px;}}")


def tabs():
    return (f"QTabBar::tab{{height:{sx(36)}px;min-width:{sx(96)}px;"
            f"font-size:{fpx(15)}px;font-weight:bold;background:#ffffff;"
            f"border:1px solid #b0b0b0;border-bottom:none;color:#000000;}}"
            f"QTabBar::tab:selected{{background:{GREEN};color:#ffffff;"
            f"border:1px solid #b0b0b0;}}"
            f"QTabWidget::pane{{border:1px solid #b0b0b0;}}")


def strip(text, color="#e3f3e3"):
    """Section header: a soft band behind the title, no card frame."""
    label = QLabel(text)
    label.setStyleSheet(f"background:{color};border-radius:{sx(4)}px;"
                        f"padding:{sx(4)}px {sx(6)}px;font-size:{fpx(14)}px;")
    return label


def arm_strip(arm_id, text):
    return strip(text, ARM_TINT[arm_id])


def touch_button(text, bg=None, height=44, font_px=15, checkable=False):
    b = QPushButton(text)
    b.setMinimumHeight(sx(height))
    b.setCheckable(checkable)
    b.setStyleSheet(solid(bg, font_px) if bg else pill(font_px))
    return b


def value_label(text="—", color=INK, font_px=14, mono=True):
    label = QLabel(text)
    label.setStyleSheet(f"font-size:{fpx(font_px)}px;color:{color};"
                        + ("font-family:monospace;" if mono else ""))
    return label


def caption(text):
    label = QLabel(text)
    label.setStyleSheet(f"font-size:{fpx(13)}px;color:#444444;")
    return label
