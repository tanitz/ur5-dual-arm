#!/usr/bin/env python3
"""Ways of grouping the program panel's function buttons, drawn to scale.

There are twenty-two controls under the step table and they currently sit in
one eight-by-three grid of identical cells. Identical was the point — no button
beside another should be a smaller target — but identical also means nothing
tells the eye where one job ends and the next begins, and twenty-two of
anything is a lot to read while holding a pendant.

So these are drawn rather than argued. Every block below is the real function
area at the width it actually has (1198 px, the program zone with the sidebar
closed), with the real labels, so the count under each one is the number of
things an operator is actually looking at.

    python3 docs/drawings/make_function_row.py

writes function_groups.svg next to this file.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

ZONE_W = 1198              # the program zone, sidebar closed, less its padding
COLUMNS = 8
GAP = 4
BTN_H = 40
ROW_GAP = 4

PAD = 40
BLOCK_GAP = 30
TITLE_H = 54
NOTE_H = 34

GREEN, RED, AMBER, PURPLE, BLUE = ("#1a7a3a", "#aa2222", "#cc7700",
                                   "#5a3a8a", "#1a5fa8")
INK, DIM, LINE, FACE = "#0a58a8", "#666666", "#c4c4c4", "#ffffff"
PLAIN, SET_BG, GROUP_BG = "#ececec", "#f3f0e4", "#f7f9fb"

FONT = "-apple-system,Segoe UI,Roboto,DejaVu Sans,sans-serif"
MONO = "DejaVu Sans Mono,Consolas,monospace"

# every control, once, with the colour it carries today
RUN = [("▶ Run", GREEN), ("⏸ Pause", AMBER), ("■ Stop", RED)]
FILE = [("📂 Load", PURPLE), ("💾 Save", PURPLE)]
SETTINGS = [("🔁 Loop", None), ("Dry run", None), ("120 mm/s", None)]
WHO = [("A", None), ("⧉ A+B", None), ("B", None)]
RECORD = [("●  MOVEJ", BLUE), ("↗  MOVEL", BLUE)]
INSERT = [("insert ▾", None), ("＋", GREEN)]
STEP = [("▷ To", None), ("✏ Edit", None), ("⎘ Dup", None), ("✕ Del", None),
        ("On/off", None), ("↑", None), ("↓", None)]


class Sheet:
    def __init__(self, width, height):
        self.width, self.height, self.parts = width, height, []

    def rect(self, x, y, w, h, fill="none", stroke=None, sw=1.0, rx=0, dash=None):
        s = ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s"'
             % (x, y, max(w, 0), max(h, 0), fill))
        if stroke:
            s += ' stroke="%s" stroke-width="%.2f"' % (stroke, sw)
        if dash:
            s += ' stroke-dasharray="%s"' % dash
        if rx:
            s += ' rx="%.1f"' % rx
        self.parts.append(s + "/>")

    def line(self, x1, y1, x2, y2, stroke=LINE, sw=1.0, dash=None):
        s = ('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" '
             'stroke-width="%.2f"' % (x1, y1, x2, y2, stroke, sw))
        if dash:
            s += ' stroke-dasharray="%s"' % dash
        self.parts.append(s + "/>")

    def text(self, x, y, s, size=12, fill="#1a1a1a", weight="normal",
             anchor="start", font=FONT, spacing=None):
        t = ('<text x="%.1f" y="%.1f" font-family="%s" font-size="%.1f" '
             'fill="%s" font-weight="%s" text-anchor="%s"'
             % (x, y, font, size, fill, weight, anchor))
        if spacing:
            t += ' letter-spacing="%.2f"' % spacing
        self.parts.append(t + ">" + _esc(s) + "</text>")

    def svg(self, title):
        return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
                'viewBox="0 0 %d %d">\n<title>%s</title>\n%s\n</svg>\n'
                % (self.width, self.height, self.width, self.height,
                   _esc(title), "\n".join(self.parts)))


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def cell_w(span=1, columns=COLUMNS):
    unit = (ZONE_W - GAP * (columns - 1)) / float(columns)
    return unit * span + GAP * (span - 1)


def button(sh, x, y, w, label, colour=None, ghost=False, h=BTN_H, bg=None):
    fill = bg or (colour or PLAIN)
    if ghost:
        sh.rect(x, y, w, h, fill=FACE, stroke="#d8d8d8", rx=3, dash="4 3")
        sh.text(x + w / 2.0, y + h / 2.0 + 4, label, size=11.5, fill="#b6b6b6",
                weight="bold", anchor="middle")
        return
    sh.rect(x, y, w, h, fill=fill, stroke=colour or LINE, rx=3)
    sh.text(x + w / 2.0, y + h / 2.0 + 4, label, size=11.5,
            fill="#ffffff" if colour else "#1a1a1a", weight="bold",
            anchor="middle")


def row(sh, x, y, items, columns=COLUMNS, ghost=False):
    """Lay items out on one line of `columns` equal cells."""
    unit = (ZONE_W - GAP * (columns - 1)) / float(columns)
    for i, (label, colour) in enumerate(items):
        button(sh, x + i * (unit + GAP), y, unit, label, colour, ghost=ghost)
    return y + BTN_H + ROW_GAP


def caption(sh, x, y, text, colour=DIM):
    sh.text(x, y, text, size=9, fill=colour, weight="bold", spacing=0.9,
            font=MONO)


# ── the options ────────────────────────────────────────────────────────────
def draw_today(sh, x, y):
    y = row(sh, x, y, RUN + FILE + SETTINGS)
    y = row(sh, x, y, WHO + RECORD + [("insert ▾", None), ("", None),
                                      ("＋", GREEN)])
    y = row(sh, x, y, STEP + [("", None)])
    return y


def draw_labelled(sh, x, y):
    """Same buttons, told apart by a caption and a band."""
    unit = (ZONE_W - GAP * (COLUMNS - 1)) / float(COLUMNS)
    caption(sh, x, y - 4, "PROGRAM")
    caption(sh, x + 5 * (unit + GAP), y - 4, "HOW IT RUNS")
    sh.rect(x + 5 * (unit + GAP) - 3, y - 1, 3 * unit + 2 * GAP + 6,
            BTN_H + 2, fill=SET_BG, rx=3)
    y = row(sh, x, y, RUN + FILE + SETTINGS)
    y += 6
    caption(sh, x, y - 4, "RECORD  —  who, and what kind of move")
    y = row(sh, x, y, WHO + RECORD + [("insert ▾", None), ("", None),
                                      ("＋", GREEN)])
    y += 6
    caption(sh, x, y - 4, "THE SELECTED STEP")
    y = row(sh, x, y, STEP + [("", None)])
    return y


def draw_selection(sh, x, y):
    y = row(sh, x, y, RUN + FILE + SETTINGS)
    y = row(sh, x, y, WHO + RECORD + [("insert ▾", None), ("", None),
                                      ("＋", GREEN)])
    caption(sh, x, y + 12, "…AND THIS ROW ONLY WHEN A STEP IS SELECTED")
    y += 18
    y = row(sh, x, y, STEP + [("", None)], ghost=True)
    return y


def draw_segmented(sh, x, y):
    """One group at a time, behind a three-way selector."""
    unit = (ZONE_W - GAP * (COLUMNS - 1)) / float(COLUMNS)
    tabs = [("Program", True), ("Teach", False), ("Step", False)]
    tw = (ZONE_W - GAP * 2) / 3.0
    for i, (label, on) in enumerate(tabs):
        button(sh, x + i * (tw + GAP), y, tw, label,
               INK if on else None, h=34)
    y += 34 + ROW_GAP + 4
    y = row(sh, x, y, RUN + FILE + SETTINGS)
    sh.rect(x, y + 4, ZONE_W, BTN_H, fill=GROUP_BG, stroke="#dfe6e8", rx=3)
    sh.text(x + ZONE_W / 2.0, y + 4 + BTN_H / 2.0 + 4,
            "Teach and Step live behind their own tab — 8 buttons on screen "
            "instead of 22",
            size=11, fill=DIM, anchor="middle")
    return y + BTN_H + 8


def draw_overflow(sh, x, y):
    """The few that are pressed constantly, and a drawer for the rest."""
    core = [("▶ Run", GREEN), ("■ Stop", RED), ("●  MOVEJ", BLUE),
            ("↗  MOVEL", BLUE), ("insert ▾", None), ("＋", GREEN),
            ("▷ To", None), ("⋯ More", None)]
    y = row(sh, x, y, core)
    y += 6
    caption(sh, x, y + 8, "⋯ MORE OPENS")
    y += 14
    rest = [("⏸ Pause", AMBER), ("📂 Load", PURPLE), ("💾 Save", PURPLE),
            ("🔁 Loop", None), ("Dry run", None), ("120 mm/s", None),
            ("✏ Edit", None), ("⎘ Dup", None)]
    y = row(sh, x, y, rest, ghost=True)
    y = row(sh, x, y, [("✕ Del", None), ("On/off", None), ("↑", None),
                       ("↓", None), ("A", None), ("⧉ A+B", None),
                       ("B", None), ("", None)], ghost=True)
    return y


OPTIONS = [
    ("0", "Today — one grid of identical cells", draw_today, 3,
     "22 on screen. Every target the same size, and nothing says where one "
     "job ends and the next begins.", "22 visible"),
    ("A", "Labelled groups, nothing hidden", draw_labelled, 3,
     "The same 22 buttons with a caption over each group and a band behind "
     "the three that are settings rather than actions. Cheapest change, and "
     "the only one that hides nothing.", "22 visible"),
    ("B", "The step row appears when a step is selected", draw_selection, 3,
     "▷ To, Edit, Dup, Del, On/off and the arrows do nothing without a "
     "selected row, so they are not there until there is one. Nothing is "
     "permanently hidden and the resting panel is a third smaller.",
     "15 at rest, 22 with a step selected"),
    ("C", "One group at a time", draw_segmented, 3,
     "Program, Teach and Step become three pages of one selector. Fewest "
     "buttons on screen by far, and the price is a mode: the button you want "
     "is sometimes one press away instead of zero.", "8 visible"),
    ("D", "The constant few, and a drawer", draw_overflow, 3,
     "Run, Stop, the two record keys, insert and To are pressed all day; "
     "Load, Save, Loop, Dry run and the step edits are not. The drawer is "
     "one press, and what is in it is out of sight rather than out of reach.",
     "8 visible, 14 in the drawer"),
]


def main():
    heights = []
    for _letter, _name, _draw, rows, _note, _count in OPTIONS:
        heights.append(TITLE_H + rows * (BTN_H + ROW_GAP) + 40 + NOTE_H)
    total = PAD * 2 + sum(heights) + BLOCK_GAP * (len(OPTIONS) - 1) + 30

    sh = Sheet(ZONE_W + PAD * 2, int(total))
    sh.rect(0, 0, sh.width, sh.height, fill="#ffffff")
    sh.text(PAD, PAD + 8, "The function row", size=22, weight="bold")
    sh.text(PAD, PAD + 30,
            "twenty-two controls under the step table, five ways to group them "
            "— drawn at the real 1198 px width",
            size=12, fill=DIM)

    y = PAD + 56
    for letter, name, draw, _rows, note, count in OPTIONS:
        sh.line(PAD, y, PAD + ZONE_W, y, stroke="#aebcc2", sw=1.2)
        sh.text(PAD, y + 22, letter, size=17, weight="bold", fill=INK, font=MONO)
        sh.text(PAD + 26, y + 22, name, size=15, weight="bold")
        sh.text(PAD + ZONE_W, y + 22, count, size=11, fill=INK, anchor="end",
                font=MONO, weight="bold")
        y = draw(sh, PAD, y + 40)
        y += 12
        sh.text(PAD, y, note, size=11.5, fill="#3f4e56")
        y += BLOCK_GAP + 6

    sh.text(PAD, sh.height - 16,
            "dual UR5 cell  ·  gui/panels/program.py  ·  drawn at the program "
            "zone's own width with the sidebar closed",
            size=9.5, fill="#9aa8ad", font=MONO)

    path = os.path.join(HERE, "function_groups.svg")
    with open(path, "w") as f:
        f.write(sh.svg("Grouping the function row"))
    print("wrote %s" % os.path.basename(path))
    for letter, name, _d, _r, _n, count in OPTIONS:
        print("  %-2s %-46s %s" % (letter, name, count))


if __name__ == "__main__":
    main()
