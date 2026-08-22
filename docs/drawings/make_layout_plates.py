#!/usr/bin/env python3
"""Scale drawings of the panel, one per candidate layout for the two-arm line.

Two columns of targets do not fit the program zone as it stands, and the whole
decision is how much room each way of finding the width actually leaves. So
these are drawn rather than argued: every plate is the real panel at its
1280x800 design size, built from the same constants the panel is built from --
`app.PROGRAM_COL_W`, the 250 px editor, the 6 px gaps -- and the step text is
clipped at the true column edge instead of being written to fit.

    python3 docs/drawings/make_layout_plates.py

writes layout_0_today.svg and layout_a..h.svg next to this file. They are
images, so anything that shows a picture will show them; the numbers below each
drawing are the widths the layout produces, not a caption on it.

See docs/program_line_design.md for what the line itself is.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))

# -- the panel's own numbers, from gui/app.py and gui/style.py --------------
W, H = 1280, 800           # style.DESIGN_W, DESIGN_H
MARGIN = 4                 # app._build outer contents margins
GAP = 6                    # spacing between the program zone and the tabs
PROGRAM_COL_W = 620        # app.PROGRAM_COL_W
EDITOR_W = 250             # panels/program.py, _build_editor maximum width
ZONE_PAD = 6               # the program page's own contents margins
ZONE_GAP = 8               # between the step table and the editor

# the two columns that do not stretch, and the one that would be added
COL_IDX, COL_KIND, COL_LINK = 28, 62, 90

# -- colours, from gui/style.py --------------------------------------------
A_COL, A_TINT = "#1a5fa8", "#e6eef8"
B_COL, B_TINT = "#8a4a1a", "#f8ece4"
GREEN, RED, AMBER, PURPLE, INK = "#1a7a3a", "#aa2222", "#cc7700", "#5a3a8a", "#0a58a8"
FACE, BG, LINE, LINE2, DIM = "#ffffff", "#f2f2f2", "#c4c4c4", "#dcdcdc", "#666666"
STRIP, SPAN_BG = "#e3f3e3", "#f0f0ee"

RAIL_W = 56                # the icon rail: one finger wide, always visible

SHEET_PAD = 40             # white space around the panel on the plate
TITLE_H = 92               # the title block above it
DIMS_H = 78                # the dimension callouts below it

FONT = "-apple-system,Segoe UI,Roboto,DejaVu Sans,sans-serif"
MONO = "DejaVu Sans Mono,Consolas,monospace"

# the same ten lines in every drawing, so only the width differs
STEPS = [
    (1,  "MOVE",   "home_A  movej", "home_B  movej", "⇉ together"),
    (2,  "MOVE",   "pick + world Z+50.0  movel",
                   "pick + world Z+50.0  movel", "⇉ together"),
    (3,  "MOVE",   "world Z-50.0  movel", "world Z-50.0  movel", "⇉ together"),
    (4,  "GRIP",   "close DO0", "close DO0", "⇉"),
    (5,  "MOVE",   None, "world Z+100.0", "⇉⇉ pair"),
    (6,  "ATTACH", None, "take hold of 'box' (origin midpoint)", ""),
    (7,  "MOVE",   None, "⟨obj⟩ place_1", "⛓ coupled"),
    (8,  "MOVE",   None, "⟨obj⟩ object RZ+45.0", "⛓ coupled"),
    (9,  "DETACH", None, "let go", ""),
    (10, "MOVE",   "home_A  movej", "·", ""),
]

ROW_H = 21


class Sheet:
    """An SVG being written. Only what these plates need."""

    def __init__(self, width, height):
        self.width, self.height = width, height
        self.parts = []
        self.clips = []
        self._n = 0

    def rect(self, x, y, w, h, fill="none", stroke=None, sw=1.0, rx=0):
        s = '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s"' % (
            x, y, max(w, 0), max(h, 0), fill)
        if stroke:
            s += ' stroke="%s" stroke-width="%.2f"' % (stroke, sw)
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
             anchor="start", font=FONT, spacing=None, clip=None):
        t = ('<text x="%.1f" y="%.1f" font-family="%s" font-size="%.1f" '
             'fill="%s" font-weight="%s" text-anchor="%s"'
             % (x, y, font, size, fill, weight, anchor))
        if spacing:
            t += ' letter-spacing="%.2f"' % spacing
        if clip:
            t += ' clip-path="url(#%s)"' % clip
        self.parts.append(t + ">" + _esc(s) + "</text>")

    def clip(self, x, y, w, h):
        """A real clip, so a column that is too narrow is drawn too narrow."""
        self._n += 1
        name = "c%d" % self._n
        self.clips.append(
            '<clipPath id="%s"><rect x="%.1f" y="%.1f" width="%.1f" '
            'height="%.1f"/></clipPath>' % (name, x, y, max(w, 0), h))
        return name

    def svg(self, title):
        return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
                'viewBox="0 0 %d %d">\n<title>%s</title>\n<defs>%s</defs>\n%s\n</svg>\n'
                % (self.width, self.height, self.width, self.height, _esc(title),
                   "".join(self.clips), "\n".join(self.parts)))


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def geometry(opt):
    """Where the width goes, for one layout. The only arithmetic on the page."""
    work = W - 2 * MARGIN
    if opt.get("rail"):
        # the rail is always there, the sidebar is what the toggle removes, and
        # the program takes whatever is left either way
        side = opt.get("sidebar_w", 0)
        prog_w = work - RAIL_W - GAP - ((side + GAP) if side else 0)
        right_w = 0
    else:
        prog_w = PROGRAM_COL_W if opt["tabs"] else work
        right_w = work - GAP - prog_w if opt["tabs"] else 0
    inner = prog_w - 2 * ZONE_PAD
    table_w = inner - ZONE_GAP - EDITOR_W if opt["editor"] else inner
    kind = 90 if opt.get("stacked") else COL_KIND
    link = 0 if opt.get("stacked") else COL_LINK
    detail = table_w - COL_IDX - kind - link
    per_arm = detail if opt.get("stacked") else detail / 2.0
    return {"work": work, "prog_w": prog_w, "right_w": right_w, "inner": inner,
            "sidebar_w": opt.get("sidebar_w", 0),
            "table_w": table_w, "kind": kind, "link": link, "detail": detail,
            "per_arm": int(round(per_arm)), "chars": int(round(per_arm / 7.0))}


# -- the pieces of the panel ------------------------------------------------
def button(sh, x, y, w, h, label, fill="#ececec", fg="#1a1a1a", size=11,
           stroke=LINE, weight="bold"):
    sh.rect(x, y, w, h, fill=fill, stroke=stroke, rx=2)
    clip = sh.clip(x + 2, y, w - 4, h)
    sh.text(x + w / 2.0, y + h / 2.0 + size * 0.36, label, size=size, fill=fg,
            weight=weight, anchor="middle", clip=clip)


def safety_bar(sh, x, y, w, h=40):
    """REAL, the dashboard controls and the cell-wide STOP. Never hidden."""
    pills = [("● REAL", RED, "#ffffff", 1.2), ("A", "#ececec", "#1a1a1a", 1),
             ("B", "#ececec", "#1a1a1a", 1), ("Power", "#ececec", "#1a1a1a", 1),
             ("Brake", "#ececec", "#1a1a1a", 1), ("Unlock", "#ececec", "#1a1a1a", 1),
             ("■ STOP", RED, "#ffffff", 2)]
    total = sum(p[3] for p in pills)
    avail = w - 4 * (len(pills) - 1)
    cx = x
    for label, fill, fg, weight in pills:
        pw = avail * weight / total
        button(sh, cx, y, pw, h, label, fill=fill, fg=fg,
               size=13 if "STOP" in label else 11)
        cx += pw + 4


def step_table(sh, x, y, w, h, g, opt):
    """The program, clipped where the real columns clip it."""
    sh.rect(x, y, w, h, fill=FACE, stroke=LINE2)
    idx_w, kind_w, link_w = COL_IDX, g["kind"], g["link"]
    detail_x = x + idx_w + kind_w
    arm_w = g["per_arm"]

    # header
    sh.line(x, y + 17, x + w, y + 17, stroke=LINE)
    heads = ([("#", x + idx_w - 4, "end"), ("step", x + idx_w + 4, "start")] +
             ([("arm / detail", detail_x + 4, "start")] if opt.get("stacked") else
              [("Arm A", detail_x + 4, "start"),
               ("Arm B", detail_x + arm_w + 4, "start"),
               ("link", x + w - link_w + 4, "start")]))
    for label, hx, anchor in heads:
        sh.text(hx, y + 12, label, size=9.5, fill=DIM, weight="bold", anchor=anchor)

    ry = y + 17
    for n, kind, a, b, link in STEPS:
        spanning = a is None
        rows = 1 if (spanning or opt.get("stacked") is not True) else 2
        if opt.get("stacked") and not spanning:
            rows = 2
        block_h = ROW_H * rows
        if ry + block_h > y + h:
            break

        if n == 7:                      # the selected line, as the panel shows it
            sh.rect(x + 1, ry, w - 2, block_h, fill="#f4fbf4")

        sh.text(x + idx_w - 4, ry + 14, str(n), size=10, fill=DIM, anchor="end")
        kind_label = kind
        if opt.get("stacked") and link:
            kind_label = "%s %s" % (kind, link.split(" ")[0])
        kc = sh.clip(x + idx_w, ry, kind_w - 2, block_h)
        sh.text(x + idx_w + 4, ry + 14, kind_label, size=10, weight="bold", clip=kc)

        if spanning:
            span_w = w - idx_w - kind_w - link_w
            sh.rect(detail_x, ry, span_w, ROW_H, fill=SPAN_BG)
            c = sh.clip(detail_x, ry, span_w - 4, ROW_H)
            sh.text(detail_x + 5, ry + 14, b, size=10.5, clip=c)
            if link and not opt.get("stacked"):
                lc = sh.clip(x + w - link_w, ry, link_w - 4, ROW_H)
                sh.text(x + w - link_w + 4, ry + 14, link, size=9.5, fill=DIM, clip=lc)
            elif link and opt.get("stacked"):
                sh.text(detail_x + span_w + link_w - 5, ry + 14, link, size=9.5,
                        fill=DIM, anchor="end")
        elif opt.get("stacked"):
            for i, (who, txt, tint, col) in enumerate(
                    (("A", a, A_TINT, A_COL), ("B", b, B_TINT, B_COL))):
                yy = ry + i * ROW_H
                sh.rect(detail_x, yy, w - idx_w - kind_w, ROW_H, fill=tint)
                c = sh.clip(detail_x, yy, w - idx_w - kind_w - 4, ROW_H)
                sh.text(detail_x + 5, yy + 14, who, size=10, fill=col,
                        weight="bold", clip=c)
                sh.text(detail_x + 18, yy + 14, txt, size=10.5, clip=c)
                sh.line(x, yy + ROW_H, x + w, yy + ROW_H, stroke="#eeeeee")
        else:
            for i, (txt, tint) in enumerate(((a, A_TINT), (b, B_TINT))):
                cx = detail_x + i * arm_w
                sh.rect(cx, ry, arm_w, ROW_H, fill=tint)
                c = sh.clip(cx, ry, arm_w - 4, ROW_H)
                sh.text(cx + 4, ry + 14, txt, size=10.5, clip=c)
            lc = sh.clip(x + w - link_w, ry, link_w - 4, ROW_H)
            sh.text(x + w - link_w + 4, ry + 14, link, size=9.5, fill=DIM, clip=lc)

        sh.line(x, ry + block_h, x + w, ry + block_h, stroke="#eeeeee")
        ry += block_h


def editor_pane(sh, x, y, w, h):
    """Two mirrored columns, and the button that copies one into the other."""
    sh.line(x - 4, y, x - 4, y + h, stroke=LINE2)
    sh.rect(x, y, w, 18, fill=STRIP, stroke=LINE2)
    sh.text(x + 5, y + 13, "Add a step", size=10.5, weight="bold")
    cy = y + 24
    for label, value in (("kind", "MOVE ▾"), ("link", "together ▾"),
                         ("", "⧉  mirror A → B")):
        sh.text(x, cy + 10, label, size=9.5, fill=DIM)
        sh.rect(x + 40, cy, w - 40, 16, fill=FACE, stroke=LINE, rx=2)
        sh.text(x + 44, cy + 12, value, size=9.5)
        cy += 21
    col_w = (w - 5) / 2.0
    for i, (who, tint, col) in enumerate((("Arm A", A_TINT, A_COL),
                                          ("Arm B", B_TINT, B_COL))):
        cx = x + i * (col_w + 5)
        sh.rect(cx, cy, col_w, 148, fill=tint, stroke=LINE2, rx=2)
        sh.text(cx + 5, cy + 13, who, size=9.5, fill=col, weight="bold")
        fy = cy + 20
        for value in ("place + offset", "pick ▾", "world ▾",
                      "Z  +50.0  mm", "movel ▾", "Here"):
            sh.rect(cx + 4, fy, col_w - 8, 16, fill=FACE, stroke=LINE, rx=2)
            sh.text(cx + 8, fy + 12, value, size=9, fill="#1a1a1a")
            fy += 20
    button(sh, x, y + h - 34, w, 30, "＋  Add step", fill=GREEN, fg="#ffffff",
           size=12, stroke=GREEN)


def jog_strip(sh, x, y, w, h=44):
    """The jog keys that survive a collapsed tab zone.

    Teaching is jog, Here, jog, Here. Options A and C hide the whole Jog page
    when the program goes wide, which puts two extra presses on every taught
    point -- so one row of it stays docked: the three targets, all six axes,
    and the step preset. The full three-column grid still lives on the Jog tab.

    Whatever collapses a zone has to call panels["jog"].release() as it goes,
    the way app._tab_changed and app.changeEvent already do. A held key whose
    button has left the screen must not still be driving an arm.
    """
    sh.rect(x, y, w, h, fill=FACE, stroke=INK)
    sh.text(x + 6, y + 12, "JOG — DOCKED", size=8, fill=INK, weight="bold",
            spacing=0.7)
    tx = x + 6
    for label, tint, col, on in (("A", A_TINT, A_COL, False),
                                 ("A+B", "#f4f0e0", "#1a1a1a", True),
                                 ("B", B_TINT, B_COL, False)):
        button(sh, tx, y + 16, 40, h - 22, label,
               fill=INK if on else tint, fg="#ffffff" if on else col, size=10)
        tx += 43
    tx += 8
    for axis in ("X", "Y", "Z", "RX", "RY", "RZ"):
        button(sh, tx, y + 16, 22, h - 22, "−", size=11, fg=DIM)
        sh.text(tx + 33, y + 16 + (h - 22) / 2.0 + 4, axis, size=9.5, fill=DIM,
                weight="bold", anchor="middle")
        button(sh, tx + 44, y + 16, 22, h - 22, "+", size=11, fg=DIM)
        tx += 72
    tx += 6
    for label in ("1 mm", "10", "50"):
        button(sh, tx, y + 16, 38, h - 22, label,
               fill=GREEN if label == "10" else "#ececec",
               fg="#ffffff" if label == "10" else "#1a1a1a", size=9.5)
        tx += 41
    sh.text(x + w - 6, y + 12, "full 3-column grid stays on the Jog tab",
            size=8.5, fill="#9aa8ad", anchor="end", font=MONO)


def icon_rail(sh, x, y, w, h, opt):
    """The always-visible rail, and the toggle at the top of it.

    Nothing on the rail can be hidden, so the way back is never behind the
    thing that took it away -- which is the failure mode of a plain Wide
    button. The lit icon is the open panel; pressing it again closes the
    sidebar and leaves the rail.
    """
    sh.rect(x, y, w, h, fill="#e4e7e9", stroke=LINE)
    open_id = opt.get("sidebar_panel") if opt.get("sidebar_w") else None

    # Toggle Sidebar, in the corner a VSCode hand already reaches for
    button(sh, x + 6, y + 6, w - 12, 34, "◀" if open_id else "▶",
           fill=FACE, fg=INK, stroke=INK, size=14)
    sh.text(x + w / 2.0, y + 52, "toggle", size=7.5, fill=DIM, anchor="middle",
            font=MONO)

    cy = y + 62
    for pid, glyph, label in (("points", "◎", "Pts"), ("vars", "⚙", "Vars"),
                              ("object", "▣", "Obj"), ("jog", "✥", "Jog")):
        lit = pid == open_id
        sh.rect(x + 4, cy, w - 8, 52, fill=INK if lit else "#ececec",
                stroke=INK if lit else LINE, rx=3)
        if lit:
            # the strip down the edge is the open panel, the way a rail says
            # which one it is without colour alone
            sh.rect(x, cy, 4, 52, fill=INK)
        sh.text(x + w / 2.0, cy + 26, glyph, size=19,
                fill="#ffffff" if lit else "#54636a", anchor="middle")
        sh.text(x + w / 2.0, cy + 43, label, size=8.5,
                fill="#ffffff" if lit else DIM, anchor="middle", weight="bold")
        cy += 56

    sh.text(x + w / 2.0, y + h - 8, "rail", size=7.5, fill="#9aa8ad",
            anchor="middle", font=MONO)


def jog_sidebar(sh, x, y, w, h, targets=1):
    """The Jog page as a left sidebar.

    With one target shown at a time the keys stay finger-sized in 320 px, and
    the price is a selector -- which is the control the panel deliberately
    removed ("no target selector stands between the operator and either arm").
    With all three kept the sidebar needs the width of the old tab zone, and
    the step editor has to become a sheet to pay for it.
    """
    sh.rect(x, y, w, h, fill=FACE, stroke=LINE)
    sh.rect(x, y, w, 20, fill=STRIP, stroke=LINE2)
    sh.text(x + 6, y + 15, "Jog", size=11, weight="bold")
    sh.text(x + w - 6, y + 15,
            "one target at a time" if targets == 1 else "all three, as today",
            size=9, fill=DIM, anchor="end", font=MONO)

    cols = [("Arm A", A_TINT, A_COL), ("A+B — synchronized", "#f4f0e0", "#1a1a1a"),
            ("Arm B", B_TINT, B_COL)]
    pad = 6
    if targets == 1:
        # the selector this layout costs
        bw = (w - 2 * pad - 8) / 3.0
        for i, (label, tint, col) in enumerate(cols):
            lit = i == 1
            button(sh, x + pad + i * (bw + 4), y + 26, bw, 30,
                   ("A", "A+B", "B")[i], fill=INK if lit else tint,
                   fg="#ffffff" if lit else col, size=11,
                   stroke=INK if lit else LINE)
        sh.text(x + pad, y + 70, "world ▾", size=10, fill=DIM)
        gy = y + 78
        gh = (y + h - gy - 46) / 6.0
        for r, axis in enumerate(("X", "Y", "Z", "RX", "RY", "RZ")):
            kw = (w - 2 * pad - 46) / 2.0
            button(sh, x + pad, gy + r * gh, kw, gh - 4, "−", size=13, fg=DIM)
            sh.text(x + pad + kw + 23, gy + r * gh + gh / 2.0, axis, size=11,
                    fill=DIM, weight="bold", anchor="middle")
            button(sh, x + pad + kw + 46, gy + r * gh, kw, gh - 4, "+", size=13,
                   fg=DIM)
    else:
        cw = (w - 2 * pad - 8) / 3.0
        for i, (label, tint, col) in enumerate(cols):
            cx = x + pad + i * (cw + 4)
            sh.rect(cx, y + 26, cw, 18, fill=tint, stroke=LINE2)
            sh.text(cx + cw / 2.0, y + 39, label, size=8.5, fill=col,
                    weight="bold", anchor="middle")
        gy = y + 48
        gh = (y + h - gy - 46) / 6.0
        for r, axis in enumerate(("X", "Y", "Z", "RX", "RY", "RZ")):
            for i in range(3):
                cx = x + pad + i * (cw + 4)
                kw = (cw - 22) / 2.0
                button(sh, cx, gy + r * gh, kw, gh - 3, "−", size=10, fg=DIM)
                sh.text(cx + kw + 11, gy + r * gh + gh / 2.0 + 3, axis, size=9,
                        fill=DIM, weight="bold", anchor="middle")
                button(sh, cx + kw + 22, gy + r * gh, kw, gh - 3, "+", size=10,
                       fg=DIM)

    # the compact readouts that live under the jog keys today
    ry = y + h - 40
    sh.rect(x + pad, ry, w - 2 * pad, 34, fill="#f7f9fb", stroke=LINE2)
    sh.text(x + pad + 5, ry + 14, "A  -412.3  188.0  505.1", size=8.5,
            fill=A_COL, font=MONO)
    sh.text(x + pad + 5, ry + 27, "B  -395.8 -190.4  504.7   gap 380.2",
            size=8.5, fill=B_COL, font=MONO)


def right_zone(sh, x, y, w, h, opt):
    """The tabs, and the three-column jog surface under them."""
    cy = y
    if not opt.get("topbar"):
        safety_bar(sh, x, cy, w)
        cy += 44
    tw = (w - 9) / 4.0
    for i, (label, on) in enumerate((("Points", 0), ("Vars", 0), ("Object", 0),
                                     ("Jog", 1))):
        button(sh, x + i * (tw + 3), cy, tw, 26, label,
               fill=GREEN if on else "#e4e4e4", fg="#ffffff" if on else "#1a1a1a",
               size=11)
    cy += 30
    sh.rect(x, cy, w, y + h - cy, fill=FACE, stroke=LINE)
    hw = (w - 18) / 3.0
    for i, (label, tint, col) in enumerate((("Arm A", A_TINT, A_COL),
                                            ("A+B — synchronized", "#f4f0e0", "#1a1a1a"),
                                            ("Arm B", B_TINT, B_COL))):
        hx = x + 5 + i * (hw + 4)
        sh.rect(hx, cy + 5, hw, 17, fill=tint, stroke=LINE2)
        sh.text(hx + hw / 2.0, cy + 17, label, size=9.5, fill=col, weight="bold",
                anchor="middle")
    gy = cy + 26
    gh = (y + h - gy - 5) / 6.0
    for r, axis in enumerate(("X", "Y", "Z", "RX", "RY", "RZ")):
        for c in range(3):
            gx = x + 5 + c * (hw + 4)
            kw = (hw - 26) / 2.0
            button(sh, gx, gy + r * gh, kw, gh - 3, "−", size=10, fg=DIM)
            sh.text(gx + kw + 13, gy + r * gh + gh / 2.0, axis, size=9.5,
                    fill=DIM, weight="bold", anchor="middle")
            button(sh, gx + kw + 26, gy + r * gh, kw, gh - 3, "+", size=10, fg=DIM)


def panel(sh, ox, oy, opt, g):
    """One whole 1280x800 screen."""
    sh.rect(ox, oy, W, H, fill=BG, stroke="#8d979b", sw=1.5)
    x, y = ox + MARGIN, oy + MARGIN
    w = W - 2 * MARGIN

    if opt.get("topbar"):
        safety_bar(sh, x, y, w)
        y += 44
    msg_h = 24
    work_h = H - 2 * MARGIN - (44 if opt.get("topbar") else 0) - msg_h - 4

    # -- the rail and its sidebar, ahead of the program on the left
    if opt.get("rail"):
        icon_rail(sh, x, y, RAIL_W, work_h, opt)
        x += RAIL_W + GAP
        if g["sidebar_w"]:
            jog_sidebar(sh, x, y, g["sidebar_w"], work_h,
                        targets=opt.get("sidebar_targets", 1))
            x += g["sidebar_w"] + GAP

    # -- the program zone
    pw = g["prog_w"]
    sh.rect(x, y, pw, work_h, fill=FACE, stroke=LINE)
    px, py = x + ZONE_PAD, y + ZONE_PAD
    pw_in, ph_in = pw - 2 * ZONE_PAD, work_h - 2 * ZONE_PAD
    tw = g["table_w"]

    sh.rect(px, py, tw, 18, fill=STRIP, stroke=LINE2)
    sh.text(px + 5, py + 13, "Program — teach & run", size=10.5, weight="bold")

    # the live position row: requirement 2, and the reason a point can be read
    # rather than only taught
    sh.rect(px, py + 21, tw, 17, fill="#f7f9fb", stroke=LINE2)
    nc = sh.clip(px, py + 21, tw - 4, 17)
    sh.text(px + 5, py + 33, "A", size=9, fill=A_COL, weight="bold", font=MONO, clip=nc)
    sh.text(px + 14, py + 33, "-412.3 188.0 505.1", size=9, fill=DIM, font=MONO, clip=nc)
    sh.text(px + 118, py + 33, "B", size=9, fill=B_COL, weight="bold", font=MONO, clip=nc)
    sh.text(px + 127, py + 33, "-395.8 -190.4 504.7", size=9, fill=DIM, font=MONO, clip=nc)
    sh.text(px + 238, py + 33, "gap 380.2", size=9, fill=DIM, font=MONO, clip=nc)

    table_y = py + 42
    buttons_h = 118
    strip_h = 48 if opt.get("jog_strip") else 0
    table_h = ph_in - 42 - buttons_h - strip_h
    step_table(sh, px, table_y, tw, table_h, g, opt)
    if strip_h:
        jog_strip(sh, px, py + ph_in - buttons_h - strip_h + 2, tw)

    by = py + ph_in - buttons_h + 6
    row = [("▲", None), ("▼", None), ("Delete", None), ("On/off", None)]
    if opt.get("wide_btn"):
        row.append(("⇔ Wide", INK))
    if opt.get("cycle_btn"):
        row.append(("Program ▸ Split ▸ Tabs", INK))
    if opt.get("sheet_btn"):
        row.append(("＋ Add", GREEN))
    bw = (tw - 4 * (len(row) - 1)) / float(len(row))
    for i, (label, accent) in enumerate(row):
        button(sh, px + i * (bw + 4), by, bw, 26, label,
               fill=FACE if accent else "#ececec", fg=accent or "#1a1a1a",
               stroke=accent or LINE, size=10.5)
    by += 30
    for i, (label, fill) in enumerate((("▶ Run", GREEN), ("⏸ Pause", AMBER),
                                       ("■ Stop", RED))):
        bw3 = (tw - 8) / 3.0
        button(sh, px + i * (bw3 + 4), by, bw3, 34, label, fill=fill, fg="#ffffff",
               size=13, stroke=fill)
    by += 38
    for i, (label, fill) in enumerate(((" Save", PURPLE), (" Load", PURPLE),
                                       ("Loop", "#ececec"), ("Dry run", "#ececec"))):
        bw4 = (tw - 12) / 4.0
        button(sh, px + i * (bw4 + 4), by, bw4, 26, label, fill=fill,
               fg="#ffffff" if fill == PURPLE else "#1a1a1a", size=10.5,
               stroke=fill if fill == PURPLE else LINE)

    if opt["editor"]:
        editor_pane(sh, px + tw + ZONE_GAP + 4, py, EDITOR_W - 4, ph_in)

    # -- the tab zone
    if opt["tabs"]:
        right_zone(sh, x + pw + GAP, y, g["right_w"], work_h, opt)

    # -- the sheet, for option B
    if opt.get("sheet"):
        sx, sw_ = px + 6, tw - 12
        sy, shh = table_y + table_h - 196, 196
        sh.rect(sx, sy, sw_, shh, fill="#ffffff", stroke=INK, sw=1.5)
        sh.parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                        'fill="none" stroke="%s" stroke-width="1.5" '
                        'stroke-dasharray="5 3"/>' % (sx, sy, sw_, shh, INK))
        sh.text(sx + 8, sy + 15, "ADD A STEP — SLIDES UP OVER THE TABLE",
                size=9, fill=INK, weight="bold", spacing=0.8)
        cw = (sw_ - 24) / 2.0
        for i, (who, tint, col) in enumerate((("Arm A", A_TINT, A_COL),
                                              ("Arm B", B_TINT, B_COL))):
            cx = sx + 8 + i * (cw + 8)
            sh.rect(cx, sy + 24, cw, 118, fill=tint, stroke=LINE2, rx=2)
            sh.text(cx + 6, sy + 38, who, size=10, fill=col, weight="bold")
            fy = sy + 45
            for value in ("place + offset", "pick ▾", "world  Z +50.0",
                          "movel ▾", "Here"):
                sh.rect(cx + 5, fy, cw - 10, 17, fill=FACE, stroke=LINE, rx=2)
                sh.text(cx + 9, fy + 13, value, size=9.5)
                fy += 20
        button(sh, sx + 8, sy + shh - 40, sw_ - 16, 32, "＋  Add step",
               fill=GREEN, fg="#ffffff", size=12, stroke=GREEN)

    # -- the message drawer
    mx, my = ox + MARGIN, oy + H - MARGIN - msg_h
    sh.rect(mx, my, w, msg_h, fill=FACE, stroke=LINE)
    sh.text(mx + 6, my + 16, "Messages", size=10, weight="bold")
    sh.text(mx + 66, my + 16, "line 5 · pair move 100.0 mm world Z at 50 mm/s",
            size=10, fill=DIM)


def dimensions(sh, ox, oy, g, opt):
    """What the drawing above actually measures, called out under the table."""
    x = ox + MARGIN + ZONE_PAD
    if opt.get("rail"):
        x += RAIL_W + GAP + ((g["sidebar_w"] + GAP) if g["sidebar_w"] else 0)
    y = oy + 14
    cols = [(COL_IDX, "%d" % COL_IDX, None, False),
            (g["kind"], "%d" % g["kind"], None, False)]
    if opt.get("stacked"):
        cols.append((g["detail"], "%d px" % g["detail"],
                     "one arm, on its own row  ≈ %d characters" % g["chars"], True))
    else:
        for _ in range(2):
            cols.append((g["per_arm"], "%d px" % g["per_arm"],
                         "≈ %d characters" % g["chars"], True))
        cols.append((g["link"], "%d" % g["link"], None, False))

    cx = x
    for width, label, note, lead in cols:
        colour = INK if lead else "#9aa8ad"
        sh.line(cx, y, cx, y + 9, stroke=colour, sw=1.2)
        sh.line(cx + width, y, cx + width, y + 9, stroke=colour, sw=1.2)
        sh.line(cx, y + 4.5, cx + width, y + 4.5, stroke=colour, sw=1.2)
        sh.text(cx + width / 2.0, y + 24, label, size=11 if lead else 9.5,
                fill=colour, anchor="middle", font=MONO,
                weight="bold" if lead else "normal")
        if note:
            sh.text(cx + width / 2.0, y + 37, note, size=9, fill=DIM,
                    anchor="middle", font=MONO)
        cx += width


def title_block(sh, ox, oy, opt, g):
    sh.text(ox, oy + 26, opt["letter"], size=30, fill=INK, weight="bold", font=MONO)
    sh.text(ox + (44 if len(opt["letter"]) < 2 else 60), oy + 24, opt["name"],
            size=21, weight="bold")
    sh.text(ox, oy + 52, opt["verdict"], size=12.5, fill="#3f4e56")
    where = "zone %d px" % g["prog_w"]
    if opt.get("rail"):
        where = "rail %d + sidebar %s   ·   program %d px" % (
            RAIL_W, ("%d" % g["sidebar_w"]) if g["sidebar_w"] else "closed",
            g["prog_w"])
    facts = "%s   ·   editor %s   ·   table %d px   ·   PER ARM %d px (≈%d chars)" % (
        where, "open" if opt["editor"] else "on demand", g["table_w"],
        g["per_arm"], g["chars"])
    sh.text(ox, oy + 72, facts, size=11, fill=DIM, font=MONO)
    sh.line(ox, oy + 82, ox + W, oy + 82, stroke="#aebcc2", sw=1.2)


OPTIONS = [
    {"id": "0_today", "letter": "0", "name": "Today, with two columns added",
     "tabs": True, "editor": True,
     "verdict": "Unreadable at the width that matters. Line 2 shows exactly what "
                "the operator would see."},
    {"id": "a_wide", "letter": "A", "name": "A Wide button collapses the tab zone",
     "tabs": False, "editor": True, "topbar": True, "wide_btn": True,
     "verdict": "The most room on offer. Needs REAL / Power / STOP moved to a "
                "full-width row first — STOP may never be hidden."},
    {"id": "b_sheet", "letter": "B", "name": "The editor becomes a sheet",
     "tabs": True, "editor": False, "sheet": True, "sheet_btn": True,
     "verdict": "panels/program.py only. Enough for most lines; the long one "
                "still clips, and the table is covered while a step is written."},
    {"id": "c_cycle", "letter": "C", "name": "One button cycling three widths",
     "tabs": False, "editor": True, "topbar": True, "cycle_btn": True,
     "verdict": "A superset of A, and a full-width Jog page as its third state. "
                "One more state to be in the wrong one of."},
    {"id": "e_strip", "letter": "E", "name": "Wide, with the jog keys docked",
     "tabs": False, "editor": True, "topbar": True, "wide_btn": True,
     "jog_strip": True,
     "verdict": "A or C, with the hole closed: one row of jog survives the "
                "collapse, so teaching stays jog-Here-jog without a mode "
                "switch. Costs table height, not width."},
    {"id": "f_side_open", "letter": "F", "name": "Icon rail, Jog sidebar open",
     "tabs": True, "editor": True, "topbar": True, "rail": True,
     "sidebar_w": 320, "sidebar_panel": "jog", "sidebar_targets": 1,
     "verdict": "The tab zone becomes a sidebar on the left and the program "
                "takes everything else. 320 px keeps the keys finger-sized by "
                "showing one target at a time — the selector the panel removed."},
    {"id": "g_side_three", "letter": "G", "name": "Icon rail, all three jog targets kept",
     "tabs": True, "editor": False, "topbar": True, "rail": True,
     "sidebar_w": 640, "sidebar_panel": "jog", "sidebar_targets": 3,
     "verdict": "No selector: A, A+B and B stay visible together, which needs "
                "the width of the old tab zone — so the step editor has to be "
                "a sheet to pay for it."},
    {"id": "h_side_shut", "letter": "H", "name": "Icon rail, sidebar closed",
     "tabs": True, "editor": True, "topbar": True, "rail": True,
     "sidebar_w": 0, "sidebar_panel": "jog",
     "verdict": "What one press on the rail leaves: the program at full width "
                "with the way back still on screen. The shared closed state of "
                "F and G."},
    {"id": "d_rows", "letter": "D", "name": "Two rows per step, and no zone change",
     "tabs": True, "editor": True, "stacked": True,
     "verdict": "More room than B, with no new control anywhere — paid for "
                "by showing about 8 steps at a time instead of 16."},
]


def main():
    sheet_w = W + 2 * SHEET_PAD
    sheet_h = TITLE_H + H + DIMS_H + 2 * SHEET_PAD
    written = []
    for opt in OPTIONS:
        g = geometry(opt)
        sh = Sheet(sheet_w, sheet_h)
        sh.rect(0, 0, sheet_w, sheet_h, fill="#ffffff")
        title_block(sh, SHEET_PAD, SHEET_PAD, opt, g)
        panel(sh, SHEET_PAD, SHEET_PAD + TITLE_H, opt, g)
        dimensions(sh, SHEET_PAD, SHEET_PAD + TITLE_H + H, g, opt)
        sh.text(SHEET_PAD, sheet_h - 14,
                "dual UR5 cell  ·  panel layout  ·  drawn to the "
                "1280x800 design size  ·  docs/program_line_design.md",
                size=9.5, fill="#9aa8ad", font=MONO)
        path = os.path.join(HERE, "layout_%s.svg" % opt["id"])
        with open(path, "w") as f:
            f.write(sh.svg("Panel layout %s — %s" % (opt["letter"], opt["name"])))
        written.append((opt["letter"], os.path.basename(path), g))

    print("%-3s %-26s %8s %9s %7s" % ("", "file", "table", "per arm", "chars"))
    for letter, name, g in written:
        print("%-3s %-26s %8d %9d %7d"
              % (letter, name, g["table_w"], g["per_arm"], g["chars"]))


if __name__ == "__main__":
    main()
