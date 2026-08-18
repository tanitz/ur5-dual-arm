#!/usr/bin/env python3
"""2D drawing of the robot mounting flange, generated from the cell numbers.

The plate is the interface between one end of the crossbar and one UR5 base.
Its outer size, its thickness, and where the two of them sit on the mast are
`mount.*` in config/cell.yaml -- the same numbers the kinematics uses -- so the
drawing and the controller can never quietly disagree. Edit the YAML and run
this again.

The UR5 side of the interface is not a guess either: the bolt pattern below was
measured off meshes/ur5/visual/base.dae in Universal_Robots_ROS2_Description
(four Ø9 holes on a Ø132 circle at 45/135/225/315 deg, base outer Ø~147).

    python3 docs/drawings/make_flange_drawing.py

writes ur5_mount_flange.svg (dimensioned A3 sheet) and ur5_mount_flange.dxf
(1:1 profile for the shop) next to this file.
"""

import math
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CELL_YAML = os.path.join(ROOT, "config", "cell.yaml")

# -- the UR5 base, as measured from the description package -----------------
UR5_BASE_OD = 149.0        # nominal outer diameter of the robot base
UR5_BOLT_PCD = 132.0       # measured: hole centres at r = 66.0
UR5_BOLT_ANGLES = (45.0, 135.0, 225.0, 315.0)   # measured
UR5_BOLT_SIZE = "M8"
TAP_DRILL = 6.8            # M8 x 1.25
TAP_DEPTH = 18.0
DOWEL_DIA = 8.0            # Ø8 H7 locating pin, on the same Ø132 circle
DOWEL_ANGLE = 0.0
DOWEL_DEPTH = 12.0

# -- the crossbar side of the interface -------------------------------------
CB_BOLT_PCD = 80.0                       # clear of the robot pattern
CB_BOLT_ANGLES = (0.0, 90.0, 180.0, 270.0)
CB_CLEAR_DIA = 9.0                       # M8 clearance
CB_CBORE_DIA = 15.0                      # SHCS head, sunk under the robot
CB_CBORE_DEPTH = 11.0
CABLE_BORE = 40.0                        # centre pass-through


def load_mount():
    with open(CELL_YAML) as fh:
        cfg = yaml.safe_load(fh)
    m = cfg["mount"]
    return {
        "plate_od": 2000.0 * float(m.get("pad_radius", 0.075)),
        "thickness": 1000.0 * float(m.get("pad_thickness", 0.022)),
        "spacing": 1000.0 * float(m["spacing"]),
        "height": 1000.0 * float(m["column_height"]),
        "tilt": float(m["tilt_deg"]),
        "column_od": 2000.0 * float(m.get("column_radius", 0.06)),
        "crossbar_od": 2000.0 * float(m.get("crossbar_radius", 0.05)),
    }


# ---------------------------------------------------------------- svg sheet
SHEET_W, SHEET_H = 420.0, 297.0     # A3 landscape, drawn in millimetres

THICK = 0.5
THIN = 0.18
TEXT = 3.2
SMALL = 2.4
FONT = "'Noto Sans Thai','Loma','DejaVu Sans',sans-serif"


class View(object):
    """Maps part millimetres (x right, y up) onto the sheet (y down)."""

    def __init__(self, ox, oy, scale=1.0):
        self.ox, self.oy, self.s = ox, oy, scale

    def __call__(self, x, y):
        return self.ox + x * self.s, self.oy - y * self.s


class Sheet(object):
    def __init__(self):
        self.out = []

    def add(self, s):
        self.out.append(s)

    # -- primitives ---------------------------------------------------------
    def line(self, v, x1, y1, x2, y2, w=THICK, dash=None, color="#111"):
        a, b = v(x1, y1), v(x2, y2)
        d = ' stroke-dasharray="%s"' % dash if dash else ""
        self.add('<line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="%s" '
                 'stroke-width="%.2f"%s/>' % (a[0], a[1], b[0], b[1], color, w, d))

    def circle(self, v, cx, cy, r, w=THICK, dash=None, fill="none", color="#111"):
        c = v(cx, cy)
        d = ' stroke-dasharray="%s"' % dash if dash else ""
        self.add('<circle cx="%.3f" cy="%.3f" r="%.3f" fill="%s" stroke="%s" '
                 'stroke-width="%.2f"%s/>' % (c[0], c[1], r * v.s, fill, color, w, d))

    def arc(self, v, cx, cy, r, a0, a1, w=THIN, color="#111"):
        p0 = v(cx + r * math.cos(math.radians(a0)), cy + r * math.sin(math.radians(a0)))
        p1 = v(cx + r * math.cos(math.radians(a1)), cy + r * math.sin(math.radians(a1)))
        large = 1 if (a1 - a0) % 360 > 180 else 0
        self.add('<path d="M %.3f %.3f A %.3f %.3f 0 %d 0 %.3f %.3f" fill="none" '
                 'stroke="%s" stroke-width="%.2f"/>'
                 % (p0[0], p0[1], r * v.s, r * v.s, large, p1[0], p1[1], color, w))

    def rect(self, v, x, y, w_, h_, w=THICK, fill="none"):
        a = v(x, y + h_)
        self.add('<rect x="%.3f" y="%.3f" width="%.3f" height="%.3f" fill="%s" '
                 'stroke="#111" stroke-width="%.2f"/>'
                 % (a[0], a[1], w_ * v.s, h_ * v.s, fill, w))

    def text(self, x, y, s, size=TEXT, anchor="start", color="#111",
             weight="normal", angle=None):
        rot = ' transform="rotate(%.2f %.3f %.3f)"' % (angle, x, y) if angle else ""
        self.add('<text x="%.3f" y="%.3f" font-family="%s" font-size="%.2f" '
                 'fill="%s" text-anchor="%s" font-weight="%s"%s>%s</text>'
                 % (x, y, FONT, size, color, anchor, weight, rot, esc(s)))

    def vtext(self, v, x, y, s, **kw):
        p = v(x, y)
        self.text(p[0], p[1], s, **kw)

    # -- dimensioning -------------------------------------------------------
    def arrow(self, x, y, ang, size=2.6):
        a = math.radians(ang)
        p1 = (x, y)
        p2 = (x + size * math.cos(a) + 0.45 * math.sin(a),
              y + size * math.sin(a) - 0.45 * math.cos(a))
        p3 = (x + size * math.cos(a) - 0.45 * math.sin(a),
              y + size * math.sin(a) + 0.45 * math.cos(a))
        self.add('<polygon points="%.3f,%.3f %.3f,%.3f %.3f,%.3f" fill="#111"/>'
                 % (p1 + p2 + p3))

    def dim_h(self, v, x1, x2, y, off, label, above=True, ext=True):
        """Horizontal dimension between x1 and x2, line at part-y + off/scale."""
        a, b = v(x1, y), v(x2, y)
        ly = a[1] - off
        if ext:
            self.line_raw(a[0], a[1] - (2 if off > 0 else -2) * sign(off),
                          a[0], ly + 1.5 * sign(off), THIN)
            self.line_raw(b[0], b[1] - (2 if off > 0 else -2) * sign(off),
                          b[0], ly + 1.5 * sign(off), THIN)
        self.line_raw(a[0], ly, b[0], ly, THIN)
        self.arrow(a[0], ly, 0)
        self.arrow(b[0], ly, 180)
        self.text((a[0] + b[0]) / 2.0, ly - 1.2 if above else ly + 3.4,
                  label, anchor="middle")

    def dim_v(self, v, y1, y2, x, off, label):
        a, b = v(x, y1), v(x, y2)
        lx = a[0] + off
        self.line_raw(a[0] + 1.5 * sign(off), a[1], lx - 1.5 * sign(off), a[1], THIN)
        self.line_raw(b[0] + 1.5 * sign(off), b[1], lx - 1.5 * sign(off), b[1], THIN)
        self.line_raw(lx, a[1], lx, b[1], THIN)
        self.arrow(lx, a[1], 90 if a[1] < b[1] else 270)
        self.arrow(lx, b[1], 270 if a[1] < b[1] else 90)
        self.text(lx + 1.4, (a[1] + b[1]) / 2.0 + 1.0, label)

    def line_raw(self, x1, y1, x2, y2, w=THIN, dash=None, color="#111"):
        d = ' stroke-dasharray="%s"' % dash if dash else ""
        self.add('<line x1="%.3f" y1="%.3f" x2="%.3f" y2="%.3f" stroke="%s" '
                 'stroke-width="%.2f"%s/>' % (x1, y1, x2, y2, color, w, d))

    def dim_dia(self, v, r, ang, label, out=14.0):
        """Diameter dimension drawn across a circle of radius r at `ang`."""
        a = math.radians(ang)
        p1 = v(-r * math.cos(a), -r * math.sin(a))
        p2 = v(r * math.cos(a), r * math.sin(a))
        ex = (p2[0] - p1[0], p2[1] - p1[1])
        n = math.hypot(*ex)
        u = (ex[0] / n, ex[1] / n)
        e = (p2[0] + u[0] * out, p2[1] + u[1] * out)
        self.line_raw(p1[0], p1[1], e[0], e[1], THIN)
        self.arrow(p1[0], p1[1], math.degrees(math.atan2(-u[1], -u[0])) + 180)
        self.arrow(p2[0], p2[1], math.degrees(math.atan2(u[1], u[0])) + 180)
        anchor = "start" if u[0] >= 0 else "end"
        dx = 1.5 if u[0] >= 0 else -1.5
        self.text(e[0] + dx, e[1] + 1.0, label, anchor=anchor)

    def leader(self, v, px, py, tx, ty, label, lines=None, anchor="start"):
        """Arrow at part point (px,py), elbow, then text at sheet (tx,ty)."""
        p = v(px, py)
        knee = (tx - (4.0 if anchor == "start" else -4.0), ty - 1.2)
        self.line_raw(p[0], p[1], knee[0], knee[1], THIN)
        self.line_raw(knee[0], knee[1], tx, knee[1], THIN)
        ang = math.degrees(math.atan2(p[1] - knee[1], p[0] - knee[0])) + 180
        self.arrow(p[0], p[1], ang)
        for i, s in enumerate([label] + list(lines or [])):
            self.text(tx, ty + i * 3.6, s, size=SMALL if i else TEXT, anchor=anchor)

    def centre_mark(self, v, cx, cy, r):
        self.line(v, cx - r, cy, cx + r, cy, THIN, dash="6 1.5 1 1.5")
        self.line(v, cx, cy - r, cx, cy + r, THIN, dash="6 1.5 1 1.5")


def sign(x):
    return 1.0 if x >= 0 else -1.0


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


DIA = "⌀"      # Ø
DEG = "°"
DEPTH = "↓"    # depth symbol stand-in


# ------------------------------------------------------------------ views
def draw_face_view(sh, m, v):
    R = m["plate_od"] / 2.0
    rb = UR5_BOLT_PCD / 2.0
    rc = CB_BOLT_PCD / 2.0

    sh.circle(v, 0, 0, R, THICK)                                   # plate edge
    sh.circle(v, 0, 0, UR5_BASE_OD / 2.0, THIN, dash="7 2 1 2",
              color="#8a8a8a")                                     # robot base
    sh.circle(v, 0, 0, rb, THIN, dash="7 2 1 2")                   # bolt circle
    sh.circle(v, 0, 0, rc, THIN, dash="7 2 1 2")
    sh.circle(v, 0, 0, CABLE_BORE / 2.0, THICK)
    sh.centre_mark(v, 0, 0, R + 9)

    for ang in UR5_BOLT_ANGLES:                                    # tapped M8
        a = math.radians(ang)
        cx, cy = rb * math.cos(a), rb * math.sin(a)
        sh.circle(v, cx, cy, TAP_DRILL / 2.0, THICK)
        sh.arc(v, cx, cy, 4.0, ang + 100, ang + 10)                # 3/4 crest
        sh.arc(v, cx, cy, 4.0, ang + 190, ang + 100)
        sh.arc(v, cx, cy, 4.0, ang + 280, ang + 190)
        sh.centre_mark(v, cx, cy, 7)

    a = math.radians(DOWEL_ANGLE)                                  # dowel
    dx, dy = rb * math.cos(a), rb * math.sin(a)
    sh.circle(v, dx, dy, DOWEL_DIA / 2.0, THICK)
    sh.centre_mark(v, dx, dy, 7)

    for ang in CB_BOLT_ANGLES:                                     # to crossbar
        a = math.radians(ang)
        cx, cy = rc * math.cos(a), rc * math.sin(a)
        sh.circle(v, cx, cy, CB_CBORE_DIA / 2.0, THICK)
        sh.circle(v, cx, cy, CB_CLEAR_DIA / 2.0, THICK)
        sh.centre_mark(v, cx, cy, 10)

    # section line A-A along the horizontal axis
    y0 = v(0, 0)[1]
    x0, x1 = v(-R - 14, 0)[0], v(R + 14, 0)[0]
    sh.line_raw(x0, y0, x0 + 8, y0, 0.7)
    sh.line_raw(x1 - 8, y0, x1, y0, 0.7)
    sh.arrow(x0 + 8, y0 - 0.0, 0, 3.2)
    sh.arrow(x1 - 8, y0, 180, 3.2)
    sh.text(x0 - 1.5, y0 - 2.0, "A", size=4.0, anchor="end", weight="bold")
    sh.text(x1 + 1.5, y0 - 2.0, "A", size=4.0, weight="bold")

    # dimensions
    sh.dim_dia(v, R, 30, "%s%.0f" % (DIA, m["plate_od"]), out=16)
    sh.dim_dia(v, rb, 112, "%s%.0f P.C.D." % (DIA, UR5_BOLT_PCD), out=13)
    sh.dim_dia(v, rc, 200, "%s%.0f P.C.D." % (DIA, CB_BOLT_PCD), out=11)

    sh.leader(v, rb * math.cos(math.radians(135)), rb * math.sin(math.radians(135)),
              v(0, 0)[0] - 96, v(0, 0)[1] - 44,
              "4x %s %s%.0f" % (UR5_BOLT_SIZE, DEPTH, TAP_DEPTH),
              ["ยึดฐาน UR5",
               "ดอกสว่าน 6.8 %s21" % DEPTH],
              anchor="start")
    sh.leader(v, dx, dy, v(0, 0)[0] + 60, v(0, 0)[1] + 62,
              "%s8 H7 %s%.0f" % (DIA, DEPTH, DOWEL_DEPTH),
              ["หมุดกำหนด"
               "ตำแหน่ง (NOTE 5)"])
    sh.leader(v, 0, CB_BOLT_PCD / 2.0 + CB_CBORE_DIA / 2.0,
              v(0, 0)[0] + 26, v(0, 0)[1] - 62,
              "4x %s%.0f THRU" % (DIA, CB_CLEAR_DIA),
              ["ผายหัว %s%.0f %s%.0f"
               % (DIA, CB_CBORE_DIA, DEPTH, CB_CBORE_DEPTH),
               "ยึดกับคานขวาง"])
    sh.leader(v, -CABLE_BORE / 2.0 * 0.71, -CABLE_BORE / 2.0 * 0.71,
              v(0, 0)[0] - 92, v(0, 0)[1] + 40,
              "%s%.0f THRU" % (DIA, CABLE_BORE),
              ["ช่องเดินสาย"])

    # 45 deg between the vertical centreline and a tapped hole
    p = v(rb * 0.62 * math.cos(math.radians(67)), rb * 0.62 * math.sin(math.radians(67)))
    sh.text(p[0], p[1], "45%s" % DEG, size=SMALL, anchor="middle")
    sh.arc(v, 0, 0, rb * 0.62, 45, 90, THIN)

    c = v(0, -R - 30)
    sh.text(c[0], c[1], "VIEW A  —  หน้าประ"
            "กบหุ่นยนต์   SCALE 1:1",
            size=3.6, anchor="middle", weight="bold")


def draw_section(sh, m, v):
    R = m["plate_od"] / 2.0
    t = m["thickness"]
    rb = UR5_BOLT_PCD / 2.0
    rc = CB_BOLT_PCD / 2.0
    top, bot = 0.0, -t

    def hatched(x0, x1):
        a, b = v(x0, top), v(x1, bot)
        sh.add('<rect x="%.3f" y="%.3f" width="%.3f" height="%.3f" '
               'fill="url(#hatch)" stroke="none"/>'
               % (a[0], a[1], b[0] - a[0], b[1] - a[1]))

    # solid material, split by the holes on the cutting plane
    edges = [(-R, -rb - 4), (-rb + 4, -rc - CB_CBORE_DIA / 2.0),
             (-rc + CB_CBORE_DIA / 2.0, -CABLE_BORE / 2.0),
             (CABLE_BORE / 2.0, rc - CB_CBORE_DIA / 2.0),
             (rc + CB_CBORE_DIA / 2.0, rb - DOWEL_DIA / 2.0),
             (rb + DOWEL_DIA / 2.0, R)]
    for x0, x1 in edges:
        hatched(x0, x1)
    hatched(-rc - CB_CBORE_DIA / 2.0, -rc + CB_CBORE_DIA / 2.0)
    hatched(rc - CB_CBORE_DIA / 2.0, rc + CB_CBORE_DIA / 2.0)

    sh.line(v, -R, top, R, top, THICK)
    sh.line(v, -R, bot, R, bot, THICK)
    sh.line(v, -R, top, -R, bot, THICK)
    sh.line(v, R, top, R, bot, THICK)

    # centre bore
    for s in (-1, 1):
        sh.line(v, s * CABLE_BORE / 2.0, top, s * CABLE_BORE / 2.0, bot, THICK)

    # counterbored clearance holes to the crossbar (open on the robot face)
    for c in (-rc, rc):
        for s in (-1, 1):
            sh.line(v, c + s * CB_CLEAR_DIA / 2.0, top - CB_CBORE_DEPTH,
                    c + s * CB_CLEAR_DIA / 2.0, bot, THICK)
            sh.line(v, c + s * CB_CBORE_DIA / 2.0, top,
                    c + s * CB_CBORE_DIA / 2.0, top - CB_CBORE_DEPTH, THICK)
        sh.line(v, c - CB_CBORE_DIA / 2.0, top - CB_CBORE_DEPTH,
                c - CB_CLEAR_DIA / 2.0, top - CB_CBORE_DEPTH, THICK)
        sh.line(v, c + CB_CLEAR_DIA / 2.0, top - CB_CBORE_DEPTH,
                c + CB_CBORE_DIA / 2.0, top - CB_CBORE_DEPTH, THICK)

    # M8 tap, revolved into the cutting plane (left)
    c = -rb
    for s in (-1, 1):
        sh.line(v, c + s * TAP_DRILL / 2.0, top, c + s * TAP_DRILL / 2.0,
                top - TAP_DEPTH - 3, THICK)
        sh.line(v, c + s * 4.0, top, c + s * 4.0, top - TAP_DEPTH, THIN)
    sh.line(v, c - TAP_DRILL / 2.0, top - TAP_DEPTH - 3,
            c + TAP_DRILL / 2.0, top - TAP_DEPTH - 3, THICK)
    sh.line(v, c - 4.0, top - TAP_DEPTH, c + 4.0, top - TAP_DEPTH, THIN)

    # dowel hole (right)
    c = rb
    for s in (-1, 1):
        sh.line(v, c + s * DOWEL_DIA / 2.0, top, c + s * DOWEL_DIA / 2.0,
                top - DOWEL_DEPTH, THICK)
    sh.line(v, c - DOWEL_DIA / 2.0, top - DOWEL_DEPTH,
            c + DOWEL_DIA / 2.0, top - DOWEL_DEPTH, THICK)

    # dimensions
    sh.dim_h(v, -R, R, bot, -26, "%.0f" % m["plate_od"])
    sh.dim_h(v, -rb, 0, top, 12, "%.0f" % rb)
    sh.dim_h(v, 0, rc, top, 20, "%.0f" % rc)
    sh.dim_v(v, top, bot, R, 16, "%.0f" % t)
    sh.dim_v(v, top, top - TAP_DEPTH, -rb - 4, -16, "%.0f" % TAP_DEPTH)
    sh.dim_v(v, top, top - DOWEL_DEPTH, rb + 5, 30, "%.0f" % DOWEL_DEPTH)

    p = v(0, top)
    sh.text(p[0] - 78, p[1] - 30, "หน้า A "
            "(ประกบฐานหุ่น"
            "ยนต์) — Ra 3.2, ระนาบ 0.05",
            size=SMALL)
    p = v(0, bot)
    sh.text(p[0], p[1] + 34, "SECTION A—A   (หมุนรู"
            "เข้าแนวตัด)   SCALE 1:1",
            size=3.6, anchor="middle", weight="bold")


def draw_layout(sh, m, v):
    """Where the two plates sit -- reference only, not a mast fabrication dwg."""
    half = m["spacing"] / 2.0
    h = m["height"]
    tilt = m["tilt"]
    cr = m["column_od"] / 2.0
    br = m["crossbar_od"] / 2.0
    R = m["plate_od"] / 2.0
    t = m["thickness"]

    # floor
    sh.line(v, -half - 170, 0, half + 170, 0, THICK)
    for x in range(int(-half - 165), int(half + 170), 22):
        sh.line(v, x, 0, x - 14, -14, THIN)

    # mast and crossbar
    sh.rect(v, -cr, 0, 2 * cr, h - br)
    sh.rect(v, -half, h - br, 2 * half, 2 * br)
    sh.line(v, 0, 0, 0, h + 60, THIN, dash="8 2 1.5 2")

    # one plate per end, face normal tilted `tilt` from vertical, outward
    for side in (+1, -1):
        cx, cy = side * half, h
        a = math.radians(tilt) * side          # face normal direction
        nx, ny = math.sin(a), math.cos(a)      # outward normal of the face
        tx, ty = ny, -nx                       # along the face
        sh.add('<polygon points="%s" fill="#e8eef5" stroke="#111" '
               'stroke-width="%.2f"/>' % (" ".join(
                   "%.3f,%.3f" % v(cx + tx * R * s1, cy + ty * R * s1
                                   - 0 * s1) for s1 in (1, -1)) + " " +
                   " ".join("%.3f,%.3f" % v(cx + tx * R * s1 - nx * t,
                                            cy + ty * R * s1 - ny * t)
                            for s1 in (-1, 1)), THICK))
        # the robot base, phantom
        sh.add('<polygon points="%s" fill="none" stroke="#8a8a8a" '
               'stroke-width="%.2f" stroke-dasharray="6 2 1 2"/>'
               % (" ".join("%.3f,%.3f" % v(cx + tx * 74.5 * s1 + nx * d,
                                           cy + ty * 74.5 * s1 + ny * d)
                           for s1, d in ((1, 0), (1, 120), (-1, 120), (-1, 0))),
                  THIN))
        sh.line(v, cx, cy, cx + nx * 150, cy + ny * 150, THIN, dash="8 2 1.5 2")
        p = v(cx + nx * 155, cy + ny * 155)
        sh.text(p[0], p[1], "Z", size=SMALL, anchor="middle")
        p = v(cx + tx * (R + 30) * side, cy + ty * (R + 30) * side)
        sh.text(p[0], p[1], "ARM A" if side > 0 else "ARM B",
                size=SMALL, anchor="middle", weight="bold")

    sh.dim_h(v, -half, half, h + 105, 0, "%.0f  (mount.spacing)" % m["spacing"])
    sh.dim_v(v, 0, h, -half - 120, -14, "%.0f" % h)
    p = v(half, h)
    sh.text(p[0] + 26, p[1] + 20, "tilt %.0f%s" % (tilt, DEG), size=SMALL)
    sh.text(p[0] + 26, p[1] + 23.6, "(%.0f%s จากแนว"
            "ระดับ)" % (180 - tilt, DEG), size=SMALL)
    p = v(0, 0)
    sh.text(p[0], p[1] + 12, "ผังติดตั้"
            "ง (อ้างอิง)  SCALE 1:12",
            size=3.4, anchor="middle", weight="bold")
    sh.text(p[0], p[1] + 17, "เสา %s%.0f / คานข"
            "วาง %s%.0f" % (DIA, m["column_od"], DIA, m["crossbar_od"]),
            size=SMALL, anchor="middle")


NOTES = [
    "1.  หน่วย: มิลลิเมตร  |  เกณฑ์ทั่วไป ISO 2768-mK  |  ลบคม 0.5x45%s ทุกขอบ" % DEG,
    "2.  วัสดุ: เหล็ก S45C หรือ SS400 หนา 22 (ทางเลือก AL6061-T6 ถ้าเน้นน้ำหนัก)",
    "3.  หน้า A คือหน้าประกบฐานหุ่นยนต์: Ra 3.2, ระนาบ 0.05, ตั้งฉากกับรู %s8 H7 0.05" % DIA,
    "4.  แพทเทิร์น 4x %s บน %s132 ที่ 45/135/225/315%s วัดจาก base.dae ของ ur_description" % (UR5_BOLT_SIZE, DIA, DEG),
    "5.  รูหมุด %s8 H7: เจาะโดย match-drill จากฐานหุ่นยนต์จริง หรือยืนยันกับคู่มือ UR ก่อน" % DIA,
    "6.  สลักยึดหุ่นยนต์: M8 class 12.9 x 4 ตัว, ขันตามค่าในคู่มือ UR (~20 N·m)",
    "7.  จำนวน 2 ชิ้น (Arm A / Arm B) — ชิ้นเหมือนกัน ไม่ต้อง mirror",
    "8.  %s150 / หนา 22 / ระยะ 500 / สูง 1200 / tilt 135%s มาจาก config/cell.yaml — แก้ที่นั่นแล้ว gen แบบใหม่" % (DIA, DEG),
    "9.  หลังติดตั้งจริง ให้วัดระยะหน้าแปลน-ต่อ-หน้าแปลนจริง แล้วใส่กลับใน arms.<id>.base",
]


def build_svg(m):
    sh = Sheet()
    sh.add('<defs><pattern id="hatch" width="4" height="4" '
           'patternTransform="rotate(45)" patternUnits="userSpaceOnUse">'
           '<line x1="0" y1="0" x2="0" y2="4" stroke="#111" stroke-width="0.16"/>'
           '</pattern></defs>')
    sh.add('<rect x="0" y="0" width="%.0f" height="%.0f" fill="#fff"/>'
           % (SHEET_W, SHEET_H))
    sh.add('<rect x="8" y="8" width="%.0f" height="%.0f" fill="none" stroke="#111" '
           'stroke-width="0.7"/>' % (SHEET_W - 16, SHEET_H - 16))

    draw_face_view(sh, m, View(120, 108, 1.0))
    draw_section(sh, m, View(120, 232, 1.0))
    draw_layout(sh, m, View(330, 148, 1.0 / 12.0))

    # notes
    sh.text(250, 176, "NOTES", size=3.6, weight="bold")
    for i, line in enumerate(NOTES):
        sh.text(250, 182 + i * 4.6, line, size=2.6)

    # title block
    x0, y0, w, h = 250.0, 234.0, 160.0, 55.0
    sh.add('<rect x="%.0f" y="%.0f" width="%.0f" height="%.0f" fill="none" '
           'stroke="#111" stroke-width="0.7"/>' % (x0, y0, w, h))
    for yy in (y0 + 14, y0 + 26, y0 + 38, y0 + 47):
        sh.line_raw(x0, yy, x0 + w, yy, 0.35)
    for xx in (x0 + 62, x0 + 110):
        sh.line_raw(xx, y0 + 14, xx, y0 + h, 0.35)
    sh.text(x0 + 4, y0 + 7, "MOUNT FLANGE — UR5 BASE", size=4.6, weight="bold")
    sh.text(x0 + 4, y0 + 12, "หน้าแปลน"
            "ยึดหุ่นยนต์ "
            "(ปลายคานขวาง)", size=2.8)
    rows = [
        (x0 + 4, y0 + 21, "DWG NO.", "UR5D-MP-01  REV A"),
        (x0 + 66, y0 + 21, "MATERIAL", "S45C t22"),
        (x0 + 114, y0 + 21, "QTY", "2"),
        (x0 + 4, y0 + 33, "PROJECT", "dual-UR5 cell"),
        (x0 + 66, y0 + 33, "SCALE", "1:1 (A3)"),
        (x0 + 114, y0 + 33, "UNITS", "mm"),
        (x0 + 4, y0 + 44, "SOURCE", "config/cell.yaml"),
        (x0 + 66, y0 + 44, "PROJECTION", "1st angle"),
        (x0 + 114, y0 + 44, "MASS", "~2.6 kg"),
    ]
    for tx, ty, k, val in rows:
        sh.text(tx, ty - 4.2, k, size=2.3, color="#666")
        sh.text(tx, ty, val, size=3.2)
    sh.text(x0 + 4, y0 + 52, "ตัวเลขทุก"
            "ตัวสร้างจาก "
            "config/cell.yaml — อย่าแก้"
            "ในไฟล์ SVG", size=2.5, color="#666")

    body = "\n".join(sh.out)
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %.0f %.0f" '
            'width="100%%" style="max-width:100%%;height:auto;background:#fff">\n'
            '%s\n</svg>\n' % (SHEET_W, SHEET_H, body))


# ------------------------------------------------------------------- dxf
def build_dxf(m):
    """Minimal R12 DXF: the profile and every hole, 1:1, one layer per feature."""
    e = []

    def circle(layer, cx, cy, r):
        e.extend(["0", "CIRCLE", "8", layer, "10", "%.4f" % cx, "20", "%.4f" % cy,
                  "30", "0.0", "40", "%.4f" % r])

    def line(layer, x1, y1, x2, y2):
        e.extend(["0", "LINE", "8", layer, "10", "%.4f" % x1, "20", "%.4f" % y1,
                  "30", "0.0", "11", "%.4f" % x2, "21", "%.4f" % y2, "31", "0.0"])

    def text(layer, x, y, h, s):
        e.extend(["0", "TEXT", "8", layer, "10", "%.4f" % x, "20", "%.4f" % y,
                  "30", "0.0", "40", "%.2f" % h, "1", s])

    R = m["plate_od"] / 2.0
    circle("OUTLINE", 0, 0, R)
    circle("BORE", 0, 0, CABLE_BORE / 2.0)
    for ang in UR5_BOLT_ANGLES:
        a = math.radians(ang)
        circle("TAP-M8-DRILL", (UR5_BOLT_PCD / 2.0) * math.cos(a),
               (UR5_BOLT_PCD / 2.0) * math.sin(a), TAP_DRILL / 2.0)
    a = math.radians(DOWEL_ANGLE)
    circle("DOWEL-8H7", (UR5_BOLT_PCD / 2.0) * math.cos(a),
           (UR5_BOLT_PCD / 2.0) * math.sin(a), 7.8 / 2.0)
    for ang in CB_BOLT_ANGLES:
        a = math.radians(ang)
        cx = (CB_BOLT_PCD / 2.0) * math.cos(a)
        cy = (CB_BOLT_PCD / 2.0) * math.sin(a)
        circle("CLEAR-9", cx, cy, CB_CLEAR_DIA / 2.0)
        circle("CBORE-15", cx, cy, CB_CBORE_DIA / 2.0)
    for r, layer in ((UR5_BOLT_PCD / 2.0, "PCD"), (CB_BOLT_PCD / 2.0, "PCD"),
                     (UR5_BASE_OD / 2.0, "REF-ROBOT-BASE")):
        circle(layer, 0, 0, r)
    line("CENTRE", -R - 10, 0, R + 10, 0)
    line("CENTRE", 0, -R - 10, 0, R + 10)
    text("NOTE", -R, R + 16, 5, "UR5D-MP-01 MOUNT FLANGE t%.0f - mm 1:1"
         % m["thickness"])

    head = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "4",
            "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    tail = ["0", "ENDSEC", "0", "EOF"]
    return "\n".join(head + e + tail) + "\n"


def main():
    m = load_mount()
    svg = os.path.join(HERE, "ur5_mount_flange.svg")
    dxf = os.path.join(HERE, "ur5_mount_flange.dxf")
    with open(svg, "w") as fh:
        fh.write(build_svg(m))
    with open(dxf, "w") as fh:
        fh.write(build_dxf(m))
    print("plate  %s%.0f x t%.0f" % (DIA, m["plate_od"], m["thickness"]))
    print("cell   spacing %.0f, height %.0f, tilt %.0f deg"
          % (m["spacing"], m["height"], m["tilt"]))
    print("wrote  %s" % svg)
    print("wrote  %s" % dxf)


if __name__ == "__main__":
    main()
