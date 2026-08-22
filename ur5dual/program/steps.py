"""
The step vocabulary a two-arm program is written in.

A line is a gesture, not a command to one robot. `MOVE` carries a column for
arm A and a column for arm B, and a `link` field that says how tightly the two
columns are tied to each other — which is not a preference but a choice of
engine:

    solo       only the filled column moves
    together   both are sent, and the line waits for both to arrive
    pair       one world delta given to both arms at matched speed
    coupled    one object frame drives both arms from one clock

`together` is not coordination. Each controller plans its own timing, so the
arms agree at the end of the line and nowhere in particular in the middle;
that is fine for approach and retreat and is refused while anything is held.
`pair` is the Jog tab's A+B column written down — world frame only, because
"base" and "tool" name a different direction at each robot, and translation
only, because RX/RY/RZ turn each wrist about its own tool and would twist a
rigid workpiece rather than turn it. Turning the pair is `coupled`, which is
the servo loop with its force and drift guards.

ATTACH is still where the change of mode happens, and it is `coupled`'s
precondition.

A column is a *target*, and an offset is not a separate kind of step — it is a
target with no named place in it. Offsets resolve when the line runs, against
where that arm actually is, which is what makes a stacking program possible
and what makes a line that ended somewhere unexpected hand the surprise to the
next one.
"""

import json
import math

import numpy as np

from ..geometry.kinematics import (
    mat_to_pose, pose_to_mat, rotvec_to_mat,
)

# every step kind, with the fields it carries
STEP_KINDS = {
    "MOVE":    ("link", "a", "b", "pair", "obj"),
    "ATTACH":  ("object", "origin"),
    "DETACH":  (),
    "BARRIER": (),
    "DELAY":   ("seconds",),
    "WHERE":   (),
    "CALL":    ("program", "repeat"),
    # cell I/O: one output to set, one input to wait for
    "OUT":     ("arm", "output", "state"),
    "WAIT_IN": ("arm", "input", "state", "timeout"),
    # flow: where a jump can land, and the three ways of taking one
    "LABEL":   ("name",),
    "JUMP":    ("target",),
    "IF":      ("source", "arm", "input", "name", "compare", "value",
                "target", "otherwise"),
    "SET_VAR": ("name", "op", "value"),
    # what the camera saw, against where the box was when the pick was taught
    "FIND":    ("into", "reference", "timeout"),
}

# What a kind is called on screen, where the operator's name for it is not
# the kind itself. The stored kind never changes with it: a program saved
# before a rename still loads, and the executor still dispatches on "BARRIER".
KIND_LABEL = {"BARRIER": "IN POSE"}

# a UR's standard digital I/O, and the variables a program can count with
IO_RANGE = 8
VAR_OPS = ("=", "+=", "-=")
COMPARES = ("==", "!=", "<", ">", "<=", ">=")
IF_SOURCES = ("input", "var")
# steps that only decide where to go next, and never touch an arm
CONTROL_KINDS = ("LABEL", "JUMP", "IF", "SET_VAR")
# What a FIND writes alongside its correction, so IF can ask whether there was
# one. A name rather than a fixed variable, because a cell may look for more
# than one thing.
FOUND_SUFFIX = "_found"

# how deep one program may call another. The limit is not about the machine —
# it is that a call tree nobody can hold in their head is a program whose next
# move nobody can predict, and this one runs two arms.
CALL_MAX_DEPTH = 5

LINKS = ("solo", "together", "pair", "coupled")
LINK_LABEL = {"solo": "", "together": "⇉ together",
              "pair": "⇉⇉ SYNC", "coupled": "⛓ coupled"}

# which offset frames each link is allowed to speak. The arm frames are the
# jog panel's, for the jog panel's reasons; `pair` keeps only the one that
# names the same direction at both robots.
ARM_FRAMES = ("world", "base", "tool")
PAIR_FRAMES = ("world",)
OBJECT_FRAMES = ("world", "object")

# which links need the arms coupled, and which need them free
NEEDS_OBJECT = ("coupled",)
NEEDS_FREE = ("solo", "together", "pair")

MOTIONS = ("movej", "movel")
_AXIS_NAMES = ("X", "Y", "Z", "RX", "RY", "RZ")


# ── targets: one column of a line ─────────────────────────────────────────
#
# A target is a plain dict so it survives JSON untouched. Four shapes, and
# the only difference between them is which of two keys is present:
#
#   {"point": "pick"}                              a taught place
#   {"point": "pick", "offset": [...], "frame": …} that place, shifted
#   {"offset": [...], "frame": …}                  shifted from where it is now
#   {"pose": [x,y,z,rx,ry,rz]}                     a pose captured with Here

def make_target(point=None, pose=None, offset=None, frame="world",
                motion=None, speed=None, pivot=None, correct_by=None, **extra):
    """A column, with the numbers normalised to what JSON can hold."""
    t = {}
    if point:
        t["point"] = str(point)
    if pose is not None:
        t["pose"] = [float(x) for x in pose]
    if offset is not None:
        t["offset"] = [float(x) for x in offset]
        t["frame"] = frame
    if motion:
        t["motion"] = motion
    if speed is not None:
        t["speed"] = float(speed)
    if pivot is not None:
        t["pivot"] = [float(x) for x in pivot]
    if correct_by:
        t["correct_by"] = str(correct_by)
    t.update({k: v for k, v in extra.items() if v is not None})
    return t


def target_kind(target):
    """Which of the four shapes this is, or None for an empty column."""
    if not target:
        return None
    if target.get("point"):
        return "point_offset" if target.get("offset") is not None else "point"
    if target.get("offset") is not None:
        return "offset"
    if target.get("pose") is not None:
        return "pose"
    return None


def target_turns(target):
    """True if the offset asks for any rotation at all.

    `pair` needs this: the same world *translation* given to two arms carries
    a workpiece between them, and the same world rotation twists it.
    """
    off = (target or {}).get("offset")
    return off is not None and float(np.max(np.abs(np.asarray(off, float)[3:]))) > 1e-9


def apply_offset(mat, offset, frame, base=None, pivot=None):
    """A 4x4, shifted by six numbers read in `frame`.

    Which side the rotation multiplies on is the whole difference between the
    frames, and it is the same distinction kinematics.py draws:

        world / base   R_new = Rk · R_old     an axis fixed in the cell
        tool / object  R_new = R_old · Rk     an axis that travels with it

    Translation and rotation in one offset happen in that order: the frame is
    turned first, about its own origin or about `pivot`, and then moved.
    """
    d = np.asarray(offset, dtype=float)
    out = np.asarray(mat, dtype=float).copy()
    R = rotvec_to_mat(d[3:])

    if frame in ("tool", "object"):
        step = np.eye(4)
        step[:3, :3] = R
        step[:3, 3] = d[:3]
        return out @ step

    if frame == "base":
        if base is None:
            raise ValueError("a base-frame offset needs the arm's base matrix")
        R_axes = np.asarray(base, dtype=float)[:3, :3]
    elif frame == "world":
        R_axes = np.eye(3)
    else:
        raise ValueError("unknown offset frame %r" % frame)

    # the rotation is about axes fixed in `frame`, so it is expressed there
    # and brought back: R_axes · Rk · R_axesᵀ pre-multiplies the pose
    R_world = R_axes @ R @ R_axes.T
    origin = np.asarray(pivot, dtype=float)[:3] if pivot is not None else out[:3, 3].copy()
    out[:3, :3] = R_world @ out[:3, :3]
    out[:3, 3] = R_world @ (out[:3, 3] - origin) + origin
    out[:3, 3] += R_axes @ d[:3]
    return out


def resolve_target(target, points, current=None, base=None, correction=None):
    """The world pose a column asks for, as a 6-vector.

    `current` is where the thing this column drives is *now*, in world — what
    an offset with no place in it offsets from, and what the tool frame is.
    `base` is that arm's base matrix, for a base-frame offset. Both are poses
    or matrices; neither is needed unless the target asks for it.

    `correction` is a rigid transform the whole target is carried by, and it
    is what a camera contributes: the box moved, so the pick taught on it
    moves with it. It multiplies from the left, which is the difference
    between "the workpiece is somewhere else" and "the wrist is turned" —
    adding an angle to the target instead would spin the tool on the spot.
    """
    kind = target_kind(target)
    if kind is None:
        raise ValueError("an empty column has no target")

    if kind in ("point", "point_offset"):
        mat = pose_to_mat(points.get(target["point"]))
    elif kind == "pose":
        mat = pose_to_mat(target["pose"])
    else:
        if current is None:
            raise ValueError("an offset with no place in it needs to know "
                             "where the arm is now")
        mat = _as_matrix(current)

    if target.get("offset") is not None:
        mat = apply_offset(
            mat, target["offset"], target.get("frame", "world"),
            base=None if base is None else _as_matrix(base),
            pivot=target.get("pivot"))
    if correction is not None:
        mat = _as_matrix(correction) @ mat
    return mat_to_pose(mat)


def _as_matrix(pose_or_mat):
    a = np.asarray(pose_or_mat, dtype=float)
    return a if a.shape == (4, 4) else pose_to_mat(a)


# ── how a column reads ────────────────────────────────────────────────────
def format_offset(offset):
    """Only the axes that are actually asked for, in mm and degrees."""
    d = np.asarray(offset, dtype=float)
    parts = []
    for i, v in enumerate(d):
        if abs(v) < 1e-9:
            continue
        parts.append("%s%+.1f" % (_AXIS_NAMES[i],
                                  v * 1000.0 if i < 3 else math.degrees(v)))
    return " ".join(parts) or "0"


def describe_target(target, with_motion=True):
    kind = target_kind(target)
    if kind is None:
        return "·"
    if kind == "pose":
        p = np.asarray(target["pose"], dtype=float)
        text = "here [%.1f %.1f %.1f]" % (p[0] * 1000, p[1] * 1000, p[2] * 1000)
    else:
        shift = ("%s %s" % (target.get("frame", "world"),
                            format_offset(target["offset"]))
                 if target.get("offset") is not None else "")
        if kind == "point":
            text = target["point"]
        elif kind == "point_offset":
            text = "%s + %s" % (target["point"], shift)
        else:
            text = shift
    if with_motion and target.get("motion"):
        text += "  " + target["motion"]
    return text


# ── migrating what is already on disk ─────────────────────────────────────
def _legacy(kind, fields):
    """The old one-thing-per-line spellings, as the line they now mean.

    Nothing on disk is rewritten. `MOVE_ARM`, `MOVE_OBJ` and `ROTATE_OBJ` are
    read, translated here, and saved back in the new shape the next time the
    program is saved.
    """
    f = dict(fields)
    if kind == "MOVE_ARM":
        arm = f.pop("arm", None)
        slot = make_target(point=f.pop("point", None),
                           motion=f.pop("motion", "movej"),
                           speed=f.pop("speed", None))
        out = {"link": "solo"}
        if arm in ("A", "B"):
            out[arm.lower()] = slot
        else:
            # an arm id that is neither leaves the line with nothing to move,
            # and the validator says which line and why
            out["arm"] = arm
        out.update(f)
        return "MOVE", out
    if kind == "MOVE_OBJ":
        return "MOVE", {"link": "coupled",
                        "obj": make_target(point=f.pop("point", None),
                                           lin_speed=f.pop("lin_speed", None),
                                           ang_speed=f.pop("ang_speed", None))}
    if kind == "ROTATE_OBJ":
        axis = str(f.pop("axis", "z")).lower()
        angle = math.radians(float(f.pop("angle_deg", 0.0)))
        rv = np.zeros(6)
        rv[3 + ("xyz".index(axis) if axis in "xyz" else 2)] = angle
        # a program written before world rotation existed meant the object's
        # own axes, and must keep meaning that
        return "MOVE", {"link": "coupled",
                        "obj": make_target(offset=rv,
                                           frame=f.pop("frame", "object"),
                                           ang_speed=f.pop("ang_speed", None))}
    if kind == "GRIP":
        # A gripper was never a different thing from a digital output, only a
        # narrower name for one. Old programs said GRIP with a column per arm;
        # they mean OUT on that arm, or on both.
        if "arm" in f:
            arm = f.pop("arm", "both")
        else:
            filled = [a for a in ("a", "b") if f.get(a)]
            arm = {"ab": "both", "a": "A", "b": "B"}.get("".join(filled), "both")
        slot = f.get("a") or f.get("b") or {}
        return "OUT", {"arm": arm,
                       "output": int(f.get("output", slot.get("output", 0))),
                       "state": bool(f.get("state", slot.get("state", True)))}
    return kind, f


def normalise(kind, fields):
    """One place where a step's spelling is settled, whichever door it came in
    by — the file loader, the panel, or a test."""
    if kind in ("MOVE_ARM", "MOVE_OBJ", "ROTATE_OBJ"):
        return _legacy(kind, fields)
    if kind == "GRIP":
        return _legacy(kind, fields)
    if kind == "MOVE":
        fields = dict(fields)
        fields.setdefault("link", "solo")
    return kind, fields


class Step:
    def __init__(self, kind, **fields):
        enabled = fields.pop("enabled", True)
        kind, fields = normalise(kind, fields)
        if kind not in STEP_KINDS:
            raise ValueError("unknown step kind %r" % kind)
        self.kind = kind
        self.fields = dict(fields)
        self.enabled = bool(enabled)

    def get(self, name, default=None):
        return self.fields.get(name, default)

    @property
    def link(self):
        return self.fields.get("link", "solo") if self.kind == "MOVE" else ""

    def slot(self, key):
        """One column, by the name it has in the file: a, b, pair or obj."""
        return self.fields.get(key) or None

    def arm_slots(self):
        return [(a, self.slot(a.lower())) for a in ("A", "B")]

    def filled_arms(self):
        return [a for a, t in self.arm_slots() if target_kind(t) is not None]

    # -- how the line reads -------------------------------------------------
    def render(self):
        """What the two-column table shows.

        Returns (span, a, b, link): `span` is text that crosses both columns
        for a step that addresses the pair as one body, and is None for a step
        that has something different to say to each arm.
        """
        f = self.fields
        if self.kind == "MOVE":
            if self.link == "pair":
                # describe_target already names the frame, and a pair move has
                # only the one it is allowed to speak
                return (describe_target(self.slot("pair"), with_motion=False),
                        None, None, LINK_LABEL["pair"])
            if self.link == "coupled":
                return ("⟨obj⟩ %s" % describe_target(self.slot("obj"),
                                                     with_motion=False),
                        None, None, LINK_LABEL["coupled"])
            return (None, describe_target(self.slot("a")),
                    describe_target(self.slot("b")), LINK_LABEL[self.link])
        if self.kind == "GRIP":
            # the tie-bar means "both at once", so it is earned by having both
            # columns rather than printed on every gripper line
            both = self.slot("a") and self.slot("b")
            return (None, _grip_text(self.slot("a")), _grip_text(self.slot("b")),
                    "⇉" if both else "")
        if self.kind == "ATTACH":
            return ("take hold of '%s' (origin %s)"
                    % (f.get("object", "object"), f.get("origin", "midpoint")),
                    None, None, "")
        if self.kind == "DETACH":
            return ("let go", None, None, "")
        if self.kind == "BARRIER":
            return ("wait for both arms to settle", None, None, "")
        if self.kind == "DELAY":
            return ("wait %.2f s" % float(f.get("seconds", 0)), None, None, "")
        if self.kind == "WHERE":
            return ("read where both arms are", None, None, "")
        if self.kind == "OUT":
            return ("%s  DO%d %s" % (_who(f.get("arm")), int(f.get("output", 0)),
                                     "ON" if f.get("state", True) else "OFF"),
                    None, None, "")
        if self.kind == "WAIT_IN":
            timeout = float(f.get("timeout", 0) or 0)
            return ("%s  wait DI%d %s%s"
                    % (_who(f.get("arm")), int(f.get("input", 0)),
                       "ON" if f.get("state", True) else "OFF",
                       "  (up to %.0f s)" % timeout if timeout else ""),
                    None, None, "")
        if self.kind == "LABEL":
            return ("%s:" % f.get("name", "?"), None, None, "◆")
        if self.kind == "JUMP":
            return ("jump to %s" % f.get("target", "?"), None, None, "↴")
        if self.kind == "IF":
            return (_if_text(f), None, None, "?")
        if self.kind == "SET_VAR":
            return ("%s %s %g" % (f.get("name", "?"), f.get("op", "="),
                                  float(f.get("value", 0))),
                    None, None, "=")
        if self.kind == "FIND":
            return ("look for the box, against %s -> %s"
                    % (f.get("reference", "?"), f.get("into", "?")),
                    None, None, "◉")
        if self.kind == "CALL":
            repeat = int(f.get("repeat", 1) or 1)
            return ("run program '%s'%s" % (f.get("program", "?"),
                                            "  ×%d" % repeat if repeat > 1 else ""),
                    None, None, "↳")
        return (self.kind, None, None, "")

    def describe(self):
        """The one-line form, for the log and for anything that is not the
        table."""
        span, a, b, link = self.render()
        if span is not None:
            return span
        if self.kind == "MOVE" and self.link == "solo":
            filled = self.filled_arms()
            if len(filled) == 1:
                return "arm %s  %s" % (filled[0],
                                       describe_target(self.slot(filled[0].lower())))
        text = "A %s | B %s" % (a, b)
        return text + ("  " + link if link and link.strip() else "")

    # -- io ----------------------------------------------------------------
    def to_dict(self):
        d = {"kind": self.kind, "enabled": self.enabled}
        d.update(self.fields)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        kind = d.pop("kind")
        enabled = d.pop("enabled", True)
        step = cls(kind, **d)
        step.enabled = enabled
        return step


def _who(arm):
    """How a step names the controller it speaks to."""
    return "both arms" if arm == "both" else "arm %s" % (arm or "?")


def _if_text(f):
    if f.get("source") == "var":
        test = "%s %s %g" % (f.get("name", "?"), f.get("compare", "=="),
                             float(f.get("value", 0)))
    else:
        test = "%s DI%d is %s" % (_who(f.get("arm")), int(f.get("input", 0)),
                                  "ON" if f.get("state", True) else "OFF")
    text = "if %s  jump %s" % (test, f.get("target", "?"))
    if (f.get("otherwise") or "").strip():
        text += "  else jump %s" % f["otherwise"]
    return text


def _grip_text(slot):
    if not slot:
        return "·"
    return "%s DO%d" % ("close" if slot.get("state", True) else "open",
                        int(slot.get("output", 0)))


class PointLibrary:
    """Named poses, in the cell's world frame.

    Arm targets and object targets share one namespace on purpose: a place
    to put the drum is a place, whether one arm or two arms take it there.
    """

    def __init__(self):
        self.points = {}          # name -> 6-vector, world frame

    def teach_arm(self, cell, arm_id, name):
        self.points[name] = np.array(cell.arms[arm_id].tcp_pose_world(), dtype=float)
        return self.points[name]

    def teach_object(self, held_object, name):
        if not held_object.held:
            raise ValueError("nothing is being held")
        self.points[name] = np.array(held_object.pose_world, dtype=float)
        return self.points[name]

    def set(self, name, pose):
        self.points[name] = np.array(pose, dtype=float)

    def get(self, name):
        if name not in self.points:
            raise KeyError("no point named %r" % name)
        return self.points[name]

    def remove(self, name):
        self.points.pop(name, None)

    def names(self):
        return sorted(self.points)

    def to_dict(self):
        return {k: [float(x) for x in v] for k, v in self.points.items()}

    def load_dict(self, d):
        self.points = {k: np.array(v, dtype=float) for k, v in (d or {}).items()}
        return self


class Program:
    def __init__(self, name="untitled"):
        self.name = name
        self.steps = []
        self.loop = False

    def add(self, step, index=None):
        if index is None:
            self.steps.append(step)
        else:
            self.steps.insert(index, step)
        return step

    def remove(self, index):
        if 0 <= index < len(self.steps):
            self.steps.pop(index)

    def move(self, index, delta):
        j = index + delta
        if 0 <= index < len(self.steps) and 0 <= j < len(self.steps):
            self.steps[index], self.steps[j] = self.steps[j], self.steps[index]
            return j
        return index

    # -- static checking ---------------------------------------------------
    def check(self, points, config=None, holding=False, surfaces=None):
        """Problems findable without moving, and warnings worth reading.

        A problem stops the program from starting; a warning is something the
        operator has to know and may well have meant. Catching either before
        the run matters more here than in a one-arm program — a coupled move
        with no ATTACH above it would otherwise be discovered by two arms
        pulling in different directions.

        `config` is the cell's, and only the calibration gate on `pair` needs
        it; without one that gate is skipped rather than guessed at.

        `surfaces` are the places taught in a plane file, when the cell has
        one with a map in it. It decides which library a FIND's `reference` is
        looked up in, because a cell that reads the box off a surface measures
        against a placement the camera recorded and a cell that solves its
        full pose measures against a point somebody jogged to. `None` means
        there is no map, and points are the only answer.
        """
        problems, warnings = [], []
        placed = {"A": False, "B": False}   # is this arm's pose known by now?
        labels = self._labels(problems)
        found = {}                          # what a FIND has looked for so far
        for i, step in enumerate(self.steps):
            if not step.enabled:
                continue
            line = i + 1
            if step.kind == "ATTACH":
                if holding:
                    problems.append("line %d: already holding something" % line)
                holding = True
            elif step.kind == "DETACH":
                if not holding:
                    problems.append("line %d: nothing is held" % line)
                holding = False
            elif step.kind in ("OUT", "WAIT_IN"):
                self._check_io(step, line, problems)
            elif step.kind in ("LABEL", "JUMP", "IF", "SET_VAR"):
                self._check_flow(step, line, labels, problems)
            elif step.kind == "FIND":
                self._check_find(step, line, points, surfaces, problems)
                found[(step.get("into") or "").strip()] = line
            elif step.kind == "MOVE":
                self._check_move(step, line, points, config, holding, placed,
                                 problems, warnings)
            elif step.kind == "CALL":
                if not (step.get("program") or "").strip():
                    problems.append("line %d: CALL names no program" % line)
                # what the called program *contains* is checked when it is
                # expanded, which happens before anything moves and needs a
                # loader this class has no business owning
            self._check_points(step, line, points, problems)
            self._check_correction(step, line, found, problems)
        if holding:
            problems.append("program ends still holding the object — add a DETACH")
        return problems, warnings

    def validate(self, points, config=None, holding=False):
        """Just the problems, for anything about to start the program."""
        return self.check(points, config, holding)[0]

    def _labels(self, problems):
        """Every place a jump can land, and a complaint about any name twice.

        A duplicate label is not a style problem: `jump top` with two `top`s
        means the program goes to one of them and nobody can say which.
        """
        labels = {}
        for i, step in enumerate(self.steps):
            if not step.enabled or step.kind != "LABEL":
                continue
            name = (step.get("name") or "").strip()
            if not name:
                problems.append("line %d: LABEL has no name" % (i + 1))
                continue
            if name in labels:
                problems.append("line %d: there is already a label called %r "
                                "on line %d" % (i + 1, name, labels[name] + 1))
                continue
            labels[name] = i
        return labels

    def _check_find(self, step, line, points, surfaces, problems):
        into = (step.get("into") or "").strip()
        reference = (step.get("reference") or "").strip()
        if not into:
            problems.append("line %d: FIND has nowhere to put what it finds"
                            % line)
        if not reference:
            problems.append("line %d: FIND needs the place the box was taught "
                            "at, to measure how far it has moved" % line)
        elif surfaces is not None:
            if reference not in surfaces:
                problems.append(
                    "line %d: nothing was taught against %r on the surface — "
                    "teach it with the box standing where the pick was taught"
                    % (line, reference))
        elif reference not in points.points:
            problems.append("line %d: no point named %r" % (line, reference))
        if float(step.get("timeout", 0) or 0) < 0:
            problems.append("line %d: a negative timeout waits for nothing"
                            % line)

    def _check_correction(self, step, line, found, problems):
        """A line corrected by a camera needs a FIND above it, in this order.

        Below it is not the same thing, and neither is somewhere a jump might
        reach: the correction has to exist when the line runs, and the only
        arrangement that can be checked on paper is the one that reads down
        the page.
        """
        for key in ("a", "b", "pair", "obj"):
            target = step.slot(key)
            name = (target or {}).get("correct_by")
            if name and name not in found:
                problems.append(
                    "line %d: corrected by %r, but nothing has looked for it "
                    "yet — a FIND has to come above the line it corrects"
                    % (line, name))

    def _check_io(self, step, line, problems):
        arm = step.get("arm")
        if arm not in ("A", "B", "both"):
            problems.append("line %d: %s needs arm A, B or both"
                            % (line, step.kind))
        key = "output" if step.kind == "OUT" else "input"
        try:
            number = int(step.get(key, 0))
        except (TypeError, ValueError):
            number = -1
        if not 0 <= number < IO_RANGE:
            problems.append("line %d: %s %s must be 0 to %d"
                            % (line, step.kind, key, IO_RANGE - 1))
        if step.kind == "WAIT_IN":
            if arm == "both":
                # Two arms have two input words, and "wait until both see it"
                # is a different instruction from "wait until this one does".
                problems.append("line %d: WAIT_IN watches one arm's inputs — "
                                "pick A or B" % line)
            if float(step.get("timeout", 0) or 0) < 0:
                problems.append("line %d: a negative timeout waits for nothing"
                                % line)

    def _check_flow(self, step, line, labels, problems):
        if step.kind == "SET_VAR":
            if not (step.get("name") or "").strip():
                problems.append("line %d: SET_VAR has no variable to set" % line)
            if step.get("op") not in VAR_OPS:
                problems.append("line %d: %r is not one of %s"
                                % (line, step.get("op"), " ".join(VAR_OPS)))
            return
        if step.kind == "LABEL":
            return                      # names were checked while collecting

        for key in ("target", "otherwise"):
            name = (step.get(key) or "").strip()
            if not name:
                if key == "target":
                    problems.append("line %d: %s has nowhere to jump to"
                                    % (line, step.kind))
                continue
            if name not in labels:
                problems.append("line %d: no label called %r" % (line, name))

        if step.kind == "IF":
            if step.get("source") not in IF_SOURCES:
                problems.append("line %d: IF tests an input or a variable, "
                                "not %r" % (line, step.get("source")))
            elif step.get("source") == "var":
                if not (step.get("name") or "").strip():
                    problems.append("line %d: IF has no variable to test" % line)
                if step.get("compare") not in COMPARES:
                    problems.append("line %d: %r is not a comparison"
                                    % (line, step.get("compare")))
            else:
                self._check_io(Step("WAIT_IN", arm=step.get("arm"),
                                    input=step.get("input", 0)),
                               line, problems)

    def _check_points(self, step, line, points, problems):
        for key in ("a", "b", "pair", "obj"):
            target = step.slot(key)
            if not target:
                continue
            name = target.get("point")
            if name and name not in points.points:
                problems.append("line %d: no point named %r" % (line, name))
            motion = target.get("motion")
            if motion and motion not in MOTIONS:
                problems.append("line %d: %r is not a motion" % (line, motion))

    def _check_move(self, step, line, points, config, holding, placed,
                    problems, warnings):
        link = step.link
        if link not in LINKS:
            problems.append("line %d: unknown link %r" % (line, link))
            return
        if link in NEEDS_OBJECT and not holding:
            problems.append("line %d: a coupled move needs an ATTACH first" % line)
        if link in NEEDS_FREE and holding:
            problems.append("line %d: moving an arm while both hold the object "
                            "would tear it out of the grippers" % line)

        filled = step.filled_arms()
        if link in ("solo", "together"):
            if "arm" in step.fields:
                problems.append("line %d: MOVE needs arm A or B" % line)
            elif not filled:
                problems.append("line %d: MOVE has no arm to move" % line)
            elif link == "solo" and len(filled) == 2:
                problems.append("line %d: a solo move drives one arm — set the "
                                "link to 'together' to send both" % line)
            elif link == "together" and len(filled) < 2:
                problems.append("line %d: 'together' needs a target for both "
                                "arms" % line)
        elif filled:
            problems.append("line %d: a %s move drives both arms; leave the arm "
                            "columns empty" % (line, link))

        if link == "pair":
            self._check_pair(step, line, config, problems)
        elif link == "coupled":
            self._check_coupled(step, line, problems)

        self._check_start_known(step, line, link, placed, warnings)

    def _check_pair(self, step, line, config, problems):
        target = step.slot("pair")
        if target_kind(target) is None:
            problems.append("line %d: a pair move needs a world offset" % line)
        else:
            if target_kind(target) != "offset":
                problems.append("line %d: a pair move is an offset, not a place "
                                "— two arms cannot stand on one point" % line)
            if target.get("frame", "world") not in PAIR_FRAMES:
                problems.append("line %d: a pair move is world frame only — "
                                "'base' and 'tool' name a different direction "
                                "at each robot" % line)
            if target_turns(target):
                problems.append("line %d: a pair move cannot rotate — each "
                                "wrist would turn about its own tool and twist "
                                "the workpiece. Attach, and turn it coupled"
                                % line)
        if config is not None and not config.translation_calibrated:
            problems.append("line %d: a pair move needs the relative base "
                            "directions measured — run "
                            "tests/check_directions_online.py --apply" % line)

    def _check_coupled(self, step, line, problems):
        target = step.slot("obj")
        if target_kind(target) is None:
            problems.append("line %d: a coupled move needs a place or an "
                            "offset for the object" % line)
        elif target.get("offset") is not None and \
                target.get("frame", "world") not in OBJECT_FRAMES:
            problems.append("line %d: a coupled offset is read in the world "
                            "frame or the object's own" % line)

    def _check_start_known(self, step, line, link, placed, warnings):
        """An offset with no place in it starts from wherever the arm was left.

        That is the point of it, and it is worth one warning the first time a
        program leans on a starting pose nothing in the program has set.
        """
        if link == "pair":
            unknown = [a for a in ("A", "B") if not placed[a]]
            if target_kind(step.slot("pair")) == "offset" and unknown:
                warnings.append("line %d: the pair is offset from wherever the "
                                "arms were left" % line)
            for a in ("A", "B"):
                placed[a] = True
            return
        if link == "coupled":
            return
        for arm, target in step.arm_slots():
            kind = target_kind(target)
            if kind is None:
                continue
            if kind == "offset" and not placed[arm]:
                warnings.append("line %d: arm %s is offset from wherever it was "
                                "left" % (line, arm))
            placed[arm] = True

    # -- calls ---------------------------------------------------------------
    def expand(self, load, chain=(), depth=0):
        """The lines this program actually runs, with every CALL flattened.

        Resolved before the run rather than during it, for the same reason
        everything else here is: a CALL that cannot be loaded, or one that
        calls itself, would otherwise be discovered with the arms already
        somewhere unpredictable.

        Returns [(step, index)], where the index is the row *in this program*
        the step came from — so the panel can highlight the CALL line while
        the program it names is running, rather than a row that is not there.
        """
        out = []
        for i, step in enumerate(self.steps):
            if step.kind != "CALL":
                out.append((step, i))
                continue
            if not step.enabled:
                continue
            name = (step.get("program") or "").strip()
            if not name:
                raise ValueError("line %d: CALL names no program" % (i + 1))
            if name in chain:
                raise ValueError("line %d: '%s' calls itself (%s)"
                                 % (i + 1, name, " → ".join(chain + (name,))))
            if depth + 1 > CALL_MAX_DEPTH:
                raise ValueError("line %d: '%s' is nested deeper than %d calls"
                                 % (i + 1, name, CALL_MAX_DEPTH))
            try:
                sub = load(name)
            except (OSError, ValueError, KeyError) as e:
                raise ValueError("line %d: cannot load '%s' (%s)"
                                 % (i + 1, name, e))
            if any(t.kind == "LABEL" and t.enabled for t in sub.steps):
                raise ValueError(
                    "line %d: '%s' contains a label, and a called program is "
                    "pasted in wherever it is called — two copies would put "
                    "two jumps on one name. Move the label into the program "
                    "that jumps to it." % (i + 1, name))
            inner = sub.expand(load, chain + (name,), depth + 1)
            for _ in range(max(1, int(step.get("repeat", 1) or 1))):
                out.extend((s, i) for s, _j in inner)
        return out

    def calls(self):
        return [s for s in self.steps if s.kind == "CALL"]

    # -- io ----------------------------------------------------------------
    def to_dict(self):
        return {"name": self.name, "loop": self.loop,
                "steps": [s.to_dict() for s in self.steps]}

    @classmethod
    def from_dict(cls, d):
        p = cls(d.get("name", "untitled"))
        p.loop = bool(d.get("loop", False))
        p.steps = [Step.from_dict(s) for s in d.get("steps", [])]
        return p

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
