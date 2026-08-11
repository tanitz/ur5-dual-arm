"""
The step vocabulary a two-arm program is written in.

Half of these address one arm — approach, retreat, open a gripper — and the
other half address the *object* both arms are holding. That split is the
whole point of the object-centric concept: while the arms are apart they are
programmed separately, and the moment they take hold of something together
they stop being two arms and become one carried object.

ATTACH is where the change of mode happens: it freezes each arm's grip and
from then on MOVE_OBJ and ROTATE_OBJ steer both arms at once. DETACH gives
them back their independence.
"""

import json

import numpy as np

# every step kind, with the fields it carries
STEP_KINDS = {
    "MOVE_ARM":   ("arm", "point", "motion", "speed"),
    "MOVE_OBJ":   ("point", "lin_speed", "ang_speed"),
    "ROTATE_OBJ": ("axis", "angle_deg", "ang_speed", "frame"),
    "ATTACH":     ("object", "origin"),
    "DETACH":     (),
    "GRIP":       ("arm", "output", "state"),
    "BARRIER":    (),
    "DELAY":      ("seconds",),
}

# which steps need the arms coupled, and which need them free
NEEDS_OBJECT = ("MOVE_OBJ", "ROTATE_OBJ", "DETACH")
NEEDS_FREE = ("MOVE_ARM",)


class Step:
    def __init__(self, kind, **fields):
        if kind not in STEP_KINDS:
            raise ValueError("unknown step kind %r" % kind)
        self.kind = kind
        self.fields = dict(fields)
        self.enabled = bool(fields.pop("enabled", True))

    def get(self, name, default=None):
        return self.fields.get(name, default)

    def describe(self):
        f = self.fields
        if self.kind == "MOVE_ARM":
            return "arm %s  %s -> %s" % (f.get("arm"), f.get("motion", "movej"),
                                         f.get("point"))
        if self.kind == "MOVE_OBJ":
            return "carry object -> %s" % f.get("point")
        if self.kind == "ROTATE_OBJ":
            return "spin object %+.1f deg about %s %s" % (
                float(f.get("angle_deg", 0)),
                "the world's" if f.get("frame") == "world" else "its own",
                f.get("axis", "z"))
        if self.kind == "ATTACH":
            return "take hold of '%s' (origin %s)" % (
                f.get("object", "object"), f.get("origin", "midpoint"))
        if self.kind == "DETACH":
            return "let go"
        if self.kind == "GRIP":
            return "arm %s  DO%s %s" % (f.get("arm"), f.get("output"),
                                        "ON" if f.get("state") else "OFF")
        if self.kind == "BARRIER":
            return "wait for both arms to settle"
        if self.kind == "DELAY":
            return "wait %.2f s" % float(f.get("seconds", 0))
        return self.kind

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
    def validate(self, points):
        """Problems findable without moving: unknown points, and coupling
        steps that appear when nothing is held.

        Catching these before the run matters more here than in a one-arm
        program — a MOVE_OBJ with no ATTACH before it would otherwise be
        discovered by two arms pulling in different directions.
        """
        problems = []
        holding = False
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
            elif step.kind in NEEDS_OBJECT and not holding:
                problems.append("line %d: %s needs an ATTACH first"
                                % (line, step.kind))
            elif step.kind in NEEDS_FREE and holding:
                problems.append("line %d: moving one arm while both hold the "
                                "object would tear it out of the grippers" % line)

            name = step.get("point")
            if name and name not in points.points:
                problems.append("line %d: no point named %r" % (line, name))
            if step.kind == "MOVE_ARM" and step.get("arm") not in ("A", "B"):
                problems.append("line %d: MOVE_ARM needs arm A or B" % line)
        if holding:
            problems.append("program ends still holding the object — add a DETACH")
        return problems

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
