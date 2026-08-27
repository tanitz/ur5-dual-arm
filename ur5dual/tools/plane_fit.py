"""
Collecting what a plane map is fitted from.

The job is the calibration. Two arms carry the box, set it down and are still
gripping it: their poses at that moment are a frame rigidly attached to it.
Take them out of the view and the camera says where the rim is. Three of those
pairs fix the map from pixels onto the surface *and* where each arm had hold —
and every one of them costs only a pick-and-place that was going to happen.

There used to be a second way in: drive one arm to each of four rim corners,
three or four times over. It worked, and it is gone. It asked an operator to
do a careful separate job to learn something the real job was already saying,
and a procedure nobody has a reason to run is a procedure that is wrong by the
time anybody tries it.

The session is deliberately free of both Qt and sockets. It is handed poses
and handed pictures, and knows nothing about where either came from — which is
what lets the whole procedure be exercised against a rendered box, with a
known answer, rather than only against a real one where the answer is what is
being measured.

Two things are asked of an operator and nothing else is:

  press with the arms on it      right after the box is set down, before
                                 anything lets go.

  press again once they are out  of the view. An arm over the box is the one
                                 thing guaranteed to be in the way of the rim,
                                 and every capture forgets the detector's
                                 temporal lock first, so a sample is a
                                 measurement rather than a blend with wherever
                                 the box was standing a moment ago.

The raw cycles outlive the fit. `config/plane_log.json` keeps every arm pose
and every corner found, so a map can be re-fitted after a box size is
corrected without carrying the box round the cell again — the same reason
`flange_log.json` sits beside the flange fit rather than inside it.
"""

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from ..vision.planar import (
    PlaneFile, PlaneMapError, box_on_plane, fit_from_cycles, stance_frame,
    placement_of,
)

DEFAULT_LOG = "config/plane_log.json"

CORNERS = 4
# And the fewest pick-and-places that fix a map. Three is the fewest that can
# disagree about where the arms stand on the box; five or six is what a real
# answer looks like, because every one of them costs only the job being done.
ENOUGH_CYCLES = 3

# What the cycles have to disagree about before they can settle anything.
#
# A map of one plane can only be fitted from a box that was on that plane
# every time: the arms holding it at different heights means it was in the air
# rather than set down, and no map explains both. Measured on this cell's
# first attempt, the tools were at heights 248 mm apart and the fit came out
# 34 mm wrong with a message blaming the wrong thing.
MAX_HEIGHT_SPREAD = 0.020
# And it has to have gone somewhere, in two directions rather than one. The
# span that matters is the *smaller* of the two the placements cover: a box
# slid up and down one line is the same argument `calibrate.spread_of` makes
# one dimension up, and it fits perfectly along that line while saying nothing
# across it. Measured on this cell's first attempt: 25 mm one way and 9 mm the
# other, which is a line.
MIN_TRAVEL = 0.020
# Turning it is what pins down where the arms have hold of it. Without any,
# the offset is loose — harmless for a pick, because the map absorbs the same
# looseness, but worth saying while the box is still in reach.
MIN_TURN = math.radians(8.0)



class PlaneFitError(RuntimeError):
    pass


class Cycled:
    """One pick-and-place: where the arms were, and what the camera then saw.

    The two halves are read at different moments on purpose — the arms are on
    the box for the first and out of the view for the second — and the box
    must not move between them. Kept together because neither is worth
    anything without the other.
    """

    def __init__(self, tools, corners, taken=None):
        self.tools = {str(a): np.asarray(m, dtype=float).reshape(4, 4)
                      for a, m in dict(tools).items()}
        self.corners = np.asarray(corners, dtype=float).reshape(CORNERS, 2)
        self.taken = taken or datetime.now().astimezone().isoformat(
            timespec="seconds")

    @property
    def frame(self):
        return stance_frame([self.tools[a] for a in sorted(self.tools)])

    def to_dict(self):
        return {"taken": self.taken,
                "tools": {a: [[float(v) for v in row] for row in m]
                          for a, m in sorted(self.tools.items())},
                "corners": [[float(v) for v in c] for c in self.corners]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["tools"], d["corners"], d.get("taken"))

    def describe(self, index=None):
        span = np.linalg.norm(
            [self.tools[a][:2, 3] for a in sorted(self.tools)][0]
            - [self.tools[a][:2, 3] for a in sorted(self.tools)][-1])
        return ("%sarms %s, %.0f mm apart"
                % ("" if index is None else "%d. " % (index + 1),
                   "+".join(sorted(self.tools)), span * 1000))


class PlaneFitSession:
    """The pick-and-places collected so far, and the map they come to.

    The two halves of one are separate presses because they happen at
    different moments: the arms on the box, then the picture once they are
    clear. A session that took the picture when the arms were recorded would
    photograph the arms every time.
    """

    def __init__(self, box_size, path=None, cycles=None, name=None):
        self.box_size = tuple(float(v) for v in box_size)
        self.path = Path(path or DEFAULT_LOG)
        self.cycles = list(cycles or [])
        self.pending_tools = None
        # What the reference will be called. Settled by the first press and
        # not editable after it, because the name is what the whole run of
        # cycles is collected under: a keystroke that landed in the field half
        # way through would finish the job under a name no program uses.
        self.name = str(name).strip() if name else None

    # -- the job as the calibration ---------------------------------------
    def ready(self, tools, name=None):
        """Where the arms stand, posed at a box that is on the table.

        Not gripping it: the pose that matters is the one a pick starts from,
        flanges parallel to the box's sides, and how far short of it they stop
        is an offset the program adds afterwards. What has to be the same
        every round is the pose *relative to the box*, because that constant
        is the one the fit solves for.
        """
        if self.name is None and name:
            self.name = str(name).strip()
        if not tools:
            raise PlaneFitError(
                "no arm is holding the box, so there is nothing to record")
        self.pending_tools = {str(a): np.asarray(m, float).reshape(4, 4)
                              for a, m in dict(tools).items()}
        return self.pending_tools

    def look(self, corners):
        """The picture taken once the arms are out of the way, closing it."""
        if not self.pending_tools:
            raise PlaneFitError(
                "stand the arms ready at the box and press first — where they "
                "were is half of this")
        cycle = Cycled(self.pending_tools, corners)
        self.cycles.append(cycle)
        self.pending_tools = None
        return cycle

    def spreads(self):
        """How much the cycles differ, in the three ways that matter."""
        heights = np.array([c.tools[a][2, 3]
                            for c in self.cycles for a in c.tools])
        frames = [c.frame for c in self.cycles]
        origins = np.array([[f[0, 2], f[1, 2]] for f in frames])
        angles = np.array([math.atan2(f[1, 0], f[0, 0]) for f in frames])
        turn = 0.0
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                turn = max(turn, abs(math.atan2(math.sin(angles[i] - angles[j]),
                                                math.cos(angles[i] - angles[j]))))
        if len(origins) < 2:
            travel = np.zeros(2)
        else:
            centred = origins - origins.mean(axis=0)
            travel = np.linalg.svd(centred, full_matrices=False)[1][:2] \
                / math.sqrt(len(origins))
        return (float(np.ptp(heights)) if len(heights) else 0.0,
                travel, turn)

    def not_on_one_surface(self):
        """The one fault that no residual can be allowed to argue with.

        A map is a map of one plane. Arms holding the box at different heights
        did not put it on one, and a fit that comes out tidy anyway has agreed
        with data whose premise is false — which is the failure this whole
        module exists to make impossible, not one to weigh against a number.
        """
        if len(self.cycles) < 2:
            return ""
        height, _travel, _turn = self.spreads()
        if height <= MAX_HEIGHT_SPREAD:
            return ""
        return ("the arms stood at heights %.0f mm apart between rounds, so "
                "the pose they were put in was not the same one relative to "
                "the box each time — stand them the same way at it, with the "
                "box flat on the table"
                % (height * 1000))

    def why_not_fitting(self):
        """What is wrong with the cycles themselves, in order of severity.

        Asked before the residual is blamed on anything, because the residual
        cannot tell these apart and the generic answer names a cause that
        cannot apply here: nobody chose the plane's axes, the arms did.
        """
        if len(self.cycles) < 2:
            return ""
        airborne = self.not_on_one_surface()
        if airborne:
            return airborne
        _height, travel, turn = self.spreads()
        if float(np.min(travel)) < MIN_TRAVEL:
            return ("the box covered %.0f by %.0f mm between rounds, which is "
                    "%s — put it down well away from where it was, and to "
                    "the side as well as along"
                    % (travel[0] * 1000, travel[1] * 1000,
                       "one place seen several times"
                       if float(np.max(travel)) < MIN_TRAVEL
                       else "a line"))
        if turn < MIN_TURN:
            return ("the box was never turned — every round has it facing "
                    "within %.1f deg of the last, and turning it is what says "
                    "where the arms have hold of it"
                    % math.degrees(turn))
        return ""

    def fit_cycles(self, height=None):
        """The map and the holds, from pick-and-places alone."""
        airborne = self.not_on_one_surface()
        if airborne:
            raise PlaneFitError(airborne)
        wrong = self.why_not_fitting()
        try:
            plane_map, offset = fit_from_cycles(
                [(c.frame, c.corners) for c in self.cycles], self.box_size,
                height=height)
        except PlaneMapError as exc:
            raise PlaneFitError(wrong or str(exc))
        return plane_map, offset

    def apply(self, plane_map, plane_path=None):
        """Put a fitted map into the file a program reads, keeping the rest."""
        store = PlaneFile.load(plane_path)
        store.set_map(plane_map)
        store.box_size = self.box_size
        store.save(plane_path)
        return store

    @property
    def collecting(self):
        """Is a run of cycles under way, so the name is settled?"""
        return bool(self.cycles or self.pending_tools)

    def taught_from_cycles(self, name, plane_path=None, height=None):
        """Fit, write the map, and record what the last cycle already taught.

        One action produced all of it: where the box is when the camera sees
        it, and where each arm had hold of it. Teaching them separately would
        be teaching a moment that was never observed.
        """
        plane_map, offset = self.fit_cycles(height=height)
        store = self.apply(plane_map, plane_path)
        last = self.cycles[-1]
        store.teach(name, placement_of(last.frame, offset))
        # The tool poses as they were in the cell, not already converted:
        # `teach_stance` takes the difference itself, against the placement just
        # recorded. Handing it `stances_from_cycle`'s answer would take that
        # difference twice.
        store.teach_stance(name, last.tools)
        store.save(plane_path)
        return store, plane_map, offset

    # -- the log -----------------------------------------------------------
    @classmethod
    def load(cls, path=None, box_size=None):
        path = Path(path or DEFAULT_LOG)
        if not path.exists():
            if box_size is None:
                raise PlaneFitError(
                    "%s does not exist yet and nothing says what size the box "
                    "is" % path)
            return cls(box_size, path)
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PlaneFitError("%s cannot be read: %s" % (path, exc))
        size = box_size or held.get("box_size")
        if size is None:
            raise PlaneFitError("%s does not say what size the box is" % path)
        try:
            cycles = [Cycled.from_dict(d) for d in held.get("cycles") or []]
        except (KeyError, TypeError, ValueError) as exc:
            raise PlaneFitError("%s is not a plane log: %s" % (path, exc))
        return cls(size, path, cycles, held.get("name"))

    def save(self, path=None):
        path = Path(path or self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({
                "saved": datetime.now().astimezone().isoformat(
                    timespec="seconds"),
                "box_size": list(self.box_size),
                "name": self.name,
                "cycles": [c.to_dict() for c in self.cycles],
            }, f, indent=2)
        self.path = path
        return path

    # -- what to tell an operator -----------------------------------------
    def describe(self):
        if self.pending_tools:
            return ("arms recorded — take them out of the view and press "
                    "again")
        if not self.cycles:
            return ("nothing collected yet — set the box down with both arms "
                    "still holding it and press")
        lines = [c.describe(i) for i, c in enumerate(self.cycles)]
        if len(self.cycles) < ENOUGH_CYCLES:
            lines.append("%d of %d: three are the fewest that can disagree "
                         "about where the arms stand"
                         % (len(self.cycles), ENOUGH_CYCLES))
        else:
            lines.append("%d pick-and-places" % len(self.cycles))
        return "\n".join(lines)
