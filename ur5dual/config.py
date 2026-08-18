"""
The cell configuration: where each arm is bolted down, how to reach it, and
how hard it is allowed to move.

Everything that needs to know the geometry — the URDF generator, the
object-centric solver, and calibration tools — reads it from here. One file,
one truth. The operator GUI intentionally does not expose cell geometry.
"""

import copy
import math
import os

import numpy as np
import yaml

from .geometry.kinematics import mat_to_xyz_rpy, xyz_rpy_to_mat

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(REPO_ROOT, "config", "cell.yaml")

ARM_IDS = ("A", "B")

DEFAULTS = {
    "cell": {"name": "dual_ur5", "world_frame": "world"},
    "arms": {
        "A": {
            "label": "Arm A",
            "ip": "192.168.250.20",
            "enabled": True,
            "ur_type": "ur5",
            "base": {"xyz": [0.0, 0.30, 0.0], "rpy": [0.0, 0.0, 0.0]},
        },
        "B": {
            "label": "Arm B",
            "ip": "192.168.250.21",
            "enabled": False,
            "ur_type": "ur5",
            "base": {"xyz": [0.0, -0.30, 0.0], "rpy": [0.0, 0.0, 3.141592653589793]},
        },
    },
    # The cell in the photo: a vertical column with a horizontal T crossbar on
    # top, one arm bolted to each end of the bar, both hanging outward and
    # down. Describing it this way means the setup panel asks for four numbers
    # a fitter can measure, instead of twelve a fitter cannot.
    "mount": {
        "style": "pedestal",        # pedestal | table | custom
        "column_height": 1.20,      # m, floor to the crossbar's mounting face
        "spacing": 0.50,            # m, between the two base flange centres
        # how far each base Z axis is tipped away from straight up, outward:
        #   0 = flange horizontal, arm stands up off a table
        #  90 = flange vertical, arm reaches straight out sideways
        # 180 = arm hangs upside down
        #
        # Per arm, because the two brackets are not guaranteed to be mirror
        # images of each other and assuming they were sent a world +Z jog off
        # in two different directions. `tilt_deg` is the fallback for both when
        # a per-arm figure has not been entered.
        #
        # These are the same two numbers the teach pendant shows under
        # Installation -> General -> Mounting, and the robot's own gravity
        # compensation already depends on them being right, so the pendant is
        # the authority.
        "tilt_deg": 120.0,
        "tilt_deg_A": None,
        "tilt_deg_B": None,
        "rotate_deg_A": 0.0,        # about the arm's own base Z, from the pendant
        "rotate_deg_B": 0.0,
        "yaw_deg": 0.0,             # rotation of the whole pair about the column
        # tube frame, drawing only — none of this affects the kinematics.
        # Kept separate from `style` on purpose: a calibrated cell switches to
        # `custom` so the preset stops overwriting measured base transforms,
        # and that has nothing to do with whether the mast is still there to
        # be drawn. Tying the two together made RViz drop the frame the moment
        # the cell was measured.
        "show_frame": True,
        "column_radius": 0.06,
        "crossbar_radius": 0.05,
        "pad_radius": 0.075,        # a UR5 base flange is 149 mm across
        "pad_thickness": 0.022,
        # A gross-error guard for the standard mast layout. Custom measured
        # cells may legitimately mount a base with its Z axis toward the mast;
        # those cells can disable this heuristic without disabling force,
        # drift, reach, or robot safety checks.
        "enforce_outward_guard": True,
    },
    "motion": {
        "backend": "rtde",          # rtde | urscript
        # What the coordinated loop actually sends, once per cycle:
        #   pose    a Cartesian target per arm; each controller solves its own
        #           IK. Simple, and how this cell has always run.
        #   joint   six joint angles per arm, solved here by closed_chain.py
        #           from one box pose. Both arms stay on the IK branch chosen
        #           at ATTACH, and the whole path is checked before it starts.
        # `joint` requires the Python FK to match these robots — run
        # tests/check_chain_online.py before switching, because a wrong DH
        # table is invisible until two arms are holding something.
        "control": "pose",          # pose | joint
        # Explicit commissioning escape hatch. It permits only held-button
        # jogging at the very low caps in coupling.py; planned/program moves
        # remain refused until directions have actually been measured.
        "allow_uncalibrated_real": False,
        "servo_rate_hz": 125.0,
        "servo_lookahead": 0.1,
        "servo_gain": 300.0,
        "servo_accel": 1.2,
        "servo_speed": 0.5,
        # An arm holding nothing reads tens of newtons here if its payload is
        # not configured to match the tool actually bolted on — the estimate
        # is motor current minus a model, and a wrong model is a standing
        # offset. So the working guard is the *rise* since the grip was taken,
        # which cancels that offset; the absolute figure stays as a backstop
        # for a genuine hard collision.
        # Measured on this cell with the payload unset: an arm holding
        # nothing reads 29-44 N standing still, and ordinary 5 mm coordinated
        # steps invent up to 66 N more. So the limit has to clear 66 N to stop
        # false trips, which leaves it too blunt to notice a bottle being
        # squeezed. Setting the payload and TCP on both pendants to match the
        # tools actually fitted is what earns the right to tighten it — expect
        # to reach 20-30 N once the model matches the hardware.
        "max_force_rise": 100.0,    # N above the reading captured at ATTACH
        "max_tcp_force": 250.0,     # N absolute backstop, hard collisions only
        "max_pair_drift": 0.004,    # m  — abort if the grasp separation moves
    },
    # Set true only by a completed touch-off calibration. Coordinated motion
    # is refused until then: with the geometry unmeasured, a world-frame jog
    # sends each arm somewhere different, which looks exactly like the arms
    # disagreeing about which way is up — because they do.
    "calibrated": False,
    # Set by the hand-taught direction calibration. Enough for the two arms to
    # carry an object left/right/up/down together, which needs only their
    # relative orientation; not enough to rotate the object, which also needs
    # the distance between the bases that motion alone cannot reveal.
    "translation_calibrated": False,
    "limits": {
        "jog_lin_max": 0.25,        # m/s
        "jog_ang_max": 1.0,         # rad/s
        "object_lin_speed": 0.05,   # m/s   default for MOVE_OBJ
        "object_ang_speed": 0.30,   # rad/s default for ROTATE_OBJ
    },
}

HEADER = """\
# Dual-UR5 cell configuration.
#
# arms.<id>.base is the transform from the cell's `world` frame to that arm's
# base, in metres and fixed-axis roll-pitch-yaw — the same convention a URDF
# <origin> tag uses. Calibration tools can overwrite arm B's transform with
# the measured value.
#
# mount.style = pedestal derives those base transforms from four measurable
# numbers instead (column height, shoulder spacing, outward tilt, pair yaw).
# Any edit to arms.<id>.base by hand or by the wizard switches it to custom.
#
# mount.show_frame only decides whether RViz draws the mast and crossbar. It
# is deliberately independent of style, so a calibrated (custom) cell still
# shows the structure it is bolted to. Set it false for a cell with no mast.
#
# motion.backend picks how the 125 Hz coordinated loop talks to the robots:
#   rtde      ur_rtde servoL/servoJ  (default)
#   urscript  a streaming URScript program uploaded to each controller
"""


def _deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ArmConfig:
    def __init__(self, arm_id, data):
        self.id = arm_id
        self._d = data

    label = property(lambda self: self._d["label"])
    ur_type = property(lambda self: self._d["ur_type"])

    @property
    def ip(self):
        return self._d["ip"]

    @ip.setter
    def ip(self, value):
        self._d["ip"] = str(value)

    @property
    def enabled(self):
        return bool(self._d["enabled"])

    @enabled.setter
    def enabled(self, value):
        self._d["enabled"] = bool(value)

    @property
    def xyz(self):
        return np.array(self._d["base"]["xyz"], dtype=float)

    @property
    def rpy(self):
        return np.array(self._d["base"]["rpy"], dtype=float)

    def base_matrix(self):
        """world -> this arm's base."""
        return xyz_rpy_to_mat(self.xyz, self.rpy)

    def set_base_matrix(self, T):
        xyz, rpy = mat_to_xyz_rpy(T)
        self._d["base"]["xyz"] = [round(float(v), 6) for v in xyz]
        self._d["base"]["rpy"] = [round(float(v), 8) for v in rpy]

    def set_base(self, xyz, rpy):
        self._d["base"]["xyz"] = [float(v) for v in xyz]
        self._d["base"]["rpy"] = [float(v) for v in rpy]


class CellConfig:
    def __init__(self, data=None, path=DEFAULT_PATH):
        self.path = path
        self._d = _deep_merge(DEFAULTS, data or {})
        self.arms = {a: ArmConfig(a, self._d["arms"][a]) for a in ARM_IDS}

    # -- io ----------------------------------------------------------------
    @classmethod
    def load(cls, path=DEFAULT_PATH):
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            data = {}
        return cls(data, path)

    def save(self, path=None):
        path = path or self.path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(HEADER)
            yaml.safe_dump(self._d, f, sort_keys=False, default_flow_style=None)
        return path

    # -- convenience -------------------------------------------------------
    @property
    def name(self):
        return self._d["cell"]["name"]

    @property
    def world_frame(self):
        return self._d["cell"]["world_frame"]

    @property
    def mount(self):
        return self._d["mount"]

    @property
    def motion(self):
        return self._d["motion"]

    # -- pedestal layout ---------------------------------------------------
    def apply_mount_preset(self):
        """Recompute both arm bases from the mount parameters.

        Arm A sits on the +Y end of the crossbar and arm B on the -Y end, each
        tipped by `tilt_deg` *away* from the column, with the whole pair yawed
        about the column by `yaw_deg`.

        The sign matters and is easy to get backwards. A roll of +t about X
        sends the base Z axis to (0, -sin t, cos t), which for the arm on the
        +Y end points back towards the column — the arms would face each other
        across the mast instead of reaching out. Hence the negated side.

        Called whenever the setup panel changes a mount number. Does nothing
        in `custom` style, which is what a calibrated cell switches to.
        """
        m = self.mount
        if m["style"] != "pedestal":
            return False

        yaw = math.radians(float(m["yaw_deg"]))
        half = float(m["spacing"]) / 2.0
        z = float(m["column_height"])

        T_yaw = xyz_rpy_to_mat([0.0, 0.0, 0.0], [0.0, 0.0, yaw])
        for arm_id, side in (("A", +1.0), ("B", -1.0)):
            tilt = math.radians(self.tilt_of(arm_id))
            spin = math.radians(float(m.get("rotate_deg_%s" % arm_id, 0.0) or 0.0))
            T_local = xyz_rpy_to_mat([0.0, side * half, z], [-side * tilt, 0.0, 0.0])
            # the pendant's "rotate" turns the arm on its own flange
            T_local = T_local @ xyz_rpy_to_mat([0, 0, 0], [0, 0, spin])
            self.arms[arm_id].set_base_matrix(T_yaw @ T_local)
        return True

    def tilt_of(self, arm_id):
        """Degrees of tilt for one arm, falling back to the shared figure."""
        per_arm = self.mount.get("tilt_deg_%s" % arm_id)
        return float(self.mount["tilt_deg"] if per_arm is None else per_arm)

    def mount_frame_matrix(self):
        """world -> the T-frame: origin in the middle of the crossbar, +Y along
        the bar towards arm A, +Z up the mast.

        Read out of where the two flanges actually are, not out of the mount
        numbers, because the flanges are what the structure is welded to. The
        preset builds the bases from `world` and so did the drawing, which is
        fine until something moves the bases relative to `world` — levelling
        the cell against gravity turns both of them by the same rotation, and
        a mast drawn from `world` alone stays standing perfectly upright while
        the arms it is holding tip over. That picture is of no cell that
        exists, and it is the one thing in the view an operator uses to judge
        whether a calibration came out sane.

        The bar joins the two flange centres, so it cannot come away from
        them. The mast is drawn plumb — square to the bar, and as near world
        up as that leaves it.

        Plumb by assumption, and deliberately: a bolted steel column is, and
        the alternative is worse. Reading the mast's direction back out of the
        pad orientations sounds more honest and is a trap — every error in how
        the arms are bolted to the bar then comes out as the whole cell
        leaning, which is a picture convincing enough to send somebody
        levelling a column that was never out of plumb. Bracket error belongs
        on the pads, where it shows as an arm sitting askew on a square bar.
        A bar tipped off level, or a foot that has walked off the floor, is
        then the signature of something claiming the cell itself moved.
        """
        pa = self.arms["A"].xyz
        pb = self.arms["B"].xyz
        T = np.eye(4)
        T[:3, 3] = (pa + pb) / 2.0

        bar = pa - pb
        length = float(np.linalg.norm(bar))
        if length < 1e-9:          # one arm on top of the other: no bar to lie along
            return T
        y = bar / length

        for candidate in (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])):
            square = candidate - float(candidate @ y) * y
            norm = float(np.linalg.norm(square))
            if norm > 1e-9:        # a bar stood on end has no plumb square to it
                z = square / norm
                break

        T[:3, 0] = np.cross(y, z)
        T[:3, 1] = y
        T[:3, 2] = z
        return T

    @property
    def calibrated(self):
        return bool(self._d.get("calibrated", False))

    @calibrated.setter
    def calibrated(self, value):
        self._d["calibrated"] = bool(value)
        if value:
            self._d["translation_calibrated"] = True

    @property
    def translation_calibrated(self):
        """True once carrying the object in a straight line is trustworthy."""
        return bool(self._d.get("translation_calibrated", False)
                    or self._d.get("calibrated", False))

    @translation_calibrated.setter
    def translation_calibrated(self, value):
        self._d["translation_calibrated"] = bool(value)

    def base_axis_outward(self, arm_id):
        """Cosine between an arm's base Z axis and the direction away from the
        mast. Positive means the arm reaches out, which is the whole point of
        the crossbar; negative means the geometry is inside out.

        "Away from the mast" is taken from where the base actually sits — its
        horizontal offset from the mast axis — rather than assumed to be +/-Y,
        so this stays meaningful after a yaw and for a hand-measured cell.
        """
        import numpy as np
        base = self.arms[arm_id].base_matrix()
        outward = base[:3, 3].copy()
        outward[2] = 0.0
        norm = float(np.linalg.norm(outward))
        if norm < 1e-9:
            return 0.0
        return float(base[:3, 2] @ (outward / norm))

    def set_custom_mount(self):
        """Mark the geometry as hand-measured, so presets stop overwriting it."""
        self._d["mount"]["style"] = "custom"

    @property
    def limits(self):
        return self._d["limits"]

    def enabled_arms(self):
        return [a for a in ARM_IDS if self.arms[a].enabled]

    def base_separation(self):
        """Straight-line distance between the two arm bases, in metres."""
        return float(np.linalg.norm(self.arms["A"].xyz - self.arms["B"].xyz))

    def a_to_b(self):
        """Transform from arm A's base to arm B's base.

        This is the number that actually matters for two-arm grasping; the
        individual world placements only decide how the cell is drawn.
        """
        from .geometry.kinematics import inv
        return inv(self.arms["A"].base_matrix()) @ self.arms["B"].base_matrix()
