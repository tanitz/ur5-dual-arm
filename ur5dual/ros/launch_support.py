"""
Turning cell.yaml into xacro arguments.

The description and the control code must agree about where the arms are, or
RViz quietly lies about a cell you are trusting with two-arm grasps. Both
read the same YAML, and this is the single place the geometry is translated
into the form xacro wants.
"""

import os

from ..config import DEFAULT_PATH, CellConfig
from ..geometry.kinematics import mat_to_xyz_rpy

DESCRIPTION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "description")
XACRO_FILE = os.path.join(DESCRIPTION_DIR, "dual_ur5.urdf.xacro")


def _fmt(values):
    return " ".join("%.9g" % float(v) for v in values)


def xacro_mappings(config=None, config_path=DEFAULT_PATH):
    """{arg: value} for the dual-arm description."""
    cfg = config or CellConfig.load(config_path)
    m = {
        "ur_type": cfg.arms["A"].ur_type,
        "a_xyz": _fmt(cfg.arms["A"].xyz),
        "a_rpy": _fmt(cfg.arms["A"].rpy),
        "b_xyz": _fmt(cfg.arms["B"].xyz),
        "b_rpy": _fmt(cfg.arms["B"].rpy),
        # drawing the mast is independent of `style`: a measured cell is still
        # bolted to the same structure, it just no longer takes its base
        # transforms from the preset
        "show_frame": "true" if cfg.mount.get("show_frame", True) else "false",
        "column_height": "%.6g" % float(cfg.mount["column_height"]),
        "column_radius": "%.6g" % float(cfg.mount.get("column_radius", 0.06)),
        "crossbar_radius": "%.6g" % float(cfg.mount.get("crossbar_radius", 0.05)),
        "pad_radius": "%.6g" % float(cfg.mount.get("pad_radius", 0.075)),
        "pad_thickness": "%.6g" % float(cfg.mount.get("pad_thickness", 0.022)),
        # the structure hangs off the flanges rather than off `world`, so it
        # follows them through a levelling or a touch-off instead of staying
        # bolt upright while the arms lean. mount.spacing and mount.yaw_deg
        # are inputs to the preset that placed the bases; the bar is drawn
        # between where those bases ended up, which is the same thing until
        # something measures the cell and different afterwards.
        "bar_length": "%.6g" % cfg.base_separation(),
    }
    frame_xyz, frame_rpy = mat_to_xyz_rpy(cfg.mount_frame_matrix())
    m["frame_xyz"] = _fmt(frame_xyz)
    m["frame_rpy"] = _fmt(frame_rpy)
    return m


def xacro_command(config=None, config_path=DEFAULT_PATH):
    """The `xacro <file> k:=v ...` argv, for launch files and for humans."""
    mappings = xacro_mappings(config, config_path)
    return ["xacro", XACRO_FILE] + ["%s:=%s" % (k, v) for k, v in mappings.items()]


def xacro_command_string(config=None, config_path=DEFAULT_PATH):
    """The same command as one shell string.

    launch's Command substitution re-splits on whitespace, and three of these
    values are space-separated vectors — without quoting, `a_xyz:=0 0.15 0.9`
    arrives as three arguments and the arm lands at the origin.
    """
    import shlex
    argv = xacro_command(config, config_path)
    return " ".join(shlex.quote(a) for a in argv)


def build_urdf(config=None, config_path=DEFAULT_PATH):
    """Run xacro now and return the URDF as a string.

    Used by the GUI, which regenerates the description whenever the mounting
    geometry is edited so RViz can be restarted onto the new cell.
    """
    import subprocess
    out = subprocess.run(xacro_command(config, config_path),
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("xacro failed: %s" % out.stderr.strip())
    return out.stdout
