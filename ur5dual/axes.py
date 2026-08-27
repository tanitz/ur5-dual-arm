"""The X an operator reads, which is not the X the cell stores.

Everything underneath — points.json, the poses handed to the arms, the
transforms in geometry — is in the cell's world frame and stays there. But on
this mast the work area lies on world -X, so the jog key marked X+ means "out
from the mast" and drives world X *down*. That flip has lived in the jog page
since the keys were built.

A readout printing the raw world number therefore contradicted the key the
operator had just held: press X+, watch x fall from -603 to -613. One constant
and two helpers, shared by the keys and by every readout, are what keep the
two telling the same story.

Rotation is deliberately left alone. The RX/RY/RZ keys drive the raw world
axes, so a flipped rx would recreate precisely the disagreement this module
exists to remove.
"""

# Only X is reversed, and only for display and for the linear jog keys.
WORLD_AXIS_SIGN = (-1.0, 1.0, 1.0)


def shown_xyz(xyz):
    """The three world translations as the jog keys count them, in metres."""
    return tuple(float(v) * sign
                 for v, sign in zip(xyz[:3], WORLD_AXIS_SIGN))


def shown_pose(pose):
    """A world 6-vector ready to print: translation flipped, rotation as-is."""
    pose = [float(v) for v in pose]
    return list(shown_xyz(pose)) + pose[3:]
