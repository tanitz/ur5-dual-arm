"""Pure geometry: poses, UR5 kinematics, the closed chain, the touch-off solver.

Nothing in here opens a socket or reads a clock. Given the same numbers it
returns the same numbers, which is why every unit test points at this package
and why it can be trusted while two arms are holding something.
"""
