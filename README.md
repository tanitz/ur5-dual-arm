# Dual UR5 cell

Two UR5s on one column, driven from a Jetson over the robots' own sockets —
no ROS in the control path, no URCap on the pendant.

Programs are written for the **object**, not for the arms. Once both grippers
hold one bottle or one drum, their TCPs are no longer free: each is fixed to
the object, and therefore to the other. So you say where the object goes, and
both arms' paths fall out of that constraint.

```
T_world_tcpA = T_world_object · grasp_A
T_world_tcpB = T_world_object · grasp_B
```

`grasp_A` and `grasp_B` are captured once, when the grippers close.

In the code this virtual workpiece frame is a `HeldObject`. Pressing ATTACH
creates its `[x, y, z, rx, ry, rz]` `pose_world` from the live TCPs: midpoint
puts it halfway between the grippers, while origins A and B provide the same
geometry in leader-follower form. The object has no CAD dimensions; it is the
frame plus the two captured grasp transforms. Translating or rotating that one
frame produces both arms' targets from the equations above.

All six degrees of freedom are driven the same way, and each can be taken in
either of two frames. Translation along the cell's own axes is what a jog
means by +X; rotation has the same choice, and it is only which side the
rotation multiplies on:

```
world axis    R_new = Rk(theta) · R_old      RZ is the vertical of the room
object axis   R_new = R_old · Rk(theta)      RZ is along the bottle
```

Turning happens about the object frame's origin, so for a midpoint grasp the
box spins between the grippers — the one pivot that costs both arms the same
travel. `command_rotate(..., pivot=p)` swings it around a place in the cell
instead.

All six buttons are on the object grid from the start. Turning needs the
distance between the two bases, which only a touch-off measures, but the error
a wrong distance produces grows with the angle — so before that measurement
exists the box may still be turned 5 degrees a press at 2 deg/s, which is
enough to check that RX turns it the way the label says and far inside what the
drift guard catches. The touch-off lifts the limit rather than unlocking the
feature.

## The panel

Two working halves at 1280x800, the shape of a teach pendant. The REAL cell
bar belongs to the right-hand tab column rather than spanning the program.

```
┌──────────────────────────┬──────────────────────────────┐
│ Program — teach & run    │ REAL|A|B|Power|Brake|Unlock|STOP
│   # step detail          │ [Points][Vars][Object][Jog]  │
│   ...                    │ Arm A | A+B | Arm B          │  6 rows
│ [▲][▼][Delete][On/off]   │ X Y Z RX RY RZ, each with −/+│
│ [▶ Run][⏸][■ Stop]       │                              │
│ [Save][Load][Loop][Dry]  │                              │
├──────────────────────────┴──────────────────────────────┤
│ Messages: newest line                         [Expand]  │
└─────────────────────────────────────────────────────────┘
```

The left half is the program and never changes: it is the document being
written, and it is what the panel is for. The right half is everything you do
to build it — the named places (`Points`), the cell's own numbers and the
touch-off (`Vars`), taking hold and driving what is held (`Object`), and the
jog grid.

1280x800 is the size every number in the layout is quoted in, not a size the
screen has to be. On startup the panel measures the desktop's work area — what
is left after its dock and the window's own title bar — and scales the whole
design down to fit, never up. On this cell's 10" panel that leaves 1232x736
and comes out at 0.92; `--fullscreen` has no frame to pay for and gets the
design size back at 1.0, which is the largest the touch targets ever are.
`--scale` overrides both.

The fixed REAL indicator, connection and dashboard controls, and the cell-wide
STOP stay in one row above the tabs. The newest message stays in a one-line
drawer at the bottom; `Expand` opens its history. Compact A/B TCP and force
readouts plus the pair gap sit below the Jog keys, and `Details` opens the full
XYZ, RX/RY/RZ, J1–J6, force, robot/safety state, TCP gap, drift, and holding
state in a separate window.

## Jogging by hand

The Jog tab drives arms, not objects. Arm A, synchronized A+B, and arm B are
three permanent columns; there is no Drive selector:

| Drive | What a press does |
|---|---|
| `Arm A` / `Arm B` | one arm, in the cell frame, that arm's base, its tool, or a joint at a time |
| `Synchronized A+B` | the same cell-frame direction to both arms at once |

All six axes are visible at once: X/Y/Z and RX/RY/RZ. In joint mode the same
rows become J1 through J6. A and B can use world, base, tool, or joint frames;
the A+B column is enabled only in world frame.

`Synchronized A+B` is the single-arm world jog issued twice, once per arm,
each direction resolved through that arm's own mounting transform. Two arms
given the same world direction travel the same way at the same speed, so a
workpiece gripped between them is carried — the pairing is in the geometry, not
in any attempt to keep two commands in step. REAL A+B therefore requires the
relative base directions to have been measured (`translation_calibrated`); run
`python3 tests/check_directions_online.py --apply` and restart the panel. SIM
does not require that hardware measurement. ATTACH and the servo loop are not
needed for this direct translation jog.

What it does not do is turn the pair as one body. RX/RY/RZ turn each wrist
about that world axis through its *own* tool, so a rigid workpiece across both
grippers gets twisted rather than rotated. Rotating one object about one pivot
is what the coordinated path on the Object tab is for, and while anything is
attached there the Jog tab's direct jog stands aside rather than racing the
servo loop for the same two controllers.

## Running

```bash
scripts/ur5dual-gui           # the control panel
scripts/ur5dual-rviz          # 3D view of the cell, read-only (see below)
scripts/ur5dual-jog --arm A   # terminal jog for one arm, for ssh sessions
scripts/ur5dual-flange-log    # record a measured flange gap against both poses
scripts/ur5dual-flange-fit    # what those gaps say about where arm B stands
python3 tests/test_*.py       # the maths, no robot needed
python3 tests/check_chain_online.py   # read-only: does our FK match the arms?
```

## REAL-only panel

The control panel now starts directly in REAL, announces that state with a
fixed red indicator, and attempts to connect both arms. There is no mode
selector. The simulated cell remains available to the automated geometry and
GUI tests, but it is not an operator mode in the panel.

`scripts/ur5dual-rviz` draws whichever is happening. It never waits for a
robot — every joint starts at zero, which reads as "no reading" rather than as
a plausible pose, and each arm snaps to its own angles when its feed opens. A
simulation, when one is running, outranks both.

The RViz bridge opens its own connection to each robot's 125 Hz feed, and
**this cell's arm A cannot serve two of those at once** — it is a CB3 running
PolyScope 3.7.2, which stalls one client mid-packet when a second attaches
and never resumes. The symptom is `arm A feed restarted` in the log every few
seconds, and the danger is that a stalled feed does not raise: every reader
keeps receiving the last sample it saw, so the screen shows an arm standing
still that is not, and the coordinated loop carries the object using a pose
that has moved on.

So run the viewer or the control panel, not both, until arm A is upgraded.
`ss -tnp | grep 30003` on the Jetson names whatever is attached — a bridge
left running from an earlier session is the usual culprit, and it does not
exit with the RViz window.

## Layout

Three rules decide which folder a module is in: `geometry/` is arithmetic and
never opens a socket, `robot/` is everything about *one* arm, and what is left
in the root is the cell — the objects that compose the other two.

```
ur5dual/
  geometry/          numbers only, no I/O, no clock
    kinematics.py      poses, transforms, the one rotation convention
    ur_kinematics.py   UR5 FK, Jacobian and IK in Python
    closed_chain.py    both arms + the box as one chain: joints from a box pose
    world.py           joints <-> Cartesian in the cell frame, for either arm
    calibration.py     touch-off, hand-taught directions, and flange pairing —
                       three ways to measure the real base-to-base transform
  robot/             one arm, and how bytes reach it
    transport/         one module per robot interface
      dashboard.py       29999  power, brakes, safety state
      script.py          30002  send URScript
      rt_stream.py       30003  125 Hz state feed
      primary.py         30001  TCP offset, which 30003 does not carry
    motion.py          URScript for one arm: jog, moves, I/O, freedrive
    arm.py             one arm + the transform that places it in the cell
    sim_arm.py         the same arm with the controller replaced by maths
    backends.py        servo backends for the coordinated loop (rtde | urscript)
  config.py          cell.yaml: mounting geometry, addresses, limits
  cell.py            both arms, keep-out and force guards
  coupling.py        object-centric core: grasp capture, carry, spin
  sim_view.py        joint states out to the viewer, at servo rate
  program/           step vocabulary and the executor that runs it
  description/       dual-arm URDF built from cell.yaml, RViz config
  ros/               joint_states bridge and the RViz launch
  gui/               the touch panel: style, widgets, one module per tab
  tools/             standalone entry points; the panel never imports these
    jog_cli.py         terminal jog for one arm
    midpoint_hold.py   two arms holding one midpoint, without the panel
    flange_log.py      a measured flange gap, with the pose it was measured in
    flange_fit.py      those gaps against cell.yaml: report, correct, or fit
config/              cell.yaml, points.json, flange_log.json, programs/
scripts/             entry points
tests/               geometry, coupling, calibration, program validation
ros2_ws/             ur_description, built here because apt needs root
```

Two directories are used but not tracked: `ros2_ws/src/` holds upstream
`ur_description` and `ros2_ws/build|install|log` are colcon's output, and
`update/` is where PolyScope `.urup` images go — 230 MB each, which is not what
git is for. See below.

## Getting the sources

```bash
git clone <this repo> UR5 && cd UR5
git clone --branch 2.13.0 --depth 1 \
  https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git \
  ros2_ws/src/Universal_Robots_ROS2_Description
(cd ros2_ws && colcon build)
```

`2.13.0` is the tag this cell's URDF was written against. PolyScope images are
not distributed here; download the one a controller needs from Universal
Robots' support site into `update/`.

## Configuration

Everything lives in `config/cell.yaml`.

`mount.style: pedestal` derives both arm bases from four numbers a fitter can
measure — column height, flange spacing, outward tilt, pair yaw. Good enough
to draw the cell and to catch gross reach errors.

`tilt_deg` is how far a base Z axis is tipped away from straight up, outward:
0 stands the arm up off a table, 90 reaches it straight out sideways, 180
hangs it upside down. The frame is drawn as round tube with a mounting pad at
each crossbar end carrying the *same* rpy as the arm base — so a base floating
clear of its pad in RViz is a tilt that is wrong, not a drawing that is rough.

The tube itself hangs off those two flanges rather than off `world`: the
crossbar is drawn between the two base origins and the mast square to it, down
the axis the brackets are tilted away from. So a calibration that turns both
bases — levelling the cell against gravity does exactly that — carries the
structure with it, instead of leaving a bolt-upright mast with the arms
leaning off the ends of it.

It is **not** good enough for two-arm grasping unless the values came from the
as-built cell. Drawing numbers can be out by a few millimetres, and between two
arms holding one object that error becomes a permanent fight. The supplied
touch-off solver uses both TCPs at the same physical point in six well-spread
positions. `tests/test_calibration.py`
measures what that buys against 0.2 mm touch repeatability — 4 points lands
within ~0.8 mm, 8 points within ~0.4 mm. Applying it switches the style to
`custom` so the preset stops overwriting the measurement.

`motion.backend` chooses how the 125 Hz coordinated loop reaches the robots:

| | |
|---|---|
| `rtde` | ur_rtde `servoL`. Less code, well travelled, already installed. |
| `urscript` | a streaming program uploaded to the controller. No dependency, works on any 3.x firmware, whole protocol visible in `backends.py`. |

Both are implemented because neither has been tried against this cell's
PolyScope 3.7.2 yet. If one fails, switch the line and reconnect.

The `urscript` stream is **request-response**: the robot asks for each target
and this side answers with the newest one. That is not a detail. The first
version pushed targets at 125 Hz into a robot-side loop that read the socket,
ran `get_inverse_kin` and called `servoj` in sequence, and each of those costs
the controller at least a control cycle — so it consumed roughly half of what
it was sent and the rest queued. Every send on the Jetson succeeded, every
guard passed, the panel reported both backends up, and the arms executed poses
from further and further in the past: from the bench, two robots that would not
move. With the robot asking, no queue can form, and `servoj` runs in its own
thread on the controller so a slow socket cannot slow the arm.

`tests/check_servo_stream.py` measures it on one arm: how many targets the
robot asked for (about 125 a second when it is well) and how far behind the
target it runs.

`motion.control` chooses *what* that loop sends, and therefore where the
inverse kinematics happens:

| | |
|---|---|
| `pose` | a Cartesian target per arm; each controller solves its own IK. |
| `joint` | six joint angles per arm, solved here by `closed_chain.py` from one box pose. What this cell runs, and what SIM has always run. |

`joint` exists because two controllers solving independently may not agree.
Eight joint solutions put a flange in the same place, and the one a controller
returns can change between cycles; on a single arm that is a lurch, and on two
arms holding one box it is the box. Solving here seeds each cycle with the
previous answer, so neither arm can leave the branch it was on when the
grippers closed — and because the joints are known in advance, the whole path
is walked before it starts. A rotation that would run arm B's wrist past its
stop is refused with the joint number in the message, rather than discovered
with the box in the air.

The cost is that the DH table has to match these robots, and the published one
does not. Every UR is measured at the factory and its own table kept in its
controller, so `transport/read_geometry` reads it off the primary interface at
connect (`KINEMATICS_INFO`) and each arm gets its own. What the two arms here
return is worth knowing before reading `ur_kinematics.py`:

| | arm A (3.7.2) | arm B (3.15.8) |
|---|---|---|
| calibration status | 0 — none held | 1 — calibrated |
| what it sends | the published table, echoed | its own measured table |
| published table is out by | 0.00 mm | 3.18 mm, 0.85° |

Arm A agreeing with the textbook is therefore the absence of a measurement
rather than a passing grade. Arm B's table has `d2 ≈ -54.6 m` and
`d3 ≈ +54.5 m`: a DH chain cannot express a small change in the angle between
two parallel joints, so the fit escapes into enormous offsets that cancel.
That is the expected parallel-joint degeneracy and not corruption — the
kinematics stays exact (0.03 mm against the arm, manipulability 0.2392 against
the published 0.2394, IK converging to 1e-8 m), and anything that "sanity
checks" those numbers by size throws away the only correct table on offer.

Whether a firmware means corrections or a finished table is settled by
measurement, not by belief: `choose_dh` scores every reading against the pose
that controller reports for the joints it is sitting at. If none of them match,
the loop says so and sends Cartesian targets instead of refusing to run — a
pose derived from what the controller reported and handed back to that
controller to solve cannot carry our error, so the fallback is safe and being
stranded with a gripped workpiece is not.

`tests/check_primary_calibration.py` dumps what each controller sends and
scores every reading of it; `tests/check_chain_online.py` says what each arm
ended up with. Both are read-only.

## Safety

Firing `movel` at both controllers is **not** coordination — each plans its own
timing, and the tens of milliseconds they differ by become millimetres of
relative error. Anything that carries a held object goes through the servo
loop, where both arms are driven from one clock.

Guards that run on every cycle of a coordinated move:

- **force** — either TCP past `motion.max_tcp_force`
- **drift** — the measured A-to-B transform wandering from the captured one
  by more than `motion.max_pair_drift`; that is the arms levering against
  each other, and it shows up before either trips on force

Guards that run before anything moves:

- reach check per arm, in that arm's own base frame
- program validation: carrying with nothing attached, moving one arm while
  both grip the same object, unbalanced ATTACH/DETACH, unknown point names

With `motion.control: joint`, three more run before anything moves, because
the joints for the whole path are known before the first one is sent:

- **reach** — either arm unable to hold its side of the box anywhere along it
- **joint limits** — the angle, the joint, and how far along it happens
- **joint speed** — what the move would demand against what a servo loop can
  actually track

None of these are available in `pose` mode: there the controllers find out for
themselves, one cycle at a time.

There is **no arm-to-arm collision checking**. Nothing in the socket interface
knows the other robot exists. That is the price of staying out of ROS, and it
is why the keep-out and force limits matter. Note that solving the closed
chain does **not** help here, and cannot: two 6-DOF arms rigidly holding one
box is twelve joints against twelve constraint equations, so there is exactly
one solution per branch and nothing spare to steer around an obstacle with.
Arms with room to spare in this situation have seven joints each.
