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

A program, an icon rail, and a sidebar the rail opens and closes. 1280x800,
the shape of a teach pendant. The cell-wide controls span everything, because
the sidebar can be closed and STOP cannot.

```
┌─────────────────────────────────────────────────────────┐
│ ● REAL | A | B | Power | Brake | Unlock |     ■ STOP     │
├──────────────────────────────────────────┬────┬─────────┤
│ Program — teach & run                    │Jog │ ◀       │
│        x      y      z     rx    ry    rz│    │         │
│ A -412.3  188.0  505.1   75.1   0.0   0.1│    │         │
│ B -395.8 -190.4  504.7  -75.1   2.9  -4.9│    │         │
│  # step   Arm A       Arm B     link     │A|AB│ ◎ Pts   │
│  1 MOVE   home_A      home_B    ⇉ togeth │ |B │ ✥ Jog   │
│  5 MOVE   world Z+100.0 ······  ⇉⇉ SYNC  │6 ax│         │
│ [▶ Run ][   insert ▾   ][＋][   ⧉ A+B    ]  │    │ ⇄  │
│ [⏸Pause][● MOVEJ][↗ MOVEL][  A  ][  B  ]    │    │    │
│ [■ Stop][▷To][✏Edit][⎘Dup][✕Del][   ↑   ]   │    │    │
│ [📂Load][💾Save][🔁Loop][ 120 mm/s ][   ↓   ] │    │    │
├──────────────────────────────────────────┴────┴─────────┤
│ Messages: newest line                         [Expand]  │
└─────────────────────────────────────────────────────────┘
```

The program is the document being written and it is what the panel is for, so
it takes every pixel the sidebar is not using: 884 px with a panel open, 1210
px with the sidebar closed. **Every panel is the same 320 px**, so switching
between the jog keys and the points list does not slide the step table
sideways under a finger — the sidebar changes what it holds, never how much
room it takes. That matters because a line now carries two columns
of targets — 350 px split five ways gave each arm 86, which is an ellipsis
rather than a column, and 884 gives each of them 391.

The rail is 56 px, it is never hidden, and it carries the panels as icons: the
named places (`◎ Pts`), the camera (`◉ Cam`) and the jog grid (`✥ Jog`). Both are laid out for the
one sidebar width: the points list stacks its teach buttons under the table
rather than beside it, and shows each place's position with the rotation on
the row's tooltip, because all six numbers want 45 characters and the sidebar
has room for half of that. The lit icon is the open
panel and pressing it again closes the sidebar — **the way back is never behind
the thing that took it away**, which is the whole reason the rail exists rather
than a bare "wide" button.

`gui/panels/cell_setup.py` and `gui/panels/objects.py` are still there and
neither is imported. The cell's geometry and its touch-off are measured with
the command-line tools rather than typed at the pendant, and taking hold is an
ATTACH step in a program rather than a button, so neither panel earned its
icon. Putting either back is one line in `widgets/rail.py` and one in
`app._build`. What that costs while they are out: there is no touch-off from
the panel — `tests/check_directions_online.py --apply` and
`scripts/ur5dual-flange-fit` are how the cell is measured — and no manual
take-hold, though the Jog page's release button still lets go of one.

`⇄` moves the rail and its panel to the other edge and writes that down
(`ui.sidebar_side`). Which hand reaches the jog keys is the operator's, and an
operator who moved them wants them moved tomorrow too. Closing or moving the
sidebar releases whatever jog button is held first: a key whose button has left
the screen must not go on driving an arm.

The panel starts maximised, with the desktop's dock and top bar still beside
it: the terminal it was started from is a swipe away and the panel has to be
findable again afterwards, so `ui.fullscreen` starts out false. The rail's `⬚`
button takes the whole screen when the whole screen is wanted — `❐` gives the
frame back — and F11 does the same from a keyboard; either way the answer is
written to `cell.yaml` and is how it starts tomorrow. That button lives in the
rail rather than the title bar because fullscreen takes the title bar with it,
and the way out of it must not go with it. `--fullscreen` and `--windowed`
override the saved answer for one run without writing anything back.

1280x800 is the size every number in the layout is quoted in, not a size the
screen has to be. On startup the panel measures what it will actually get and
scales the whole design down to fit, never up. Maximised that is the desktop's
work area less what the title bar will take — on this cell's 10" panel,
1232x736, which comes out at 0.92. Fullscreen has no frame to pay for and gets
the design size back at 1.0, the largest the touch targets ever are. `--scale`
overrides both.

What it is not is a window resized to 1280x800 and left where the window
manager put it, which is what it used to do on a 1280x800 screen: bottom edge
under the dock, and a maximise button with nothing left to give.

Every message also goes to the terminal the panel was started from. The drawer
holds 400 lines and is one line tall until it is opened, so a panel run from a
shell would otherwise lose the line that says why a camera did not open, or
which arm refused a command, behind whatever came next.

The fixed REAL indicator, connection and dashboard controls, and the cell-wide
STOP stay in one row above everything. The newest message stays in a one-line
drawer at the bottom; `Expand` opens its history. Compact A/B TCP and force
readouts plus the pair gap sit below the Jog keys, and `Details` opens the full
XYZ, RX/RY/RZ, J1–J6, force, robot/safety state, TCP gap, drift, and holding
state in a separate window.

## Writing a program

A line is a gesture, not a command to one robot. `MOVE` carries a column for
arm A and a column for arm B, and a `link` field that says how tightly the two
are tied — which is not a preference but a choice of engine:

| link | what happens | engine | needs |
|---|---|---|---|
| `solo` | only the filled column moves | `movej`/`movel`, one arm | — |
| `together` | both are sent, and the line waits for **both** | two moves + a joint wait | nothing attached |
| `pair` | **one** world delta to both arms at matched speed | two moves, same v/a | `translation_calibrated`, translation only |
| `coupled` | one object frame drives both from one clock | the 125 Hz servo loop | an `ATTACH` above it — *not offered on the panel, see below* |

`together` is not coordination. Each controller plans its own timing, so the
arms agree at the end of the line and nowhere in particular in the middle;
that is right for approach and retreat, and it is refused while anything is
held. `pair` is the Jog tab's A+B column written down — world frame only,
because "base" and "tool" name a different direction at each robot, and
translation only, because RX/RY/RZ turn each wrist about its own tool and
would twist a rigid workpiece rather than turn it. Its guards run in the
arrival wait at 50 Hz rather than at 125, which is what the translation-only
rule is paying for. Turning the pair is `coupled`.

A column is a *target*, and an offset is not a separate kind of step — it is a
target with no named place in it:

| target | means |
|---|---|
| place | go to the taught place |
| place + offset | go there, shifted — approach and retreat without a second point |
| offset | shift from wherever this arm is **when the line runs** |
| here | the pose the `⌖ Here` button captured, written into the step |

Offsets resolve at run time. That is what makes a stacking or feeding program
possible, and it means a line that ended somewhere unexpected hands the
surprise to the next one — so an offset with no place in it, in a program that
has not sent that arm anywhere yet, is a warning under the step list rather
than a refusal.

Frames follow the same rule as jogging, for the same reason: a single arm gets
`world`, `base` and `tool`; a `pair` line gets `world` alone; a `coupled` line
gets `world` or the object's own, plus a pivot.

The live row above the table is arm A, arm B, the gap, and the object frame
while one is held — **all six numbers each**, in world, millimetres and
degrees, under one header so the columns line up and a difference between the
two arms is a difference in one column.

Position alone was half a readout. A place is a pose, and orientation is the
half of it that a `⌖ Here` capture carries and that nothing else on this page
shows, so a step recorded from a wrist turned the wrong way looked exactly
like one that was not. The row is there at all because teaching used to mean
leaving for the Jog page's `Details` window to read a number that belongs
where the step is written.

The function grid is four rows on a twelve-unit column, and the unit is what
keeps every target big rather than every button identical: `insert ▾` needs the
room to show a step kind and `＋` does not, so they take five units and one
rather than the same cell each. The narrowest control is 97 px across and 40
tall.

The columns carry the grouping. Run, Pause, Stop and Load stand in the left
one, so the four things that act on the program as a whole are under the same
thumb. `⧉ A+B` sits directly over `A` and `B` and is exactly as wide as the two
of them together, which is the layout saying what the button means. `↑` and `↓`
stack at the right end of the two rows that act on the selected step.

`Dry run` and `On/off` are not on the panel. The executor still walks a program
without commanding either arm — the tests run on that — but nothing on screen
turns it on, so a Run is always a real one. A step saved as disabled still
loads, still shows a `·` for its number and is still skipped; there is no
longer a button that switches it back.

A step is recorded, not typed. `● MOVEJ` and `↗ MOVEL` write where the arms
already are into a line — the arm is the input device a pendant has, and the
`for` selector says whether that line is arm A, arm B, or both at once. What
they record is a pose baked into the step rather than a name: it replays what
was taught and does not follow a point re-taught later, which is the trade the
Points tab exists to offer the other side of.

Everything with nothing to record comes off the insert list, and both routes
open the step editor when there is more to say. `✏ Edit`, or a double-tap on
the row, opens it again. The editor is a dialog rather than a column beside
the table for one measurable reason: a 252 px editor standing next to the list
costs each arm the difference between 554 px of step text and 196.

In that dialog the two arm columns are shown side by side, and which of them
you fill decides the link — one column is `solo`, both is `together`. There is
no way to state the contradiction the validator would otherwise have to catch.

`▷ To` runs the selected line and nothing else, which is how you drive to a
taught position without running what surrounds it. It reads the highlighted
row rather than Qt's idea of the current cell, because those differ after a
selection is cleared and this one moves an arm.

`ATTACH`, `DETACH` and the coupled carry are not on the insert list. They are
the two-arm grasp — one object frame driving both arms through the 125 Hz
servo loop — and the only thing it does that `SYNC OFFSET` cannot is *turn* a
workpiece while both arms hold it, which this cell's work does not ask for.
Nothing was deleted: `steps.py` still knows the kinds, the executor still runs
them, `coupling.py` is untouched, and a program saved with them still loads,
still validates and still opens in the editor as what it is. Putting them back
is two lines in `panels/program.py`. `Teach obj` went with them, because an
object point is the frame of something both arms are holding and nothing on
the panel can take hold any more.

`OUT` drives one of the eight digital outputs on arm A's controller, arm B's,
or both — a gripper was never a different thing from an output, only a
narrower name for one, and old programs that said GRIP load as `OUT`. `WAIT IN`
holds the program where it is until an input reads the way it asks; a timeout
of zero waits as long as it takes, and any other timeout **fails the program**
rather than carrying on, because carrying on is how an arm reaches into a
fixture that is not ready.

`LABEL`, `JUMP`, `IF` and `SET VAR` are how a program stops being a straight
line. A label is a name a jump can land on; `IF` tests a digital input or a
variable and jumps one way if it holds and another if it does not. Labels are
resolved against the *flattened* plan before the run starts, so every jump has
somewhere to land before an arm moves, two labels sharing a name is a problem
on paper, and a program that only jumps — `top: jump top` — is stopped with a
message rather than left spinning a core. Every arm move resets that counter,
so a real loop can run all day.

`CALL` runs another saved program and carries on below it, optionally several
times over. A called program may not contain a `LABEL`: it is pasted in
wherever it is called, and two copies would put two jumps on one name. The whole call tree is expanded before the run starts rather than
followed during it: a call that names nothing, calls itself, nests deeper than
five levels, or points at a file that is not there is a line of red text under
the step list, not a stop half way through a move. Every step a call drags in
reports the CALL's own row while it runs, because the called program has no
rows of its own on the screen.

`MOVE_ARM`, `MOVE_OBJ` and `ROTATE_OBJ` are how this was spelled before, and
programs saved in that spelling still load: `Step.from_dict` translates them
into `MOVE` lines and nothing on disk is rewritten until it is saved again.

`docs/program_line_design.md` is the design this was chosen from, and
`docs/drawings/make_layout_plates.py` draws the panel layouts that were
weighed against each other, to scale.

## Seeing the work

A camera finds a box on a surface and the program corrects a taught pick by
where it actually is. The correction is a rigid transform of the box about its
own middle, which is not the same as adding an angle to the wrist — and the
target grammar already speaks it: `place + offset` in the world frame with the
box's centre as the `pivot`.

A RealSense needs `pyrealsense2`; PyPI has an `aarch64` wheel for this
Jetson's Python 3.10.

```bash
sudo apt install -y python3-pip
pip3 install pyrealsense2
sudo cp scripts/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

The udev rules are not optional and their absence has one symptom: `Frame
didn't arrive within 5000`. The PyPI wheel is built with the RSUSB backend,
which does not use `/dev/video*` at all — it talks to the device through
libusb and needs **write** access to its node under `/dev/bus/usb`, which is
`root:root 664` until udev says otherwise. `pipeline.start()` succeeds over
the control endpoint and no frame ever arrives. Unplug the camera and plug it
back in after installing them.

```
ur5dual/vision/
  camera.py     where a frame comes from: a RealSense, or a rendered one
  rim.py        the opening's four corners, and depth's verdict on them
  detect.py     solvePnP, outlier rejection and tracking over those corners
  service.py    the camera on its own thread, and the newest thing it saw
  calibrate.py  where the lens is, from places the arm and the camera both saw
```

Each sensor does what it is good at. The colour image finds the four corners
of the opening — it has no depth discontinuity to fall into, so it answers for
the far rim as readily as the near one — and `solvePnP` turns them plus
`vision.box_size` into metres. Depth is never asked where the corners are. It
is asked afterwards how far away the near rim actually is, and the difference
between that and where the solved pose puts it is the `size check` line on the
Camera tab: a right size agrees to a few millimetres on this cell, and a wrong
one is out by hundreds.

Corners come from contour hulls rather than Hough lines. A box with work
standing proud of its rim breaks the edge Canny draws, and a broken loop is an
open curve — measured here, a rim traced over 2769 pixels reported a contour
area of 127, so every test a closed quadrilateral would pass rejected it, and
the Hough version found the opening in 0 frames out of 100. The hull repairs
the break exactly, because the break lies along a straight side and the hull's
own edge is that same line; candidates are then ranked by hull area and by how
much of the contour really lies along the rectangle, which is what a cluttered
workshop is filtered by now. Detection went to 81 frames in 100, and 114 in
120 once the tracker's hold is counted.

The two size fields on the Camera tab are the only way in, and what is
settled in them is kept: `vision.box_sizes` holds the last eight openings that
were dialled in, newest first, and the dropdown above the fields offers them
back. The size the cell is currently set to is always the top line, so a
crate swapped in for an afternoon is one tap away from being swapped back.

`vision.roi` is therefore normally `null` — the whole picture is searched. A
window is still honoured if one is set, but it fails quietly when a box drifts
out of it: the opening is not reported missing, it is reported at the window's
own edge. Six pixels of clipping moved the pose 34 mm here at a reprojection
error of 1.6 px, and only the depth cross-check saw it.

A result with reprojection error over `max_reprojection` is rejected, a corner
jump over `max_corner_jump` must repeat for `confirm_frames`, and accepted
corners are smoothed before pose is solved again. `HOLD` keeps the picture
from flashing, but `VisionService.fresh()` never hands that stale pose to
`FIND`; a robot only receives `LOCKED`, `TRACKING`, or `RELOCKED` data.

`scripts/ur5dual-open-box-pose` is the same detector on a terminal: live at
the frame rate with a window and an overlay, `--npz` for one saved frame,
`--replay` for that frame in a loop with no lens, and `--measure` to let depth
choose the opening from the standard crate sizes instead of being told.

Set `vision.log_enabled` and a Camera session writes
`logs/camera_openbox_*.csv`: raw and filtered corners, raw and filtered
transforms, depth, reprojection error and the `LOCKING/TRACKING/HOLD/REJECT`
state for every frame, plus detector settings in the metadata line. It is the
evidence for tuning the filter, instead of judging accuracy from a moving
overlay — and it is off by default, because a row a frame is tens of megabytes
per shift for a question nobody is asking most days.

Calibration is the other half, and it is the half that is easy to believe and
hard to get right. `solve_from_points` is Kabsch over places the arm and the
camera both saw: exact on clean data from three points up, and against the
2–3 mm of depth noise a D435i actually has, eight points land the camera about
9 mm out — ten times the detector's own error. **The residual it reports is
always smaller than the error it has**, so a good-looking fit is not evidence.
Worse, points along a line fit *perfectly* — an RMS of 0.000000000 m — with the
rotation 9.5° wrong, which is what `spread_of` exists to catch: it returns the
three principal spans, and a calibration where one of them is near zero is a
calibration solved from a line.

`SimCamera` still exercises the complete Camera/FIND path before a lens is
available. `tests/test_vision.py` additionally projects a known tilted opening
into an image and requires the detector to recover its translation within
5 mm and reprojection within 1 px. Camera acquisition and detection remain on
the `VisionService` thread, so neither edge extraction nor the four-frame
initial lock blocks Qt's UI thread.

`FIND` is how a detection reaches a program. It stores not where the box is
but **how far it has moved** — the rigid transform from the place the pick was
taught at to where the box is now — and a `MOVE` target carried by that
correction is moved the same way the box was. That is the difference between
"the workpiece is somewhere else" and "the wrist is turned": the correction
multiplies from the left, so a pick 200 mm from the box's middle swings round
with it instead of spinning on the spot.

```
FIND   look for the box, against box_home -> part
IF     part_found == 0   jump  no_part
MOVE   A  pick + world Z+50   movel      moved by part
```

`box_home` is taught once, with the box standing where the picks were taught
against it, and it lives in the ordinary point library rather than in a corner
of the camera's own settings. The Camera tab used to have a `⌖ Teach box home`
button for it; it wrote a pose in the *camera's* frame, which is only the
cell's frame once `vision.camera_to_world` has been calibrated, so on an
uncalibrated cell it taught a point that read as authoritative and was not.
Teach it from the Points tab like any other place.

Finding nothing is not an error. `FIND` writes `<name>_found` as 0 and carries
on, so a program branches on it rather than stopping; what *is* an error is a
line corrected by something no `FIND` has looked for, which is refused on
paper, and a correction larger than `vision.max_correction` (100 mm, 30°),
which is refused at run time. That gate is not tidiness: a detection is the
one pose in a program nobody taught, and nothing in this cell checks one arm
against the other.

The camera tab shows **one picture with both in it**: the lens image with the
colourised depth laid over it, and the 3D cuboid the detector found drawn on
top by projecting its eight camera-frame landmarks back through the lens. `lens` and
`depth` on their own are a press away, because a blend hides a lens that has
gone dark and a depth image that has gone empty equally well.

Blended rather than side by side because what an operator is checking is
whether the rectangle sits on the thing they meant — one question about one
place in one image. Depth in colour rather than a grey ramp because a
millimetre of depth is a fraction of a grey level and a whole step of hue: a
box 80 mm off a table is a shade of grey away from it and a different colour
altogether. The overlay is scaled by the picture rather than by the box it
sits in — a 4:3 image in a wider view does not fill it, and a rectangle drawn
at the view's scale sits off the object by the difference.

A camera that opens but never delivers a frame — `Frame didn't arrive within
5000` — is almost always permissions rather than bandwidth or a profile. The
PyPI `pyrealsense2` wheel is built with the RSUSB backend, which does not use
`/dev/video*` at all: it talks to the device through libusb and needs **write**
access to its node under `/dev/bus/usb`, which is `root:root 664` until udev
says otherwise. `pipeline.start()` succeeds over the control endpoint and no
frame ever arrives. `scripts/99-realsense-libusb.rules` is that rule set:

```bash
sudo cp scripts/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

then unplug the camera and plug it back in.

## Jogging by hand

The Jog page drives arms, not objects. Arm A, synchronized A+B and arm B are
three targets, and one is on screen at a time:

| Drive | What a press does |
|---|---|
| `Arm A` / `Arm B` | one arm, in the cell frame, that arm's base, its tool, or a joint at a time |
| `Synchronized A+B` | the same cell-frame direction to both arms at once |

All six axes are visible at once: X/Y/Z and RX/RY/RZ. In joint mode the same
rows become J1 through J6. A and B can use world, base, tool, or joint frames;
A+B is enabled only in world frame.

The page used to keep all three targets up together, on the argument that a
selector can leave an operator pressing a key for the arm they are not looking
at. It is a 320 px sidebar now (`layout_f_side_open.svg`), which cannot hold
three columns of finger-sized keys — 122x67 for one target, 37 across for
three. So the argument was paid for rather than dropped: the live target's
button is filled rather than ticked, the band over the keys is the arm's own
colour and names it, the keys carry that tint, and **changing target releases
whatever key is held** so a press meant for one arm can never be inherited by
the other. Nothing on the page is narrower than 74 px.

What the width buys is on the other side of the screen: while jogging, each
arm's column in the step table is 391 px instead of 92.

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
scripts/ur5dual-snap          # live camera; s saves a JPEG and its depth
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
    steps.py           the two-column line, its targets, and what it refuses
    executor.py        one engine per link: solo/together, pair, coupled
  description/       dual-arm URDF built from cell.yaml, RViz config
  ros/               joint_states bridge and the RViz launch
  gui/               the touch panel: style, widgets, one module per panel
    widgets/rail.py    the icon rail, and the sidebar it opens and closes
    panels/step_edit.py  one line, in a dialog: a column editor per arm
  tools/             standalone entry points; the panel never imports these
    jog_cli.py         terminal jog for one arm
    midpoint_hold.py   two arms holding one midpoint, without the panel
    flange_log.py      a measured flange gap, with the pose it was measured in
    flange_fit.py      those gaps against cell.yaml: report, correct, or fit
config/              cell.yaml, points.json, flange_log.json, programs/
scripts/             entry points
tests/               geometry, coupling, calibration, the program and its panel
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
