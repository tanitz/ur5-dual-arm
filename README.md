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
│  1 MOVE   home_A      home_B    ⇉ togeth │ |B │ ◉ Cam   │
│  5 MOVE   world Z+100.0 ······  ⇉⇉ SYNC  │6 ax│ ⇄ Com   │
│  6 SEND   MC1 START             →        │    │ ✥ Jog   │
│  7 RECV   wait for DONE from MC1 ←       │    │         │
│ [▶ Run ][   insert ▾   ][＋][   ⧉ A+B    ]  │    │ ⇄  │
│ [⏸Pause][● MOVEJ][↗ MOVEL][  A  ][  B  ]    │    │    │
│ [■ Stop][▷To][✏Edit][⎘Dup][✕Del][   ↑   ]   │    │    │
│ [📂Load][▶10%][💾Save][🔁Loop][120 mm/s][ ↓  ] │    │    │
├──────────────────────────────────────────┴────┴─────────┤
│ Messages: newest line                         [Expand]  │
└─────────────────────────────────────────────────────────┘
```

The program is the document being written and takes every pixel the sidebar is
not using: about 692 px with a panel open, 1210 px with the sidebar closed.
**Every panel is the same 512 px**, so switching
between the jog keys and the points list does not slide the step table
sideways under a finger — the sidebar changes what it holds, never how much
room it takes. The program still keeps both arm target columns visible while
the camera and setup panels gain more working room.

The rail is 56 px, it is never hidden, and it carries the panels as icons: the
named places (`◎ Pts`), the camera (`◉ Cam`) and the jog grid (`✥ Jog`). Both are laid out for the
one sidebar width: the points list stacks its teach buttons under the table
rather than beside it, and shows each place's position with the rotation on
the row's tooltip, because all six numbers want 45 characters. The lit icon is the open
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

## Desktop and browser together

The optional browser UI is served by the same process as the desktop panel.
There is still one `Cell`, one `Executor`, one camera service and one owner of
the robot sockets. A command arriving over HTTP is queued onto Qt's main
thread before it touches any of them; the resulting state is then broadcast
over WebSocket to every open browser. A Run pressed in a browser therefore
lights up as running on the desktop, and a Pause, box-size change or program
edit made on the desktop appears in every browser.

Install the two optional web dependencies once:

```sh
python3 -m pip install -r requirements-web.txt
```

For a browser on the same computer:

```sh
scripts/ur5dual-gui --web
# open http://127.0.0.1:8765
```

For another computer or tablet on a trusted robot-cell LAN:

```sh
scripts/ur5dual-web
```

The terminal prints a URL containing this run's address and a new random
access token — the address is read from the interface facing the default
route, not from the hostname, so it is one another device can actually reach.
Open that complete URL on the other device; the browser keeps the token in
session storage and removes it from the address bar.

A fresh token each run means a bookmark stops working after every restart.
`--web-token` decides that instead — a fixed string keeps one URL valid across
restarts, and `none` serves the LAN with no token at all, which puts both arms
under the control of anything that can reach the port:

```sh
scripts/ur5dual-web --web-token cellkey   # http://<address>:8765/?token=cellkey
scripts/ur5dual-web --web-token none      # http://<address>:8765/
```

Holding a jog button in a browser sends a press and then a heartbeat every
100ms over the state WebSocket. Heartbeat arrival never waits behind Qt or
camera work; the desktop timer refreshes `speedl`/`speedj` while that stream is
live and drops the key if the browser goes quiet for longer than
`WEB_JOG_GRACE`. The controller also kills motion after its shorter `WATCHDOG`
when refreshes stop, so a lost browser, page or network cannot leave an old
velocity command running.

The browser has the same two-column shape as the desktop: Program remains in
one column while Points, Camera, Communication or Jog occupies the other. `⇄`
moves that sidebar to the other edge and each browser remembers its own side
and selected tab. The Communication page contains TCP, UDP and Modbus machines
in the same **Com** tab, matching the desktop rather than reviving the old Net
and Modbus split. Robot state, the current program, points, communication
library, camera settings and run/pause/stop state are shared.

Camera pictures use their own binary WebSocket rather than repeated JPEG HTTP
requests. The browser opens that stream only while Camera is visible and Live
is running, displays new JPEG frames as they arrive, and closes it again on a
different tab so image traffic can never queue ahead of Jog or state messages.

A machine added, renamed, re-protocoled or deleted in a browser appears in the
desktop tab's own tables, and a machine set up on the desktop appears in every
browser; both surfaces read the one status line, so a `⇄ Test` started from a
tablet says on the desktop whether the address answered. The browser's JSON
editor is refused whole rather than in part: an unnamed machine or datum is an
error there, where reading past it would delete a line somebody is looking at,
even though the same entry is skipped when a hand-edited `cell.yaml` is loaded
so that the panel still opens.

Web Jog has an additional ownership and heartbeat guard. Only one browser may
hold a jog command, it sends a heartbeat while the pointer is down, and a
pointer release or WebSocket close explicitly halts that session. If the close
cannot be delivered, the robot command watchdog stops motion in about 400 ms
and the desktop clears ownership after `WEB_JOG_GRACE`. The web STOP is still a
software stop and does not replace the cell's hard-wired emergency stop.

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

`▶ %` is how fast the run goes, and it governs the whole program rather than
one line of it. 100% is what the cell may do: `limits.max_lin_speed` for a
line one arm runs by itself, and `limits.object_lin_speed` for a `pair` or
`coupled` line, where both arms are pushing the same workpiece and the
coupled-carry ceiling applies instead. A speed is still stamped on a step when
it is recorded and still saved with it — it is what the line was taught at —
but it is no longer what the line runs at, so a program taught at a crawl and
one taught at speed both answer to the same dial.

It sits beside `Load` rather than out with the record box, because it belongs
to the left column: the things that act on the program as a whole, under the
thumb that presses Run. Turning it while a program is running is allowed and
takes effect on the next line sent — the move already on the wire finishes at
the speed it was given. It comes up at 10% every launch and is never written
to `cell.yaml`, on purpose: a dial that remembers comes back at whatever the
last shift left it on, and the failure that matters is the one where somebody
turned it up to prove a cycle time, closed the panel, and the next person to
press Run gets that speed on a program they have not watched move.

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

### Talking to the machine next door

`SEND DATA` hands a machine one of the things it was set up to be sent, and
`RECV DATA` holds the program until a machine sends what was set up. Both
carry two **names** and no address:

    SEND  MC1  START
    RECV  MC1  DONE     up to 30 s

The machine and the datum are set up once on the **Com** tab and the program
only ever picks from those lists — the same bargain the point
library makes for places. It is what makes a line checkable on paper: a step
naming a machine that was deleted off the tab is a line of red text under the
step list, where the same line carrying `10.1.68.200:2000` could only ever be
caught by a program that hangs.

Two steps rather than one with a direction field. Sending is over the moment
it is sent; waiting is a step that can time out, and a line that reads the
same either way hides which of the two it is. A `SEND` therefore never waits
for a reply — the reply is the `RECV` on the next line, which is a line the
operator can see and put a timeout on.

`RECV` makes the same bargain `WAIT IN` does with a digital input: a timeout
of **0 waits as long as it takes**, which is what a cell fed by a slow machine
wants, and any other timeout **fails the program** rather than carrying on as
though the signal had arrived. Carrying on is how an arm reaches into a
fixture that is not ready.

Its optional `into` puts the answer in a variable, so an `IF` below can branch
on it. A reply that is a number is stored as that number; one that is not is
stored as `1`, because it arriving is the whole of what it said.

Anything already waiting is thrown away before a `SEND`. A reply left over
from the last cycle would satisfy the next `RECV` the instant it looked, and a
handshake that passes on stale data moves an arm for a reason nothing on
screen shows.

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
  planar.py     where the box is on the surface it slides on — three numbers
ur5dual/tools/
  plane_fit.py  collecting the placements a plane map is fitted from
```

Each sensor does what it is good at. The colour image finds the four corners
of the opening — it has no depth discontinuity to fall into, so it answers for
the far rim as readily as the near one — and `solvePnP` turns them plus
`vision.box_size` into metres. Depth is never asked where the corners are. It
is asked afterwards how far away the near rim actually is, and the difference
between that and where the solved pose puts it is the `size check` line on the
Camera tab: a right size agrees to a few millimetres on this cell, and a wrong
one is out by hundreds.

### Height is not camera Z

The pose is in the camera's frame, and its Z is range along the lens axis.
That is the same thing as height only for a lens pointing straight down, and
this cell's points down *and along* — measured off two captures of one box on
one table, camera Z differed by 102 mm, of which 114 mm was slide across the
table and, along the table's own normal, none of it was height. Read camera Z
as a height and every sideways move looks like the box got taller.

Open **Camera -> Measure Surface + Box** and give it the two captures it asks
for: the box on the table with the sheet out of sight, then the printed
ChArUco sheet lying *where the box stood*. Press `Apply + Save`. That writes
the table into `vision.surface`
as a plane in the camera's frame — and, where the pair measured it, the real
`vision.box_size` alongside — and every detection then carries how far the rim
stands off that surface. It is the line on the Camera tab that reads

```
height 103 mm above surface (+3 mm)
```

The first number is the rim's height above the table; the bracketed one is
that against `vision.box_size`'s wall, so a box standing on the surface that
was measured reads near zero. Sliding it across the table must not move
either. If they drift, the surface has moved, the camera has been knocked, or
the wall height is wrong — and a cell reading only camera Z cannot tell you
which, because camera Z was never going to hold still anyway.

Until the board has been run the line says so rather than guessing, and
nothing else changes: the surface is measured in the camera's frame, so it is
true only while the camera is where it was when the sheet was read. Move the
bracket and measure it again.

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
state for every frame, plus detector settings in the metadata line. It also
carries `surface_height_m` beside camera Z, which is how a slide test is read:
push the box along the table, and the one column that should not move is the
height while `filtered_z_m` moves by 88 mm per 100 mm. It is the
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

`FIND` is how a detection reaches a program, and it measures that in whichever
of the two ways the cell has been set up for: through `config/plane.json` when
a plane map is in it, and through `vision.camera_to_world` otherwise. A cell
with neither is refused. It used to run the second one against an identity
transform, which puts every detection in the camera's own frame and produces a
correction that is confidently wrong — caught, if at all, by `max_correction`
noticing the size of it.

### Height above the floor, not above the crate

`vision.surface` is the surface the boxes stand on, so the height line answers
how far a box stands proud of *that* — on this cell, off a crate lid. Two
things are missing before the same number is a height in the room, and only
one of them is a measurement anybody has to take.

Which way is up is not one of them. The D435**i** has an accelerometer, and at
rest the only force on it is gravity, so true vertical is something this cell
can read rather than assume:

```
scripts/ur5dual-level                  # what gravity says about the surface
scripts/ur5dual-level --floor 152      # ...and the floor 152 mm below it
```

With no arguments it says how far the measured surface is out of level, and
what that costs — a surface 2 degrees out reads a 100 mm box 0.06 mm short,
and moves its true height by 10 mm across 300 mm of slide. Worth knowing
before a plane measured at one end of a crate is used at the other.

`--floor` supplies the part gravity cannot: where zero is. Measure once with a
tape, floor to the surface the boxes stand on, and it writes `vision.floor` —
the surface dropped by that much and squared to **gravity** rather than to the
crate, so a crate that is out of level does not tilt the room's idea of height
along with it. The Camera tab then reads

```
height 249 mm above floor (97 mm proud)
```

both numbers, because a cell needs both: how high in the room the rim is, and
how far the box stands proud of whatever it was put down on. `surface_height_m`
and `floor_height_m` are both columns in the session CSV.

It uses the Camera tab's existing stream, so no second process competes for
the RealSense. The camera must not move between the box and board captures.

### The hold, taught as a distance

There are two ways to write the same pick, and the difference is what is
stored. A taught point plus a correction keeps the grip as a place in the
cell, and `FIND` carries it. A **hold** keeps it as a distance from the box —
"80 mm along its long side, 60 mm above the rim" — and `FIND` puts that
distance where the box now is. Both land in the same place. Only one of them
survives the map being fitted again: re-fit and every placement moves,
including the one a pick was taught against, and a world pose taught beside it
does not move with it. A distance does.

Which way the box is facing is read off the tool's *orientation*, not off the
line between the two arms — which is also what makes "parallel to the sides"
the whole of what an operator has to get right. That was the first design, and a real cell showed
what was wrong with it: two arms taking hold of a box from opposite sides put
their tool centres on opposite faces of it, a couple of centimetres apart —
measured here, 23 mm — and a direction taken from a baseline that short is
worth nothing. A gripper with hold of the box turns with it, so its
orientation says which way it faces directly and needs no baseline at all.
Whatever constant sits between that tool and the box is absorbed into the
offset the fit already solves for.

The Camera tab exposes this as one `Create Home Box` button. Its popup keeps
the live camera image beside the complete Jog panel: Arm A / synchronized A+B
/ Arm B, hold or step motion, Cartesian world/base/tool or joints, four speed
presets, and X/Y/Z/RX/RY/RZ. Jog the selected gripper tip or pair to the safe
pose immediately before the pick, keep the box visible, and press
`Save arm + object`. That one press takes a fresh camera reading and records it
together with the selected TCP pose(s), so the box cannot be nudged between
two separate teaching steps.

With a plane map, the popup stores `box_home` as a placement on that surface
and stores the TCP as that arm's stance relative to the box; `FIND` can use it
immediately. With a calibrated `vision.camera_to_world`, it stores the 6D box
pose as `box_home` and the TCP as `box_home_A_pre_pick` or
`box_home_B_pre_pick` in the point library. If neither coordinate conversion
exists, it stores a guarded **fixed home** instead: the camera-frame detection
is used only to confirm that the box has not moved, then `FIND` publishes the
absolute pre-pick TCP pose(s) saved at that home. A fixed home refuses a moved
box; following one requires a plane map or calibrated camera-to-world, because
a camera-frame movement cannot safely be applied to a world-frame arm without
that conversion.

```
FIND   look for the box, against box_home -> part
IF     part_found == 0   jump  no_part
MOVE   A  at part + world Z+50 | B  at part + world Z+50   together
MOVE   A  at part              | B  at part                together
ATTACH box
```

Each column is given its own arm's hold. Selecting which happens in the
executor rather than inside `resolve_target`, so that function stays what it
is — a name and a place — and the one thing that knows which arm a column
drives stays in the one place that already knew.

A hold needs no `moved by`, and the editor hides that row when a column is
holding: carrying a hold by a correction as well would apply the box's
movement twice.

Either way `FIND` stores not where the box is but **how far it has moved** — the rigid transform from the place the pick was
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
against it. Where it lives depends on which of the two ways below the cell
measures with: a point in the ordinary library for the six-number way, and a
placement in `config/plane.json` for the three-number one.

The Camera tab used to write `box_home` directly from the camera frame, which
is only the cell frame once `vision.camera_to_world` has been calibrated. The
new popup is sound for the reason that version was not: it accepts either a
placement from an existing plane map or a pose transformed through a calibrated
camera, and refuses to save when neither route exists.

### Three numbers instead of six

A box that always lies flat, the same way up, at the same height on the same
surface has three degrees of freedom — it slides in x and y and turns about
the surface's normal. The other three are not merely unused. They are the
three this detector is worst at, measured on this cell's own capture at one
pixel of corner noise:

| | yaw | roll / pitch | x, y | range |
|---|---|---|---|---|
| noise at 1 px | **0.20°** | 1.07° | 1.0 mm | **3.35 mm** |
| moved by a 1% error in `box_size` | **0.0000°** | 0.0000° | 0.9, 1.4 mm | **11.8 mm** |

The second row is not a coincidence but a property of the problem: feed
`solvePnP` an object size 1% wrong and the whole answer scales by exactly 1%
about the optical axis, while the rotation does not move by so much as 4e-16.
Range takes almost all of it.

So `planar.py` asks for neither range nor tilt. It asks where a pixel lands on
one known surface, which for a fixed camera and a fixed plane is a homography
— eight numbers, and *exact* rather than a fit that happens to be close. On
ideal data the residual is 0.0000 mm at every position tried, including
positions well outside the ones it was fitted from.

What that buys beyond accuracy is that `vision.camera_to_world` stops
mattering. Where the lens is, which way it points, what `box_size` says, what
the lens does to straight lines — all of it collapses into those eight
numbers, fitted from the box itself at the height it actually sits. The
placement that could not be got below about 9 mm is not improved; it is made
unnecessary.

The bill comes due in one place. The map is a map of *one* plane, and a box
whose rim sits higher than the plane it was fitted at is reported shifted
along the line of sight by the height error times the tangent of the camera's
angle from vertical — 1.88 mm per millimetre at this cell's 62°. Ten
millimetres of variation in box height is twenty-five millimetres of miss. If
the boxes stop being identical, `detect.py`'s full pose is the right tool.

That tangent is also the one argument about where to bolt the camera, and it
points the opposite way from `detect.py`'s. Full 6-DOF wants an oblique view,
because foreshortening is what fixes tilt: out-of-plane noise falls from 1.07°
to 0.30° between a square-on view and 60°. A plane map wants the view as near
overhead as the rig allows, which improves both of its terms at once — at 35°
the height penalty is 0.70 mm/mm instead of 1.88 and yaw noise 0.35° instead
of 0.59°. Nothing in software is worth as much as moving the bracket.

### Surface-map calibration

There was a way to say where the box is that involved driving one arm to each
of four rim corners, three or four times over. It worked and it is gone,
because doing the job says the same thing for nothing: two arms carry the box,
set it down, and are still gripping it. Their poses at that moment are a frame rigidly
attached to the box — not the box's frame, because nobody measured where the
grips sit on it, but one that differs from it by the *same* unknown every
time.

That is what makes it solvable with three unknowns rather than eleven. The map
has eight degrees of freedom and the grip-to-box offset has three, but the
eight follow in closed form once the three are guessed: propose an offset, and
every cycle's rim corners are known in the cell, which is a plain homography
fit. Only the three are searched, and the inner fit's residual scores them.

The fitting data still comes from the work: the box stands on the table, both
arms are recorded at a repeatable pose relative to it, then they swing clear
and the camera records its corners. Move the box and repeat. Three rounds are
the mathematical minimum for a map; five or six is what a real answer looks
like. This is cell calibration data, separate from the day-to-day
`Create Home Box` popup, which consumes the fitted map in one press.

How far short of the box the arms stop does not matter, and that is the point
of storing a stance rather than a grip: what has to be the same every round is
the pose *relative to the box*, and closing the last few centimetres is an
offset the program adds afterwards — `at part + world Z-40`.

Two properties are worth knowing before trusting it.

The first is that the offset is fixed only up to a shift unless the box is
turned through a range between cycles — the same argument `spread_of` makes
one dimension up, and the reason the flange fit refuses readings taken along a
line. Here it does not have to be refused, because the shift is a gauge: the
map is fitted *through* that same offset, so the frame the two of them agree
to call the box's middle may not be its middle, and every number derived from
the pair is still right. Measured against a rendered box, the arms land within
1.2 mm of where the box needs them while the offset itself is metres from
being a physical description of anything.

The second is that a sample must be a measurement. `OpenBoxDetector` smooths
corners across frames and holds them through a jump, which is right for a
picture somebody is watching and wrong here: a sample taken after the box
moved would otherwise be part of where it used to be. Every capture forgets
the temporal lock first. Measured without that, a calibration built from
blends of consecutive placements put the arms 128 mm out.

### What it refuses

A map fitted from fewer than three pick-and-places is refused: two determine
the eleven numbers between them and say nothing about whether they are right,
and disagreement is the only evidence a fit ever offers.

A fit whose residual exceeds 30 mm is refused as well. That threshold is not a
noise limit — the simulated camera, which rasterises its box onto whole
pixels, fits to about 2 mm — it is there for the one mistake that does not
look like one: plane axes handed the wrong way round, which pairs every corner
with the one diagonally opposite. Measured, that fits to 179.9 mm against the
same samples' 2.0 mm, so anything between the two settles it.

Which image corner is which is not assumed either. `rim.order_corners` labels
them by where they sit in the picture, and that rolls by one as the box turns
— measured on this cell's geometry, already at -25°. So the labels are re-read
from what the map says: fit with them as they came, let that map say which
corner each one really is, and fit again.

The last one is not a refusal at all any more, and it used to be the sharpest.
A map is a map of one plane and that plane is the rim of one box, so the
opening's size being a live control on the Camera tab while the map was a
single file meant that changing the crate produced not an error but an answer
— the wrong rectangle fitted through a map of the wrong plane. Swap a 200 mm
crate for a 230 mm one and the rim it reads is 30 mm off the surface the map
knows, which at 62° is 56 mm of lateral miss with nothing on screen to say so.

So the record follows the crate. `config/plane.json` with a 200x100x100 box in
front of the lens is `config/plane_200x100x100.json`, and the log beside it
the same; changing the size on the Camera tab opens that crate's record, and
putting the first crate back finds its map again untouched. A cell that runs
three crates keeps three maps and never chooses between them by hand.

`PlaneFile.fits_box` stays, because it guards a different thing: the path
decides which file to open and the check decides whether what was opened can
be believed. A path can be typed and a file can be edited.

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

## The machines beside the cell

One **Com** tab on the rail — Communication — holding every machine the cell
talks to whatever it speaks. The protocol is a **dropdown on the machine**, not a tab to have
picked correctly first — a cell has *machines*, and which of them answers
Modbus is a detail of how it is reached:

| protocol | what a datum is | how waiting works |
|---|---|---|
| **TCP** | a string or a byte frame | the machine sends, and messages split on the link's terminator |
| **UDP** | the same, as datagrams | one datagram is one message, terminator or not |
| **Modbus TCP** | a number at a register address | nothing arrives — the cell reads the same register until it says what the step wanted |

The tab is a list of machines and, under it, the named data items on whichever
one is selected. `＋ Add` on the top list sets up a machine:

| field | |
|---|---|
| name | what a program line calls it — `MC1`, `PLC1` |
| protocol | TCP, UDP or Modbus TCP. The fields below it swap when this moves, rather than half of them sitting greyed out |
| address, port | `10.1.68.200`, `2000` (choosing Modbus offers 502) |
| connect in | how long *dialling* may take. Not how long a `RECV` waits — that is on the step |
| ends with | TCP/UDP: what marks the end of one message, typed `\r\n`. It is added to everything sent and is what incoming bytes are split on. Empty for a machine that frames its own packets |
| data / encoding | `string` with an encoding, or `hex` for a frame written `02 41 03` |
| device id | Modbus only: the unit/slave id, usually 1 |

`＋ Add` on the lower list names one datum on that machine:

| field | |
|---|---|
| name | what the program line picks — `START`, `DONE`, `READY` |
| way | `send` (the cell may put it on the wire), `recv` (the cell may wait for it), or `both` |
| payload | TCP/UDP: the text or bytes it carries |
| matches | `exact`, `contains` or `prefix` — for machines that append a sequence number nobody asked for |
| table, register, value | Modbus: `coil`/`discrete in`/`holding reg`/`input reg`, the address, and the number to write or wait for |

Changing a machine's protocol keeps its data items — they are what that
machine carries — but a payload of `ST` is not a register value, so the tab
says so beside the list that has to be gone through.

The direction is enforced rather than decorative: a machine's "cycle done" is
not something this cell may send, so it never appears in a `SEND` picker, and
the checker refuses a line that names it there anyway. Modbus's two input
tables are read-only in the protocol itself, and a `send` item on one is
refused in the dialog that created it.

`⇄ Test` opens the machine now and says whether it answered — the point is to
fail on a tab, with a cursor in the field that is wrong, rather than three
lines into a program with an arm already somewhere. `💾 Save` writes them into
the `comms:` block of `cell.yaml`; edits take effect on the wire immediately,
and the file is only where they survive a restart.

Connections open on the first step that uses one and stay open. Nothing
redials on its own: a machine that dropped the connection is a fact the
operator needs told, and a link that silently redialled would turn a cable
pulled out of a socket into a program that runs half a cycle late. Correcting
a machine's port hangs up on it so the next step redials; naming another datum
on it does not.

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
at. The 512 px sidebar still presents one target at a time so its motion keys
stay large and the active arm remains unmistakable. The live target's
button is filled rather than ticked, the band over the keys is the arm's own
colour and names it, the keys carry that tint, and **changing target releases
whatever key is held** so a press meant for one arm can never be inherited by
the other. Nothing on the page is narrower than 74 px.

The wider sidebar gives the camera and setup panels more working room while
the program still retains both arm columns.

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

`comms.links` is the machines the cell talks to — a name, a protocol, an
address, and the named data items on it. Written by the Communication tab's 💾 Save,
which rewrites that block alone and leaves every other key exactly as it found
it.
The terminator is stored the way it is typed (`"\r\n"`), so the file can be
read and hand-edited without a hex dump. See
[The machines beside the cell](#the-machines-beside-the-cell).

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
