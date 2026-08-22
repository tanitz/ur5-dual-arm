# One line, both arms — a draft

Today a program line addresses one thing. `MOVE_ARM` names an arm, `MOVE_OBJ`
names the object, and a two-arm approach is written as two lines that happen
to be next to each other:

```
 3  MOVE_ARM   arm A  movel -> pick_A
 4  MOVE_ARM   arm B  movel -> pick_B
 5  BARRIER
```

Those two lines are not two things. They are one gesture — both arms go to
the panel — written twice and then stapled together with a barrier. The arms
also run it strictly one after the other, because `_move_arm` waits for
arrival before the executor moves on, so a step that should take four seconds
takes eight.

This draft makes the line the gesture: one row, two columns, and a field that
says how tightly the two columns are tied to each other.

## The line

```
┌ Program — teach & run ───────────────────────────────────────────────────┐
│ now   A  -412.3  188.0  505.1  |  B  -395.8 -190.4  504.7  |  gap 380.2  │
├───┬────────┬────────────────────┬────────────────────┬───────────────────┤
│ # │ step   │ Arm A              │ Arm B              │ link              │
├───┼────────┼────────────────────┼────────────────────┼───────────────────┤
│ 1 │ MOVE   │ home_A     movej   │ home_B     movej   │ ⇉ together        │
│ 2 │ MOVE   │ pick  Z+50 movel   │ pick_B     movel   │ ⇉ together        │
│ 3 │ MOVE   │ Z−50       movel   │ Z−50       movel   │ ⇉ together        │
│ 4 │ GRIP   │ close DO0          │ close DO0          │ ⇉                 │
│ 5 │ MOVE   │ world  Z +100 ················· 50 mm/s │ ⇉⇉ pair           │
│ 6 │ ATTACH │ midpoint ······························· │                   │
│ 7 │ MOVE   │ ⟨obj⟩ place_1 ·························· │ ⛓ coupled         │
│ 8 │ MOVE   │ ⟨obj⟩ object RZ +45 ···················· │ ⛓ coupled         │
│ 9 │ MOVE   │ ⟨obj⟩ world  Z +80 ····················· │ ⛓ coupled         │
│10 │ DETACH │ ······································· │                   │
│11 │ MOVE   │ home_A     movej   │ ·                  │                   │
└───┴────────┴────────────────────┴────────────────────┴───────────────────┘
```

A step that has nothing to say about one arm shows `·` in that column and
leaves it standing. A step that addresses the pair as one body spans both
columns, which is the visual difference between "two arms" and "one object".

## `link` — how tightly the two columns are tied

This is the field the whole design turns on, and it is not a preference. Each
value is a different machine underneath, with different things it can and
cannot do.

| link | what happens | engine | needs |
|---|---|---|---|
| *(blank)* | only the filled column moves | `movej`/`movel`, one arm | — |
| `together` | both columns are sent, then the line waits for **both** to arrive | two `movel`s + a joint wait | nothing attached |
| `pair` | **one** world delta given to both arms at matched speed | two `movel`s, same `v`/`a` | `translation_calibrated`, nothing attached, translation only |
| `coupled` | one object frame drives both arms from one clock | the 125 Hz servo loop | an `ATTACH` above it |

`together` is the honest name for what two lines and a barrier already do —
it is *not* coordination. Each controller plans its own timing, so the two
arms are in the same place at the end of the line and nowhere in particular
in the middle. It is for approach and retreat, when the arms are not holding
the same thing. It cannot be used while something is attached, and the
validator says so.

`pair` is the Jog tab's `A+B` column written down. Both arms are given the
same world direction, each resolved through its own mounting transform, and
the same speed — so the paths are congruent and a workpiece gripped between
them is carried without either arm being told about the other. Two rules come
straight from the jog panel and are not negotiable: **world frame only**,
because "base" and "tool" name a different direction at each robot, and
**translation only**, because RX/RY/RZ turn each wrist about its own tool and
would twist a rigid workpiece rather than turn it. Turning the pair is what
`coupled` is for.

`coupled` is `MOVE_OBJ` and `ROTATE_OBJ` as they are now, unchanged
underneath: the closed-chain solver, the force and drift guards on every
cycle, the pre-walked joint path.

Between `pair` and `coupled` there is a real gap worth naming: `pair` has no
125 Hz guard. Its guards run in the arrival wait loop at 50 Hz — force on
either TCP, and the A-to-B transform against what it was when the line
started — and it halts both arms on either. That is enough for a slow
translation and is not enough for anything else, which is the reason for the
translation-only rule rather than an extra excuse for it.

## What goes in a column

One editor for both columns, four ways to fill it. This is where the offsets
live: an offset is not a step kind, it is a target with no named place in it.

| target | JSON | means |
|---|---|---|
| place | `{"point": "pick_A"}` | go to the taught place |
| place + offset | `{"point": "pick_A", "offset": [0,0,0.05,0,0,0], "frame": "world"}` | go to the place, shifted — approach and retreat without teaching two points |
| offset only | `{"offset": [0,0,-0.05,0,0,0], "frame": "tool"}` | shift from wherever this arm is when the line runs |
| here | `{"pose": [-0.412, 0.188, 0.505, 0, 3.14, 0]}` | a pose captured with the **Here** button, written into the step |
| — | omitted | this arm does not move |

Offset frames follow the same rule as jogging, for the same reason:

| column | frames offered |
|---|---|
| Arm A / Arm B alone | `world`, `base`, `tool` |
| `pair` | `world` only |
| `coupled` | `world`, `object`, and an optional pivot point |

A `coupled` offset carries rotation in the same six numbers — `[0,0,0.05,0,0,0.785]`
in the `object` frame is "up 50 mm and turn 45° about its own Z" — so
`ROTATE_OBJ` stops being a separate kind and becomes a line with a rotation
in its offset. The uncalibrated rotation limit in `coupling.py` still applies
to it and is untouched.

## Getting the current position

The panel can only record a position into a point today. Three additions, all
of them reading the same live numbers:

**A live row above the program.** Arm A's TCP, arm B's TCP, the gap, and the
object frame when something is held — in world, in millimetres and degrees.
The Program tab is where a position is needed while writing, and sending the
operator to the Jog tab's `Details` window to read it is the reason points get
taught that nobody wanted.

**A `Here` button on each column of the step editor.** It fills that column
with the arm's current pose, as a literal `pose` written into the step, with a
second button to promote it to a named point instead. A pose baked into the
step is deterministic and replays exactly; the trade is that it is a number in
a program rather than a name in the library, so it does not follow a
re-taught point. Both are offered because both are wanted: named places for
the fixtures, baked poses for a one-off.

**A `WHERE` step.** Logs A, B, the object pose, the gap and both joint vectors
into the message drawer when it runs. It moves nothing. It is for reading back
what a dry run actually computed, and for finding out where a program stopped.

## The file, before and after

```json
{"kind": "MOVE_ARM", "arm": "A", "point": "pick_A", "motion": "movel"}
{"kind": "MOVE_ARM", "arm": "B", "point": "pick_B", "motion": "movel"}
{"kind": "BARRIER"}
```

becomes

```json
{"kind": "MOVE", "link": "together",
 "a": {"point": "pick_A", "motion": "movel", "speed": 0.12},
 "b": {"point": "pick_B", "motion": "movel", "speed": 0.12}}
```

and the two-armed shapes that have no spelling today:

```json
{"kind": "MOVE", "link": "pair",
 "pair": {"offset": [0, 0, 0.10, 0, 0, 0], "frame": "world", "speed": 0.05}}

{"kind": "MOVE", "link": "coupled",
 "obj": {"point": "place_1", "lin_speed": 0.05, "ang_speed": 0.2}}

{"kind": "MOVE", "link": "coupled",
 "obj": {"offset": [0, 0, 0, 0, 0, 0.785], "frame": "object"}}
```

`GRIP` gets the same two columns, so closing both grippers is one line:

```json
{"kind": "GRIP", "a": {"output": 0, "state": true},
                 "b": {"output": 0, "state": true}, "settle": 0.4}
```

`ATTACH`, `DETACH`, `DELAY`, `BARRIER` and `WHERE` are unchanged and span both
columns. `BARRIER` survives even though every `MOVE` now waits for its own
arms — it is still the way to say "settle" after a gripper or before a
measurement.

Old programs keep working: `Step.from_dict` maps `MOVE_ARM` to a `MOVE` with
one column filled, `MOVE_OBJ` to `link: coupled` with a point, and
`ROTATE_OBJ` to `link: coupled` with a rotation offset. Nothing on disk has to
be rewritten, and `config/programs/six_axis_dry_check.json` is the test of
that.

## The editor

```
┌ Add a step ──────────────────────────────┐
│ kind   [ MOVE ▾ ]                        │
│ link   [ together ▾ ]   ⧉ mirror A → B   │
├──────────────────┬───────────────────────┤
│ Arm A            │ Arm B                 │
│ ( ) place  [▾]   │ ( ) place  [▾]        │
│ (•) place+offset │ (•) place+offset      │
│ ( ) offset only  │ ( ) offset only       │
│ ( ) here  [Here] │ ( ) here  [Here]      │
│ frame  [world ▾] │ frame  [world ▾]      │
│  X   0.0    mm   │  X   0.0    mm        │
│  Y   0.0    mm   │  Y   0.0    mm        │
│  Z  +50.0   mm   │  Z  +50.0   mm        │
│ RX RY RZ   0 deg │ RX RY RZ   0 deg      │
│ motion [movel ▾] │ motion [movel ▾]      │
│ speed  120  mm/s │ speed  120  mm/s      │
├──────────────────┴───────────────────────┤
│            ＋  Add step                  │
└──────────────────────────────────────────┘
```

`mirror A → B` copies the left column to the right, which is how most lines
in a two-arm program are actually written. Choosing `pair` or `coupled`
collapses the two columns into one, because there is only one thing to say.

## What the validator gains

On top of what it checks today:

- `together` or `pair` while something is attached — the arms would tear the
  object out of the grippers or fight the servo loop for the same controllers
- `pair` with any rotation in its offset — twists the workpiece, does not turn it
- `pair` with `frame` other than `world`
- `pair` without `translation_calibrated` — the same gate the Jog tab's A+B
  column already has
- `coupled` on a line with `a`/`b` columns filled — the object drives both
- a `MOVE` with no column filled at all
- an offset-only target in the first line of a program, which is a relative
  move from an unknown starting pose (warning, not a refusal)

## What it costs

| file | change |
|---|---|
| `program/steps.py` | `MOVE` kind, the target grammar, `describe()` per column, `from_dict` migration, the new validation rules |
| `program/executor.py` | `_move_arm` splits into "resolve a target to a world pose" and "send"; a joint arrival wait over a set of arms; the `pair` path with its 50 Hz guards |
| `gui/panels/program.py` | two-column table, the two-column editor, the live position row |
| `tests/test_program.py` | resolution of each target type, the migration, each new validation rule |

Nothing in `coupling.py`, `cell.py` or `robot/` moves. `pair` is built from
`arm.movel_world` and `cell.relative_transform`, both of which exist.

## Three things decided

1. **`pair` exists.** Translation only, world frame only, and gated on
   `translation_calibrated` — the same gate the Jog tab's A+B column already
   has. It is the one two-arm motion without a 125 Hz guard behind it, which
   is what the translation-only rule is paying for.

2. **Offsets resolve at run time.** `Z−50` moves 50 mm down from wherever the
   arm is when the line executes. That is what makes stacking and feeding
   programs possible, and it means a line that ends somewhere unexpected hands
   the surprise to the next line. The validator warns on an offset-only target
   in the first line of a program, where the starting pose is whatever the
   operator left the arm at.

3. **One `MOVE` kind, migrated on load.** `MOVE_ARM`, `MOVE_OBJ` and
   `ROTATE_OBJ` become `MOVE` with `a`/`b`/`pair`/`obj` slots. `Step.from_dict`
   translates the old spellings, so nothing on disk is rewritten and
   `config/programs/six_axis_dry_check.json` is the test that it works.

## Order of work

1. `program/steps.py` — the target grammar and its resolution, `MOVE`,
   `from_dict` migration, `describe()` per column, the new validation rules.
   All of it testable without a robot.
2. `tests/test_program.py` — each target type, the migration, each rule.
3. `program/executor.py` — split target resolution from sending, a joint
   arrival wait over a set of arms, the `pair` path and its 50 Hz guards.
4. `gui/panels/program.py` — the two-column table, the two-column editor,
   the live position row, `Here`, `WHERE`.

## Where the width comes from

Two columns of targets do not fit the program zone as it stands. At the
1280x800 design size the work area is 1264 px, the program zone is capped at
`PROGRAM_COL_W = 620`, and the step editor takes 250 of that — so the table
has 350 px. Five columns leave this much for each arm:

Five columns — `#` 28, step 62, Arm A, Arm B, link 90 — leave this much for
each arm:

| layout | table width | per arm | characters at 13 px |
|---|---|---|---|
| today | 350 | 85 | ~12 |
| editor hidden | 608 | 214 | ~31 |
| tabs hidden, editor open | 1002 | 411 | ~59 |
| tabs hidden, editor hidden | 1260 | 540 | ~77 |

Option D below reaches 232 px without moving anything, by spending the link
column instead of the zone.

`docs/drawings/make_layout_plates.py` draws all five of these to scale from
the same constants, one SVG per option, with the step text clipped at the true
column edge rather than written to fit.

`pick + world Z+50.0  movel` is 26 characters. Twelve is not a column, it is
an ellipsis.

One rule constrains every option below: **this is a touch panel, so a zone is
opened and closed by a button, never by dragging a splitter handle.** A 4 px
handle is not a finger target, and a layout an operator can half-drag is a
layout that ends up wrong.

### A — a Wide button that collapses the tab zone

```
┌ Program ─────────────────────────────────┐   ┌ Program ──────────────────────────────────────────────────┐
│ …                        │ REAL … STOP   │   │ REAL  A  B  Power  Brake  Unlock            ■ STOP        │
│ # step   A       B  link │ [Points][Jog] │ → │ # step   Arm A            Arm B          link      [▶ ⇔] │
│                          │               │   │                                                          │
│ [▶ Run] [⏸] [■]   [⇔ Wide]│              │   │ 1 MOVE   home_A movej    home_B movej    ⇉ together      │
└──────────────────────────┴───────────────┘   └──────────────────────────────────────────────────────────┘
```

The most room, and it has a prerequisite that is not optional: the REAL
indicator, the connection and dashboard controls and the cell-wide STOP live
in the right zone today, and **STOP may never be hidden**. So this option
moves that bar to a full-width row above both zones — a change to what the
README calls "one row above the tabs", and the reason it is worth making is
that it also stops STOP from being a right-hand-column thing at all.

### B — the editor becomes a sheet

```
┌ Program ────────────────────────────────┐
│ # step   Arm A          Arm B     link  │
│ 1 MOVE   home_A movej   home_B…   ⇉     │
│ …                                       │
│ [▲][▼][Delete][On/off]      [＋ Add]    │
└─────────────────────────────────────────┘
        pressing ＋ slides the editor up over the table
```

The 250 px editor is only wanted while a step is being written. Making it a
sheet that slides over the table gives the table the whole program zone the
rest of the time, and it changes nothing outside `panels/program.py` — the
tabs, the safety bar and the message drawer are untouched. 30 characters an
arm is enough for the common line and clips the long one.

### C — one button cycling three widths

`[Program | Split | Tabs]` — the program zone owns everything, the two share
as they do today, or the tabs own everything. A superset of A, and the third
state is worth something on its own: jogging with three columns of six axes
wants the width more than the program does. Same STOP prerequisite as A, and
one more state for an operator to be in the wrong one of.

### E — A or C, with the jog keys docked

```
┌ REAL  A  B  Power  Brake  Unlock                    ■ STOP ┐
│ # step   Arm A              Arm B         link      [▶ ⇔]  │
│ 1 MOVE   home_A movej       home_B movej  ⇉ together        │
│ …                                                          │
├────────────────────────────────────────────────────────────┤
│ JOG — DOCKED  [A][A+B][B]  −X+ −Y+ −Z+ −RX+ −RY+ −RZ+  10  │
├────────────────────────────────────────────────────────────┤
│ [▲][▼][Delete][On/off][⇔ Wide]      [▶ Run][⏸][■ Stop]     │
└────────────────────────────────────────────────────────────┘
```

A and C both hide the whole Jog page when the program goes wide, and teaching
is *jog, Here, jog, Here* — so hiding it puts two extra presses on every taught
point. One row of jog stays docked instead: the three targets, all six axes,
the step preset. The full three-column grid still lives on the Jog tab, where
the width is worth having.

It costs table height rather than width — 411 px an arm is untouched, and about
two rows of the list pay for it.

Whatever collapses a zone must call `panels["jog"].release()` on the way, the
same as `app._tab_changed` and `app.changeEvent` already do. A held jog key
whose button has left the screen must not still be driving an arm; that is the
one part of this option that is not cosmetic.

### F, G, H — an icon rail, and Jog as a sidebar on the left

```
┌ REAL  A  B  Power  Brake  Unlock                        ■ STOP ┐
├──┬──────────────────┬──────────────────────────────────────────┤
│◀ │ Jog              │ # step   Arm A          Arm B     link   │
│  │ [A][A+B][B]      │ 1 MOVE   home_A movej   home_B…   ⇉      │
│◎ │ world ▾          │ 2 MOVE   pick + world…  pick + …  ⇉      │
│⚙ │  −   X   +       │ …                                        │
│▣ │  −   Y   +       │                                          │
│✥ │  −   Z   +       │ [▲][▼][Delete]        [▶ Run][⏸][■]      │
└──┴──────────────────┴──────────────────────────────────────────┘
  rail    sidebar 320                program 890
```

The tab zone stops being a fixed half and becomes a sidebar the toggle can
close. A 56 px rail carries the four panels as icons and never disappears, so
**the way back is not hidden behind the thing that took it away** — which is
the one real flaw in a bare Wide button. The lit icon is the open panel;
pressing it again closes the sidebar.

| plate | sidebar | editor | per arm |
|---|---|---|---|
| F | 320 px, one jog target at a time | open | 217 |
| G | 640 px, all three targets | sheet | 186 |
| H | closed | open | 380 |

F and G differ on one thing, and it is not width. **A 320 px sidebar cannot
show three columns of finger-sized keys, so F brings back a target selector —
the control this panel deliberately removed.** The README says "no target
selector stands between the operator and either arm", and
`test_gui_layout.py` asserts `not hasattr(jog, "target_combo")`. F contradicts
both on purpose; G keeps the rule and pays for it by making the step editor a
sheet, ending up with less room per arm than option B does.

H is what one press leaves either way, and it is the state the program is
mostly read in: 380 px an arm with the editor still open.

Two consequences carry over from the options above. The safety bar has to move
to a full-width row, because the sidebar can close and STOP cannot. And
closing the sidebar must call `panels["jog"].release()`, or a held jog key goes
on driving an arm from behind a closed panel.

One thing this layout changes that the drawings cannot show: jogging moves to
the **left** hand and the program to the right, reversing where both have been
for the whole life of the panel.

### D — two rows per step, and no zone change at all

```
┌──┬───────────┬──────────────────────────────────────┐
│ #│ step      │ arm / detail                         │
├──┼───────────┼──────────────────────────────────────┤
│ 1│ MOVE ⇉    │ A  home_A  movej                     │
│  │           │ B  home_B  movej                     │
│ 2│ MOVE ⇉    │ A  pick + world Z+50.0  movel        │
│  │           │ B  pick + world Z+50.0  movel        │
│ 5│ MOVE ⇉⇉   │    world Z+100.0            ⇉⇉ pair  │
│ 7│ MOVE ⛓    │    ⟨obj⟩ place_1          ⛓ coupled  │
└──┴───────────┴──────────────────────────────────────┘
```

This one refuses the premise. The link stops being a column and becomes a
glyph in the step column, the two arms share one detail column, and each takes
a row of its own: 232 px apiece with the panel exactly as it is — more than
option B gets by hiding the editor, and with no new control anywhere.

A single-arm line still takes one row, because the empty arm has nothing to
show, and a coupled or pair line stays a single spanning row. The cost is that
a two-armed program shows about half as many steps at a time — roughly 8 rather
than 16 on the 10" panel — and that "one line, both arms" becomes true of the
step and not of the row.
