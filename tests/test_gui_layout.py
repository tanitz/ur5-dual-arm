"""The C1 jog layout and its routing, without a display or robots."""

import math
import os
import sys
import tempfile
import time

import numpy as np
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtCore import Qt                    # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from ur5dual.axes import WORLD_AXIS_SIGN   # noqa: E402
from ur5dual.cell import Cell             # noqa: E402
from ur5dual.config import CellConfig     # noqa: E402
from ur5dual.gui import style as S        # noqa: E402
from ur5dual.gui.app import MainWindow    # noqa: E402
from ur5dual.geometry.kinematics import mat_to_pose as _mat_to_pose  # noqa: E402
from ur5dual.geometry.kinematics import pose_to_mat as _pose_to_mat  # noqa: E402
from ur5dual.program.steps import Step                    # noqa: E402
from ur5dual.tools.plane_fit import PlaneFitSession       # noqa: E402
from ur5dual.vision.planar import PlaneFile, rim_corners  # noqa: E402


fail = 0


def check(name, ok, detail=""):
    global fail
    print(("  ok   " if ok else "  FAIL ") + name +
          (("  " + detail) if detail else ""))
    if not ok:
        fail += 1


app = QApplication.instance() or QApplication([])
S.set_scale(1.0)
# The panel saves as an operator presses things — the sidebar side, the taught
# camera point, the colour band. Pointed at the real cell file this test would
# quietly overwrite a taught cell with whatever it pressed last, so it is given
# a copy in a temporary directory to scribble on.
config = CellConfig.load()
config.path = os.path.join(tempfile.mkdtemp(prefix="ur5dual-layout-"),
                           "cell.yaml")
# And the simulated source, whatever the real cell is set to: the camera checks
# below are written against a known box, and a RealSense plugged into the bench
# would answer them with whatever is in front of it — as well as being taken
# away from whichever panel is actually using it.
config.vision["source"] = "sim"
config.vision["home_references"] = {}
# The surface checks below press Fit and Teach, which write files. Left at
# their defaults those are the repo's own config/plane.json and plane_log.json
# — the map the cell actually runs on — so they are pointed at a scratch
# directory for the same reason cell.yaml is.
_surface_dir = tempfile.mkdtemp(prefix="ur5dual-surface-")
config.vision["plane_file"] = os.path.join(_surface_dir, "plane.json")
config.vision["plane_log"] = os.path.join(_surface_dir, "plane_log.json")
# Nor does it inherit the box an operator taught on the real cell: a point
# tapped on a tote at the far side of a room is not on the simulated box, and
# the panel would rightly report that it is not — a true answer to a question
# this file is not asking.
config.vision["target_uv"] = None
# Same for the opening's size and the sizes used before it. The camera checks
# below are quoted against the 600 x 400 box the simulated source draws, and a
# cell set to whatever crate is on the bench this week would answer them
# truthfully with "no four-sided opening" — again, not the question here.
config.vision["box_size"] = [0.60, 0.40, 0.20]
config.vision["box_sizes"] = [[0.30, 0.22]]
config.vision["auto_size"] = False
window = MainWindow(cell=Cell(config, simulated=True),
                    connect_on_start=False)
window.resize(1280, 800)
window.show()
app.processEvents()
jog = window.panels["jog"]

print("the full-width page")
check("all three targets exist", list(jog.grids) == ["A", "AB", "B"])
check("but only the chosen one is on screen — the sidebar is 320 px wide and "
      "three columns of finger-sized keys do not fit",
      [t for t in jog.grids if jog.target_columns[t].isVisible()] == [jog.target],
      str([t for t in jog.grids if jog.target_columns[t].isVisible()]))
check("the choice is three buttons, not a closed drop-down",
      not hasattr(jog, "target_combo")
      and sorted(jog.target_btns) == ["A", "AB", "B"]
      and all(b.isVisible() for b in jog.target_btns.values()))
check("and the live one is the one that is filled in",
      jog.target_btns[jog.target].isChecked()
      and not any(b.isChecked() for t, b in jog.target_btns.items()
                  if t != jog.target))
check("STOP is global rather than owned by the Jog page",
      window.stop_btn.isVisible() and window.stop_btn.parent() is not jog)
check("the REAL mode selector is gone", not hasattr(window, "mode_combo"))
check("the REAL control bar spans the whole panel",
      # It used to sit above the tab column. That column is a sidebar now and
      # a sidebar can be closed, so anything that must never disappear cannot
      # live inside it -- least of all STOP.
      abs(window.safety_bar.width() - window.work.width()) <= 4,
      "%d vs %d" % (window.safety_bar.width(), window.work.width()))
bar_controls = ([window.real_indicator] + list(window.conn_btns.values()) +
                list(window.dashboard_btns.values()) + [window.stop_btn])
check("REAL, arm, dashboard and STOP controls share one row",
      len({button.y() for button in bar_controls}) == 1)
check("the log starts collapsed", not window.msg_box.isVisible())
check("all six Cartesian axes are always shown on the target that is up",
      [i for i, row in enumerate(jog.grids[jog.target].rows)
       if row[0].isVisibleTo(jog)] == [0, 1, 2, 3, 4, 5])

print("the rail, and the sidebar it opens")
check("the rail is there before anything is pressed", window.rail.isVisible())
check("the jog panel is what opens first", window.sidebar.isVisible()
      and window.sidebar.currentWidget() is jog)


def body_order():
    return [window.body.itemAt(i).widget() for i in range(window.body.count())]


check("the sidebar sits between the program and the rail",
      body_order() == [window.program_page, window.sidebar, window.rail])

wide_before = window.program_page.width()
window.rail.buttons["jog"].click()
app.processEvents()
check("pressing the lit icon closes the sidebar", not window.sidebar.isVisible())
check("and the program takes the width back",
      window.program_page.width() > wide_before + 200,
      "%d -> %d" % (wide_before, window.program_page.width()))
check("the rail does not go with it — the way back stays on screen",
      window.rail.isVisible())
window.rail.buttons["jog"].click()
app.processEvents()
check("pressing it again opens the same panel", window.sidebar.isVisible()
      and window.sidebar.currentWidget() is jog)

with_jog = window.program_page.width()
window.rail.buttons["points"].click()
app.processEvents()
check("another icon swaps which panel is open, without closing it",
      window.sidebar.isVisible()
      and window.sidebar.currentWidget() is window.panels["points"])
check("and the program does not move when it does — every panel is the same "
      "width, so the step table never slides sideways under a finger",
      window.program_page.width() == with_jog,
      "%d vs %d" % (with_jog, window.program_page.width()))
check("the points list is laid out for that width, not squeezed into it",
      window.panels["points"].table.width() > 250,
      "%d px" % window.panels["points"].table.width())
window.rail.buttons["jog"].click()
app.processEvents()

# A held key whose button has left the screen must not still be driving an arm.
# The same rule the tab switch and the window deactivation already follow.
jog.frame_combo.setCurrentIndex(0)
jog.motion_combo.setCurrentIndex(0)
jog.motion_combo.setCurrentIndex(0)                 # hold mode, not step
jog.grids["A"].buttons[1].pressed.emit()
app.processEvents()
check("a held key is recorded as held", jog.grids["A"].held is not None,
      str(jog.grids["A"].held))
window.rail.buttons["jog"].click()          # close it with the key still held
app.processEvents()
check("closing the sidebar releases a held jog key",
      jog.grids["A"].held is None, str(jog.grids["A"].held))
window.rail.buttons["jog"].click()
app.processEvents()

print("the window itself")
# The preference is written to cell.yaml, so the test writes to a copy of it.
window.cell.config.path = os.path.join(tempfile.mkdtemp(), "cell.yaml")
check("it starts maximised rather than fullscreen — the desktop's dock and "
      "top bar stay reachable",
      not window.wants_fullscreen())
window.show_window()
app.processEvents()
check("so that is how it starts",
      window.isMaximized() and not window.isFullScreen())
check("and it fits the panel screen with a title bar on it — a window bigger "
      "than the screen is one the maximise button cannot help",
      window.minimumSizeHint().width() <= 1280
      and window.minimumSizeHint().height() <= 800 - 40,
      "%dx%d" % (window.minimumSizeHint().width(),
                 window.minimumSizeHint().height()))
window.rail.full_btn.click()
app.processEvents()
check("the rail takes the whole screen when that is wanted",
      window.isFullScreen())
check("and that answer is remembered",
      CellConfig.load(window.cell.config.path).ui.get("fullscreen") is True)
window.rail.full_btn.click()
app.processEvents()
check("pressing it again gives the frame back — the only way out of "
      "fullscreen that fullscreen does not hide",
      not window.isFullScreen() and window.isMaximized())
# back to the 1280x800 panel the rest of the checks are quoted in — offscreen,
# the window manager's idea of maximised is whatever the dummy screen is.
window.showNormal()
window.resize(1280, 800)
app.processEvents()

print("the camera tab")
window.rail.buttons["camera"].click()
app.processEvents()
cam = window.panels["camera"]
check("the rail carries it", "camera" in window.rail.buttons)
check("and it opens in the sidebar", window.sidebar.currentWidget() is cam)
check("the opening's own size is typed in, in millimetres",
      cam.length_spin.value() == 600 and cam.width_spin.value() == 400,
      "%s x %s" % (cam.length_spin.value(), cam.width_spin.value()))
check("the sizes used before are on a dropdown, the live one at the top",
      [cam.size_combo.itemData(i) for i in range(cam.size_combo.count())]
      == ["auto", (600, 400, 200), (300, 220, 200)],
      "%s" % [cam.size_combo.itemText(i)
              for i in range(cam.size_combo.count())])
cam.length_spin.setValue(200)
cam.width_spin.setValue(100)
cam.height_spin.setValue(70)
cam._size_settled()
check("a size dialled in joins them, at the top, and is written down",
      [cam.size_combo.itemData(i) for i in range(cam.size_combo.count())]
      == ["auto", (200, 100, 70), (600, 400, 200), (300, 220, 200)]
      and CellConfig.load(cam.app.cell.config.path).vision["box_sizes"][0]
      == [0.20, 0.10, 0.07],
      "%s" % [cam.size_combo.itemText(i)
              for i in range(cam.size_combo.count())])
cam._size_picked(2)
check("and picking one off the list puts it back in the fields, and in use",
      (cam.length_spin.value(), cam.width_spin.value(),
       cam.height_spin.value()) == (600, 400, 200)
      and cam.app.vision.config["box_size"] == (0.60, 0.40, 0.20)
      and not cam.app.vision.config["auto_size"],
      "%s x %s x %s" % (cam.length_spin.value(), cam.width_spin.value(),
                          cam.height_spin.value()))
cam._size_picked(0)
check("Auto is a real mode rather than a label over the fixed size",
      cam.app.vision.config["auto_size"]
      and not cam.length_spin.isEnabled() and not cam.width_spin.isEnabled()
      and not cam.height_spin.isEnabled())
check("and nothing else is asked for — no thresholds to tune by hand",
      not any(hasattr(cam, name) for name in
              ("min_height", "max_height", "near", "far", "tolerance")))
check("the buttons that are not used are gone",
      not any(hasattr(cam, name) for name in ("once_btn", "teach_btn")))
check("at the same width as every other panel, so the program does not move",
      window.program_page.width() == with_jog,
      "%d vs %d" % (with_jog, window.program_page.width()))
def _read_live(tries=60):
    """Turn Live on until a frame with a pose in it has been painted.

    Live is the only way in now that Once has gone, so the panel has to be
    driven the way an operator drives it: start the camera, let the tick that
    paints the tab run, and wait for the detector's temporal lock to confirm.
    The generous try count is not the product being slow — this file runs
    beside a dozen other test processes, and a wall-clock timeout on a loaded
    Jetson fails for reasons that have nothing to do with the code under it.
    """
    cam.live_btn.setChecked(True)
    cam._toggle_live()
    for _ in range(tries):
        app.processEvents()
        cam.tick()
        if cam.view.pixmap() is not None and "mm" in cam.found_lbl.text():
            return True
        time.sleep(0.05)
    return False


found_live = _read_live()
check("Live reads the camera and finds the simulated box",
      found_live, cam.found_lbl.text().split("\n")[0])
check("and while it runs the button offers to stop it",
      cam.live_btn.text().endswith("Stop") and cam.live_btn.isChecked(),
      cam.live_btn.text())
check("the result shows all six pose components, and which frame they are in",
      "XYZ mm" in cam.found_lbl.text()
      and "RPY deg" in cam.found_lbl.text()
      and "camera R/F/H" in cam.found_lbl.text(), cam.found_lbl.text())
_tall = cam.found_lbl.height()
check("and the block is five lines whether or not the size check or the "
      "surface has anything to say — nothing under it walks down the panel",
      len(cam.found_lbl.text().split("\n")) == 5
      and cam.found_lbl.height() == _tall
      and cam.setup_lbl.height() == cam.setup_lbl.minimumHeight(),
      "%d lines, %d px" % (len(cam.found_lbl.text().split("\n")), _tall))
check("one of them answers the height question camera Z is misread as, "
      "and says so even on a cell whose surface has never been measured",
      "height" in cam.found_lbl.text(),
      cam.found_lbl.text().split("\n")[2])

# Uncalibrated, the numbers above are right/forward/height relative to the
# camera and its measured surface.  Once calibrated the full PnP pose is
# carried into world instead, because robot world and camera R/F/H do not
# share an origin or yaw.
# Once `camera_to_world` has been solved they are the frame the arms are
# taught in, and the label has to follow them — a world reading under a
# "camera frame" label, or the reverse, is the one mistake here that looks
# exactly like a correct answer.
_camera_xyz = cam.app.vision.latest.detection.centre * 1000
_vision = cam.app.cell.config.vision
_vision["camera_to_world"] = {"xyz": [1.0, 0.0, 0.5], "rpy": [0.0, 0.0, 0.0]}
_vision["calibrated"] = True
cam.tick()
_world_line = cam.found_lbl.text().split("\n")[0]
_world_xyz = [float(v) for v in _world_line.split()[2:5]]
check("a calibrated cell reads the box out where the arms live, not where "
      "the lens does",
      "world" in _world_line and "camera frame" not in _world_line,
      _world_line)
check("and the numbers are the camera's, carried through camera_to_world — "
      "X in the sign the jog keys count in, like every other world readout",
      abs(_world_xyz[0] - WORLD_AXIS_SIGN[0] * (_camera_xyz[0] + 1000)) < 0.2
      and abs(_world_xyz[2] - (_camera_xyz[2] + 500)) < 0.2,
      "%s -> %s" % (_camera_xyz, _world_xyz))
_vision["calibrated"] = False
_vision["camera_to_world"] = {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}
cam.tick()
check("and an uncalibrated cell goes back to saying camera R/F/H, rather "
      "than dressing an identity transform up as a world answer",
      "camera R/F/H" in cam.found_lbl.text().split("\n")[0],
      cam.found_lbl.text().split("\n")[0])
painted = cam.view.pixmap()
check("and draws the picture the detector was handed",
      painted is not None and not painted.isNull())
# the last reading stays readable after the lens is let go, and the checks
# below repaint from it — no thread is left running behind the rest of the file
cam.live_btn.setChecked(False)
cam._toggle_live()
check("and once stopped it offers to start again",
      cam.live_btn.text().endswith("Live") and not cam.live_btn.isChecked(),
      cam.live_btn.text())
check("the old target and HSV detector controls are gone",
      not any(hasattr(cam, name) for name in
              ("target_btn", "clear_target_btn", "colour_btn", "sample_btn",
               "hsv_box", "sliders")))
check("the wall depth and the search window are still stated, on one line",
      "wall" in cam.setup_lbl.text()
      and "whole picture" in cam.setup_lbl.text()
      and "\n" not in cam.setup_lbl.text(), cam.setup_lbl.text())
check("and depth is reported as a check on the size that was typed",
      "size check" in cam.found_lbl.text(),
      cam.found_lbl.text().split("\n")[-1])


print("\nthe home-box popup")
check("the old inline surface workflow is gone",
      not any(hasattr(cam, name) for name in
              ("teach_ref_btn", "drop_btn", "ref_combo", "surface_lbl")))
check("one explicit button opens the replacement workflow",
      cam.home_box_btn.text() == "Create Home Box"
      and "gripper" in cam.home_box_btn.toolTip(),
      cam.home_box_btn.text())
check("surface and box measurement has its own guided workflow",
      "Surface" in cam.surface_btn.text()
      and "surface" in cam.surface_btn.toolTip().lower(),
      cam.surface_btn.text())
cam.surface_btn.click()
for _ in range(10):
    app.processEvents()
    cam.tick()
    time.sleep(0.01)
surface_dialog = cam.surface_dialog
check("the replacement popup stages the box before the board",
      surface_dialog.isVisible()
      and surface_dialog.box_btn.text() == "1 Capture Box"
      and surface_dialog.board_btn.text() == "2 Capture Surface",
      surface_dialog.status.text())
check("and the measurement is saved back into the main GUI workflow",
      surface_dialog.apply_btn.text() == "Apply + Save"
      and not hasattr(surface_dialog, "jog_panel"))
surface_dialog.close()
check("without a full coordinate-frame calibration it offers a camera R/F/H "
      "box home instead of refusing to save",
      "camera R/F/H box home" in cam.home_box_lbl.text(),
      cam.home_box_lbl.text())

cam._open_home_box()
for _ in range(30):
    app.processEvents()
    cam.tick()
    if cam.home_box_dialog.camera_view.pixmap() is not None:
        break
    time.sleep(0.02)
fixed_dialog = cam.home_box_dialog
fixed_dialog.name_edit.setText("fixed_home")
fixed_dialog.jog_panel._select_target("AB")
fixed_dialog._save()
check("saving without calibration creates a FIND reference immediately",
      window.executor.taught_on_surface() == {"fixed_home"}
      and "fixed_home" in config.vision["home_references"]
      and "camera_rfh" in config.vision["home_references"]["fixed_home"]
      and "fixed_home" in CellConfig.load(config.path).vision[
          "home_references"],
      fixed_dialog.status.text())
check("the uncalibrated save is clearly identified as a camera R/F/H home",
      "camera R/F/H box home" in fixed_dialog.status.text(),
      fixed_dialog.status.text())
fixed_dialog.close()

RIM = config.vision["sim_plane_z"] - 0.20
GRIP = {"A": (0.012, 0.0), "B": (-0.011, 0.0)}
_real_tcp = {a: arm.tcp_matrix_world for a, arm in window.cell.arms.items()}


def _hold_like(x, y, yaw):
    for arm_id, (gx, gy) in GRIP.items():
        cos, sin = math.cos(yaw), math.sin(yaw)
        # turned with the box, because a gripper with hold of it is
        pose = _pose_to_mat([x + cos * gx - sin * gy,
                             y + sin * gx + cos * gy, RIM, 0, 0, yaw])
        window.cell.arms[arm_id].tcp_matrix_world = (
            lambda held=pose: held.copy())


# Build the independent surface calibration that the one-press reference
# teaching consumes. This is cell setup, not part of the home-box popup.
CYCLES = [(-0.05, -0.03, 0.0), (0.05, -0.03, math.radians(14)),
          (0.04, 0.04, math.radians(-11)), (-0.04, 0.03, math.radians(20))]
session = PlaneFitSession(config.vision["box_size"])
window.vision.start()
for _x, _y, _yaw in CYCLES:
    window.vision.camera.place(centre=(_x, -_y), yaw=-_yaw)
    _hold_like(_x, _y, _yaw)
    session.ready({arm_id: arm.tcp_matrix_world()
                   for arm_id, arm in window.cell.arms.items()},
                  name="calibration")
    window.vision.refit()
    session.look(window.vision.fresh(3.0).detection.corners)
store, _map, _offset = session.taught_from_cycles(
    "calibration", window.surface_path(), height=RIM)
store.forget("calibration")
store.save(window.surface_path())
window.reload_surface(store)
cam._show_home_box()

# Put one gripper at the intended pre-pick and leave the box in the camera.
_x, _y, _yaw = (0.02, 0.01, math.radians(7))
window.vision.camera.place(centre=(_x, -_y), yaw=-_yaw)
_hold_like(_x, _y, _yaw)
cam._open_home_box()
for _ in range(30):
    app.processEvents()
    cam.tick()
    if cam.home_box_dialog.camera_view.pixmap() is not None:
        break
    time.sleep(0.02)
dialog = cam.home_box_dialog
check("the popup contains a live camera image",
      dialog.isVisible() and dialog.camera_view.pixmap() is not None)
check("and exposes the complete Jog controls from the main panel",
      list(dialog.jog_panel.target_btns) == ["A", "AB", "B"]
      and dialog.jog_panel.motion_combo.count() == 2
      and dialog.jog_panel.frame_combo.count() == 4
      and len(dialog.jog_panel.preset_btns) == 4
      and all(grid.axis_count == 6
              for grid in dialog.jog_panel.grids.values()))
dialog.name_edit.setText("box_home")
dialog.jog_panel._select_target("AB")
program_panel = window.panels["program"]
program_panel.program.steps = [
    Step("FIND", into="part", reference="box_home", timeout=5.0)]
program_panel.refresh()
check("before teaching, an existing FIND correctly reports the missing "
      "reference", "box_home" in program_panel.problems.text(),
      program_panel.problems.text())
dialog._save()
held = window.surface.stance("box_home")
check("one press stores the detected object and both synchronized arm poses "
      "together",
      "box_home" in window.surface.references
      and held is not None and sorted(held) == ["A", "B"],
      dialog.status.text())
check("and FIND can use the new home-box reference immediately",
      window.executor.taught_on_surface() == {"box_home"}
      and not program_panel.problems.text(), program_panel.problems.text())
check("the popup reports both saved TCPs as pre-pick poses",
      "pre-pick arm A TCP" in dialog.status.text()
      and "arm B TCP" in dialog.status.text(), dialog.status.text())
dialog.close()

print("\nand a different crate is a different record, not a refusal")
_taught_path = window.surface_path()
cam.length_spin.setValue(200)
cam.width_spin.setValue(100)
cam.height_spin.setValue(100)
cam._size_settled()            # what finishing the field does
check("changing the size opens that crate's record instead",
      "200x100x100" in os.path.basename(window.surface_path())
      and window.surface_path() != _taught_path,
      os.path.basename(window.surface_path()))
check("which has nothing in it yet, so the panel offers a new box home",
      not window.surface.ready
      and "Ready to teach a camera R/F/H box home" in cam.home_box_lbl.text(),
      cam.home_box_lbl.text()[:48])
check("and a FIND has no name to offer for a crate never taught",
      window.executor.taught_on_surface() == set())
cam.length_spin.setValue(600)
cam.width_spin.setValue(400)
cam.height_spin.setValue(200)
cam._size_settled()
check("putting the first crate back finds its map again, untouched",
      window.surface.ready and "box_home" in window.surface.references
      and window.surface_path() == _taught_path,
      window.surface.description)

for _arm_id, _method in _real_tcp.items():       # give the arms back
    window.cell.arms[_arm_id].tcp_matrix_world = _method

cam.live_btn.setChecked(False)
cam._toggle_live()


def _sample(pixmap, count=400):
    """A scatter of pixels, enough to tell one rendering from another."""
    image = pixmap.toImage()
    out = []
    for i in range(count):
        colour = image.pixelColor((i * 37) % image.width(),
                                  (i * 53) % image.height())
        out.append((colour.red(), colour.green(), colour.blue()))
    return np.array(out)


shots = {}
for mode in ("lens", "blend", "depth"):
    cam._set_mode(mode)
    app.processEvents()
    shots[mode] = _sample(cam.view.pixmap())
check("the lens and the depth arrive as one picture, not two panes",
      cam.view.height() == cam.view.pixmap().height(),
      "%d" % cam.view.height())
check("the blend is not simply the lens",
      np.abs(shots["lens"].astype(int) - shots["blend"].astype(int)).mean() > 5,
      "%.1f apart" % np.abs(shots["lens"].astype(int)
                            - shots["blend"].astype(int)).mean())
check("nor simply the depth",
      np.abs(shots["depth"].astype(int) - shots["blend"].astype(int)).mean() > 5,
      "%.1f apart" % np.abs(shots["depth"].astype(int)
                            - shots["blend"].astype(int)).mean())


def _colourfulness(shot):
    return float(np.mean(np.abs(shot[:, 0] - shot[:, 1])
                         + np.abs(shot[:, 1] - shot[:, 2])))


check("and the depth is rendered in colour, not a grey ramp — a millimetre is "
      "a fraction of a grey level and a whole step of hue",
      _colourfulness(shots["depth"]) > 20,
      "%.1f" % _colourfulness(shots["depth"]))
cam._set_mode("blend")
app.processEvents()

check("the reading is not left running after a single shot",
      not window.vision.running)
window.rail.buttons["jog"].click()
app.processEvents()

print("switching target lets go first")
jog._select_target("A")
app.processEvents()
jog.grids["A"].buttons[1].pressed.emit()
app.processEvents()
check("a key is held on arm A", jog.grids["A"].held is not None)
jog._select_target("B")
app.processEvents()
check("choosing arm B releases it rather than carrying the press over",
      jog.grids["A"].held is None and jog.grids["B"].held is None)
check("and arm B's keys are what is on screen now",
      jog.target_columns["B"].isVisible()
      and not jog.target_columns["A"].isVisible())
jog._select_target("AB")
app.processEvents()

print("either edge will do")
side_before = window.cell.config.ui["sidebar_side"]
window.cell.config.path = os.path.join(tempfile.mkdtemp(), "cell.yaml")
window.rail.swap_btn.click()
app.processEvents()
check("the swap puts the rail and its panel on the other edge",
      body_order() == [window.rail, window.sidebar, window.program_page])
check("and the preference is written down, not just held",
      yaml.safe_load(open(window.cell.config.path))["sidebar_side" if False
                                                    else "ui"]["sidebar_side"]
      != side_before)
window.rail.swap_btn.click()
app.processEvents()
check("swapping back restores the original order",
      body_order() == [window.program_page, window.sidebar, window.rail])

print("frame switching")
jog.frame_combo.setCurrentIndex(1)
app.processEvents()
check("base-frame A and B remain available",
      jog.grids["A"].buttons[6].isEnabled() and
      jog.grids["B"].buttons[6].isEnabled())
check("but synchronized A+B is world-frame only",
      not any(button.isEnabled() for button in jog.grids["AB"].buttons))
jog.frame_combo.setCurrentIndex(3)
app.processEvents()
check("joint mode relabels all six rows",
      jog.grids["A"].axis_labels[0].text() == "J1" and
      jog.grids["A"].axis_labels[5].text() == "J6")

print("commands reach the column that was pressed")
jog.frame_combo.setCurrentIndex(0)
jog.motion_combo.setCurrentIndex(1)
before_a = window.cell.arms["A"].tcp_pose_world().copy()
before_b = window.cell.arms["B"].tcp_pose_world().copy()
jog.grids["A"].buttons[1].click()       # A, displayed X+
app.processEvents()
after_a = window.cell.arms["A"].tcp_pose_world().copy()
after_b = window.cell.arms["B"].tcp_pose_world().copy()
check("an A key moves A", np.linalg.norm(after_a[:3] - before_a[:3]) > 1e-5)
check("and leaves B alone", np.allclose(after_b, before_b, atol=1e-10))

before = {a: window.cell.arms[a].tcp_pose_world().copy() for a in ("A", "B")}
jog.grids["AB"].buttons[5].click()      # synchronized Z+
app.processEvents()
after = {a: window.cell.arms[a].tcp_pose_world().copy() for a in ("A", "B")}
check("an A+B key moves both arms in the displayed direction",
      all(after[a][2] > before[a][2] for a in ("A", "B")))

print("the detail window")
jog.tick()
jog.details_btn.click()
app.processEvents()
detail_a = jog.detail_monitors["A"]
check("Details shows pose, rotation and all six joints",
      all(label.text() != "—" for label in
          detail_a.pose_lbls + detail_a.joint_lbls))
check("Details shows force plus robot and safety states",
      detail_a.force_lbl.text() != "—" and "/" in detail_a.mode_lbl.text())
check("Details shows pair gap, drift and holding state",
      jog.detail_pair.sep_lbl.text() != "—" and
      bool(jog.detail_pair.state_lbl.text()))
jog.detail_dialog.close()

window.close()

real_window = MainWindow(connect_on_start=False)
check("normal startup builds a REAL cell directly",
      not real_window.cell.simulated and real_window.mode == "real")
if not real_window.cell.config.translation_calibrated:
    real_jog = real_window.panels["jog"]
    check("uncalibrated REAL A+B is blocked before it can change the gap",
          not any(button.isEnabled()
                  for button in real_jog.grids["AB"].buttons))
    check("the REAL A+B refusal tells the operator how to calibrate it",
          "check_directions_online.py --apply" in real_jog.note.text())
real_window.close()
print("\nFAILURES: %d" % fail)
sys.exit(1 if fail else 0)
