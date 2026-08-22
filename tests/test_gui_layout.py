"""The C1 jog layout and its routing, without a display or robots."""

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

from ur5dual.cell import Cell             # noqa: E402
from ur5dual.config import CellConfig     # noqa: E402
from ur5dual.gui import style as S        # noqa: E402
from ur5dual.gui.app import MainWindow    # noqa: E402


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
check("the result shows all six pose components",
      "6D POSE" in cam.found_lbl.text()
      and "XYZ mm" in cam.found_lbl.text()
      and "RPY deg" in cam.found_lbl.text(), cam.found_lbl.text())
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
check("the wall depth and the search window are still stated",
      "mm deep" in cam.setup_lbl.text()
      and "whole picture" in cam.setup_lbl.text(), cam.setup_lbl.text())
check("and depth is reported as a check on the size that was typed",
      "size check" in cam.found_lbl.text(),
      cam.found_lbl.text().split("\n")[-1])


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
