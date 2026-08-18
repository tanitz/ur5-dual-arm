"""The C1 jog layout and its routing, without a display or robots."""

import os
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
window = MainWindow(cell=Cell(CellConfig.load(), simulated=True),
                    connect_on_start=False)
window.resize(1280, 800)
window.show()
app.processEvents()
jog = window.panels["jog"]

print("the full-width page")
check("A, synchronized A+B and B are visible together",
      list(jog.grids) == ["A", "AB", "B"] and
      all(jog.grids[t].isVisible() for t in jog.grids))
check("the old drive selector is gone", not hasattr(jog, "target_combo"))
check("STOP is global rather than owned by the Jog page",
      window.stop_btn.isVisible() and window.stop_btn.parent() is not jog)
check("the REAL mode selector is gone", not hasattr(window, "mode_combo"))
check("the REAL control bar occupies only the right tab column",
      abs(window.safety_bar.width() - window.tabs.width()) <= 2)
bar_controls = ([window.real_indicator] + list(window.conn_btns.values()) +
                list(window.dashboard_btns.values()) + [window.stop_btn])
check("REAL, arm, dashboard and STOP controls share one row",
      len({button.y() for button in bar_controls}) == 1)
check("the log starts collapsed", not window.msg_box.isVisible())
check("all six Cartesian axes are always shown",
      all([i for i, row in enumerate(grid.rows) if row[0].isVisible()]
          == [0, 1, 2, 3, 4, 5] for grid in jog.grids.values()))

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
