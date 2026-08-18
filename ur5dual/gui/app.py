"""The dual-UR5 control panel.

The program stays on the left and the teach/run tools stay in tabs on the
right. REAL cell controls and STOP sit immediately above those tabs, leaving
the program column untouched while keeping robot ownership visible.

Messages are a one-line drawer along the bottom.  The Jog tab therefore gets
the full right-hand width for three permanent targets — arm A, synchronized
A+B, and arm B — instead of spending a third of that width on status.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QMainWindow,
    QLabel, QPlainTextEdit, QTabWidget, QVBoxLayout, QWidget,
)

REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from ur5dual.robot.backends import BackendError                     # noqa: E402
from ur5dual.cell import Cell                                 # noqa: E402
from ur5dual.config import ARM_IDS, DEFAULT_PATH, CellConfig  # noqa: E402
from ur5dual.coupling import Coordinator, CouplingError       # noqa: E402
from ur5dual.gui import style as S                            # noqa: E402
from ur5dual.gui.panels.cell_setup import CellSetupPanel      # noqa: E402
from ur5dual.gui.panels.jog import JogPanel                   # noqa: E402
from ur5dual.gui.panels.objects import ObjectPanel            # noqa: E402
from ur5dual.gui.panels.points import PointsPanel             # noqa: E402
from ur5dual.gui.panels.program import ProgramPanel           # noqa: E402
from ur5dual.program.executor import Executor                 # noqa: E402
from ur5dual.program.steps import PointLibrary                # noqa: E402

# a correctly configured arm reads only a few newtons at rest
PAYLOAD_SUSPECT_N = 15.0

POINTS_FILE = os.path.join(REPO_ROOT, "config", "points.json")
PROGRAMS_DIR = os.path.join(REPO_ROOT, "config", "programs")

# half the 1280x800 design width, less the margins between the two columns
PROGRAM_COL_W = 620
# and what it gives back when the screen is smaller than that. The step table
# reads fine narrow; the tabs beside it hold grids of touch buttons that do not.
PROGRAM_COL_MIN_W = 470

# What a window manager keeps for a title bar. Only ever used to decide how
# much of the screen the layout may plan on; run --fullscreen and the panel
# gets all of it and its design size back.
TITLE_BAR_PX = 40


class MainWindow(QMainWindow):
    # Qt widgets may only be touched from the thread that created them, and
    # two threads here are not it: the program executor and the coordinated
    # servo feed both log. Routing every message through a signal lets Qt
    # queue it onto the GUI thread. Calling appendPlainText directly from a
    # worker corrupts the document's internal cursor and takes the process
    # down with it, which is what "Cannot queue arguments of type
    # 'QTextCursor'" is warning about just before the segfault.
    log_message = pyqtSignal(str)

    def __init__(self, config_path=DEFAULT_PATH, cell=None,
                 connect_on_start=True):
        super().__init__()
        self.setWindowTitle("Dual UR5 control")

        # This installation is a REAL-only control panel.  Supplying a cell is
        # kept as a test seam; normal startup builds the two live arms here.
        self.cell = cell or Cell(CellConfig.load(config_path), simulated=False)
        self.cell.listeners.append(self.log)
        self.points = PointLibrary()
        self.executor = Executor(self.cell, self.points)
        self.executor.on_log = self.log
        self.coordinator_start_error = None
        self.mode = "real"
        self.grip_output = 0
        self.object_count = 0
        self.programs_dir = PROGRAMS_DIR
        os.makedirs(PROGRAMS_DIR, exist_ok=True)

        self.log_message.connect(self._append_log)
        self._build()
        self._load_points()
        for panel in self.panels.values():
            if hasattr(panel, "refresh"):
                panel.refresh()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(60)

        self._refresh_connect_buttons()
        if connect_on_start and not self.cell.simulated:
            # Claim the RViz channel before either 30003 feed is opened; arm A
            # cannot serve the panel and viewer at the same time.
            for _ in range(3):
                self.cell.publish_sim_view(lease=20.0)
                time.sleep(0.03)
            self.cell.connect()
            self._refresh_connect_buttons()

    # ---- layout ----------------------------------------------------------
    def _build(self):
        self.safety_bar = self._build_safety_bar()
        self.panels = {
            "program": ProgramPanel(self),
            "points": PointsPanel(self),
            "vars": CellSetupPanel(self),
            "objects": ObjectPanel(self),
            "jog": JogPanel(self),
        }
        self.panels["vars"].geometry_changed.connect(self._geometry_changed)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(S.sx(4), S.sx(4), S.sx(4), S.sx(4))
        outer.setSpacing(S.sx(4))
        work = QWidget()
        body = QHBoxLayout(work)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(S.sx(6))
        # The program keeps its useful editing width; everything it gives back
        # belongs to the three-column Jog surface on the right.
        body.addWidget(self._build_program(), 1)

        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(S.sx(4))
        right_v.addWidget(self.safety_bar, 0)
        right_v.addWidget(self._build_tabs(), 1)
        body.addWidget(right, 2)
        outer.addWidget(work, 1)
        outer.addWidget(self._build_messages(), 0)
        self.setCentralWidget(central)

    def _build_program(self):
        """The left half, and it never changes: the program being written."""
        page = self.panels["program"]
        # Half the 1280x800 design width, and it holds that half: capped so a
        # bigger screen gives its extra pixels to the tabs rather than
        # stretching the step table, floored so the tabs' own minimum width
        # cannot squeeze the program into a strip on the panel itself.
        page.setMinimumWidth(S.sx(PROGRAM_COL_MIN_W))
        page.setMaximumWidth(S.sx(PROGRAM_COL_W))
        return page

    def _build_safety_bar(self):
        """Controls that must remain visible regardless of the selected tab."""
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(S.sx(4))

        self.real_indicator = QLabel("REAL  ●")
        self.real_indicator.setMinimumHeight(S.sx(52))
        self.real_indicator.setAlignment(Qt.AlignCenter)
        self.real_indicator.setStyleSheet(
            f"background:{S.RED};color:white;font-weight:bold;"
            f"font-size:{S.fpx(14)}px;border-radius:{S.sx(6)}px;")
        row.addWidget(self.real_indicator, 1)

        self.conn_btns = {}
        for arm_id in ARM_IDS:
            button = S.touch_button("Connect %s" % arm_id, S.GREEN, height=52,
                                    font_px=13)
            button.clicked.connect(lambda _c=False, a=arm_id: self._toggle(a))
            row.addWidget(button, 1)
            self.conn_btns[arm_id] = button

        self.dashboard_btns = {}
        for label, command in (("Power", "power on"),
                               ("Brakes", "brake release"),
                               ("Unlock", "unlock protective stop")):
            button = S.touch_button(label, height=52, font_px=12)
            button.clicked.connect(lambda _c=False, c=command: self._dashboard(c))
            row.addWidget(button, 1)
            self.dashboard_btns[label] = button

        self.stop_btn = S.touch_button("■  STOP", S.RED, height=52, font_px=18)
        self.stop_btn.clicked.connect(self._stop_everything)
        row.addWidget(self.stop_btn, 2)
        return page

    def _build_messages(self):
        """A single-line log that expands only when its history is wanted."""
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(3))

        row = QHBoxLayout()
        row.setSpacing(S.sx(6))
        title = S.strip("Messages")
        title.setFixedWidth(S.sx(100))
        row.addWidget(title, 0)
        self.last_message = QLabel("Ready")
        self.last_message.setStyleSheet(
            f"font-size:{S.fpx(12)}px;color:#333333;padding-left:{S.sx(4)}px;")
        row.addWidget(self.last_message, 1)
        self.messages_btn = S.touch_button("Expand ▲", height=34, font_px=12,
                                           checkable=True)
        self.messages_btn.toggled.connect(self._toggle_messages)
        row.addWidget(self.messages_btn, 0)
        v.addLayout(row)

        self.msg_box = QPlainTextEdit()
        self.msg_box.setReadOnly(True)
        self.msg_box.setMaximumBlockCount(400)
        self.msg_box.setStyleSheet(f"font-size:{S.fpx(12)}px;")
        self.msg_box.setFixedHeight(S.sx(130))
        self.msg_box.hide()
        v.addWidget(self.msg_box, 1)
        return page

    def _toggle_messages(self, expanded):
        self.msg_box.setVisible(expanded)
        self.messages_btn.setText("Collapse ▼" if expanded else "Expand ▲")

    def _build_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(S.tabs())
        self.tabs.addTab(self.panels["points"], "Points")
        self.tabs.addTab(self.panels["vars"], "Vars")
        self.tabs.addTab(self.panels["objects"], "Object")
        jog = self.panels["jog"]
        self.tabs.addTab(jog, "Jog")
        # By widget, not by index: adding a tab must not change the opening page.
        self.tabs.setCurrentWidget(jog)
        self.tabs.currentChanged.connect(self._tab_changed)
        return self.tabs

    def _geometry_changed(self):
        """The mounting numbers moved, so everything drawn from them is stale."""
        self.panels["jog"].mode_changed()
        for panel in self.panels.values():
            if hasattr(panel, "refresh"):
                panel.refresh()

    # ---- shared services the panels call --------------------------------
    def log(self, text):
        """Safe from any thread — the append happens on the GUI thread."""
        from time import strftime
        self.log_message.emit("%s  %s" % (strftime("%H:%M:%S"), text))

    def _append_log(self, line):
        self.msg_box.appendPlainText(line)
        self.last_message.setText(line)

    def attach(self, origin="midpoint"):
        """Freeze the current grip and bring up the coordinated servo loop.

        Lives here rather than on the Object tab because taking hold is also
        the first thing anyone wants from the Jog tab — being told to go and
        press a button on another tab is how the step gets missed.

        Returns True if the arms are now holding something.
        """
        from ur5dual.coupling import HeldObject

        # A SIM midpoint grasp always begins from the known, well-conditioned
        # pickup posture. Capturing arbitrary HOME angles is especially bad
        # when J5 is zero: that is a UR wrist singularity, so a tiny Cartesian
        # box jog can demand a very large and visually erratic joint motion.
        # Centralising this here covers ATTACH on the Object tab and "Take
        # hold now" on the Jog tab, not only the dedicated SIM test button.
        if self.cell.simulated and origin == "midpoint":
            self.cell.sim_ready_pose()

        arm_ids = tuple(self.cell.connected_ids)
        if not arm_ids:
            self.log("attach: no arm is connected")
            return False
        if origin == "midpoint" and len(arm_ids) < 2:
            self.log("attach: a midpoint origin needs both arms connected "
                     "— only %s is up" % arm_ids[0])
            return False
        if origin in ("A", "B") and origin not in arm_ids:
            self.log("attach: origin is arm %s, which is not connected" % origin)
            return False

        self.object_count += 1
        try:
            obj = HeldObject("object%d" % self.object_count)
            obj.capture(self.cell, arm_ids, origin)
        except CouplingError as e:
            self.log("attach failed: %s" % e)
            return False

        for arm_id, joint, margin in self.cell.wound_up_joints(arm_ids):
            self.log("arm %s J%d is %.0f deg from its stop — unwind it a full "
                     "turn before coordinated work or it will protective-stop"
                     % (arm_id, joint + 1, __import__("math").degrees(margin)))

        self.executor.object = obj
        self.log("attached '%s' with %s (span %s)"
                 % (obj.name, "+".join(obj.arm_ids),
                    "—" if obj.span() is None else "%.1f mm" % (obj.span() * 1000)))
        for arm_id, force in sorted(
                (a, float(np.linalg.norm(self.cell.force_vector(a))))
                for a in arm_ids):
            if force > PAYLOAD_SUSPECT_N:
                self.log("arm %s reads %.0f N while holding nothing — its "
                         "payload/TCP is not set to match the tool fitted, so "
                         "the force guard is running loose" % (arm_id, force))

        if not self.start_coordinator():
            self.log("attached, but the servo loop did not start — object "
                     "moves are unavailable until it does")
        for panel in self.panels.values():
            if hasattr(panel, "refresh"):
                panel.refresh()
        return True

    def detach(self):
        self.stop_coordinator()
        self.executor.object.release()
        self.coordinator_start_error = None
        self.log("detached")
        for panel in self.panels.values():
            if hasattr(panel, "refresh"):
                panel.refresh()

    def start_coordinator(self):
        """Bring up the servo backends for whatever is currently held."""
        obj = self.executor.object
        if not obj.held:
            return False
        self.coordinator_start_error = None
        try:
            coordinator = Coordinator(self.cell,
                                      drive_robots=(self.mode == "real"))
            coordinator.start(obj)
        except (BackendError, CouplingError, OSError) as e:
            self.coordinator_start_error = str(e)
            # a refusal is a message for the operator, not a crash on the way
            # out of a button press
            self.log("servo loop not started: %s" % e)
            if isinstance(e, (BackendError, OSError)):
                self.log("if that looks like a connection problem, try the "
                         "other motion.backend in cell.yaml "
                         "(rtde <-> urscript) and reconnect")
            return False
        self.executor.coordinator = coordinator
        self.coordinator_start_error = None
        return True

    def stop_coordinator(self):
        if self.executor.coordinator is not None:
            self.executor.coordinator.shutdown()
            self.executor.coordinator = None

    def save_points(self):
        os.makedirs(os.path.dirname(POINTS_FILE), exist_ok=True)
        with open(POINTS_FILE, "w") as f:
            json.dump(self.points.to_dict(), f, indent=2)
        return POINTS_FILE

    def _load_points(self):
        try:
            with open(POINTS_FILE) as f:
                self.points.load_dict(json.load(f))
        except (OSError, ValueError):
            pass

    # ---- connection ------------------------------------------------------
    def _toggle(self, arm_id):
        arm = self.cell.arms[arm_id]
        if arm.connected:
            arm.disconnect()
            self.log("arm %s disconnected" % arm_id)
        else:
            self.cell.connect([arm_id])
        self._refresh_connect_buttons()

    def _refresh_connect_buttons(self):
        sim = self.cell.simulated
        for arm_id, button in self.conn_btns.items():
            connected = self.cell.arms[arm_id].connected
            if sim:
                # A simulated arm is always "connected" and there is nothing to
                # dial. Offering Connect here would be offering an action whose
                # only possible outcome is a timeout.
                button.setText("%s simulated" % arm_id)
                button.setStyleSheet(S.pill(13))
                button.setEnabled(False)
                button.setToolTip("simulated test cell")
                continue
            button.setEnabled(True)
            button.setToolTip("")
            button.setText(("Drop %s" if connected else "Connect %s") % arm_id)
            button.setStyleSheet(S.solid(S.RED if connected else S.GREEN, 13))

    def _dashboard(self, command):
        if self.cell.simulated:
            self.log("'%s' needs a controller — this cell is simulated. "
                     "Switch to REAL mode first" % command)
            return
        for arm_id in self.cell.connected_ids:
            try:
                with self.cell.arms[arm_id].dashboard() as db:
                    self.log("arm %s  %s -> %s" % (arm_id, command, db.send(command)))
            except OSError as e:
                self.log("arm %s  %s failed: %s" % (arm_id, command, e))

    def _stop_everything(self):
        for panel in self.panels.values():
            if hasattr(panel, "release"):
                panel.release()
        self.executor.stop()
        self.stop_coordinator()
        self.cell.halt()
        self.log("STOP")

    # ---- reactions -------------------------------------------------------
    def _tab_changed(self, _index):
        # Leaving either jog surface with a button still held must not latch
        # the command on after the button is no longer visible.
        self.panels["jog"].release()
        self.panels["objects"].release()
        for panel in self.panels.values():
            if hasattr(panel, "refresh"):
                panel.refresh()

    def _tick(self):
        # Keep the drawing fed whatever is or is not happening. The coordinated
        # loop also publishes, at servo rate, while it is carrying something —
        # but it only exists between ATTACH and DETACH, and RViz has to follow
        # a single arm being jogged, and an idle cell being looked at, just as
        # faithfully. Both write to the same channel and the newest frame wins,
        # so the overlap costs nothing.
        self.cell.publish_sim_view()

        # A dead servo loop must not leave the app claiming to hold something.
        # Left as-is it answers every press with the same stale complaint and
        # offers no way back; letting go makes the state honest and puts the
        # Take hold button in front of the operator again.
        coordinator = self.executor.coordinator
        if coordinator is not None and not coordinator.alive \
                and not self.executor.running:
            self.log("servo loop ended — letting go so you can take hold again")
            self.detach()

        for arm_id in ARM_IDS:
            arm = self.cell.arms[arm_id]
            while arm.stream_events:
                self.log(arm.stream_events.pop(0))
        for panel in self.panels.values():
            if hasattr(panel, "tick"):
                panel.tick()

    # ---- lifecycle -------------------------------------------------------
    def changeEvent(self, event):
        # a jog button held while the window loses focus never emits released()
        if event.type() == event.ActivationChange and not self.isActiveWindow():
            self.panels["jog"].release()
            self.panels["objects"].release()
        super().changeEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        self.executor.stop()
        self.stop_coordinator()
        self.cell.disconnect()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Dual UR5 control panel")
    parser.add_argument("--config", default=DEFAULT_PATH)
    parser.add_argument("--scale", type=float, default=None,
                        help="UI scale; 1.0 is the 1280x800 panel design size. "
                             "Left out, the panel measures the screen and fits "
                             "itself to it")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    palette = app.palette()
    palette.setColor(QPalette.Window, QColor("#f4f4f4"))
    app.setPalette(palette)

    # How much room the layout will really have, measured before anything is
    # built — every widget below asks sx() for its size exactly once, so the
    # scale has to be settled first.
    #
    # Fullscreen gets the screen. Anything else gets availableGeometry, which
    # is the screen less the desktop's own dock and top bar, less a title bar
    # on top of that: a maximised window keeps its frame inside the work area,
    # so the layout has that much less height than availableGeometry reports
    # and nothing in Qt will tell us how much before a window exists.
    # Over-reserving costs a few unused pixels along the bottom;
    # under-reserving puts the window back off the edge of the screen, which is
    # the whole thing being fixed here.
    screen = app.primaryScreen()
    area = screen.geometry() if args.fullscreen else screen.availableGeometry()
    height = area.height() - (0 if args.fullscreen else TITLE_BAR_PX)
    if args.scale:
        S.set_scale(args.scale)
    else:
        S.fit_to(area.width(), height)

    window = MainWindow(args.config)
    if args.fullscreen:
        window.showFullScreen()
    elif S.UI_SCALE < 1.0:
        # Having had to shrink to fit, take every pixel the desktop will give:
        # maximised, the window manager fits the frame to the work area, and
        # what the layout gets over its minimum goes to the jog grid and the
        # message log rather than to wallpaper.
        window.showMaximized()
    else:
        window.resize(S.sx(S.DESIGN_W), S.sx(S.DESIGN_H))
        window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
