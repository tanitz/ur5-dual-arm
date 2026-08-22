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
    QLabel, QPlainTextEdit, QStackedWidget, QVBoxLayout, QWidget,
)

REPO_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from ur5dual.robot.backends import BackendError                     # noqa: E402
from ur5dual.cell import Cell                                 # noqa: E402
from ur5dual.config import ARM_IDS, DEFAULT_PATH, CellConfig  # noqa: E402
from ur5dual.coupling import Coordinator, CouplingError       # noqa: E402
from ur5dual.gui import style as S                            # noqa: E402
from ur5dual.gui.panels.jog import JogPanel                   # noqa: E402
from ur5dual.gui.panels.camera import CameraPanel             # noqa: E402
from ur5dual.gui.panels.points import PointsPanel             # noqa: E402
from ur5dual.gui.panels.program import ProgramPanel           # noqa: E402
from ur5dual.gui.widgets.rail import IconRail                 # noqa: E402
from ur5dual.program.executor import Executor                 # noqa: E402
from ur5dual.program.steps import PointLibrary                # noqa: E402
from ur5dual.vision.planar import PlaneFile                    # noqa: E402
from ur5dual.vision.service import VisionService              # noqa: E402

# a correctly configured arm reads only a few newtons at rest
PAYLOAD_SUSPECT_N = 15.0

POINTS_FILE = os.path.join(REPO_ROOT, "config", "points.json")
PROGRAMS_DIR = os.path.join(REPO_ROOT, "config", "programs")

# One width for every panel the sidebar can hold, and the reason is the program
# rather than the panels: a sidebar that resized itself per page moved the step
# table sideways every time an operator went from the jog keys to the points
# list and back. 320 is what the jog keys need — layout_f_side_open.svg, one
# target at a time — and the points list is laid out to that same width.
SIDEBAR_W = 320

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
        self.vision = None
        self.executor = Executor(self.cell, self.points)
        self.executor.on_log = self.log
        # The camera belongs to the app, not to its tab: a program that asks
        # for a detection must not depend on which panel is open, and the
        # reading has to survive the sidebar being closed.
        self.vision = VisionService(self.cell.config.vision, log=self.log)
        self.executor.vision = self.vision
        # The surface the box slides on, owned here beside the camera and for
        # the same reason: a program that corrects a pick must not depend on
        # which tab an operator happened to leave open. A file that is not
        # there yet loads as an empty one; a file that is there and unreadable
        # is worth stopping for, because the alternative is a cell that
        # quietly measures the box a different way than it was set up to.
        self.surface = PlaneFile.load(self.cell.config.vision.get("plane_file"))
        self.executor.surface = self.surface
        self.log("surface: %s" % self.surface.description)
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
            "camera": CameraPanel(self),
            "jog": JogPanel(self),
        }

        self.rail = IconRail()
        self.rail.selected.connect(self._rail_selected)
        self.rail.toggled.connect(self._toggle_sidebar)
        self.rail.swapped.connect(self._swap_sidebar_side)
        self.rail.fullscreen.connect(self._toggle_fullscreen)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(S.sx(4), S.sx(4), S.sx(4), S.sx(4))
        outer.setSpacing(S.sx(4))
        # The cell-wide controls span everything now. They used to sit above
        # the tab column, and a tab column that can be closed is no place for
        # a STOP button — the sidebar may go away and STOP may not.
        outer.addWidget(self.safety_bar, 0)

        self.work = QWidget()
        self.body = QHBoxLayout(self.work)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(S.sx(6))
        self.program_page = self._build_program()
        self.sidebar = self._build_sidebar()
        outer.addWidget(self.work, 1)
        outer.addWidget(self._build_messages(), 0)
        self.setCentralWidget(central)
        self._apply_sidebar()

    def _build_program(self):
        """The program being written, and now the zone that grows.

        The cap is gone. It existed to stop a bigger screen from stretching a
        three-column step table, and the table has two columns of targets in
        it now — every pixel the sidebar gives back is a pixel one of the two
        arms can be read in. The floor stays: the sidebar's own width must not
        squeeze the program into a strip.
        """
        page = self.panels["program"]
        page.setMinimumWidth(S.sx(PROGRAM_COL_MIN_W))
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

    def _build_sidebar(self):
        """One panel at a time, behind the rail, in a stack rather than tabs.

        The tab bar is gone because the rail is the tab bar — and unlike a tab
        bar it stays put when the panel it names is closed.
        """
        stack = QStackedWidget()
        for panel_id in ("points", "camera", "jog"):
            stack.addWidget(self.panels[panel_id])
        stack.setCurrentWidget(self.panels["jog"])
        return stack

    # ---- the rail, the sidebar, and which edge they are on ---------------
    def _ui_state(self):
        ui = self.cell.config.ui
        panel_id = ui.get("sidebar_panel", "jog")
        if panel_id not in self.panels:
            panel_id = "jog"
        side = "left" if ui.get("sidebar_side") == "left" else "right"
        return panel_id, bool(ui.get("sidebar_open", True)), side

    def _apply_sidebar(self):
        """Lay the three pieces out for the current side and open state.

        Called for every change rather than each caller doing its own half of
        the work: the order of the widgets, the sidebar's width, what the rail
        is showing and which panel is on top all have to agree, and one place
        that sets all four cannot leave them disagreeing.
        """
        panel_id, is_open, side = self._ui_state()

        # a held jog key must not survive the panel it belongs to going away
        self.panels["jog"].release()

        for widget in (self.rail, self.sidebar, self.program_page):
            self.body.removeWidget(widget)
        order = ([self.rail, self.sidebar, self.program_page] if side == "left"
                 else [self.program_page, self.sidebar, self.rail])
        for widget in order:
            self.body.addWidget(widget, 1 if widget is self.program_page else 0)

        self.sidebar.setCurrentWidget(self.panels[panel_id])
        self.sidebar.setFixedWidth(S.sx(SIDEBAR_W))
        self.sidebar.setVisible(is_open)
        self.rail.show_state(panel_id, is_open, side)

        for panel in self.panels.values():
            if hasattr(panel, "refresh"):
                panel.refresh()

    def _rail_selected(self, panel_id):
        """An icon press opens that panel, or closes the one already open."""
        ui = self.cell.config.ui
        current, is_open, _side = self._ui_state()
        if panel_id == current and is_open:
            ui["sidebar_open"] = False
        else:
            ui["sidebar_panel"] = panel_id
            ui["sidebar_open"] = True
        self._apply_sidebar()

    def _toggle_sidebar(self):
        ui = self.cell.config.ui
        _panel, is_open, _side = self._ui_state()
        ui["sidebar_open"] = not is_open
        self._apply_sidebar()

    # ---- the window itself ----------------------------------------------
    def wants_fullscreen(self):
        return bool(self.cell.config.ui.get("fullscreen", False))

    def show_window(self, fullscreen=None):
        """Put the panel on screen as large as it is allowed to be.

        Maximised is the default rather than fullscreen: the desktop's dock and
        top bar stay where the operator can reach them, which is how this panel
        is used — a terminal is a swipe away and the panel has to be findable
        again afterwards. Fullscreen is the rail's ⬚ button away for the times
        the whole screen is wanted.

        Either way it is the window manager that decides the size. The old
        "resize to the design size and hope" left a 1280x800 window floating in
        the middle of a 1280x800 screen with its bottom edge under the dock,
        and a maximise button that had nothing left to give.
        """
        if fullscreen is None:
            fullscreen = self.wants_fullscreen()
        if fullscreen:
            self.showFullScreen()
        else:
            self.showMaximized()
        self.rail.show_fullscreen(fullscreen)

    def _toggle_fullscreen(self):
        """Fill the screen or give the frame back, and remember which."""
        fullscreen = not self.isFullScreen()
        self.cell.config.ui["fullscreen"] = fullscreen
        self.show_window(fullscreen)
        try:
            # only the ui block: a window button must not commit geometry
            self.cell.config.save_ui()
        except OSError as e:
            self.log("could not save the layout: %s" % e)

    def keyPressEvent(self, event):
        # F11 is what every other program on this desktop uses, and a keyboard
        # is what a developer has when the touchscreen is showing the wrong
        # thing. The rail button is the operator's way to the same place.
        if event.key() == Qt.Key_F11:
            self._toggle_fullscreen()
            return
        super().keyPressEvent(event)

    def _swap_sidebar_side(self):
        """Move the rail and its panel to the other edge, and remember it.

        Which hand reaches the jog keys is the operator's, not the layout's,
        and an operator who moved them wants them moved tomorrow too — so this
        is written to cell.yaml rather than held for the session.
        """
        ui = self.cell.config.ui
        _panel, _open, side = self._ui_state()
        ui["sidebar_side"] = "left" if side == "right" else "right"
        self._apply_sidebar()
        try:
            # only the ui block: the swap must not commit unsaved geometry
            self.cell.config.save_ui()
        except OSError as e:
            self.log("could not save the layout: %s" % e)
        self.log("jog and the panels moved to the %s" % ui["sidebar_side"])

    # ---- shared services the panels call --------------------------------
    def log(self, text):
        """Safe from any thread — the append happens on the GUI thread.

        Every line also goes to the terminal. The drawer holds 400 lines and
        is one line tall until it is opened, so a panel started from a shell —
        which is how this one is started — would otherwise lose the message
        that says why a camera did not open, or which arm refused a command,
        behind whatever came after it.
        """
        from time import strftime
        line = "%s  %s" % (strftime("%H:%M:%S"), text)
        print(line, flush=True)
        self.log_message.emit(line)

    def _append_log(self, line):
        self.msg_box.appendPlainText(line)
        # The drawer is one line tall until it is opened, and a message with
        # newlines in it — the camera's install instructions, say — would push
        # the work area up by however many it carries. The rest is in the
        # history behind Expand, and all of it is on the terminal.
        self.last_message.setText(line.splitlines()[0])

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
        super().changeEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        self.executor.stop()
        self.vision.stop()
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
    parser.add_argument("--fullscreen", action="store_true",
                        help="fill the screen for this run, whatever the "
                             "saved ui.fullscreen says")
    parser.add_argument("--windowed", action="store_true",
                        help="run inside the desktop's window frame for this "
                             "run, maximised to the work area")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    palette = app.palette()
    palette.setColor(QPalette.Window, QColor("#f4f4f4"))
    app.setPalette(palette)

    # Fullscreen or maximised has to be decided here rather than after the
    # window is built, because it decides how much room the layout has and
    # every widget below asks sx() for its size exactly once. The saved
    # preference decides — maximised unless the operator pressed ⬚ — and either
    # flag overrides it for one run without writing anything back.
    fullscreen = args.fullscreen or (
        not args.windowed
        and CellConfig.load(args.config).ui.get("fullscreen", False))

    # Fullscreen gets the screen. A framed window gets availableGeometry, which
    # is the screen less the desktop's own dock and top bar, less a title bar
    # on top of that: a maximised window keeps its frame inside the work area,
    # so the layout has that much less height than availableGeometry reports
    # and nothing in Qt will tell us how much before a window exists.
    # Over-reserving costs a few unused pixels along the bottom;
    # under-reserving puts the window back off the edge of the screen, which is
    # the whole thing being fixed here.
    screen = app.primaryScreen()
    area = screen.geometry() if fullscreen else screen.availableGeometry()
    height = area.height() - (0 if fullscreen else TITLE_BAR_PX)
    if args.scale:
        S.set_scale(args.scale)
    else:
        S.fit_to(area.width(), height)

    window = MainWindow(args.config)
    # Maximised rather than resized to the design size: what the layout gets
    # over its minimum goes to the jog grid and the message log rather than to
    # wallpaper, and a window already the size of the screen leaves the
    # maximise button nothing to do but nothing.
    window.show_window(fullscreen)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
