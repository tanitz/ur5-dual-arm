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
import socket
import sys
import threading
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
from ur5dual.config import ARM_IDS, DEFAULT_PATH, CellConfig, in_repo  # noqa: E402
from ur5dual.coupling import Coordinator, CouplingError       # noqa: E402
from ur5dual.gui import style as S                            # noqa: E402
from ur5dual.comms import LinkLibrary, LinkService            # noqa: E402
from ur5dual.gui.panels.jog import JogPanel                   # noqa: E402
from ur5dual.gui.panels.camera import CameraPanel             # noqa: E402
from ur5dual.gui.panels.network import NetworkPanel           # noqa: E402
from ur5dual.gui.panels.points import PointsPanel             # noqa: E402
from ur5dual.gui.panels.program import ProgramPanel           # noqa: E402
from ur5dual.gui.widgets.rail import RAIL_ITEMS, IconRail     # noqa: E402
from ur5dual.program.executor import Executor                 # noqa: E402
from ur5dual.program.steps import PointLibrary, Program       # noqa: E402
from ur5dual.axes import shown_pose                            # noqa: E402
from ur5dual.vision.planar import PlaneFile, sized_path        # noqa: E402
from ur5dual.vision.service import VisionService              # noqa: E402

# a correctly configured arm reads only a few newtons at rest
PAYLOAD_SUSPECT_N = 15.0

POINTS_FILE = os.path.join(REPO_ROOT, "config", "points.json")
PROGRAMS_DIR = os.path.join(REPO_ROOT, "config", "programs")

# One width for every panel the sidebar can hold. Moving 15% of the 1280 px
# design width from the program editor to this column gives the camera and
# setup panels room without making the step table jump when tabs are changed.
SIDEBAR_W = 320 + round(S.DESIGN_W * 0.15)

# half the 1280x800 design width, less the margins between the two columns
PROGRAM_COL_W = 620
# and what it gives back when the screen is smaller than that. The step table
# reads fine narrow; the tabs beside it hold grids of touch buttons that do not.
PROGRAM_COL_MIN_W = 470

# What a window manager keeps for a title bar. Only ever used to decide how
# much of the screen the layout may plan on; run --fullscreen and the panel
# gets all of it and its design size back.
TITLE_BAR_PX = 40

# How long a browser may go quiet before the desktop lets go of its jog key.
# This is not what stops the arm: a held jog refreshes speedl/speedj every
# M.REFRESH and the controller kills the motion M.WATCHDOG after the last one,
# so the arm has already coasted to a stop before this expires. The window only
# has to outlast wifi jitter, which is why it is not tighter.
WEB_JOG_GRACE = 0.75


def lan_address(bind_host):
    """The address to type into a browser for a server bound to `bind_host`.

    A wildcard bind has no address of its own, and the hostname is no help to
    a tablet that cannot resolve it, so the answer is the address on the
    interface facing the default route — the one the cell LAN arrives on. The
    UDP socket only names a local endpoint; nothing is sent.
    """
    if bind_host not in ("0.0.0.0", "::", ""):
        return bind_host
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))     # TEST-NET-1: reserved, never routed
        return probe.getsockname()[0]
    except OSError:
        return socket.gethostname()
    finally:
        probe.close()


def connection_status(simulated, state):
    """What the top strip says, what colour it is, and what it says on hover.

    The band used to be red always, because red once meant REAL — these are
    real robots and not a simulation. On a panel that is REAL-only that is a
    warning nobody can act on and nobody keeps reading, and it cost the one
    thing a status strip is for: with the colour already spent, an arm that
    had dropped off looked exactly like a cell that was fine.

    So the colour is the connection now, and REAL is a word instead:

        green   every arm answering — the cell is as it should be
        amber   some but not all, which is the state worth catching early
        red     nothing answering, on a cell that expects to
        slate   simulated, which is neither

    The dots stay. Colour alone is not a status an operator who cannot pick
    green out of amber can read, and this is the line they check before
    pressing Run.
    """
    marks = "   ".join("%s %s" % (arm, "●" if state.get(arm) else "○")
                       for arm in ARM_IDS)
    if simulated:
        return "SIM     %s" % marks, S.SLATE, "simulated test cell"

    answering = [a for a in ARM_IDS if state.get(a)]
    colour = (S.GREEN if len(answering) == len(ARM_IDS)
              else S.AMBER if answering else S.RED)
    tooltip = "\n".join("arm %s: %s" % (a, "connected" if state.get(a)
                                         else "not connected")
                        for a in ARM_IDS)
    return "REAL     %s" % marks, colour, tooltip


class MainWindow(QMainWindow):
    # Qt widgets may only be touched from the thread that created them, and
    # two threads here are not it: the program executor and the coordinated
    # servo feed both log. Routing every message through a signal lets Qt
    # queue it onto the GUI thread. Calling appendPlainText directly from a
    # worker corrupts the document's internal cursor and takes the process
    # down with it, which is what "Cannot queue arguments of type
    # 'QTextCursor'" is warning about just before the segfault.
    log_message = pyqtSignal(str)
    web_command = pyqtSignal(object)
    # A Test started from a browser opens the socket on a worker thread, for
    # the reason the desktop's own Test does not: the answer must not hold the
    # HTTP request, and a timing-out address would otherwise hold the whole Qt
    # loop. The line it produces comes back through here.
    comm_probe = pyqtSignal(str)

    def __init__(self, config_path=DEFAULT_PATH, cell=None,
                 connect_on_start=True, web_host=None, web_port=8765,
                 web_token=None):
        super().__init__()
        self.setWindowTitle("Dual UR5 control")

        # This installation is a REAL control panel: startup builds the two
        # live arms here. A cell handed in from outside is the one seam past
        # that — the tests use it, and so does `--sim`, which hands in the
        # simulated stand-ins. They answer every question below the way a
        # controller does and send nothing to one.
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
        # The machines the cell stands next to, and the sockets to them —
        # owned here for the reason the camera is. The library is one object
        # the two tabs edit in place; `links_changed` is how an edit reaches
        # the config and the open connections.
        self._links = self.cell.config.link_library()
        self.links = LinkService(self._links, log=self.log)
        self.executor.links = self.links
        self.surface = None
        self.reload_surface()
        self.coordinator_start_error = None
        # What `start_coordinator` reads before it opens servo backends. A
        # simulated cell closes the loop through its own arms instead of
        # through two controllers, which is the whole of the substitution.
        self.mode = "sim" if self.cell.simulated else "real"
        self.grip_output = 0
        self.object_count = 0
        self.programs_dir = PROGRAMS_DIR
        os.makedirs(PROGRAMS_DIR, exist_ok=True)

        self.web_hub = None
        self.web_server = None
        self._web_jog_owner = None
        self._web_jog_target = None
        self._web_jog_axis = None
        self._web_jog_session = None
        # client id -> when that browser was last heard from, stamped on the
        # HTTP thread rather than on Qt's; see _start_web.
        self._web_jog_seen = {}
        self._web_jog_lock = threading.Lock()

        self.log_message.connect(self._append_log)
        self.web_command.connect(self._handle_web_command)
        self.comm_probe.connect(self._comm_probed)
        self._build()
        self._load_points()
        for panel in self.panels.values():
            if hasattr(panel, "refresh"):
                panel.refresh()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(60)

        self._refresh_connection()
        if web_host:
            self._start_web(web_host, web_port, web_token)
        if connect_on_start and not self.cell.simulated:
            # Claim the RViz channel before either 30003 feed is opened; arm A
            # cannot serve the panel and viewer at the same time.
            for _ in range(3):
                self.cell.publish_sim_view(lease=20.0)
                time.sleep(0.03)
            self.cell.connect()
            self._refresh_connection()

    # ---- layout ----------------------------------------------------------
    def surface_path(self):
        """The plane file for the box the cell is set to, not just *a* file.

        Derived rather than configured, so that changing the crate on the
        Camera tab changes which map is in use with nothing else to remember.
        """
        return sized_path(in_repo(self.cell.config.vision.get("plane_file")),
                          self.cell.config.vision.get("box_size"))

    def reload_surface(self, store=None):
        """Take up a map or a reference that was just fitted or taught.

        The executor is handed the same object rather than a path, so a
        program run straight after the fit measures against what is on screen
        — and does not need the panel restarted to notice. Called with nothing
        it reads the file for whatever box the cell is now set to, which is
        what a change of crate does.
        """
        self.surface = store or PlaneFile.load(self.surface_path())
        self.executor.surface = self.surface
        self.log("surface: %s" % self.surface.description)
        return self.surface

    # ---- the machines beside the cell ------------------------------------
    def link_library(self):
        """The one library the Communication tab edits, and the executor reads.

        One object rather than a copy: a panel holding its own would let the
        cell have two answers to what it is connected to, and a program would
        run against whichever was saved last.
        """
        return self._links

    def links_changed(self):
        """An edit made on the Communication tab, taken up now, not at Save.

        Adding a machine and pressing Test has to reach the wire, and so does
        a program run before anybody remembers to save — the file is where
        this survives a restart, not where it takes effect.
        """
        self.cell.config.set_link_library(self._links)
        self.links.reload(self._links)

    def save_links(self):
        self.cell.config.set_link_library(self._links)
        path = self.cell.config.save_comms()
        self.links.reload(self._links)
        self.log("machines saved to %s" % path)
        return path

    def _build(self):
        self.safety_bar = self._build_safety_bar()
        self.panels = {
            "program": ProgramPanel(self),
            "points": PointsPanel(self),
            "camera": CameraPanel(self),
            "net": NetworkPanel(self),
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
        """One line saying what the cell is talking to. No buttons.

        It carried seven widgets: Connect/Drop per arm, Power, Brakes, Unlock,
        and STOP. All seven are gone at the operator's asking, and each for its
        own reason. The five in the middle were never pressed — this panel
        connects on start, and these robots are powered and released from
        their own pendants. STOP went last.

        That leaves this cell three ways to stop, and it is worth being able
        to name them: `■ Stop` on the Program tab ends the run, the pendant's
        E-stop cuts the power, and `stop_all` from the browser panel still
        calls `_stop_everything` — which is why that method stays whole below
        rather than going with the button.

        Nothing was deleted here either. `_toggle` and `_dashboard` still
        work, and `conn_btns`/`dashboard_btns` are still the dicts they filled
        — empty, so every loop over them is a no-op. Putting any of the seven
        back is one `row.addWidget` in this method.

        With no button left to stand beside, the band is a strip rather than a
        52 px slab: the height it was is height the program now has. What it
        is painted is `connection_status`, not this method.
        """
        page = QWidget()
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(S.sx(4))

        # Named `real_indicator` still, because it is still that: what it adds
        # is which arms are answering, which is the only thing the buttons
        # beside it were being read for.
        self.real_indicator = QLabel("REAL")
        self.real_indicator.setMinimumHeight(S.sx(30))
        self.real_indicator.setAlignment(Qt.AlignCenter)
        # Colour and text both come from `_refresh_connection`, which runs
        # before this is ever shown: a band painted here as well would be a
        # second answer to what the cell is doing, and the wrong one for the
        # first frame after every connect.
        row.addWidget(self.real_indicator, 1)

        self.conn_btns = {}
        self.dashboard_btns = {}
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

        The stack is built from the rail's own list rather than from a second
        one written here. The two lists drifting apart is not a hypothetical:
        adding an icon and forgetting this line gives a rail button that lights
        up and shows the panel that was already open, which reads as a panel
        that failed to load rather than as a stack that never held it.
        """
        stack = QStackedWidget()
        for panel_id, _glyph, _label, _tip in RAIL_ITEMS:
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
        self._refresh_connection()

    def _refresh_connection(self):
        """The one line that says what the cell is talking to."""
        sim = self.cell.simulated
        state = {a: sim or self.cell.arms[a].connected for a in ARM_IDS}
        text, colour, tooltip = connection_status(sim, state)
        self.real_indicator.setText(text)
        self.real_indicator.setToolTip(tooltip)
        self.real_indicator.setStyleSheet(
            f"background:{colour};color:white;font-weight:bold;"
            f"font-size:{S.fpx(14)}px;border-radius:{S.sx(4)}px;"
            f"letter-spacing:{S.sx(1)}px;")

        # Empty unless a Connect/Drop button has been put back on the bar.
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
        """Everything down: jog released, program stopped, servo loop shut,
        arms halted.

        No button on this window calls it any more — see `_build_safety_bar`.
        The browser panel's `stop_all` does, and it is what a STOP put back on
        the bar would be wired to.
        """
        self._release_web_jog()
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
        if self._web_jog_owner:
            if not self._web_jog_live(self._web_jog_owner,
                                      self._web_jog_session):
                self._release_web_jog("web jog heartbeat lost — stopped")
            elif self.panels["jog"].hold_mode and self._web_jog_axis:
                jog = self.panels["jog"]
                reason = jog._blocked_reason(self._web_jog_target)
                if reason:
                    self._release_web_jog(reason)
                else:
                    row, sign = self._web_jog_axis
                    jog._ticked(self._web_jog_target, row, sign)
        self._publish_web_state()

    # ---- browser UI -----------------------------------------------------
    def _start_web(self, host, port, token=None):
        """Start the optional HTTP server beside this hardware-owning UI.

        The token decides who may drive the arms from a browser. Left as None
        a LAN run mints a fresh random one, which is safe but changes the URL
        every restart; a fixed string keeps one bookmark working, and "none"
        serves the LAN with no token at all.
        """
        import secrets
        from ur5dual.web import WebHub, WebServer

        def dispatch(payload):
            # Jog liveness is stamped here, on the HTTP thread, and not where
            # the command finally runs. Polling two arms and encoding a camera
            # frame can hold Qt's loop past WEB_JOG_GRACE, and a browser that
            # is still pressing must not lose the key — and take a 400 on every
            # queued heartbeat — because the desktop was slow to listen.
            if payload.get("action") in ("jog_press", "jog_heartbeat"):
                self._mark_web_jog(payload)
            request = {"payload": payload, "done": __import__("threading").Event(),
                       "result": None, "error": None}
            self.web_command.emit(request)
            if not request["done"].wait(3.0):
                raise TimeoutError("desktop did not answer the web command")
            if request["error"] is not None:
                raise request["error"]
            return request["result"]

        self.web_hub = WebHub(dispatch, jog_touch=self._mark_web_jog)
        # Preview is produced on the capture thread before the slower box
        # detector runs. JPEG encoding is pure NumPy/OpenCV, so it stays off
        # Qt's UI/safety loop and can follow the RealSense source cadence.
        self.vision.on_preview = self._publish_web_preview
        remote = host not in ("127.0.0.1", "localhost", "::1")
        if token is None:
            token = secrets.token_urlsafe(18) if remote else None
        elif token.strip().lower() in ("", "none", "off"):
            token = None
        self.web_server = WebServer(self.web_hub, host, port, token=token)
        url = self.web_server.start()
        if remote:
            url = "http://%s:%d/%s" % (lan_address(host), int(port),
                                       "?token=%s" % token if token else "")
        self._publish_web_state()
        note = ""
        if remote:
            note = " (trusted LAN only)" if token else " (trusted LAN, no token)"
        self.log("web UI: %s%s" % (url, note))

    def _handle_web_command(self, request):
        try:
            request["result"] = self._execute_web_command(request["payload"])
        except Exception as exc:
            request["error"] = exc
        finally:
            request["done"].set()

    def _execute_web_command(self, command):
        """Run one checked browser command on Qt's main thread."""
        action = command.get("action")
        program = self.panels["program"]
        camera = self.panels["camera"]
        jog = self.panels["jog"]

        if action == "stop_all":
            self._stop_everything()
        elif action == "program_run":
            program._run()
        elif action == "program_pause":
            if not self.executor.running:
                raise ValueError("no program is running")
            program._pause()
        elif action == "program_stop":
            program._stop()
        elif action == "program_load":
            name = os.path.basename(str(command.get("name") or ""))
            path = os.path.join(self.programs_dir, name +
                                ("" if name.endswith(".json") else ".json"))
            if not name or not os.path.isfile(path):
                raise ValueError("program not found")
            program.program = Program.load(path)
            program.loop_btn.setChecked(program.program.loop)
            program.refresh()
            self.log("loaded program %s from web" % name)
        elif action == "program_set":
            value = command.get("program")
            if not isinstance(value, dict):
                raise ValueError("program must be a JSON object")
            replacement = Program.from_dict(value)
            problems, _warnings = replacement.check(
                self.points, self.cell.config,
                surfaces=self.executor.taught_on_surface())
            if problems:
                raise ValueError("; ".join(problems[:3]))
            program.program = replacement
            program.loop_btn.setChecked(replacement.loop)
            program.refresh()
            self.log("program updated from web")
        elif action == "program_save":
            name = os.path.basename(program.program.name.strip())
            if not name or name in (".", ".."):
                raise ValueError("program needs a valid name")
            path = os.path.join(self.programs_dir, name +
                                ("" if name.endswith(".json") else ".json"))
            program.program.save(path)
            self.log("saved program to %s from web" % path)
        elif action == "point_teach":
            arm_id = str(command.get("arm") or "")
            name = str(command.get("name") or "").strip()
            if arm_id not in ARM_IDS or not name:
                raise ValueError("point needs an arm and a name")
            if not self.cell.arms[arm_id].connected:
                raise ValueError("arm %s is not connected" % arm_id)
            self.points.teach_arm(self.cell, arm_id, name)
            self.panels["points"]._refresh_everywhere()
            self.log("taught %s from arm %s on web" % (name, arm_id))
        elif action == "point_delete":
            name = str(command.get("name") or "")
            if name not in self.points.points:
                raise ValueError("point not found")
            self.points.remove(name)
            self.panels["points"]._refresh_everywhere()
        elif action == "points_save":
            self.save_points()
            self.log("points saved from web")
        elif action == "comm_set":
            value = command.get("links")
            if not isinstance(value, list) or any(
                    not isinstance(link, dict) for link in value):
                raise ValueError("links must be a JSON list of machines")
            for link in value:
                if not isinstance(link.get("data", []), list):
                    raise ValueError("a machine's data must be a JSON list")
            candidate = LinkLibrary.from_list(value)
            # from_list drops what cannot be a machine or a datum, because a
            # hand-edited cell.yaml must still open the panel. Typed into the
            # browser's editor the same silence would delete a line the
            # operator is looking at, so here the drop is the error.
            if len(candidate.links) != len(value):
                raise ValueError("every machine needs a name")
            if [len(link["data"]) for link in candidate.links] != [
                    len(link.get("data") or []) for link in value]:
                raise ValueError("every data item needs a name")
            problems = candidate.check()
            if problems:
                raise ValueError("; ".join(problems[:3]))
            self._links.links = candidate.links
            self.links_changed()
            self.panels["net"].refresh()
            program.refresh()
            # _say rather than log: the tab's status line is what both
            # surfaces show, so the desktop says where the change came from
            # instead of silently growing a machine nobody on it added.
            self.panels["net"]._say("Communication updated from web")
        elif action == "comm_save":
            problems = self._links.check()
            if problems:
                raise ValueError("; ".join(problems[:3]))
            self.save_links()
            self.panels["net"].refresh()
            self.panels["net"].status.setText(
                "saved %d machine(s)" % len(self._links.links))
        elif action == "comm_test":
            name = str(command.get("name") or "")
            if self._links.get(name) is None:
                raise ValueError("machine not found")
            import threading

            def probe():
                self.comm_probe.emit(self.links.probe(name))

            threading.Thread(target=probe, name="ur5dual-com-probe",
                             daemon=True).start()
        elif action == "camera_mode":
            mode = str(command.get("mode") or "")
            if mode not in camera.mode_btns:
                raise ValueError("unknown camera mode")
            camera._set_mode(mode)
        elif action == "camera_live":
            camera.live_btn.setChecked(bool(command.get("running")))
            camera._toggle_live()
        elif action == "camera_source":
            source = str(command.get("source") or "")
            if source not in ("sim", "realsense"):
                raise ValueError("unknown camera source")
            camera.source_combo.setCurrentText(source)
        elif action == "camera_auto_size":
            camera._apply_auto_size(bool(command.get("enabled")))
            camera._fill_sizes()
            camera._write_setup()
            camera.app.cell.config.save_vision()
        elif action == "camera_box":
            values = command.get("box_mm")
            if not isinstance(values, list) or len(values) != 3:
                raise ValueError("box_mm needs length, width and height")
            values = [int(v) for v in values]
            if not (50 <= values[0] <= 2000 and
                    50 <= values[1] <= 2000 and 10 <= values[2] <= 2000):
                raise ValueError("box size is outside the allowed range")
            for spin, value in zip(
                    (camera.length_spin, camera.width_spin, camera.height_spin),
                    values):
                spin.setValue(value)
            camera._size_settled()
        elif action == "jog_config":
            if self._web_jog_owner:
                raise ValueError("release the web jog key first")
            if "target" in command:
                target = str(command["target"])
                if target not in jog.grids:
                    raise ValueError("unknown jog target")
                jog._select_target(target)
            if "frame" in command:
                from ur5dual.gui.panels.jog import ARM_FRAMES
                frame = str(command["frame"])
                if frame not in ARM_FRAMES:
                    raise ValueError("unknown jog frame")
                jog.frame_combo.setCurrentIndex(ARM_FRAMES.index(frame))
            if "hold_mode" in command:
                jog.motion_combo.setCurrentIndex(
                    0 if bool(command["hold_mode"]) else 1)
            if "preset" in command:
                preset = int(command["preset"])
                if preset not in range(4):
                    raise ValueError("unknown jog preset")
                jog._set_preset(preset)
        elif action == "jog_press":
            client = str(command.get("client_id") or "")
            session = str(command.get("session_id") or client)
            target = str(command.get("target") or jog.target)
            row, sign = int(command.get("row", -1)), int(command.get("sign", 0))
            if not client or not session or target not in jog.grids or row not in range(6) \
                    or sign not in (-1, 1):
                raise ValueError("invalid jog command")
            if (self._web_jog_owner and self._web_jog_owner != client and
                    self._web_jog_live(self._web_jog_owner,
                                       self._web_jog_session)):
                raise ValueError("jog is owned by another browser")
            reason = jog._blocked_reason(target)
            if reason:
                raise ValueError(reason)
            self._web_jog_owner = client
            self._web_jog_target = target
            self._web_jog_axis = (row, sign)
            self._web_jog_session = session
            jog._pressed(target, row, sign)
        elif action == "jog_heartbeat":
            # dispatch() already stamped the arrival; this only says whose key
            # it is, so a second browser cannot hold it open.
            client = str(command.get("client_id") or "")
            session = str(command.get("session_id") or client)
            if (client, session) != (self._web_jog_owner,
                                     self._web_jog_session):
                raise ValueError("this browser does not own jog")
        elif action == "jog_release":
            client = str(command.get("client_id") or "")
            session = str(command.get("session_id") or client)
            # Releases are idempotent. A late packet from an old press must
            # neither produce a 400 storm nor stop a newer jog session.
            if (client, session) == (self._web_jog_owner,
                                     self._web_jog_session):
                self._release_web_jog()
        else:
            raise ValueError("unknown action: %s" % action)

        self._publish_web_state()
        return {"action": action}

    def _comm_probed(self, text):
        """What a browser's Test found, said on the tab that owns the link."""
        self.panels["net"]._say(text)
        self._publish_web_state()

    def _mark_web_jog(self, command):
        client = str(command.get("client_id") or "")
        session = str(command.get("session_id") or client)
        if client and session:
            with self._web_jog_lock:
                self._web_jog_seen[(client, session)] = time.monotonic()

    def _web_jog_live(self, client, session=None):
        """Has this browser been heard from inside the grace window?"""
        session = str(session or client or "")
        with self._web_jog_lock:
            seen = self._web_jog_seen.get((client, session), 0.0)
        return (time.monotonic() - seen) <= WEB_JOG_GRACE

    def _release_web_jog(self, message=None):
        if self._web_jog_target:
            self.panels["jog"]._released(self._web_jog_target)
        self._web_jog_owner = None
        self._web_jog_target = None
        self._web_jog_axis = None
        self._web_jog_session = None
        with self._web_jog_lock:
            self._web_jog_seen.clear()
        if message:
            self.log(message)

    def _web_state(self):
        program_panel = self.panels["program"]
        camera = self.panels["camera"]
        jog = self.panels["jog"]
        arms = {}
        for arm_id in ARM_IDS:
            arm = self.cell.arms[arm_id]
            pose, pose_text = [], "—"
            if arm.connected:
                try:
                    shown = shown_pose(arm.tcp_pose_world())
                    pose = ([v * 1000 for v in shown[:3]] +
                            list(np.degrees(shown[3:])))
                    pose_text = "%+.0f %+.0f %+.0f mm" % tuple(pose[:3])
                except (OSError, RuntimeError, ConnectionError, ValueError):
                    pass
            arms[arm_id] = {"connected": bool(arm.connected), "pose": pose,
                            "pose_text": pose_text}

        rows = []
        for index, step in enumerate(program_panel.program.steps):
            span, a, b, link = step.render()
            rows.append({"index": index + 1 if step.enabled else "·",
                         "kind": step.kind, "a": span or a, "b": "" if span else b,
                         "link": link})
        files = sorted(os.path.splitext(name)[0] for name in
                       os.listdir(self.programs_dir) if name.endswith(".json"))
        points = []
        for name in self.points.names():
            pose = shown_pose(self.points.get(name))
            values = ([v * 1000 for v in pose[:3]] +
                      list(np.degrees(pose[3:])))
            points.append({"name": name, "pose": values})
        sizes = []
        for i in range(camera.size_combo.count()):
            value = camera.size_combo.itemData(i)
            if isinstance(value, tuple):
                sizes.append(list(value))
        return {
            "arms": arms,
            # The browser has no red band of its own to read, and a tablet
            # across the room is exactly where mistaking a simulated cell for
            # two live arms costs the most. So the panel's own word travels
            # with the state rather than being inferred from anything.
            "simulated": bool(self.cell.simulated),
            "last_message": self.last_message.text(),
            "program": {
                "name": program_panel.program.name,
                "raw": program_panel.program.to_dict(), "rows": rows,
                "files": files, "running": bool(self.executor.running),
                "paused": bool(self.executor.paused),
                "current": int(self.executor.current),
                "problem": program_panel.problems.text(),
            },
            "points": points,
            "comm": {
                "links": self._links.to_list(),
                "status": self.panels["net"].status.text(),
            },
            "camera": {
                "running": bool(self.vision.running), "mode": camera.mode,
                "reading": camera.found_lbl.text(),
                "source": camera.source_combo.currentText(), "sizes": sizes,
                "auto_size": bool(self.cell.config.vision.get(
                    "auto_size", False)),
                "box_mm": [camera.length_spin.value(),
                           camera.width_spin.value(), camera.height_spin.value()],
            },
            "jog": {
                "target": jog.target, "frame": jog.frame,
                "hold_mode": bool(jog.hold_mode), "preset": jog.preset,
                "size": jog.size_lbl.text(), "note": jog.note.text(),
                "web_owned": bool(self._web_jog_owner),
            },
        }

    def _publish_web_state(self):
        if self.web_hub is None:
            return
        self.web_hub.publish(self._web_state())

    def _publish_web_preview(self, frame):
        """Capture-thread path: raw frame to binary WebSocket without Qt."""
        if self.web_hub is None or not self.web_hub.camera_streaming():
            return
        camera = self.panels["camera"].web_jpeg(frame, quality=74)
        if camera is not None:
            self.web_hub.publish_camera(camera)

    # ---- lifecycle -------------------------------------------------------
    def changeEvent(self, event):
        # a jog button held while the window loses focus never emits released()
        if event.type() == event.ActivationChange and not self.isActiveWindow():
            self.panels["jog"].release()
        super().changeEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        self.vision.on_preview = None
        self._release_web_jog()
        self.executor.stop()
        self.vision.stop()
        if self.web_server is not None:
            self.web_server.stop()
        self.links.close_all()
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
    parser.add_argument("--sim", action="store_true",
                        help="simulated arms: the panel, the programs, the "
                             "jog and the servo loop all run and nothing is "
                             "sent to a controller. The camera still follows "
                             "vision.source; the Camera tab switches it to "
                             "'sim' on a machine with no lens")
    parser.add_argument("--web", action="store_true",
                        help="serve the synchronized browser UI")
    parser.add_argument("--web-host", default="127.0.0.1",
                        help="web bind address; use 0.0.0.0 on a trusted LAN")
    parser.add_argument("--web-port", type=int, default=8765,
                        help="browser UI port (default: 8765)")
    parser.add_argument("--web-token", default=None,
                        help="fixed browser access token, so the URL survives "
                             "a restart; 'none' serves the LAN with no token "
                             "at all. Left out, every run mints a new one")
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

    # Simulated arms are built here rather than inside the window, so the one
    # place that decides whether this run can move a robot is the command line
    # that started it. Everything below the window keeps asking `cell.arms[id]`
    # the same questions either way.
    cell = (Cell(CellConfig.load(args.config), simulated=True)
            if args.sim else None)
    window = MainWindow(args.config, cell=cell,
                        web_host=args.web_host if args.web else None,
                        web_port=args.web_port,
                        web_token=args.web_token)
    # Maximised rather than resized to the design size: what the layout gets
    # over its minimum goes to the jog grid and the message log rather than to
    # wallpaper, and a window already the size of the screen leaves the
    # maximise button nothing to do but nothing.
    window.show_window(fullscreen)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
