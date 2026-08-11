"""
The dual-UR5 control panel.

Two halves of the window, the shape of a teach pendant. The left half is the
program: the step table and the buttons that run it, always in view, because
that is the document being written and it is what the panel is for. The right
half is everything you do to build it, in tabs — Points, Vars, Object, Jog.

The Jog tab carries the cell down its own left column: mode, connect, STOP,
both arms' readouts, the pair numbers and the message log. Jogging is where an
operator stands and watches, so the state sits beside the buttons that change
it rather than in a strip the program column would otherwise pay for.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QGridLayout, QHBoxLayout, QMainWindow,
    QPlainTextEdit, QTabWidget, QVBoxLayout, QWidget,
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
from ur5dual.gui.widgets.monitor import ArmMonitor, PairMonitor  # noqa: E402
from ur5dual.program.executor import Executor                 # noqa: E402
from ur5dual.program.steps import PointLibrary                # noqa: E402

# a correctly configured arm reads only a few newtons at rest
PAYLOAD_SUSPECT_N = 15.0

POINTS_FILE = os.path.join(REPO_ROOT, "config", "points.json")
PROGRAMS_DIR = os.path.join(REPO_ROOT, "config", "programs")

# half the 1280x800 design width, less the margins between the two columns
PROGRAM_COL_W = 620
PROGRAM_COL_MIN_W = 520


class MainWindow(QMainWindow):
    # Qt widgets may only be touched from the thread that created them, and
    # two threads here are not it: the program executor and the coordinated
    # servo feed both log. Routing every message through a signal lets Qt
    # queue it onto the GUI thread. Calling appendPlainText directly from a
    # worker corrupts the document's internal cursor and takes the process
    # down with it, which is what "Cannot queue arguments of type
    # 'QTextCursor'" is warning about just before the segfault.
    log_message = pyqtSignal(str)

    def __init__(self, config_path=DEFAULT_PATH):
        super().__init__()
        self.setWindowTitle("Dual UR5 control")

        # Built simulated, matching the mode selector's default. Opening a
        # panel next to two robots must not be the act that reaches out to
        # them; REAL is a thing you choose.
        self.cell = Cell(CellConfig.load(config_path), simulated=True)
        self.cell.listeners.append(self.log)
        self.points = PointLibrary()
        self.executor = Executor(self.cell, self.points)
        self.executor.on_log = self.log
        self.coordinator_start_error = None
        # SIM until told otherwise. The panel comes up next to two robots that
        # can reach each other, and a default that moves them is a default that
        # is wrong exactly once.
        self.mode = "sim"
        self._rviz_hint_given = False
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

    # ---- layout ----------------------------------------------------------
    def _build(self):
        # the cell column is built first and handed to the Jog tab: the panels
        # below already reach into these widgets (mode, connect buttons, the
        # log), and the tab is only where they are shown
        status = self._build_status()
        self.panels = {
            "program": ProgramPanel(self),
            "points": PointsPanel(self),
            "vars": CellSetupPanel(self),
            "objects": ObjectPanel(self),
            "jog": JogPanel(self, status),
        }
        self.panels["vars"].geometry_changed.connect(self._geometry_changed)

        central = QWidget()
        body = QHBoxLayout(central)
        body.setContentsMargins(S.sx(4), S.sx(4), S.sx(4), S.sx(4))
        body.setSpacing(S.sx(6))
        body.addWidget(self._build_program(), 1)
        body.addWidget(self._build_tabs(), 1)
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

    def _build_status(self):
        page = QWidget()
        # not fixed: the readouts need this much, the jog grid beside them
        # wants every pixel that is left
        page.setMinimumWidth(S.sx(330))
        page.setMaximumWidth(S.sx(370))
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(S.sx(4))

        # The mode sits above everything, because it is the answer to the only
        # question an operator has on walking up to this panel: is pressing a
        # button on it going to move a robot.
        v.addWidget(S.strip("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "SIM — RViz moves, the arms do not",
            "REAL — the arms move",
        ])
        self.mode_combo.setMinimumHeight(S.sx(46))
        self.mode_combo.setStyleSheet(S.combo())
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        v.addWidget(self.mode_combo)

        v.addWidget(S.strip("Cell"))
        grid = QGridLayout()
        grid.setSpacing(S.sx(4))
        self.conn_btns = {}
        for col, arm_id in enumerate(ARM_IDS):
            button = S.touch_button("Connect %s" % arm_id, S.GREEN, height=40,
                                    font_px=13)
            button.clicked.connect(lambda _c=False, a=arm_id: self._toggle(a))
            grid.addWidget(button, 0, col)
            self.conn_btns[arm_id] = button

        # Only useful, and only offered, while the cell is simulated. It
        # restores both arms to their configured ready posture.
        self.ready_btn = S.touch_button("Ready pose", S.BLUE, height=40,
                                        font_px=13)
        self.ready_btn.clicked.connect(self._sim_ready_pose)
        grid.addWidget(self.ready_btn, 0, len(ARM_IDS))

        for col, (label, command) in enumerate((("Power on", "power on"),
                                                ("Brakes", "brake release"),
                                                ("Unlock", "unlock protective stop"))):
            button = S.touch_button(label, height=36, font_px=12)
            button.clicked.connect(lambda _c=False, c=command: self._dashboard(c))
            grid.addWidget(button, 1, col)
        v.addLayout(grid)

        self.stop_btn = S.touch_button("■   STOP", S.RED, height=58, font_px=20)
        self.stop_btn.clicked.connect(self._stop_everything)
        v.addWidget(self.stop_btn)

        self.monitors = {}
        for arm_id in ARM_IDS:
            v.addWidget(S.arm_strip(arm_id, "Arm %s" % arm_id))
            monitor = ArmMonitor(arm_id)
            self.monitors[arm_id] = monitor
            v.addWidget(monitor)

        v.addWidget(S.strip("Both arms"))
        self.pair = PairMonitor()
        v.addWidget(self.pair)

        v.addWidget(S.strip("Messages"))
        self.msg_box = QPlainTextEdit()
        self.msg_box.setReadOnly(True)
        self.msg_box.setMaximumBlockCount(400)
        self.msg_box.setStyleSheet(f"font-size:{S.fpx(12)}px;")
        v.addWidget(self.msg_box, 1)
        return page

    def _build_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(S.tabs())
        self.tabs.addTab(self.panels["points"], "Points")
        self.tabs.addTab(self.panels["vars"], "Vars")
        self.tabs.addTab(self.panels["objects"], "Object")
        jog = self.panels["jog"]
        self.tabs.addTab(jog, "Jog")
        # by widget, not by index: adding a tab must not change which page the
        # panel opens on, and it opens on the one with the cell down its side
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

    def _sim_ready_pose(self):
        if self.executor.object.held:
            self.log("let go before moving the arms to the ready posture — "
                     "the grip was captured where they are now")
            return
        self.cell.sim_ready_pose()

    def _mode_changed(self, index):
        """Switch between driving the drawing and driving the robots.

        The servo loop is torn down and rebuilt rather than switched in place.
        Half of what the mode decides is settled when the loop starts — which
        backend each arm gets, whether the solver runs, which guards are armed
        — and a loop that changed its mind about those mid-flight would be a
        loop nobody could reason about while two arms held a workpiece.

        Whatever is held stays held. Only the question of where its targets go
        is being answered differently.
        """
        wanted = "real" if index == 1 else "sim"
        if wanted == self.mode:
            return
        self.mode = wanted
        self.stop_coordinator()
        # Whatever was held was held by the arms that are about to be replaced,
        # and its grasp transforms were measured against them. Carrying that
        # across would be claiming a grip that no longer refers to anything.
        if self.executor.object.held:
            self.executor.object.release()
            self.log("let go of the object — its grip was captured on the "
                     "other set of arms")
        self.cell.set_simulated(wanted == "sim")
        self._refresh_connect_buttons()
        self.panels["jog"].mode_changed()
        self.panels["objects"].refresh()
        self.log("mode: %s" % (
            "REAL — object moves drive both arms"
            if wanted == "real" else
            "SIM — nothing is sent to a robot; the arms here are arithmetic"))
        if wanted == "real":
            # Announce ownership before opening 30003.  The RViz bridge polls
            # loopback at 50 Hz and drops its fallback robot sockets as soon
            # as it sees this; a short handover prevents old controllers from
            # ever having to choose between two real-time clients.
            for _ in range(3):
                # Connecting can block while a powered-off arm times out.  The
                # lease keeps RViz from reclaiming 30003 during that quiet
                # interval; normal 60 ms heartbeats take over once connected.
                self.cell.publish_sim_view(lease=20.0)
                time.sleep(0.03)
            self.cell.connect()
            self._refresh_connect_buttons()
        elif not self._rviz_hint_given:
            self._rviz_hint_given = True
            self.log("run scripts/ur5dual-rviz to watch it; RViz receives both "
                     "SIM and REAL joint states from this panel")

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
        self.ready_btn.setVisible(sim)
        for arm_id, button in self.conn_btns.items():
            connected = self.cell.arms[arm_id].connected
            if sim:
                # A simulated arm is always "connected" and there is nothing to
                # dial. Offering Connect here would be offering an action whose
                # only possible outcome is a timeout.
                button.setText("%s simulated" % arm_id)
                button.setStyleSheet(S.pill(13))
                button.setEnabled(False)
                button.setToolTip("switch to REAL mode to reach the controller")
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

        limit = float(self.cell.config.motion["max_tcp_force"])
        for arm_id in ARM_IDS:
            arm = self.cell.arms[arm_id]
            while arm.stream_events:
                self.log(arm.stream_events.pop(0))
        for arm_id, monitor in self.monitors.items():
            monitor.update_from(self.cell.arms[arm_id], limit)
        self.pair.update_from(self.cell, self.executor.object,
                              float(self.cell.config.motion["max_pair_drift"]))
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
                        help="UI scale; 1.0 is the 1280x800 panel design size")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    if args.scale:
        S.set_scale(args.scale)

    app = QApplication(sys.argv)
    palette = app.palette()
    palette.setColor(QPalette.Window, QColor("#f4f4f4"))
    app.setPalette(palette)

    window = MainWindow(args.config)
    window.resize(S.sx(1280), S.sx(800))
    window.showFullScreen() if args.fullscreen else window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
