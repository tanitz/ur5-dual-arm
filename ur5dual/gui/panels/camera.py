"""Camera tab for the known-size open-box detector."""

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit,
    QSpinBox, QVBoxLayout, QWidget,
)

from ...axes import shown_xyz
from ...geometry.kinematics import mat_to_pose, mat_to_rpy
from ...vision.detect import (
    height_check, image_relative_rotation, image_relative_xyz, size_check,
    surface_plane,
)
from ...vision.planar import PlaneMapError, box_on_plane
from ...vision.charuco import BoardError, find_board, printed_board
from ...tools.board_check import Staged, _apply_live, rank_rims, rim_candidates
from .. import style as S
from .jog import JogPanel


VIEW_W, VIEW_H = 308, 195
VIEW_MODES = ("lens", "blend", "depth")
BLEND = 0.45
NEAR, FAR = 0.30, 1.60

# What an opening may measure, in millimetres. Wide enough for anything this
# cell can reach into and narrow enough that a slipped digit is refused rather
# than solved for: a 40 mm box would still produce a confident pose, metres
# out, and only the depth check underneath would ever say so.
SIZE_MIN, SIZE_MAX, SIZE_STEP = 50, 2000, 10
# How many openings the dropdown keeps. A shift works through a handful
# of crates, not a catalogue, and a list longer than the panel is a list
# that has to be scrolled to find the box that is actually on the bench.
SIZE_MEMORY = 8
AUTO_SIZE = "auto"

class SurfaceBoxCalibrationDialog(QDialog):
    """The board tool's two captures, hosted by the main camera service."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.app = panel.app
        self.board = printed_board()
        self.staged = Staged()
        self.reading = None
        self.rims = []
        self.rim_index = 0
        self.rim = None
        self.pose = None
        self._held = False
        self.setWindowTitle("Measure Surface + Box")
        self.setModal(False)

        body = QVBoxLayout(self)
        body.setContentsMargins(S.sx(10), S.sx(10), S.sx(10), S.sx(10))
        body.setSpacing(S.sx(7))
        body.addWidget(S.strip("Camera surface — box then ChArUco board"))

        self.camera_view = QLabel("starting camera…")
        self.camera_view.setFixedSize(S.sx(640), S.sx(420))
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet(
            "background:#101418;border:1px solid #c4c4c4;color:#aab4ba;")
        body.addWidget(self.camera_view, 0, Qt.AlignHCenter)

        help_text = QLabel(
            "Step 1: leave the box on its support and capture its rim. "
            "Step 2: remove the box, put the printed ChArUco board exactly "
            "where it stood without moving the camera, then capture and save.")
        help_text.setWordWrap(True)
        help_text.setStyleSheet(f"font-size:{S.fpx(12)}px;color:#59656d;")
        body.addWidget(help_text)

        self.result_lbl = QLabel("")
        self.result_lbl.setWordWrap(True)
        self.result_lbl.setStyleSheet(
            f"font-size:{S.fpx(12)}px;font-family:monospace;color:#333;")
        body.addWidget(self.result_lbl)
        self.status = QLabel("Step 1 — waiting for a box rim")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"font-size:{S.fpx(12)}px;color:#59656d;")
        body.addWidget(self.status)

        actions = QHBoxLayout()
        self.box_btn = S.touch_button(
            "1 Capture Box", S.BLUE, height=44, font_px=13)
        self.box_btn.clicked.connect(self._capture_box)
        self.next_btn = S.touch_button("Next Rim", height=44, font_px=13)
        self.next_btn.clicked.connect(self._next_rim)
        self.board_btn = S.touch_button(
            "2 Capture Surface", S.BLUE, height=44, font_px=13)
        self.board_btn.clicked.connect(self._capture_board)
        self.apply_btn = S.touch_button(
            "Apply + Save", S.GREEN, height=44, font_px=13)
        self.apply_btn.clicked.connect(self._apply)
        reset_btn = S.touch_button("Reset", height=44, font_px=13)
        reset_btn.clicked.connect(self._reset)
        close_btn = S.touch_button("Close", height=44, font_px=13)
        close_btn.clicked.connect(self.close)
        for button, stretch in ((self.box_btn, 2), (self.next_btn, 1),
                                (self.board_btn, 2), (self.apply_btn, 2),
                                (reset_btn, 1), (close_btn, 1)):
            actions.addWidget(button, stretch)
        body.addLayout(actions)
        self.resize(S.sx(700), S.sx(600))
        self._update_controls()

    def open_for_measurement(self):
        self._reset()
        if not self.app.vision.running:
            self.app.vision.start()
        self.show()
        self.raise_()
        self.activateWindow()
        self.show_reading(self.app.vision.latest)

    def _say(self, text, colour=S.AMBER, hold=True):
        self.status.setText(text)
        self.status.setStyleSheet(
            f"font-size:{S.fpx(12)}px;color:{colour};")
        self._held = hold

    def _reset(self):
        self.staged = Staged()
        self.rims = []
        self.rim_index = 0
        self.rim = None
        self.pose = None
        self._held = False
        self.result_lbl.setText("")
        self._say("Step 1 — leave the box in view and capture its rim",
                  "#59656d", hold=False)
        self._update_controls()

    def _next_rim(self):
        if len(self.rims) < 2:
            self._say("No other rim is visible")
            return
        self.rim_index = (self.rim_index + 1) % len(self.rims)
        self.rim = self.rims[self.rim_index]
        self._say("Showing rim %d of %d" %
                  (self.rim_index + 1, len(self.rims)), hold=False)
        self._update_controls()

    def _capture_box(self):
        if self.reading is None or self.reading.frame is None or self.rim is None:
            self._say("No box rim is ready to capture")
            return
        self.staged.take_box(self.reading.frame, self.rim)
        self.pose = None
        self._held = False
        self._say("Box captured — remove it and put the board where it stood",
                  S.GREEN, hold=False)
        self._update_controls()

    def _capture_board(self):
        if self.pose is None:
            self._say("The ChArUco board is not ready to capture")
            return
        self.staged.take_board(self.pose)
        opening, height, why = self.staged.opening()
        if opening is None:
            self._say(why)
        else:
            self.result_lbl.setText(
                "opening %.1f × %.1f mm    height %.1f mm" %
                (opening.length * 1000, opening.width * 1000, height * 1000))
            self._say("Surface captured — check the measurement, then Apply",
                      S.GREEN, hold=False)
        self._update_controls()

    def _apply(self):
        message = _apply_live(
            self.app.cell.config, self.staged, self.app.cell.config.path,
            self.app.log)
        if not message.startswith("wrote "):
            self._say(message)
            return

        vision = self.app.cell.config.vision
        self.app.vision.config.update(vision)
        detector = self.app.vision.detector
        if detector is not None:
            detector.box_size = tuple(vision.get("box_size") or detector.box_size)
            detector.surface = surface_plane(vision.get("surface"))
            detector.reset()
        self.panel._load_size()
        self.panel._write_setup()
        self.panel._take_up_record()
        self.panel._show_surface_measurement()
        self.panel._show_home_box()
        self._say("Saved surface and measured box size", S.GREEN)

    def _update_controls(self):
        stage = self.staged.stage
        self.box_btn.setEnabled(stage == 1 and self.rim is not None)
        self.next_btn.setEnabled(stage == 1 and len(self.rims) > 1)
        self.board_btn.setEnabled(stage == 2 and self.pose is not None)
        self.apply_btn.setEnabled(self.staged.board is not None)

    def _picture(self, frame):
        rgb = self.panel._color_rgb(frame)
        if rgb is None:
            return None
        rgb = np.ascontiguousarray(rgb).copy()
        if self.rim is not None and self.staged.stage == 1:
            cv2.polylines(rgb, [np.rint(self.rim).astype(np.int32)], True,
                          (240, 170, 40), 3, cv2.LINE_AA)
        if self.staged.box is not None:
            cv2.polylines(rgb,
                          [np.rint(self.staged.box[0]).astype(np.int32)], True,
                          (90, 170, 240), 2, cv2.LINE_AA)
        if self.pose is not None:
            for corner in np.rint(self.pose.corners).astype(int):
                cv2.circle(rgb, tuple(corner), 4, (80, 220, 80), -1,
                           cv2.LINE_AA)
        return self.panel._scaled(rgb, self.camera_view)

    def show_reading(self, reading):
        if not self.isVisible():
            return
        self.reading = reading
        if reading.frame is None:
            self.camera_view.setText(reading.error or "no frame yet")
            return
        frame = reading.frame
        if self.staged.stage == 1:
            try:
                self.rims = rank_rims(
                    frame, rim_candidates(
                        frame, self.app.cell.config.vision.get("roi")))
                if self.rims:
                    self.rim_index %= len(self.rims)
                    self.rim = self.rims[self.rim_index]
                    if not self._held:
                        self._say("Step 1 — rim %d of %d is ready" %
                                  (self.rim_index + 1, len(self.rims)),
                                  S.GREEN, hold=False)
                else:
                    self.rim = None
                    if not self._held:
                        self._say("Step 1 — no box rim found", hold=False)
            except (BoardError, ValueError) as exc:
                self.rim = None
                if not self._held:
                    self._say(str(exc), hold=False)
        else:
            try:
                self.pose = find_board(frame, self.board)
                if not self._held:
                    self._say("Step 2 — board found with %d corners" %
                              len(self.pose.corners), S.GREEN, hold=False)
            except BoardError as exc:
                self.pose = None
                if not self._held:
                    self._say("Step 2 — %s" % exc, hold=False)
        picture = self._picture(frame)
        if picture is not None:
            self.camera_view.setPixmap(picture)
        self._update_controls()


class HomeBoxDialog(QDialog):
    """Teach a box reference and its pre-pick TCP in one guided window."""

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.app = panel.app
        self._saved = False
        # This row is rewritten at the panel's tick, so a refusal needs
        # somewhere to say it is still being read.
        self._held = False
        self.setWindowTitle("Create Home Box")
        self.setModal(False)

        body = QVBoxLayout(self)
        body.setContentsMargins(S.sx(10), S.sx(10), S.sx(10), S.sx(10))
        body.setSpacing(S.sx(7))
        body.addWidget(S.strip("Home box — camera and pre-pick pose"))

        content = QHBoxLayout()
        content.setSpacing(S.sx(10))
        left = QVBoxLayout()
        left.setSpacing(S.sx(7))

        self.camera_view = QLabel("starting camera…")
        self.camera_view.setFixedSize(S.sx(520), S.sx(330))
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setStyleSheet(
            "background:#101418;border:1px solid #c4c4c4;color:#aab4ba;")
        left.addWidget(self.camera_view, 0, Qt.AlignHCenter)

        fields = QHBoxLayout()
        fields.setSpacing(S.sx(6))
        fields.addWidget(S.caption("reference"))
        self.name_edit = QLineEdit("box_home")
        self.name_edit.setMinimumHeight(S.sx(36))
        self.name_edit.setStyleSheet(S.field())
        fields.addWidget(self.name_edit, 1)
        left.addLayout(fields)

        help_text = QLabel(
            "Select Arm A, synchronized A+B, or Arm B on the Jog panel. Move "
            "the gripper tip(s) to the safe pose immediately before the pick, "
            "keep the box visible, then save both readings together.")
        help_text.setWordWrap(True)
        help_text.setStyleSheet(f"font-size:{S.fpx(12)}px;color:#59656d;")
        left.addWidget(help_text)

        self.status = QLabel("Waiting for a box detection")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"font-size:{S.fpx(12)}px;color:#59656d;")
        left.addWidget(self.status)
        left.addStretch(1)

        actions = QHBoxLayout()
        self.save_btn = S.touch_button(
            "Save arm + object", S.BLUE, height=44, font_px=13)
        self.save_btn.clicked.connect(self._save)
        close_btn = S.touch_button("Close", height=44, font_px=13)
        close_btn.clicked.connect(self.close)
        actions.addWidget(self.save_btn, 2)
        actions.addWidget(close_btn, 1)
        left.addLayout(actions)

        self.jog_panel = JogPanel(self.app)
        self.jog_panel.setFixedWidth(S.sx(320))
        content.addLayout(left, 1)
        content.addWidget(self.jog_panel, 0)
        body.addLayout(content, 1)
        self.resize(S.sx(880), S.sx(720))

    def open_for_teach(self):
        self._saved = False
        self._held = False
        self.status.setStyleSheet(
            f"font-size:{S.fpx(12)}px;color:#59656d;")
        if not self.app.vision.running:
            self.app.vision.start()
        self.show()
        self.raise_()
        self.activateWindow()
        self.show_reading(self.app.vision.latest)

    def show_reading(self, reading):
        if not self.isVisible():
            return
        self.jog_panel.tick()
        if reading.frame is None:
            self.camera_view.setText(reading.error or "no frame yet")
            return
        self.camera_view.setPixmap(
            self.panel._picture(reading, self.camera_view))
        if reading.detection is None:
            if not (self._saved or self._held):
                self.status.setText(reading.why_not())
            self.save_btn.setEnabled(False)
        else:
            if not (self._saved or self._held):
                self.status.setText(
                    "Object found at camera XYZ %+.1f  %+.1f  %+.1f mm" %
                    tuple(reading.detection.centre * 1000))
            self.save_btn.setEnabled(True)

    def _selected_arms(self):
        target = self.jog_panel.target
        arm_ids = tuple(sorted(self.app.cell.arms)) if target == "AB" \
            else (target,)
        missing = [arm_id for arm_id in arm_ids
                   if not self.app.cell.arms[arm_id].connected]
        if missing:
            raise PlaneMapError(
                "arm %s is not connected" % " and ".join(missing))
        return arm_ids

    def _save(self):
        self.jog_panel.release()
        # Releasing the row also releases its colour, the same pair
        # `open_for_teach` sets: the live readout that resumes here is not the
        # refusal that was standing, and reading it in amber says it is.
        self._held = False
        self.status.setStyleSheet(
            f"font-size:{S.fpx(12)}px;color:#59656d;")
        name = self.name_edit.text().strip() or "box_home"
        try:
            arm_ids = self._selected_arms()
            self.app.vision.refit()
            reading = self.app.vision.fresh(3.0)
            if reading.detection is None:
                raise PlaneMapError(
                    "camera cannot see the box: %s" % reading.why_not())
            tools = {arm_id: self.app.cell.arms[arm_id].tcp_matrix_world()
                     for arm_id in arm_ids}
            store = self.app.surface
            if store is not None and store.ready:
                wrong = store.fits_box(self.panel._size())
                if wrong:
                    raise PlaneMapError(wrong)
                placement = box_on_plane(
                    reading.detection.corners, store.plane_map(),
                    self.panel._size())
                store.teach(name, placement)
                store.teach_stance(name, tools)
                store.save(self.app.surface_path())
                self.app.reload_surface(store)
                object_text = placement.describe()
            elif self.app.cell.config.vision.get("calibrated"):
                camera_to_world = self.app.executor._camera_to_world()
                object_pose = reading.detection.matrix_in(camera_to_world)
                self.app.points.set(name, mat_to_pose(object_pose))
                for arm_id, tool in tools.items():
                    self.app.points.set("%s_%s_pre_pick" % (name, arm_id),
                                        mat_to_pose(tool))
                self.app.save_points()
                object_text = "6D object pose in world frame"
            else:
                # Save exactly the R/F/H pose printed on the camera card so
                # FIND cannot mix it with raw optical PnP XYZ later.
                vision = self.app.cell.config.vision
                relative_xyz = image_relative_xyz(
                    reading.detection, reading.frame,
                    vision.get("surface"), vision.get("floor"))
                relative_rotation = image_relative_rotation(
                    reading.detection, vision.get("surface"),
                    vision.get("floor"))
                if relative_xyz is None or relative_rotation is None:
                    raise PlaneMapError(
                        "Create Home Box needs camera R/F/H — run Measure "
                        "Surface + Box first")
                relative_pose = np.eye(4)
                relative_pose[:3, :3] = relative_rotation
                relative_pose[:3, 3] = relative_xyz
                refs = self.app.cell.config.vision.setdefault(
                    "home_references", {})
                refs[name] = {
                    "camera_rfh": [float(v) for v in mat_to_pose(
                        relative_pose)],
                    # With one arm this is the taught box-home/pick point.
                    # With two it is their midpoint, so a yaw correction
                    # carries both TCPs around the same physical centre.
                    "anchor_world": [float(v) for v in np.mean(
                        [tool[:3, 3] for tool in tools.values()], axis=0)],
                    "tools": {
                        arm_id: [float(v) for v in mat_to_pose(tool)]
                        for arm_id, tool in tools.items()},
                    "box_size": [float(v) for v in self.panel._size()],
                }
                self.app.cell.config.save_vision()
                object_text = "camera R/F/H box home"
        except (PlaneMapError, RuntimeError, OSError) as exc:
            self.status.setText(str(exc))
            self.status.setStyleSheet(
                f"font-size:{S.fpx(12)}px;color:{S.AMBER};")
            self._held = True
            return
        # Every storage route feeds FIND, but its editor takes a snapshot of
        # the available references when it opens. Refresh the panels after
        # either route so a just-taught name is immediately offered instead
        # of appearing only after a restart or an unrelated tab change.
        self.app.panels["points"].refresh()
        self.app.panels["program"].refresh()
        self.status.setStyleSheet(
            f"font-size:{S.fpx(12)}px;color:{S.GREEN};")
        self._saved = True
        tcp_text = "; ".join(
            "arm %s TCP %+.1f %+.1f %+.1f mm"
            % (arm_id, *(mat_to_pose(tool)[:3] * 1000))
            for arm_id, tool in tools.items())
        self.status.setText(
            "Saved %s: %s; pre-pick %s" % (name, object_text, tcp_text))
        self.panel._show_home_box()
        self.app.log("home box %s saved with %s pre-pick pose" %
                     (name, "+".join(arm_ids)))

    def closeEvent(self, event):
        self.jog_panel.release()
        super().closeEvent(event)

    def changeEvent(self, event):
        if event.type() == event.ActivationChange and not self.isActiveWindow():
            self.jog_panel.release()
        super().changeEvent(event)


class CameraPanel(QWidget):
    """Live image, the opening's size, and the filtered pose FIND reads."""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.mode = "lens"
        self._said = None
        self._saved_size = None      # what is on disk, so a pass of focus
                                     # through the fields does not rewrite it
        configured = self.app.cell.config.vision.get("box_size") or (
            0.60, 0.40, 0.20)
        # Older box_sizes entries contain L/W only.  Preserve the height that
        # was global when this panel opened while those entries are migrated
        # to L/W/H; do not let a newly typed height leak into every old size.
        self._legacy_height = float(
            configured[2] if len(configured) >= 3 else 0.20)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(S.sx(6), S.sx(6), S.sx(6), S.sx(6))
        layout.setSpacing(S.sx(5))
        layout.addWidget(S.strip("Camera — open-box pose"))

        self.view = QLabel("no frame yet")
        self.view.setFixedSize(S.sx(VIEW_W), S.sx(VIEW_H))
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet(
            "background:#101418;border:1px solid #c4c4c4;")
        layout.addWidget(self.view)

        modes = QHBoxLayout()
        modes.setSpacing(S.sx(4))
        self.mode_btns = {}
        for key in VIEW_MODES:
            button = S.touch_button(key, height=32, font_px=12,
                                    checkable=True)
            button.clicked.connect(lambda _checked=False, name=key:
                                   self._set_mode(name))
            modes.addWidget(button, 1)
            self.mode_btns[key] = button
        layout.addLayout(modes)

        self.found_lbl = QLabel("—")
        self.found_lbl.setWordWrap(True)
        self.found_lbl.setStyleSheet(
            f"font-size:{S.fpx(11)}px;font-family:monospace;color:#333333;"
            f"background:#f7f9fb;border:1px solid #dcdcdc;"
            f"padding:{S.sx(3)}px {S.sx(6)}px;")
        layout.addWidget(self._pin(self.found_lbl, 4))

        self.source_combo = QComboBox()
        self.source_combo.addItems(["sim", "realsense"])
        self.source_combo.setCurrentText(
            self.app.cell.config.vision.get("source", "sim"))
        self.source_combo.setMinimumHeight(S.sx(34))
        self.source_combo.setStyleSheet(S.combo())
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        layout.addLayout(self._row("source", self.source_combo))

        # The opening's own size, which is not a preference but a measurement
        # the solver is given: four corners in a picture are the same four
        # corners whatever they are made of, so the metres come entirely from
        # the first two numbers. Height controls the projected wall below that
        # face; unlike length and width it is not inferred from the top rim.
        # The depth reading under the pose says whether length and width fit.
        #
        # Picked far more often than typed, so the list comes first: a cell
        # runs the same two or three crates all week, and dialling all three
        # dimensions afresh each time is needless room for error.
        self.size_combo = QComboBox()
        self.size_combo.setMinimumHeight(S.sx(34))
        self.size_combo.setStyleSheet(S.combo())
        self.size_combo.activated.connect(self._size_picked)
        layout.addLayout(self._row("box mm", self.size_combo))

        self.length_spin = self._size_spin()
        self.width_spin = self._size_spin()
        self.height_spin = self._size_spin(minimum=10)
        sizes = QHBoxLayout()
        sizes.setSpacing(S.sx(4))
        # Aligned under the dropdown rather than given a caption of their own:
        # the three fields are how a size that is not on the list gets onto it,
        # and a second label saying "box mm" would read as a second setting.
        indent = S.caption("")
        indent.setFixedWidth(S.sx(52))
        sizes.addWidget(indent)
        sizes.addWidget(self.length_spin, 1)
        sizes.addWidget(S.caption("×"), 0)
        sizes.addWidget(self.width_spin, 1)
        sizes.addWidget(S.caption("×"), 0)
        sizes.addWidget(self.height_spin, 1)
        layout.addLayout(sizes)

        dimensions = QHBoxLayout()
        dimension_indent = S.caption("")
        dimension_indent.setFixedWidth(S.sx(52))
        dimensions.addWidget(dimension_indent)
        dimensions.addWidget(S.caption("length"), 1)
        dimensions.addWidget(S.caption(""), 0)
        dimensions.addWidget(S.caption("width"), 1)
        dimensions.addWidget(S.caption(""), 0)
        dimensions.addWidget(S.caption("height"), 1)
        layout.addLayout(dimensions)

        self.setup_lbl = QLabel()
        self.setup_lbl.setWordWrap(True)
        self.setup_lbl.setStyleSheet(
            f"font-size:{S.fpx(11)}px;color:#59656d;")
        layout.addWidget(self._pin(self.setup_lbl, 1))

        self.live_btn = S.touch_button(
            "▶ Live", S.GREEN, height=40, font_px=13, checkable=True)
        self.live_btn.clicked.connect(self._toggle_live)
        layout.addWidget(self.live_btn)
        self._show_live()

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setStyleSheet(
            f"font-size:{S.fpx(11)}px;color:{S.AMBER};")
        layout.addWidget(self.note)

        workflows = QHBoxLayout()
        workflows.setSpacing(S.sx(4))
        self.surface_btn = S.touch_button(
            "Measure Surface + Box", S.PURPLE, height=42, font_px=13)
        self.surface_btn.setToolTip(
            "capture the box, then the ChArUco board where it stood; save "
            "vision.surface and the measured box size")
        self.surface_btn.clicked.connect(self._open_surface_measurement)
        workflows.addWidget(self.surface_btn, 1)
        self.surface_dialog = SurfaceBoxCalibrationDialog(self)

        self.home_box_btn = S.touch_button(
            "Create Home Box", S.BLUE, height=42, font_px=13)
        self.home_box_btn.setToolTip(
            "open the camera and X/Y/Z jog controls, then store the box "
            "reference together with the gripper's pre-pick pose")
        self.home_box_btn.clicked.connect(self._open_home_box)
        workflows.addWidget(self.home_box_btn, 1)
        layout.addLayout(workflows)
        self.home_box_lbl = QLabel("")
        self.home_box_lbl.setWordWrap(True)
        self.home_box_lbl.setStyleSheet(
            f"font-size:{S.fpx(11)}px;color:#59656d;")
        layout.addWidget(self.home_box_lbl)
        self.home_box_dialog = HomeBoxDialog(self)

        layout.addStretch(1)
        self._set_mode(self.mode)
        self._load_size()
        self._write_setup()
        self._show_surface_measurement()
        self._show_home_box()

    # -- camera surface and metric box -----------------------------------
    def _open_surface_measurement(self):
        self.surface_dialog.open_for_measurement()

    def _show_surface_measurement(self):
        vision = self.app.cell.config.vision
        surface = surface_plane(vision.get("surface"))
        if surface is not None:
            text = ("Surface measured: %.0f mm from lens, tilt %.1f deg" %
                    (surface.distance * 1000, np.degrees(surface.tilt)))
            self.surface_btn.setText("✓ Surface + Box")
            self.surface_btn.setStyleSheet(S.solid(S.GREEN, 13))
        else:
            text = "Surface has not been measured"
            self.surface_btn.setText("Measure Surface + Box")
            self.surface_btn.setStyleSheet(S.solid(S.PURPLE, 13))
        self.surface_btn.setToolTip(text)

    # -- home box --------------------------------------------------------
    def _open_home_box(self):
        self.home_box_dialog.open_for_teach()

    def _show_home_box(self):
        store = getattr(self.app, "surface", None)
        if store is not None and store.ready:
            names = sorted(store.references)
            text = ("Ready to teach against the surface map"
                    if not names else "Saved references: %s" % ", ".join(names))
            warn = bool(store.fits_box(self._size()))
        elif self.app.cell.config.vision.get("calibrated"):
            text = "Ready to teach with calibrated camera-to-world"
            warn = False
        else:
            current = self._size()
            names = []
            for name, record in (
                    self.app.cell.config.vision.get(
                        "home_references") or {}).items():
                taught = (record.get("box_size")
                           if isinstance(record, dict) else None)
                if not taught or (len(taught) == len(current)
                                  and max(abs(float(a) - float(b))
                                          for a, b in zip(taught, current))
                                      <= 0.001):
                    names.append(name)
            names.sort()
            text = ("Ready to teach a camera R/F/H box home"
                    if not names else
                    "Box home references: %s" % ", ".join(names))
            warn = False
        self.home_box_lbl.setStyleSheet(
            f"font-size:{S.fpx(11)}px;"
            f"color:{S.AMBER if warn else '#59656d'};")
        self.home_box_lbl.setText(text)

    @staticmethod
    def _pin(label, lines, font_px=11, pad=6):
        """Hold a label to a fixed number of lines.

        Word-wrapped labels grow and shrink with what they say, and everything
        under them moves when they do — an operator reading the pose watches
        the size fields walk down the panel as a `size check` line appears and
        goes. The ones with something under them are given room for the most
        they ever say and keep it.

        The height is worked out from the size the stylesheet asks for, not
        from `fontMetrics()`: a label measured before it is polished is still
        carrying the default font, which is half again as tall and reserves
        half a panel of nothing.
        """
        label.setFixedHeight(int(lines * S.fpx(font_px) * 1.35) + S.sx(pad))
        label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        return label

    def _size_spin(self, minimum=SIZE_MIN):
        spin = QSpinBox()
        spin.setRange(minimum, SIZE_MAX)
        spin.setSingleStep(SIZE_STEP)
        spin.setSuffix(" mm")
        spin.setMinimumHeight(S.sx(34))
        spin.setStyleSheet(S.field())
        spin.setAlignment(Qt.AlignRight)
        # Live while it is turned, saved when it is let go: an operator
        # dialling a size wants the wireframe to follow the dial, and the file
        # on disk wants one write rather than one per digit.
        spin.valueChanged.connect(self._size_changed)
        spin.editingFinished.connect(self._size_settled)
        return spin

    def _size(self):
        """Length, width and height from the three millimetre fields."""
        return (self.length_spin.value() / 1000.0,
                self.width_spin.value() / 1000.0,
                self.height_spin.value() / 1000.0)

    def _size_changed(self):
        # Touching either numeric field is an explicit fixed-size choice.
        # Otherwise auto-size would replace the value again on the next
        # camera frame and make the controls appear to do nothing.
        self._apply_auto_size(False)
        size = self._size()
        self.app.cell.config.vision["box_size"] = list(size)
        self.app.vision.config["box_size"] = size
        # Reach into the running detector rather than restarting the camera:
        # a lens that reopens on every turn of the dial is a lens that spends
        # its time warming up instead of showing what the new size looks like.
        detector = self.app.vision.detector
        if detector is not None:
            detector.box_size = size
            detector.reset()
        self._fill_sizes()
        self._write_setup()
        if hasattr(self, "home_box_lbl"):
            self._show_home_box()

    def _take_up_record(self):
        """Open the record for the crate the fields now name.

        A different crate is a different plane and a different map, so both
        the record being collected and the one a program reads follow the
        size. Done when the number has settled rather than on every digit: a
        spin box passing through 20, 200, 2000 would otherwise open three
        maps, and the one an operator was typing towards is the only one they
        meant.
        """
        if not hasattr(self, "home_box_lbl"):
            return
        self.app.reload_surface()
        self._show_home_box()

    def _size_settled(self):
        size = self._size()
        if size == self._saved_size:
            return
        self._saved_size = size
        self._remember(size)
        self._fill_sizes()
        self.app.cell.config.save_vision()
        self.app.log("box is now %.0f × %.0f × %.0f mm"
                     % tuple(v * 1000 for v in size))
        self._take_up_record()

    def _remembered(self):
        """The openings on file, newest first, in whole millimetres."""
        out = []
        for entry in self.app.cell.config.vision.get("box_sizes") or ():
            try:
                height = entry[2] if len(entry) >= 3 else self._legacy_height
                pair = (int(round(float(entry[0]) * 1000)),
                        int(round(float(entry[1]) * 1000)),
                        int(round(float(height) * 1000)))
            except (IndexError, TypeError, ValueError):
                continue        # a hand-edited line is skipped, not fatal
            if pair not in out:
                out.append(pair)
        return out

    def _remember(self, size):
        """Put a settled size at the top of the list the dropdown offers.

        Settled, not typed: every digit of a number on its way to 200 is a
        valid opening in passing, and a list of those is a record of the
        typing rather than of the boxes.
        """
        pair = tuple(int(round(v * 1000)) for v in size[:3])
        kept = [pair] + [s for s in self._remembered() if s != pair]
        self.app.cell.config.vision["box_sizes"] = [
            [length / 1000.0, width / 1000.0, height / 1000.0]
            for length, width, height in kept[:SIZE_MEMORY]]
        profiles = self.app.cell.config.vision["box_sizes"]
        self.app.vision.config["box_sizes"] = profiles
        detector = self.app.vision.detector
        if detector is not None:
            detector.set_box_sizes(profiles)

    def _fill_sizes(self):
        """Rebuild the dropdown: what is dialled in now, then what came before.

        Rebuilt rather than appended to, so the line showing is always the
        size the solver was actually handed — a list that lags the fields is a
        list that names one box while the wireframe is drawn round another.
        """
        current = (self.length_spin.value(), self.width_spin.value(),
                   self.height_spin.value())
        offered = [current] + [s for s in self._remembered() if s != current]
        self.size_combo.blockSignals(True)
        self.size_combo.clear()
        self.size_combo.addItem("Auto (depth)", AUTO_SIZE)
        for length, width, height in offered[:SIZE_MEMORY]:
            self.size_combo.addItem("%d × %d × %d mm" %
                                    (length, width, height),
                                    (length, width, height))
        self.size_combo.setCurrentIndex(
            0 if self.app.cell.config.vision.get("auto_size", False) else 1)
        self.size_combo.blockSignals(False)

    def _size_picked(self, index):
        """A size chosen off the list, which is a size already settled on."""
        pair = self.size_combo.itemData(index)
        if pair == AUTO_SIZE:
            self._apply_auto_size(True)
            self._fill_sizes()
            self._write_setup()
            self.app.cell.config.save_vision()
            self.app.log("box opening size is now automatic (depth)")
            return
        if not isinstance(pair, (tuple, list)) or len(pair) != 3:
            return
        self._apply_auto_size(False)
        for spin, millimetres in zip(
                (self.length_spin, self.width_spin, self.height_spin), pair):
            spin.blockSignals(True)
            spin.setValue(int(millimetres))
            spin.blockSignals(False)
        self._size_changed()
        self._size_settled()

    def _apply_auto_size(self, enabled):
        """Switch the running detector and both config copies together."""
        enabled = bool(enabled)
        self.app.cell.config.vision["auto_size"] = enabled
        self.app.vision.config["auto_size"] = enabled
        self.length_spin.setEnabled(not enabled)
        self.width_spin.setEnabled(not enabled)
        self.height_spin.setEnabled(not enabled)
        detector = self.app.vision.detector
        if detector is not None:
            detector.auto_size = enabled
            detector.reset()

    def _row(self, label, widget):
        row = QHBoxLayout()
        row.setSpacing(S.sx(4))
        caption = S.caption(label)
        caption.setFixedWidth(S.sx(52))
        row.addWidget(caption)
        row.addWidget(widget, 1)
        return row

    def _load_size(self):
        """Put the configured opening into the two fields, quietly."""
        size = self.app.cell.config.vision.get("box_size") or (0.60, 0.40, 0.20)
        self._saved_size = tuple(float(v) for v in size)
        for spin, metres in zip(
                (self.length_spin, self.width_spin, self.height_spin), size[:3]):
            spin.blockSignals(True)
            spin.setValue(int(round(float(metres) * 1000)))
            spin.blockSignals(False)
        # The box the cell is set to counts as the one most recently used,
        # even on a panel nobody has typed into yet: without this the size
        # showing at startup is the one size the dropdown cannot get back to
        # once it has been dialled away from. Written to disk by the next
        # save rather than by opening the tab.
        self._remember(self._saved_size)
        enabled = bool(self.app.cell.config.vision.get("auto_size", False))
        self.length_spin.setEnabled(not enabled)
        self.width_spin.setEnabled(not enabled)
        self.height_spin.setEnabled(not enabled)
        self._fill_sizes()

    def _write_setup(self):
        vision = self.app.cell.config.vision
        size = vision.get("box_size") or (0.60, 0.40, 0.20)
        height = float(size[2]) if len(size) > 2 else 0.20
        roi = vision.get("roi")
        # One line, and it stays one: this label has the size fields and the
        # whole surface block under it, and a second line that comes and goes
        # with a ROI walks all of them down the panel.
        self.setup_lbl.setText(
            "%s, wall %.0f mm — %s"
            % ("auto size" if vision.get("auto_size", False) else "fixed size",
               height * 1000,
               "whole picture" if not roi else
               "ROI %d %d %d %d" % tuple(int(v) for v in roi)))

    def _set_mode(self, key):
        self.mode = key
        for name, button in self.mode_btns.items():
            button.setChecked(name == key)
        reading = self.app.vision.latest
        if reading.frame is not None:
            self.view.setPixmap(self._picture(reading))

    def _source_changed(self):
        vision = self.app.cell.config.vision
        vision["source"] = self.source_combo.currentText()
        self.app.vision.config.update(vision)
        if self.app.vision.running:
            self.app.vision.stop()
            self.app.vision.start()
        self.app.log("camera source is now %s" % vision["source"])

    def _toggle_live(self):
        if self.live_btn.isChecked():
            self.app.vision.start()
        else:
            self.app.vision.stop()
        self._show_live()

    def _show_live(self):
        """The button says what pressing it does next, in its own colour.

        Read from the service rather than from the press: a camera another
        process is holding refuses to start, and a button left reading "Stop"
        over a lens that never opened is a button that lies about the state of
        the cell.
        """
        running = self.app.vision.running
        self.live_btn.setChecked(running)
        self.live_btn.setText("■ Stop" if running else "▶ Live")
        self.live_btn.setStyleSheet(
            S.solid(S.RED if running else S.GREEN, 13))

    def _show(self, reading):
        if reading.frame is None:
            self.view.setText(reading.error or "no frame yet")
            self.found_lbl.setText("—")
            return
        self.view.setPixmap(self._picture(reading))
        found = reading.detection
        if found is None:
            self.found_lbl.setText(reading.why_not())
            return
        # A calibrated cell shows robot-world coordinates.  Before that, a
        # measured surface can still provide useful metric coordinates in the
        # picture: right, forward along the surface, and height.  That is not
        # dressed up as world — its origin and yaw still belong to the camera.
        pose, frame_name = found.matrix(), "camera frame"
        if self.app.cell.config.vision.get("calibrated"):
            pose = found.matrix_in(self.app.executor._camera_to_world())
            frame_name = "world - robot base"
        centre = pose[:3, 3]
        if frame_name.startswith("world"):
            # once it is world, it is compared against the arm readouts by eye,
            # so it carries their sign for X as well
            centre = np.array(shown_xyz(centre))
        rotation = pose[:3, :3]
        if not self.app.cell.config.vision.get("calibrated"):
            vision = self.app.cell.config.vision
            relative = image_relative_xyz(
                found, reading.frame, vision.get("surface"),
                vision.get("floor"))
            relative_rotation = image_relative_rotation(
                found, vision.get("surface"), vision.get("floor"))
            if relative is not None and relative_rotation is not None:
                centre = relative
                rotation = relative_rotation
                frame_name = "camera R/F/H"
        rpy = np.degrees(mat_to_rpy(rotation))
        if frame_name == "camera R/F/H":
            # An unmarked rectangle is identical after half a turn.  Report
            # the nearer equivalent so corner labelling cannot make RZ jump
            # between (for example) +2 and -178 degrees.
            rpy[2] = (rpy[2] + 90.0) % 180.0 - 90.0
        # Five lines, always five. Every one of them is unconditional, which
        # is the whole rule: what used to walk the panel about was a line that
        # came and went with whether the size check had anything to report.
        #
        # The height line earns its place by answering the question the XYZ
        # line above it is read as answering and, uncalibrated, does not: that
        # Z is range along the lens axis, so on this cell's oblique view it
        # moves when the box slides and the box never changed height. It stays
        # once the frame above turns into world, because it is measured
        # against the surface the box actually stands on rather than against
        # whatever height the robot's own origin sits at.
        self.found_lbl.setText(
            "XYZ mm  %+7.1f %+7.1f %+7.1f   %s\n"
            "RPY deg %+7.1f %+7.1f %+7.1f\n"
            "%s\n"
            "rmse %.1f px  depth %.0f mm  %s\n"
            "%s" %
            (centre[0] * 1000, centre[1] * 1000, centre[2] * 1000, frame_name,
             rpy[0], rpy[1], rpy[2],
             height_check(found),
             found.reprojection_error, found.depth_center * 1000,
             found.state, size_check(found) or "size check: —"))

    def _picture(self, reading, view=None):
        view = view or self.view
        frame = reading.frame
        rgb = self._composite(frame)
        canvas = QPixmap(view.width(), view.height())
        canvas.fill(QColor("#101418"))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing, True)
        if rgb is None:
            painter.setPen(QPen(QColor("#6b7a82"), 1))
            painter.drawText(8, view.height() // 2,
                             "this source sends no colour")
        else:
            pane = self._scaled(rgb, view)
            left = (canvas.width() - pane.width()) // 2
            top = (canvas.height() - pane.height()) // 2
            painter.drawPixmap(left, top, pane)
            scale = pane.width() / float(frame.depth.shape[1])
            if self.app.cell.config.vision.get("roi"):
                self._draw_roi(painter, reading, scale, left, top)
            if reading.detection is not None:
                self._draw_box(painter, frame, reading.detection,
                               scale, left, top)
        painter.end()
        return canvas

    def _draw_roi(self, painter, reading, scale, left, top):
        """The search window, when a cell has chosen to have one.

        Most do not: the detector searches the whole picture unless
        `vision.roi` says otherwise, and a frame drawn round the whole picture
        is a frame that says nothing.
        """
        roi = (reading.notes.get("roi")
               or self.app.cell.config.vision.get("roi"))
        x1, y1, x2, y2 = [int(round(float(v) * scale)) for v in roi]
        painter.save()
        painter.translate(left, top)
        painter.setPen(QPen(QColor("#ffa500"), 2))
        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        painter.drawText(x1 + 4, y1 + 15, "ROI")
        painter.restore()

    def _draw_box(self, painter, frame, found, scale, left, top):
        pixels = [self._project(frame, point, scale)
                  for point in found.landmarks_3d()]
        if any(point is None for point in pixels):
            return
        painter.save()
        painter.translate(left, top)
        painter.setPen(QPen(QColor(S.GREEN), 2))
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0),
                     (4, 5), (5, 6), (6, 7), (7, 4),
                     (0, 4), (1, 5), (2, 6), (3, 7)):
            painter.drawLine(int(pixels[a][0]), int(pixels[a][1]),
                             int(pixels[b][0]), int(pixels[b][1]))
        # Put the display axes on the middle of the rim opposite the uppermost
        # visible edge rather than in the opening's centre. Conventional
        # colours: +X red, +Y green, +Z blue.
        axis_length = min(found.size) * 0.28
        origin_3d = found.display_axis_origin()
        axis_3d = [origin_3d + found.rotation[:, i] * axis_length
                   for i in range(3)]
        origin = self._project(frame, origin_3d, scale)
        endpoints = [self._project(frame, point, scale) for point in axis_3d]
        if origin is not None:
            ox, oy = map(int, origin)
            for label, endpoint, colour in zip(
                    ("X", "Y", "Z"), endpoints,
                    ("#ff3030", "#28d948", "#3088ff")):
                if endpoint is None:
                    continue
                ex, ey = map(int, endpoint)
                painter.setPen(QPen(QColor(colour), 3))
                painter.drawLine(ox, oy, ex, ey)
                painter.setPen(QPen(QColor(colour), 2))
                painter.drawText(ex + 3, ey - 3, label)
            painter.setBrush(QColor("#ffffff"))
            painter.setPen(QPen(QColor("#202020"), 1))
            painter.drawEllipse(ox - 3, oy - 3, 6, 6)
        painter.restore()

    @staticmethod
    def _project(frame, point, scale):
        if point[2] <= 1e-6:
            return None
        k = frame.intrinsics
        return ((point[0] * k.fx / point[2] + k.cx) * scale,
                (point[1] * k.fy / point[2] + k.cy) * scale)

    def _composite(self, frame):
        colour = self._color_rgb(frame)
        depth = self._depth_rgb(frame.depth)
        if self.mode == "depth" or colour is None:
            return depth
        if self.mode == "lens":
            return colour
        known = frame.depth > 0
        out = colour.astype(np.float32).copy()
        out[known] = (out[known] * (1.0 - BLEND) +
                      depth[known].astype(np.float32) * BLEND)
        return out.astype(np.uint8)

    def _scaled(self, rgb, view=None):
        view = view or self.view
        rgb = np.ascontiguousarray(rgb)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, 3 * width,
                       QImage.Format_RGB888)
        return QPixmap.fromImage(image.copy()).scaled(
            view.width(), view.height(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation)

    @staticmethod
    def _color_rgb(frame):
        if frame.color is None:
            return None
        colour = np.asarray(frame.color)
        return np.dstack([colour] * 3) if colour.ndim == 2 else colour[:, :, ::-1]

    @staticmethod
    def _depth_rgb(depth):
        known = depth > 0
        scaled = np.zeros(depth.shape, dtype=np.uint8)
        if np.any(known):
            ramp = (FAR - np.clip(depth, NEAR, FAR)) / (FAR - NEAR)
            scaled[known] = (ramp[known] * 255).astype(np.uint8)
        rgb = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)[:, :, ::-1]
        rgb[~known] = 0
        return rgb

    def refresh(self):
        self.source_combo.setCurrentText(
            self.app.cell.config.vision.get("source", "sim"))
        self._load_size()
        self._write_setup()
        self._show_surface_measurement()

    def tick(self):
        self._show_live()
        if not self.app.vision.running:
            return
        reading = self.app.vision.latest
        self._show(reading)
        self.surface_dialog.show_reading(reading)
        self.home_box_dialog.show_reading(reading)
        saying = (reading.why_not()
                  if reading.detection is None and reading.frame is not None
                  else "")
        self.note.setText(saying)
        if saying != self._said:
            self._said = saying
            if saying:
                self.app.log("camera: %s" % saying)

    def release(self):
        self.home_box_dialog.jog_panel.release()
