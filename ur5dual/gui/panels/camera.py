"""Camera tab for the known-size open-box detector."""

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget,
)

from ...geometry.kinematics import mat_to_rpy
from ...vision.detect import size_check
from .. import style as S


VIEW_W, VIEW_H = 308, 231
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
        layout.addWidget(self.found_lbl)

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
        layout.addWidget(self.setup_lbl)

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
        layout.addStretch(1)
        self._set_mode(self.mode)
        self._load_size()
        self._write_setup()

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
        self.setup_lbl.setText(
            "%s — wall %.0f mm deep\n"
            "%s" % ("automatic size from depth" if
                     vision.get("auto_size", False) else "fixed box size",
                     height * 1000,
                    "searching the whole picture" if not roi else
                    "ROI %d %d %d %d — orange frame"
                    % tuple(int(v) for v in roi)))

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
        rpy = np.degrees(mat_to_rpy(found.rotation))
        self.found_lbl.setText(
            "6D POSE — camera frame\n"
            "XYZ mm  %+7.1f %+7.1f %+7.1f\n"
            "RPY deg %+7.1f %+7.1f %+7.1f\n"
            "rmse %.1f px  depth %.0f mm  %s\n"
            "%s" %
            (found.centre[0] * 1000, found.centre[1] * 1000,
             found.centre[2] * 1000, rpy[0], rpy[1], rpy[2],
             found.reprojection_error, found.depth_center * 1000,
             found.state, size_check(found)))

    def _picture(self, reading):
        frame = reading.frame
        rgb = self._composite(frame)
        canvas = QPixmap(self.view.width(), self.view.height())
        canvas.fill(QColor("#101418"))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.Antialiasing, True)
        if rgb is None:
            painter.setPen(QPen(QColor("#6b7a82"), 1))
            painter.drawText(8, self.view.height() // 2,
                             "this source sends no colour")
        else:
            pane = self._scaled(rgb)
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
        # Pose axes replace the old centre cross. Conventional colours:
        # +X red, +Y green, +Z blue, all projected from the solved 3D pose.
        axis_length = min(found.size) * 0.28
        origin_3d = found.centre
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

    def _scaled(self, rgb):
        rgb = np.ascontiguousarray(rgb)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, 3 * width,
                       QImage.Format_RGB888)
        return QPixmap.fromImage(image.copy()).scaled(
            self.view.width(), self.view.height(), Qt.KeepAspectRatio,
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

    def tick(self):
        self._show_live()
        if not self.app.vision.running:
            return
        reading = self.app.vision.latest
        self._show(reading)
        saying = (reading.why_not()
                  if reading.detection is None and reading.frame is not None
                  else "")
        self.note.setText(saying)
        if saying != self._said:
            self._said = saying
            if saying:
                self.app.log("camera: %s" % saying)

    def release(self):
        pass
