"""Motion correction window: automatic and manual shifts of the projections. The original projections
(cine), sinogram and linogram are shown on the top row, the motion corrected ones below, with a plot
of the shifts."""

import numpy as np
import qt

from . import display
from . import motion

VIEW_SIZE = 260
PLOT_HEIGHT = 170
PLOT_WIDTH = 3 * VIEW_SIZE + 20
CINE_INTERVAL_MS = 120
DX_COLOUR = (255, 150, 40)    # lateral
DZ_COLOUR = (60, 200, 255)    # axial


def _fitTo(image, size):
    return display.resampleCubicTo(image, size, size, 0.0)


def shiftPlot(dx, dz, slotDetectors, width, height=PLOT_HEIGHT):
    """RGB plot of the shifts (pixels) over the views in acquisition order: dx orange, dz blue,
    zero and +-1 pixel lines in grey, a white line between detectors."""
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    n = len(dx)
    if n == 0:
        return canvas
    limit = max(2.0, float(np.max(np.abs(np.concatenate([dx, dz])))) * 1.15)
    margin = 6

    def yOf(value):
        return np.clip(np.round((height - 1) / 2.0 - value / limit * ((height - 1) / 2.0 - margin)),
                       0, height - 1).astype(int)

    def xOf(index):
        return np.clip(np.round(margin + index * (width - 1 - 2 * margin) / max(1, n - 1)),
                       0, width - 1).astype(int)

    for value, grey in ((0.0, 110), (1.0, 60), (-1.0, 60)):
        canvas[yOf(np.array(value)), :] = grey
    for i in range(1, n):
        if slotDetectors[i] != slotDetectors[i - 1]:
            canvas[:, xOf(np.array(i - 0.5))] = 200
    for values, colour in ((dz, DZ_COLOUR), (dx, DX_COLOUR)):
        xs = np.linspace(0, n - 1, (width - 2 * margin) * 2)
        ys = np.interp(xs, np.arange(n), values)
        px, py = xOf(xs), yOf(ys)
        for offset in (-1, 0, 1):
            canvas[np.clip(py + offset, 0, height - 1), px] = colour
        canvas[yOf(np.asarray(values)), xOf(np.arange(n))] = (255, 255, 255)
    return canvas


class MotionCorrectionWindow:
    """Modal window. exec() returns (dx, dz) per slot when the user accepts, else None."""

    def __init__(self, parent, frames, slots, energyWindows, peakWindow, angles, slotDetectors,
                 clockwise=True, pixelMm=None, initial=None, log=print):
        self.log = log
        self.slots = slots
        self.slotDetectors = np.asarray(slotDetectors, dtype=int)
        self.angles = np.asarray(angles, dtype=float)
        self.times = motion.timeIndex(self.slotDetectors)
        self.pixelMm = pixelMm
        self.stack = motion.slotStack(frames, slots, energyWindows, peakWindow)
        self.order = motion.displayOrder(self.angles, clockwise)  # views along the rotation
        n = self.stack.shape[0]
        if initial is not None and len(initial[0]) == n:
            self.dx, self.dz = np.array(initial[0], dtype=float), np.array(initial[1], dtype=float)
        else:
            self.dx, self.dz = np.zeros(n), np.zeros(n)
        self.corrected = None
        self.petColors = display.petColorTable()
        self.renderer = display.PixmapRenderer(VIEW_SIZE)
        self.plotRenderer = display.PixmapRenderer(PLOT_WIDTH)
        self._build(parent)
        self._refresh()

    # ---------------- UI ----------------
    def _build(self, parent):
        self.dialog = qt.QDialog(parent)
        self.dialog.setWindowTitle("Motion correction")
        self.dialog.setModal(True)
        layout = qt.QHBoxLayout(self.dialog)

        views = qt.QGridLayout()
        layout.addLayout(views, 1)
        self.views = {}
        for column, title in enumerate(("Projection", "Sinogram", "Linogram")):
            header = qt.QLabel(f"<b>{title}</b>")
            header.setAlignment(qt.Qt.AlignCenter)
            views.addWidget(header, 0, column + 1)
        for row, (kind, title) in enumerate((("original", "Original"), ("corrected", "Motion<br>corrected"))):
            label = qt.QLabel(f"<b>{title}</b>")
            label.setAlignment(qt.Qt.AlignCenter)
            views.addWidget(label, row + 1, 0)
            for column, image in enumerate(("projection", "sinogram", "linogram")):
                view = qt.QLabel()
                view.setFixedSize(VIEW_SIZE, VIEW_SIZE)
                view.setAlignment(qt.Qt.AlignCenter)
                view.setStyleSheet("background-color: black;")
                views.addWidget(view, row + 1, column + 1)
                self.views[(kind, image)] = view

        sliders = qt.QGridLayout()
        self.cineCheck = qt.QCheckBox("Cine")
        self.cineCheck.setChecked(True)
        self.cineCheck.setToolTip("Play the projections along the rotation")
        self.cineCheck.connect("toggled(bool)", self._onCineToggled)
        sliders.addWidget(self.cineCheck, 0, 0)
        self.projSlider = qt.QSlider(qt.Qt.Horizontal)
        self.projSlider.setRange(0, self.stack.shape[0] - 1)
        self.projSlider.connect("valueChanged(int)", lambda value: self._refreshProjections())
        self.projSlider.connect("sliderPressed()", lambda: self.cineCheck.setChecked(False))
        sliders.addWidget(self.projSlider, 0, 1)
        self.projLabel = qt.QLabel("")
        sliders.addWidget(self.projLabel, 0, 2)
        sliders.addWidget(qt.QLabel("Sinogram row:"), 1, 0)
        self.rowSlider = qt.QSlider(qt.Qt.Horizontal)
        self.rowSlider.setRange(0, self.stack.shape[1] - 1)
        rowSums = self.stack.sum(axis=(0, 2))
        self.rowSlider.setValue(int(np.argmax(rowSums)) if rowSums.size else 0)
        self.rowSlider.connect("valueChanged(int)", lambda value: self._refreshImages())
        sliders.addWidget(self.rowSlider, 1, 1)
        self.rowLabel = qt.QLabel("")
        sliders.addWidget(self.rowLabel, 1, 2)
        sliders.setColumnStretch(1, 1)
        views.addLayout(sliders, 3, 1, 1, 3)
        caption = qt.QLabel("Projections and sinogram rows run along the rotation, first view at the top of the "
                            "sinogram; the linogram (axial profile) has the views left to right and axial row 1 "
                            "at the top. Photopeak window, gates summed. Drag the yellow line on the projections "
                            "(left mouse button) to pick the sinogram row; every view is scaled to the "
                            "maximum on that line.")
        caption.setWordWrap(True)
        caption.setStyleSheet("color: gray;")
        views.addWidget(caption, 4, 1, 1, 3)
        self.plotView = qt.QLabel()
        self.plotView.setFixedSize(PLOT_WIDTH, PLOT_HEIGHT)
        views.addWidget(self.plotView, 5, 1, 1, 3)
        legend = qt.QLabel("Shift per view (pixels) in acquisition order: <span style='color:#ff9628'>"
                           "lateral dx</span>, <span style='color:#3cc8ff'>axial dz</span>; grey lines at "
                           "0 and &plusmn;1 pixel, white lines between detectors.")
        legend.setWordWrap(True)
        views.addWidget(legend, 6, 1, 1, 3)

        self.lineFilters = []
        for kind in ("original", "corrected"):
            self._installLineDrag(self.views[(kind, "projection")])

        self.cineTimer = qt.QTimer()
        self.cineTimer.setInterval(CINE_INTERVAL_MS)
        self.cineTimer.connect("timeout()", self._cineStep)

        side = qt.QVBoxLayout()
        layout.addLayout(side, 0)

        autoGroup = qt.QGroupBox("Automatic")
        autoForm = qt.QFormLayout(autoGroup)
        self.axialCheck = qt.QCheckBox("Axial (up/down, linogram)")
        self.axialCheck.setChecked(True)
        self.lateralCheck = qt.QCheckBox("Lateral (sideways, centre of mass)")
        self.lateralCheck.setChecked(True)
        self.thresholdSpin = self._doubleSpin(0.0, 5.0, 0.1, motion.DEFAULT_THRESHOLD, " px",
                                              "Shifts smaller than this are left alone")
        self.smoothingSpin = qt.QSpinBox()
        self.smoothingSpin.setRange(1, 15)
        self.smoothingSpin.setSingleStep(2)
        self.smoothingSpin.setValue(motion.DEFAULT_SMOOTHING)
        self.smoothingSpin.setToolTip("Axial: median filter over this many views in acquisition time, so a "
                                      "single noisy view is not moved")
        self.maxShiftSpin = self._doubleSpin(1.0, 30.0, 1.0, motion.DEFAULT_MAX_SHIFT, " px",
                                             "Largest axial shift searched")
        detectButton = qt.QPushButton("Detect")
        detectButton.setToolTip("Replace the shifts with the automatically detected ones")
        detectButton.connect("clicked()", self.onDetect)
        autoForm.addRow(self.axialCheck)
        autoForm.addRow(self.lateralCheck)
        autoForm.addRow("Threshold:", self.thresholdSpin)
        autoForm.addRow("Smoothing:", self.smoothingSpin)
        autoForm.addRow("Max axial:", self.maxShiftSpin)
        autoForm.addRow(detectButton)
        side.addWidget(autoGroup)

        manualGroup = qt.QGroupBox("Manual")
        manualForm = qt.QFormLayout(manualGroup)
        nTimes = int(self.times.max()) + 1
        self.detectorCombo = qt.QComboBox()
        self.detectorCombo.addItem("All")
        for detector in np.unique(self.slotDetectors):
            self.detectorCombo.addItem(f"Detector {int(detector)}")
        self.detectorCombo.setToolTip("Axial movements are the same on every detector; a sideways movement "
                                      "shows differently on each, so shift those per detector")
        self.fromSpin = qt.QSpinBox()
        self.fromSpin.setRange(1, nTimes)
        self.toSpin = qt.QSpinBox()
        self.toSpin.setRange(1, nTimes)
        self.toSpin.setValue(nTimes)
        for spin in (self.fromSpin, self.toSpin):
            spin.setToolTip(f"Time step 1-{nTimes}: the n-th view of every detector is acquired at the same time")
        self.manualDxSpin = self._doubleSpin(-30.0, 30.0, 0.5, 0.0, " px", "Lateral shift added (+ = right)")
        self.manualDzSpin = self._doubleSpin(-30.0, 30.0, 0.5, 0.0, " px", "Axial shift added (+ = down)")
        addButton = qt.QPushButton("Add to these views")
        addButton.connect("clicked()", self.onManual)
        manualForm.addRow("Detector:", self.detectorCombo)
        manualForm.addRow("From step:", self.fromSpin)
        manualForm.addRow("To step:", self.toSpin)
        manualForm.addRow("dx:", self.manualDxSpin)
        manualForm.addRow("dz:", self.manualDzSpin)
        manualForm.addRow(addButton)
        side.addWidget(manualGroup)

        resetButton = qt.QPushButton("Reset (no correction)")
        resetButton.connect("clicked()", self.onReset)
        side.addWidget(resetButton)
        self.statusLabel = qt.QLabel("")
        self.statusLabel.setWordWrap(True)
        side.addWidget(self.statusLabel)
        side.addStretch(1)

        buttons = qt.QHBoxLayout()
        okButton = qt.QPushButton("Use correction")
        okButton.setDefault(True)
        okButton.setToolTip("Reconstruct from the corrected projections (all energy windows and gates)")
        okButton.connect("clicked()", self.dialog.accept)
        cancelButton = qt.QPushButton("Cancel")
        cancelButton.connect("clicked()", self.dialog.reject)
        buttons.addWidget(okButton)
        buttons.addWidget(cancelButton)
        side.addLayout(buttons)

    def _installLineDrag(self, view):
        from .dialog import LineDragFilter
        rows = self.stack.shape[1]

        def move(y):
            self.rowSlider.setValue(int(min(rows - 1, max(0, y / VIEW_SIZE * rows))))
        try:
            lineFilter = LineDragFilter(move)
            view.installEventFilter(lineFilter)
            self.lineFilters.append(lineFilter)  # keep alive
        except Exception:
            self.log("Reference line dragging not available")

    @staticmethod
    def _doubleSpin(minimum, maximum, step, value, suffix, tooltip):
        spin = qt.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(1)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setToolTip(tooltip)
        return spin

    # ---------------- actions ----------------
    def onDetect(self):
        dx, dz = motion.detectMotion(self.stack, self.angles, self.slotDetectors,
                                     maxShift=self.maxShiftSpin.value, smoothing=self.smoothingSpin.value,
                                     threshold=self.thresholdSpin.value, axial=self.axialCheck.checked,
                                     lateral=self.lateralCheck.checked)
        self.dx, self.dz = dx, dz
        self.log(f"Motion detection: {motion.describe(dx, dz)}")
        self._refresh()

    def onManual(self):
        first, last = sorted((self.fromSpin.value - 1, self.toSpin.value - 1))
        mask = (self.times >= first) & (self.times <= last)
        if self.detectorCombo.currentIndex > 0:
            detector = int(np.unique(self.slotDetectors)[self.detectorCombo.currentIndex - 1])
            mask &= self.slotDetectors == detector
        self.dx[mask] += self.manualDxSpin.value
        self.dz[mask] += self.manualDzSpin.value
        self._refresh()

    def onReset(self):
        self.dx[:] = 0.0
        self.dz[:] = 0.0
        self._refresh()

    # ---------------- display ----------------
    def _refresh(self):
        self.corrected = motion.correctedStack(self.stack, self.dx, self.dz)
        self._refreshImages()
        self.plotView.setPixmap(self.plotRenderer.toPixmap(
            shiftPlot(self.dx, self.dz, self.slotDetectors, PLOT_WIDTH)))
        text = motion.describe(self.dx, self.dz)
        if self.pixelMm and text != "None":
            text += f"; 1 px = {self.pixelMm:.2f} mm"
        self.statusLabel.setText("Correction: " + text)

    def _colour(self, image, vmax):
        """PET-coloured, stretched to the square view (aspect not kept)."""
        return self.renderer.toPixmap(display.colorize(_fitTo(image, VIEW_SIZE), vmax or 1.0, self.petColors, 1.0))

    def _refreshImages(self):
        row = self.rowSlider.value
        self.rowLabel.setText(f"{row + 1} / {self.stack.shape[1]}")
        original, corrected = self.stack[self.order], self.corrected[self.order]
        line = original[:, row, :]  # the reference line on every projection sets the display maxima
        sinoMax = float(line.max())
        linoMax = float(line.sum(axis=1).max())
        for kind, data in (("original", original), ("corrected", corrected)):
            self.views[(kind, "sinogram")].setPixmap(self._colour(data[:, row, :], sinoMax))
            self.views[(kind, "linogram")].setPixmap(self._colour(data.sum(axis=2).T, linoMax))
        self._refreshProjections()

    def _refreshProjections(self):
        index = min(self.projSlider.value, self.stack.shape[0] - 1)
        slot = int(self.order[index])
        self.projLabel.setText(f"{index + 1} / {self.stack.shape[0]}  (detector {int(self.slotDetectors[slot])})")
        row, rows = self.rowSlider.value, self.stack.shape[1]
        projMax = float(self.stack[:, row, :].max()) or float(self.stack.max()) or 1.0
        for kind, data in (("original", self.stack), ("corrected", self.corrected)):
            rgb = display.grayscale(_fitTo(data[slot], VIEW_SIZE), projMax)
            display.drawLine(rgb, row, rows)
            self.views[(kind, "projection")].setPixmap(self.renderer.toPixmap(rgb))

    def _onCineToggled(self, on):
        if on:
            self.cineTimer.start()
        else:
            self.cineTimer.stop()

    def _cineStep(self):
        self.projSlider.blockSignals(True)
        self.projSlider.setValue((self.projSlider.value + 1) % self.stack.shape[0])
        self.projSlider.blockSignals(False)
        self._refreshProjections()

    def exec(self):
        if self.cineCheck.checked:
            self.cineTimer.start()
        try:
            accepted = self.dialog.exec_() if hasattr(self.dialog, "exec_") else self.dialog.exec()
            return (self.dx.copy(), self.dz.copy()) if accepted else None
        finally:
            self.cineTimer.stop()
            self.renderer.cleanup()
            self.plotRenderer.cleanup()
