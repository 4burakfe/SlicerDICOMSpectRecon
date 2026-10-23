"""Motion correction window: automatic and manual shifts of the projections, with the sinogram and
linogram before and after the correction and a plot of the shifts."""

import numpy as np
import qt

from . import display
from . import motion

VIEW_SIZE = 300
PLOT_HEIGHT = 170
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
                 pixelMm=None, initial=None, log=print):
        self.log = log
        self.slots = slots
        self.slotDetectors = np.asarray(slotDetectors, dtype=int)
        self.angles = np.asarray(angles, dtype=float)
        self.times = motion.timeIndex(self.slotDetectors)
        self.pixelMm = pixelMm
        self.stack = motion.slotStack(frames, slots, energyWindows, peakWindow)
        self.order = np.argsort(self.angles, kind="stable")  # sinogram rows by angle
        n = self.stack.shape[0]
        if initial is not None and len(initial[0]) == n:
            self.dx, self.dz = np.array(initial[0], dtype=float), np.array(initial[1], dtype=float)
        else:
            self.dx, self.dz = np.zeros(n), np.zeros(n)
        self.corrected = None
        self.renderer = display.PixmapRenderer(VIEW_SIZE)
        self.plotRenderer = display.PixmapRenderer(2 * VIEW_SIZE + 10)
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
        for column, title in enumerate(("Before", "After")):
            header = qt.QLabel(f"<b>{title}</b>")
            header.setAlignment(qt.Qt.AlignCenter)
            views.addWidget(header, 0, column)
            for row, kind in ((1, "sinogram"), (2, "linogram")):
                view = qt.QLabel()
                view.setFixedSize(VIEW_SIZE, VIEW_SIZE)
                view.setAlignment(qt.Qt.AlignCenter)
                view.setStyleSheet("background-color: black;")
                views.addWidget(view, row, column)
                self.views[(title, kind)] = view
        rowLayout = qt.QHBoxLayout()
        rowLayout.addWidget(qt.QLabel("Sinogram row:"))
        self.rowSlider = qt.QSlider(qt.Qt.Horizontal)
        self.rowSlider.setRange(0, self.stack.shape[1] - 1)
        rowSums = self.stack.sum(axis=(0, 2))
        self.rowSlider.setValue(int(np.argmax(rowSums)) if rowSums.size else 0)
        self.rowSlider.connect("valueChanged(int)", lambda value: self._refreshImages())
        rowLayout.addWidget(self.rowSlider, 1)
        self.rowLabel = qt.QLabel("")
        rowLayout.addWidget(self.rowLabel)
        views.addLayout(rowLayout, 3, 0, 1, 2)
        caption = qt.QLabel("Top: sinogram of the chosen row, bottom: linogram (axial profile, left = row 1). "
                            "Angles run from top to bottom; gates are summed.")
        caption.setWordWrap(True)
        caption.setStyleSheet("color: gray;")
        views.addWidget(caption, 4, 0, 1, 2)
        self.plotView = qt.QLabel()
        self.plotView.setFixedSize(2 * VIEW_SIZE + 10, PLOT_HEIGHT)
        views.addWidget(self.plotView, 5, 0, 1, 2)
        legend = qt.QLabel("Shift per view (pixels) in acquisition order: <span style='color:#ff9628'>"
                           "lateral dx</span>, <span style='color:#3cc8ff'>axial dz</span>; grey lines at "
                           "0 and &plusmn;1 pixel, white lines between detectors.")
        legend.setWordWrap(True)
        views.addWidget(legend, 6, 0, 1, 2)

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
            shiftPlot(self.dx, self.dz, self.slotDetectors, 2 * VIEW_SIZE + 10)))
        text = motion.describe(self.dx, self.dz)
        if self.pixelMm and text != "None":
            text += f"; 1 px = {self.pixelMm:.2f} mm"
        self.statusLabel.setText("Correction: " + text)

    def _refreshImages(self):
        row = self.rowSlider.value
        self.rowLabel.setText(f"{row + 1} / {self.stack.shape[1]}")
        before, after = self.stack[self.order], self.corrected[self.order]
        sinoMax = float(before[:, row, :].max()) or 1.0
        linoMax = float(before.sum(axis=2).max()) or 1.0
        for title, data in (("Before", before), ("After", after)):
            sinogram = _fitTo(data[:, row, :], VIEW_SIZE)
            linogram = _fitTo(data.sum(axis=2), VIEW_SIZE)
            self.views[(title, "sinogram")].setPixmap(self.renderer.toPixmap(display.grayscale(sinogram, sinoMax)))
            self.views[(title, "linogram")].setPixmap(self.renderer.toPixmap(display.grayscale(linogram, linoMax)))

    def exec(self):
        try:
            accepted = self.dialog.exec_() if hasattr(self.dialog, "exec_") else self.dialog.exec()
            return (self.dx.copy(), self.dz.copy()) if accepted else None
        finally:
            self.renderer.cleanup()
            self.plotRenderer.cleanup()
