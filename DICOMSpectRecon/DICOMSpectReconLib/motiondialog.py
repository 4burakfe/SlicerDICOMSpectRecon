"""Motion correction window: automatic and manual shifts of the projections. A fast reconstruction of the
axial slice at the reference line, the projections (cine), the summed sinogram and the linogram are shown
on the top row, the motion corrected ones below, with a plot of the shifts."""

import numpy as np
import qt

from . import display
from . import motion

VIEW_SIZE = 240
PLOT_HEIGHT = 170
PLOT_WIDTH = 4 * VIEW_SIZE + 30
RECON_DELAY_MS = 1000         # the slice is reconstructed after 1 s without changes
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
                 clockwise=True, pixelMm=None, initial=None, reconParams=None, log=print):
        self.log = log
        # the main window's reconstruction settings (algorithm, iterations, subsets, prior, filter)
        self.reconParams = reconParams or {"algorithm": "OSEM", "iters": 3, "subsets": 6, "filter": "None"}
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
        self.sliceRecon = None        # motion.SliceReconstructor, made on first use
        self.originalSlice = None     # (row, image) of the uncorrected slice
        self.sinoImages = {}          # (kind, "sinogram" / "linogram") -> RGB without the frame marker
        self.petColors = display.petColorTable()
        self.renderer = display.PixmapRenderer(VIEW_SIZE)
        self.plotRenderer = display.PixmapRenderer(PLOT_WIDTH)
        self._build(parent)
        self._newManualBase()
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
        for column, title in enumerate(("Axial slice (fast)", "Projection", "Summed sinogram", "Linogram")):
            header = qt.QLabel(f"<b>{title}</b>")
            header.setAlignment(qt.Qt.AlignCenter)
            views.addWidget(header, 0, column + 1)
        for row, (kind, title) in enumerate((("original", "Original"), ("corrected", "Motion<br>corrected"))):
            label = qt.QLabel(f"<b>{title}</b>")
            label.setAlignment(qt.Qt.AlignCenter)
            views.addWidget(label, row + 1, 0)
            for column, image in enumerate(("axial", "projection", "sinogram", "linogram")):
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
        sliders.addWidget(qt.QLabel("Line:"), 1, 0)
        self.rowSlider = qt.QSlider(qt.Qt.Horizontal)
        self.rowSlider.setRange(0, self.stack.shape[1] - 1)
        self.rowSlider.setValue(self.stack.shape[1] // 2)   # start at the middle slice
        self.rowSlider.connect("valueChanged(int)", lambda value: self._refreshImages())
        sliders.addWidget(self.rowSlider, 1, 1)
        self.rowLabel = qt.QLabel("")
        sliders.addWidget(self.rowLabel, 1, 2)
        sliders.setColumnStretch(1, 1)
        views.addLayout(sliders, 3, 1, 1, 4)
        caption = qt.QLabel("Drag the yellow line on the projections (left mouse button) or turn the mouse "
                            "wheel over the axial, sinogram or linogram views to pick the axial level; the wheel "
                            "over the projections steps through them. The axial slice at the line is "
                            "reconstructed 1 s after the last change with the main window's settings "
                            f"({self.reconDescription()}), original and corrected on the same scale. "
                            "The sinogram is all rows summed along the axis, views along the rotation from "
                            "the top; the linogram has the views left to right and axial row 1 at the top. "
                            "Photopeak window, gates summed.")
        caption.setWordWrap(True)
        caption.setStyleSheet("color: gray;")
        views.addWidget(caption, 4, 1, 1, 4)
        self.plotView = qt.QLabel()
        self.plotView.setFixedSize(PLOT_WIDTH, PLOT_HEIGHT)
        views.addWidget(self.plotView, 5, 1, 1, 4)
        legend = qt.QLabel("Shift per view (pixels) in acquisition order: <span style='color:#ff9628'>"
                           "lateral dx</span>, <span style='color:#3cc8ff'>axial dz</span>; grey lines at "
                           "0 and &plusmn;1 pixel, white lines between detectors.")
        legend.setWordWrap(True)
        views.addWidget(legend, 6, 1, 1, 4)

        self.lineFilters = []
        self.wheelFilters = []
        for kind in ("original", "corrected"):
            self._installLineDrag(self.views[(kind, "projection")])
            self._installWheel(self.views[(kind, "projection")], self.projSlider, stopsCine=True)
            # wheel up = next slice up (superior), as in the main window: one row up the projection
            self._installWheel(self.views[(kind, "axial")], self.rowSlider, reverse=True)
            for image in ("sinogram", "linogram"):
                self._installWheel(self.views[(kind, image)], self.rowSlider)

        self.reconTimer = qt.QTimer()
        self.reconTimer.setSingleShot(True)
        self.reconTimer.setInterval(RECON_DELAY_MS)
        self.reconTimer.connect("timeout()", self._reconstructSlice)
        self.reconStep = qt.QTimer()  # runs the slice one row at a time between window events
        self.reconStep.setInterval(0)
        self.reconStep.connect("timeout()", self._stepSlice)
        self.reconJob = None          # (row, request number, step generators, results) of the running slice
        self.reconRequest = 0

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
        self.lateralCheck.setChecked(False)   # off by default: attenuation bends the centre of mass curve
        self.lateralCheck.setToolTip("Sudden sideways jumps of the centre of mass between two views; smooth "
                                     "changes (attenuation, activity entering the field of view) are left alone")
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
        self.modeCombo = qt.QComboBox()
        self.modeCombo.addItems(["Single view", "Range"])
        self.modeCombo.setToolTip("Shift one time step, or every step from - to")
        self.manualDxSpin = self._doubleSpin(-30.0, 30.0, 0.1, 0.0, " px",
                                             "Lateral shift added to the chosen views (+ = right), applied live")
        self.manualDzSpin = self._doubleSpin(-30.0, 30.0, 0.1, 0.0, " px",
                                             "Axial shift added to the chosen views (+ = down), applied live")
        self.manualLabel = qt.QLabel("")
        self.manualLabel.setWordWrap(True)
        self.manualLabel.setStyleSheet("color: gray;")
        self.modeCombo.connect("currentIndexChanged(int)", lambda index: self._onManualSelection())
        self.fromSpin.connect("valueChanged(int)", lambda value: self._onManualSelection(value - 1))
        self.toSpin.connect("valueChanged(int)", lambda value: self._onManualSelection(value - 1))
        self.detectorCombo.connect("currentIndexChanged(int)", lambda index: self._onManualSelection())
        self.manualDxSpin.connect("valueChanged(double)", lambda value: self._applyManual())
        self.manualDzSpin.connect("valueChanged(double)", lambda value: self._applyManual())
        manualForm.addRow("Mode:", self.modeCombo)
        manualForm.addRow("Detector:", self.detectorCombo)
        manualForm.addRow("Step / from:", self.fromSpin)
        manualForm.addRow("To step:", self.toSpin)
        manualForm.addRow("dx:", self.manualDxSpin)
        manualForm.addRow("dz:", self.manualDzSpin)
        manualForm.addRow(self.manualLabel)
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

    def _installWheel(self, view, slider, stopsCine=False, reverse=False):
        from .dialog import WheelScrollFilter

        def step(direction):
            if stopsCine:
                self.cineCheck.setChecked(False)
            if reverse:
                direction = -direction
            slider.setValue(max(slider.minimum, min(slider.maximum, slider.value + direction)))
        try:
            wheelFilter = WheelScrollFilter(step)
            view.installEventFilter(wheelFilter)
            self.wheelFilters.append(wheelFilter)  # keep alive
        except Exception:
            self.log("Mouse wheel scrolling not available")

    def _showStep(self, step):
        """Stop the cine and show the projection of a manual time step (on the chosen detector)."""
        self.cineCheck.setChecked(False)
        detectors = np.unique(self.slotDetectors)
        index = self.detectorCombo.currentIndex
        detector = int(detectors[index - 1]) if index > 0 else int(detectors[0])
        slots = np.where((self.slotDetectors == detector) & (self.times == step))[0]
        if slots.size:
            self.projSlider.setValue(int(np.where(self.order == slots[0])[0][0]))

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
        self._newManualBase()
        self._refresh()

    # Manual: dx / dz are added live to the chosen views on top of the correction they had when they
    # were chosen; choosing other views keeps what was added and starts again from 0.
    def _manualMask(self):
        single = self.modeCombo.currentIndex == 0
        self.toSpin.setEnabled(not single)
        first = self.fromSpin.value - 1
        last = first if single else self.toSpin.value - 1
        first, last = sorted((first, last))
        mask = (self.times >= first) & (self.times <= last)
        if self.detectorCombo.currentIndex > 0:
            detector = int(np.unique(self.slotDetectors)[self.detectorCombo.currentIndex - 1])
            mask &= self.slotDetectors == detector
        return mask

    def _newManualBase(self):
        self.manualBase = (self.dx.copy(), self.dz.copy())
        self.manualMask = self._manualMask()
        for spin in (self.manualDxSpin, self.manualDzSpin):
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)
        self._updateManualLabel()

    def _updateManualLabel(self):
        selected = np.where(self.manualMask)[0]
        if not selected.size:
            self.manualLabel.setText("No views chosen")
            return
        dx, dz = self.dx[selected], self.dz[selected]
        if selected.size == 1:
            self.manualLabel.setText(f"View {int(selected[0]) + 1}: dx {dx[0]:+.2f} px, dz {dz[0]:+.2f} px")
        else:
            self.manualLabel.setText(f"{selected.size} views: dx {dx.min():+.2f} to {dx.max():+.2f} px, "
                                     f"dz {dz.min():+.2f} to {dz.max():+.2f} px")

    def _onManualSelection(self, step=None):
        self._newManualBase()
        self._showStep(self.fromSpin.value - 1 if step is None else step)

    def _applyManual(self):
        dx, dz = self.manualBase[0].copy(), self.manualBase[1].copy()
        dx[self.manualMask] += self.manualDxSpin.value
        dz[self.manualMask] += self.manualDzSpin.value
        self.dx, self.dz = dx, dz
        self._updateManualLabel()
        self._refresh()

    def onReset(self):
        self.dx[:] = 0.0
        self.dz[:] = 0.0
        self._newManualBase()
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

    def _colour(self, image, vmax, lineRow=None, rows=None, asArray=False):
        """PET-coloured, stretched to the square view (aspect not kept), with the reference line if given."""
        rgb = display.colorize(_fitTo(image, VIEW_SIZE), vmax or 1.0, self.petColors, 1.0)
        if lineRow is not None:
            display.drawLine(rgb, lineRow, rows)
        return rgb if asArray else self.renderer.toPixmap(rgb)

    def _showFrameMarkers(self):
        """The projection shown, as a line on the sinograms (across) and linograms (up and down)."""
        index, count = min(self.projSlider.value, self.stack.shape[0] - 1), self.stack.shape[0]
        for (kind, image), rgb in self.sinoImages.items():
            marked = display.drawFrameMarker(rgb.copy(), index, count, vertical=image == "linogram")
            self.views[(kind, image)].setPixmap(self.renderer.toPixmap(marked))

    def _refreshImages(self):
        row = self.rowSlider.value
        self.rowLabel.setText(f"{row + 1} / {self.stack.shape[1]}")
        original, corrected = self.stack[self.order], self.corrected[self.order]
        rows = self.stack.shape[1]
        line = original[:, row, :]  # the reference line on every projection sets the display maxima
        sinoMax = float(original.sum(axis=1).max())
        linoMax = float(line.sum(axis=1).max())
        for kind, data in (("original", original), ("corrected", corrected)):
            self.sinoImages[(kind, "sinogram")] = self._colour(data.sum(axis=1), sinoMax, asArray=True)
            self.sinoImages[(kind, "linogram")] = self._colour(data.sum(axis=2).T, linoMax, row, rows, asArray=True)
        self._refreshProjections()
        self._scheduleRecon()

    # ---------------- fast axial slice ----------------
    def _scheduleRecon(self):
        """(Re)start the 1 s wait: every change (line, shifts) restarts it, so nothing is computed
        while the user is still dragging or typing."""
        self.reconRequest += 1  # a slice still running for older settings is not shown
        self.reconTimer.start()

    def _reconstructSlice(self):
        """Start the slice. Slicer's Python does not run threads while the window waits for events,
        so the work is done in small steps (one row each) from a zero-interval timer instead: the
        window stays responsive and a newer change abandons the running slice."""
        row, params = self.rowSlider.value, self.reconParams
        try:
            if self.sliceRecon is None:
                subsets = params.get("subsets", 1) if params.get("algorithm") != "FBP" else 1
                self.sliceRecon = motion.SliceReconstructor(self.angles, self.stack.shape[2], subsets)
        except Exception as e:
            self.log(f"Fast slice reconstruction failed: {e}")
            return
        steps, results = [], {}
        if self.originalSlice is not None and self.originalSlice[0] == row:
            results["original"] = self.originalSlice[1]
        else:
            steps.append(("original", motion.reconstructSliceSteps(self.sliceRecon, self.stack, row, params,
                                                                   self.pixelMm, self.log)))
        steps.append(("corrected", motion.reconstructSliceSteps(self.sliceRecon, self.corrected, row, params,
                                                                self.pixelMm, self.log)))
        self.reconJob = (row, self.reconRequest, steps, results)
        self.reconStep.start()

    def _stepSlice(self):
        if self.reconJob is None:
            self.reconStep.stop()
            return
        row, request, steps, results = self.reconJob
        if request != self.reconRequest:   # settings changed: the newer request starts after its wait
            self.reconJob = None
            self.reconStep.stop()
            return
        try:
            kind, generator = steps[0]
            result = next(generator)
        except StopIteration:
            result = None
            steps.pop(0)
        except Exception as e:
            self.reconJob = None
            self.reconStep.stop()
            self.log(f"Fast slice reconstruction failed: {e}")
            return
        if result is not None:
            results[kind] = result
            steps.pop(0)
            if kind == "original":
                self.originalSlice = (row, result)
        if steps:
            return
        self.reconJob = None
        self.reconStep.stop()
        if "original" in results and "corrected" in results:
            self._showSlice(row, results["original"], results["corrected"])

    def _showSlice(self, row, original, corrected):
        vmax = float(original.max()) or float(corrected.max()) or 1.0
        for kind, image in (("original", original), ("corrected", corrected)):
            view = self.views[(kind, "axial")]
            view.setPixmap(self._colour(image, vmax))
            view.setToolTip(f"Axial slice at line {row + 1}: {self.reconDescription()}")

    def reconDescription(self):
        params = self.reconParams
        algorithm = params.get("algorithm", "OSEM")
        if algorithm == "FBP":
            text = f"FBP {params['fbp']['window']}"
        elif algorithm == "BSREM":
            bsrem = params["bsrem"]
            text = f"BSREM {bsrem['iters']} it x {params.get('subsets')} subsets, beta {bsrem['beta']:g}"
        else:
            text = f"OSEM {params.get('iters')} it x {params.get('subsets')} subsets"
        if params.get("filter") not in (None, "None"):
            text += f", {params['filter']} filter"
        return text + "; 2D, no attenuation, scatter or RR"

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
        self._showFrameMarkers()

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
            self.reconTimer.stop()
            self.reconStep.stop()
            self.reconJob = None
            self.renderer.cleanup()
            self.plotRenderer.cleanup()
