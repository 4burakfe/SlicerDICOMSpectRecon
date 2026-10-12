"""Interactive preview window shown when a SPECT projection series is loaded."""

import logging
import os
import traceback

import ctk
import numpy as np
import qt
import slicer

from .nmdata import (adjacentScatterWindows, cleanCtFiles, defaultPeakIndex, describeTiming,
                     readAcquisitionInfo, readNmInfo, scatterWindowProblem)
from .reconstruction import (SCALING_MODES, SCALING_VIEWS, ReconstructionAborted, SpectReconLogic,
                             applyFilter, autoScatterWeight, filterDescription, pytomographyVersion,
                             scaleCounts, scatterDescription)
from . import display
from . import quantification as quant

DATETIME_FORMAT = "yyyy-MM-dd HH:mm:ss"
FWHM_TOOLTIP = "Gaussian full width at half maximum (FWHM = 2.355 x sigma)"


def _slider(minimum, maximum, step, decimals, value, suffix="", toolTip=""):
    """Slider with an attached spin box (the usual Slicer parameter control)."""
    slider = ctk.ctkSliderWidget()
    slider.decimals = decimals
    slider.singleStep = step
    slider.pageStep = step
    slider.minimum = minimum
    slider.maximum = maximum
    slider.value = value
    if suffix:
        slider.suffix = suffix
    if toolTip:
        slider.setToolTip(toolTip)
    return slider


def _dateTimeEdit():
    edit = qt.QDateTimeEdit()
    edit.setDisplayFormat(DATETIME_FORMAT)
    edit.setCalendarPopup(True)
    return edit


def setDateTimeEdit(edit, isoText):
    if isoText:
        edit.setDateTime(qt.QDateTime.fromString(str(isoText)[:19], "yyyy-MM-ddTHH:mm:ss"))


def dateTimeFromEdit(edit):
    """Python datetime from a QDateTimeEdit (via text, which is PythonQt-safe)."""
    import datetime
    return datetime.datetime.strptime(edit.dateTime.toString(DATETIME_FORMAT), "%Y-%m-%d %H:%M:%S")


def _shrinkFont(widget, points=1.0):
    """Slightly smaller font for a dense settings panel (children inherit it)."""
    try:
        font = qt.QFont(widget.font)
        if font.pointSizeF() > 0:
            font.setPointSizeF(max(7.0, font.pointSizeF() - points))
        elif font.pixelSize() > 0:
            font.setPixelSize(max(9, font.pixelSize() - 1))
        widget.setFont(font)
    except Exception:
        pass  # cosmetic only


def _group(title):
    """Group box with a tight grid layout."""
    box = qt.QGroupBox(title)
    grid = qt.QGridLayout(box)
    grid.setContentsMargins(6, 4, 6, 4)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(3)
    return box, grid


def _row(*widgets):
    """Widgets side by side with equal width."""
    row = qt.QHBoxLayout()
    row.setSpacing(4)
    for widget in widgets:
        widget.setSizePolicy(qt.QSizePolicy.Preferred, qt.QSizePolicy.Preferred)
        row.addWidget(widget, 1)
    return row


BUTTON_BLUE = "#1f6fd1"
BUTTON_GREEN = "#1b6e2d"
BUTTON_ORANGE = "#e07b00"


def _colorButton(button, color):
    """Solid coloured push button; greyed out when disabled."""
    button.setMinimumHeight(30)
    button.setStyleSheet(
        f"QPushButton {{ background-color: {color}; color: white; font-weight: bold; border: none; "
        f"border-radius: 4px; padding: 4px 10px; }}"
        f"QPushButton:hover {{ border: 2px solid rgba(255, 255, 255, 120); }}"
        f"QPushButton:pressed {{ padding-top: 6px; }}"
        f"QPushButton:disabled {{ background-color: #9a9a9a; color: #e8e8e8; }}")


def _compactCombo():
    """Combo box that does not widen the panel for long entries (full text in the popup/tooltip)."""
    combo = qt.QComboBox()
    combo.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(6)
    return combo


def _collapsibleGroup(title):
    """Checkable group whose contents show only while it is checked.
    Returns (group, body grid)."""
    box = qt.QGroupBox(title)
    box.setCheckable(True)
    box.setChecked(False)
    outer = qt.QVBoxLayout(box)
    outer.setContentsMargins(6, 2, 6, 2)
    body = qt.QWidget()
    grid = qt.QGridLayout(body)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(6)
    grid.setVerticalSpacing(3)
    outer.addWidget(body)
    body.setVisible(False)
    box.connect("toggled(bool)", body.setVisible)
    return box, grid


def scatterMethodName(params):
    if params.get("indexLower") is None:
        return "None"
    return "TEW" if params.get("indexUpper") is not None else "DEW"


def scatterWeightText(weight):
    """Stored form of the dual window scatter weight: 'auto' or the number."""
    return "auto" if weight is None else f"{float(weight):g}"


def sameScatterWeight(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) < 1e-6


class WheelScrollFilter(qt.QObject):
    """Forwards mouse wheel events over a preview panel to a slider."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def eventFilter(self, obj, event):
        if event.type() == qt.QEvent.Wheel:
            try:
                delta = event.angleDelta().y()
            except AttributeError:
                delta = event.delta()
            if delta != 0:
                self.callback(1 if delta > 0 else -1)
            return True
        return False


class BusyDialogGuard(qt.QObject):
    """While a reconstruction runs, Escape and the window's close button stop it
    instead of closing the dialog underneath a running computation."""

    def __init__(self, isBusy, requestStop):
        super().__init__()
        self.isBusy = isBusy
        self.requestStop = requestStop

    def eventFilter(self, obj, event):
        if not self.isBusy():
            return False
        if event.type() == qt.QEvent.Close:
            event.ignore()
            self.requestStop()
            return True
        if event.type() == qt.QEvent.KeyPress and event.key() == qt.Qt.Key_Escape:
            self.requestStop()
            return True
        return False


# ---------------------------------------------------------------------------
# Preview dialog
# ---------------------------------------------------------------------------
class SpectReconDialog:
    # Panel size is computed from the screen at startup; these are the limits.
    # Display windowing is defined in display.py
    MIN_VIEW_SIZE = 150
    MAX_VIEW_SIZE = 460
    SIDE_PANEL_WIDTH = 550

    def __init__(self, inputDcm, title, ctCandidates=None):
        self.inputDcm = inputDcm
        self.ctCandidates = list(ctCandidates or [])
        self.ctFiles = self.ctCandidates[0]["files"] if self.ctCandidates else None
        self.ctMatchType = self.ctCandidates[0]["match"] if self.ctCandidates else ""
        self.ctArray = None          # CT (HU) resampled onto the reconstruction grid
        self.ctGeometryKey = None
        self.poppedCursors = 0
        self.projCineTimer = None
        self.gateCineTimer = None
        self.wheelFilters = []
        self.logic = None
        self.reconImage = None       # unfiltered static recon in native OSEM units (cached)
        self.reconStats = None
        self.reconKey = None
        self.filteredImage = None
        self.appliedParams = None
        self.info = {}
        self.reconArray = None
        self.reconMax = 1.0
        self.gateRawImages = []      # unfiltered per-gate recons in native units (cached)
        self.gateRawStats = []
        self.gateImages = []         # scaled + filtered per-gate recons, handed to Slicer
        self.gateArrays = []
        self.gateMax = 1.0
        self.gateReconKey = None
        self.gateIndex = 0
        self.projArray = None
        self.projMax = 1.0
        self.projDetectors = None
        self.projGates = None
        self.busy = False
        self.abortRequested = False
        self.lastRunFailed = False   # shown in the action panel instead of "settings changed"
        self.acUserSet = False       # true once the user touches the AC checkbox
        self.ctPreviewFailed = False # CT could not be shown in the preview (AC is unaffected)
        self.ctCleaned = False       # ctFiles already filtered/sorted by cleanCtFiles
        self.rawProjection = None    # cached pixel data and frame vectors for the preview
        self._petTable = None        # PET-DICOM colours, read once

        nmInfo = readNmInfo(inputDcm)
        self.energyWindows = nmInfo["windows"]
        self.gateNumbers = nmInfo["gateNumbers"]
        self.headerWarnings = nmInfo["warnings"]
        self.nGates = nmInfo["nGates"]
        self.isGated = self.nGates > 1

        # Camera, isotope, timing and patient data for quantification / SUV
        try:
            self.acquisition = readAcquisitionInfo(inputDcm)
        except Exception as e:
            logging.warning("Could not read acquisition info:\n" + traceback.format_exc())
            self.acquisition = {"warnings": [f"Acquisition info could not be read: {e}"],
                                "windows": self.energyWindows}
        self.calibrations = []
        self.quantLocked = False
        self.units = "counts"

        self.viewSize = self._computeViewSize()
        self.renderer = display.PixmapRenderer(self.viewSize)
        self._buildUI(title)

    def _availableScreenSize(self):
        """Usable screen area, across Qt versions."""
        try:
            geometry = slicer.util.mainWindow().screen().availableGeometry()
            return geometry.width(), geometry.height()
        except Exception:
            pass
        try:
            geometry = qt.QApplication.desktop().availableGeometry()
            return geometry.width(), geometry.height()
        except Exception:
            return 1280, 800

    def _gridShape(self):
        return (3 if self.isGated else 2), (2 if self.ctFiles else 1)

    def _computeViewSize(self):
        """Largest panel size whose grid still fits on this screen without scrolling."""
        screenWidth, screenHeight = self._availableScreenSize()
        columns, rows = self._gridShape()
        byHeight = (screenHeight - 140 - rows * 80) / rows
        byWidth = (screenWidth - self.SIDE_PANEL_WIDTH - 90) / columns
        size = int(min(byHeight, byWidth))
        return max(self.MIN_VIEW_SIZE, min(self.MAX_VIEW_SIZE, size))

    # ---------------- UI ----------------
    def _buildUI(self, title):
        self.dialog = qt.QDialog(slicer.util.mainWindow())
        self.dialog.setWindowTitle(f"SPECT OSEM Reconstruction - {title}")
        self.dialog.setModal(True)

        # Views on the left, controls in a narrow column on the right, so the window
        # stays short enough for small screens
        mainLayout = qt.QHBoxLayout(self.dialog)

        viewsWidget = qt.QWidget()
        panels = qt.QGridLayout(viewsWidget)
        panels.setContentsMargins(0, 0, 0, 0)
        scroll = qt.QScrollArea()
        scroll.setWidget(viewsWidget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(qt.QFrame.NoFrame)
        mainLayout.addWidget(scroll, 1)

        column = 0
        if self.isGated:
            self.gatedView, _, self.gatedInfo, self.gatedPanel = self._makePanel(
                panels, f"Gated reconstruction ({self.nGates} gates)", 0, column, withSlider=False)
            self.gatedFusionView, _, self.gatedFusionInfo, self.gatedFusionPanel = self._makePanel(
                panels, "Gated fusion (on CT)", 1, column, withSlider=False)

            gateCineRow = qt.QHBoxLayout()
            self.gateCineCheckBox = qt.QCheckBox("Cine gates")
            self.gateCineCheckBox.setChecked(True)
            self.gateCineCheckBox.connect("toggled(bool)", self.onGateCineToggled)
            self.gateFpsSpin = qt.QSpinBox()
            self.gateFpsSpin.setRange(1, 60)
            self.gateFpsSpin.setValue(10)
            self.gateFpsSpin.setSuffix(" fps")
            self.gateFpsSpin.connect("valueChanged(int)", self.onGateFpsChanged)
            gateCineRow.addWidget(self.gateCineCheckBox)
            gateCineRow.addWidget(self.gateFpsSpin)
            gateCineRow.addStretch(1)
            panels.addLayout(gateCineRow, 2, column)
            column += 1

        staticTitle = "Static reconstruction (summed gates)" if self.isGated else "Reconstruction (axial)"
        self.reconView, self.reconSlider, self.reconInfo, _ = self._makePanel(panels, staticTitle, 0, column)
        self.fusionView, _, self.fusionInfo, self.fusionPanel = self._makePanel(
            panels, "Fusion (SPECT on CT)", 1, column, withSlider=False)

        opacityRow = qt.QHBoxLayout()
        opacityRow.addWidget(qt.QLabel("Fusion opacity:"))
        self.opacitySlider = qt.QSlider(qt.Qt.Horizontal)
        self.opacitySlider.setRange(0, 100)
        self.opacitySlider.setValue(100)
        self.opacitySlider.connect("valueChanged(int)", lambda v: self.updateSliceViews(self.reconSlider.value))
        opacityRow.addWidget(self.opacitySlider)
        panels.addLayout(opacityRow, 2, column)
        column += 1

        self.projView, self.projSlider, self.projInfo, _ = self._makePanel(panels, "Raw projections", 0, column)
        self.ctView, _, self.ctInfo, self.ctPanel = self._makePanel(
            panels, "CT (resampled to SPECT)", 1, column, withSlider=False)

        cineRow = qt.QHBoxLayout()
        self.cineCheckBox = qt.QCheckBox("Cine")
        self.cineCheckBox.setChecked(True)
        self.cineCheckBox.connect("toggled(bool)", self.onCineToggled)
        self.fpsSpin = qt.QSpinBox()
        self.fpsSpin.setRange(1, 60)
        self.fpsSpin.setValue(10)
        self.fpsSpin.setSuffix(" fps")
        self.fpsSpin.connect("valueChanged(int)", self.onFpsChanged)
        cineRow.addWidget(self.cineCheckBox)
        cineRow.addWidget(self.fpsSpin)
        cineRow.addStretch(1)
        panels.addLayout(cineRow, 2, column)

        self.reconSlider.connect("valueChanged(int)", self.updateReconView)
        self.projSlider.connect("valueChanged(int)", self.updateProjView)

        # Mouse wheel over any axial panel scrolls through slices
        self._installWheelScroll(self.reconView, self.reconSlider)
        self._installWheelScroll(self.fusionView, self.reconSlider)
        self._installWheelScroll(self.ctView, self.reconSlider)
        self._installWheelScroll(self.projView, self.projSlider, stopsCine=True)
        if self.isGated:
            self._installWheelScroll(self.gatedView, self.reconSlider)
            self._installWheelScroll(self.gatedFusionView, self.reconSlider)

        if not self.ctFiles:
            self.fusionPanel.setVisible(False)
            self.ctPanel.setVisible(False)
            self.opacitySlider.setVisible(False)
            if self.isGated:
                self.gatedFusionPanel.setVisible(False)

        # ---- right hand control column ----
        # Right column: scrollable settings on top, a fixed action panel at the bottom
        rightColumn = qt.QWidget()
        rightColumn.setFixedWidth(self.SIDE_PANEL_WIDTH)
        _shrinkFont(rightColumn)
        rightLayout = qt.QVBoxLayout(rightColumn)
        rightLayout.setContentsMargins(0, 0, 0, 0)
        rightLayout.setSpacing(4)
        mainLayout.addWidget(rightColumn, 0)

        sideScroll = qt.QScrollArea()
        sideScroll.setWidgetResizable(True)
        sideScroll.setFrameShape(qt.QFrame.NoFrame)
        sideWidget = qt.QWidget()
        side = qt.QVBoxLayout(sideWidget)
        side.setContentsMargins(0, 0, 4, 0)
        side.setSpacing(4)
        sideScroll.setWidget(sideWidget)
        rightLayout.addWidget(sideScroll, 1)

        # Created first (currentParams reads it); placed at the bottom of the panel
        self.scalingCombo = _compactCombo()
        self.scalingCombo.addItems(SCALING_MODES)
        self.scalingCombo.setItemData(0, "Native OSEM units x number of views: the counts each voxel "
                                         "contributes over the whole acquisition. Keeps the effect of "
                                         "attenuation and scatter correction. Recommended.", qt.Qt.ToolTipRole)
        self.scalingCombo.setItemData(1, "PyTomography's native output: expected counts per voxel "
                                         "per projection view.", qt.Qt.ToolTipRole)
        self.scalingCombo.setItemData(2, "Scales the image so its total equals the detected photopeak "
                                         "counts minus the scatter estimate. Cancels the global effect "
                                         "of attenuation correction.", qt.Qt.ToolTipRole)
        self.scalingCombo.setCurrentIndex(0)

        # Row 1: OSEM | post-filter
        reconGroup, reconForm = _group("OSEM - static" if self.isGated else "OSEM")
        self.itersSpin = qt.QSpinBox()
        self.itersSpin.setRange(1, 200)
        self.itersSpin.setValue(4)
        self.itersSpin.setToolTip("Conventional SPECT: 2-4 iterations are typical with a post-filter")
        self.subsetsSpin = qt.QSpinBox()
        self.subsetsSpin.setRange(1, 128)
        self.subsetsSpin.setValue(6)
        self.subsetsSpin.setToolTip("Best if it divides the number of projection angles (e.g. 6 or 10 for 60 views)")
        reconForm.addWidget(qt.QLabel("Iterations:"), 0, 0)
        reconForm.addWidget(self.itersSpin, 0, 1)
        reconForm.addWidget(qt.QLabel("Subsets:"), 1, 0)
        reconForm.addWidget(self.subsetsSpin, 1, 1)
        reconForm.setRowStretch(2, 1)

        filterGroup, filterForm = _group("Post-filter - static" if self.isGated else "Post-filter")
        self.filterCombo = qt.QComboBox()
        self.filterCombo.addItems(["None", "Gaussian", "Butterworth", "Hann"])
        self.filterCombo.setCurrentIndex(2)  # Butterworth by default
        self.fwhmSpin = _slider(0.5, 30.0, 0.5, 1, 10.0, " mm", FWHM_TOOLTIP)
        self.cutoffSpin = _slider(0.05, 1.0, 0.01, 2, 0.45, " x Nyq",
                                  "Butterworth: conventional SPECT 0.4-0.5 x Nyquist; lower (~0.2-0.3) for "
                                  "low-count studies.\nHann: the response reaches zero at the cutoff, so it "
                                  "smooths more than Butterworth at the same value.")
        self.orderSpin = _slider(1, 20, 1, 0, 10, "", "Conventional SPECT: order 5-10")
        for row, (label, widget) in enumerate([("Filter:", self.filterCombo), ("FWHM:", self.fwhmSpin),
                                               ("Cutoff:", self.cutoffSpin), ("Order:", self.orderSpin)]):
            filterForm.addWidget(qt.QLabel(label), row, 0)
            filterForm.addWidget(widget, row, 1)
        self.filterCombo.connect("currentIndexChanged(int)", self.onFilterChanged)
        self.onFilterChanged(self.filterCombo.currentIndex)
        side.addLayout(_row(reconGroup, filterGroup))

        # Gated: OSEM and filter in two columns
        if self.isGated:
            gatedGroup, gatedForm = _group(f"Gated - {self.nGates} gates")
            side.addWidget(gatedGroup)
            self.gatedEnableCheckBox = qt.QCheckBox("Reconstruct gates")
            self.gatedEnableCheckBox.setChecked(True)
            self.gatedEnableCheckBox.setToolTip(f"Runs {self.nGates} separate reconstructions")
            self.gatedItersSpin = qt.QSpinBox()
            self.gatedItersSpin.setRange(1, 200)
            self.gatedItersSpin.setValue(2)
            self.gatedItersSpin.setToolTip("Gated SPECT: 2 iterations with 10-12 subsets is a common choice")
            self.gatedSubsetsSpin = qt.QSpinBox()
            self.gatedSubsetsSpin.setRange(1, 128)
            self.gatedSubsetsSpin.setValue(10)
            self.gatedFilterCombo = qt.QComboBox()
            self.gatedFilterCombo.addItems(["None", "Gaussian", "Butterworth", "Hann"])
            self.gatedFilterCombo.setCurrentIndex(2)
            self.gatedFwhmSpin = _slider(0.5, 30.0, 0.5, 1, 12.0, " mm", FWHM_TOOLTIP)
            self.gatedCutoffSpin = _slider(0.05, 1.0, 0.01, 2, 0.35, " x Nyq",
                                           "Gates hold fewer counts than the static image, so a lower cutoff is usual")
            self.gatedOrderSpin = _slider(1, 20, 1, 0, 10)
            gatedForm.addWidget(self.gatedEnableCheckBox, 0, 0, 1, 2)
            left = [("Iterations:", self.gatedItersSpin), ("Subsets:", self.gatedSubsetsSpin)]
            right = [("Filter:", self.gatedFilterCombo), ("FWHM:", self.gatedFwhmSpin),
                     ("Cutoff:", self.gatedCutoffSpin), ("Order:", self.gatedOrderSpin)]
            for row, (label, widget) in enumerate(left):
                gatedForm.addWidget(qt.QLabel(label), row + 1, 0)
                gatedForm.addWidget(widget, row + 1, 1)
            for row, (label, widget) in enumerate(right):
                gatedForm.addWidget(qt.QLabel(label), row, 2)
                gatedForm.addWidget(widget, row, 3)
            gatedForm.setColumnStretch(1, 1)
            gatedForm.setColumnStretch(3, 1)
            self.gatedFilterCombo.connect("currentIndexChanged(int)", self.onGatedFilterChanged)
            self.onGatedFilterChanged(self.gatedFilterCombo.currentIndex)

        # Row 2: attenuation correction | scatter correction
        acGroup, acForm = _group("Attenuation correction")
        self.acCheckBox = qt.QCheckBox("CT attenuation map")
        self.acCheckBox.connect("clicked(bool)", self.onAcClicked)
        self.ctCombo = _compactCombo()
        self.ctCombo.setToolTip("CT series found for this study; the first is the best match")
        for candidate in self.ctCandidates:
            self.ctCombo.addItem(candidate["label"])
            self.ctCombo.setItemData(self.ctCombo.count - 1, candidate["label"], qt.Qt.ToolTipRole)
        self.ctCombo.connect("currentIndexChanged(int)", self.onCtSeriesChanged)
        self.acNote = qt.QLabel("")
        self.acNote.setWordWrap(True)
        self.ctBrowseButton = qt.QPushButton("CT folder...")
        self.ctBrowseButton.setAutoDefault(False)
        self.ctBrowseButton.setToolTip("Choose a folder with the CT series manually")
        self.ctBrowseButton.connect("clicked()", self.onBrowseCt)
        self.showOnCtCheckBox = qt.QCheckBox("Show on CT")
        self.showOnCtCheckBox.setChecked(True)
        self.showOnCtCheckBox.setToolTip("After loading: load the CT into Slicer (if it is not loaded yet) "
                                         "and show the SPECT as a colour overlay on it")
        acForm.addWidget(self.acCheckBox, 0, 0, 1, 2)
        acForm.addWidget(self.ctCombo, 1, 0, 1, 2)
        acForm.addWidget(self.acNote, 2, 0, 1, 2)
        acForm.addWidget(self.ctBrowseButton, 3, 0)
        acForm.addWidget(self.showOnCtCheckBox, 3, 1)
        acForm.setRowStretch(4, 1)

        scatterGroup, scatterForm = _group("Scatter correction")
        self.scatterCheckBox = qt.QCheckBox("Energy window scatter")
        self.peakCombo = _compactCombo()
        self.lowerCombo = _compactCombo()
        self.upperCombo = _compactCombo()
        for combo in (self.lowerCombo, self.upperCombo):
            combo.addItem("None")
        for window in self.energyWindows:
            self.peakCombo.addItem(window["label"])
            self.lowerCombo.addItem(window["label"])
            self.upperCombo.addItem(window["label"])
        self.scatterWeightSpin = qt.QDoubleSpinBox()
        self.scatterWeightSpin.setRange(0.0, 5.0)
        self.scatterWeightSpin.setSingleStep(0.05)
        self.scatterWeightSpin.setDecimals(2)
        self.scatterWeightSpin.setSpecialValueText("Auto")  # 0 = automatic
        self.scatterWeightSpin.setValue(0.0)
        self.scatterWeightSpin.setToolTip(
            "Dual energy window scatter weight k: scatter = k x lower window counts.\n"
            "Auto: 0.5 x photopeak width / lower window width (PyTomography's estimate).\n"
            "Enter a value to match a vendor setting, e.g. GE Xeleris 'Scatter weight for Tc99m' = 1.1.")
        self.scatterNote = qt.QLabel("")
        self.scatterNote.setWordWrap(True)
        scatterForm.addWidget(self.scatterCheckBox, 0, 0, 1, 2)
        for row, (label, widget) in enumerate([("Peak:", self.peakCombo), ("Lower:", self.lowerCombo),
                                               ("Upper:", self.upperCombo), ("Weight:", self.scatterWeightSpin)]):
            scatterForm.addWidget(qt.QLabel(label), row + 1, 0)
            scatterForm.addWidget(widget, row + 1, 1)
        scatterForm.addWidget(self.scatterNote, 5, 0, 1, 2)
        scatterForm.setColumnStretch(1, 1)
        scatterForm.setRowStretch(6, 1)
        side.addLayout(_row(acGroup, scatterGroup))
        self.peakCombo.connect("currentIndexChanged(int)", self.onPeakWindowChanged)
        self.scatterCheckBox.connect("toggled(bool)", lambda checked: self._updateScatterState())
        self.lowerCombo.connect("currentIndexChanged(int)", lambda index: self._updateScatterState())
        self.upperCombo.connect("currentIndexChanged(int)", lambda index: self._updateScatterState())
        self.scatterWeightSpin.connect("valueChanged(double)", lambda value: self._updateScatterState())
        self._initScatterWindows()

        # Quantification and SUV (collapsed until switched on)
        self._buildQuantificationUI(side)

        # Output count scaling, at the bottom (applied after OSEM, so changing it does not re-reconstruct)
        scalingRow = qt.QHBoxLayout()
        scalingLabel = qt.QLabel("Output scaling:")
        scalingLabel.setToolTip("Relative counts - not a quantitative (Bq/ml) calibration; "
                                "use Quantification for Bq/ml")
        scalingRow.addWidget(scalingLabel)
        scalingRow.addWidget(self.scalingCombo, 1)
        side.addLayout(scalingRow)

        side.addWidget(qt.QLabel("Log:"))
        self.logBox = qt.QPlainTextEdit()
        self.logBox.setReadOnly(True)
        self.logBox.setMinimumHeight(120)
        side.addWidget(self.logBox, 1)

        # Fixed action panel: always visible below the scrolling settings
        actionPanel = qt.QFrame()
        actionPanel.setFrameShape(qt.QFrame.StyledPanel)
        actionLayout = qt.QVBoxLayout(actionPanel)
        actionLayout.setContentsMargins(6, 4, 6, 6)
        actionLayout.setSpacing(3)
        rightLayout.addWidget(actionPanel, 0)

        self.progressLabel = qt.QLabel("")
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 1)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(False)
        self.progressBar.setMaximumHeight(8)
        self.staleNote = qt.QLabel("")
        self.staleNote.setWordWrap(True)
        actionLayout.addWidget(self.progressLabel)
        actionLayout.addWidget(self.progressBar)
        actionLayout.addWidget(self.staleNote)

        self.applyButton = qt.QPushButton("Reconstruct")
        self.applyButton.setDefault(True)  # Enter in a spin box = Reconstruct, never closes dialog
        self.applyButton.setToolTip("Reconstruct with the current settings and update the preview")
        self.loadButton = qt.QPushButton("Load into Slicer")
        self.loadButton.setAutoDefault(False)
        self.loadButton.setEnabled(False)
        self.loadButton.setToolTip("Load the previewed reconstruction (enabled when the preview matches the settings)")
        self.cancelButton = qt.QPushButton("Cancel")
        self.cancelButton.setAutoDefault(False)
        buttonRow = qt.QHBoxLayout()
        buttonRow.setSpacing(6)
        for button, color in ((self.cancelButton, BUTTON_ORANGE), (self.applyButton, BUTTON_BLUE),
                              (self.loadButton, BUTTON_GREEN)):
            _colorButton(button, color)
            buttonRow.addWidget(button)
        actionLayout.addLayout(buttonRow)

        self.applyButton.connect("clicked()", self.onApply)
        self.loadButton.connect("clicked()", self.onLoad)
        self.cancelButton.connect("clicked()", self.onCancelClicked)

        self.busyGuard = BusyDialogGuard(lambda: self.busy, self.requestStop)
        self.dialog.installEventFilter(self.busyGuard)

        self._watchSettings()
        self._updateCtState()
        self._updateLoadState()
        self._resizeDialog()

    def _buildQuantificationUI(self, side):
        acq = self.acquisition

        # Quantification: off by default; a calibration drives AC, scatter and photopeak
        self.quantGroup, form = _collapsibleGroup("Quantification (Bq/ml)")
        self.quantGroup.setToolTip("Convert the static image to Bq/ml with a saved camera calibration "
                                   "(made in the SPECT Calibration module)")
        side.addWidget(self.quantGroup)
        self.calibrationCombo = _compactCombo()
        self.libraryButton = qt.QPushButton("Library...")
        self.libraryButton.setAutoDefault(False)
        self.libraryButton.setToolTip("Choose the calibration library JSON file")
        self.libraryLabel = qt.QLabel("")
        self.libraryLabel.setWordWrap(True)
        self.quantNote = qt.QLabel("")
        self.quantNote.setWordWrap(True)
        self.durationSpin = qt.QDoubleSpinBox()
        self.durationSpin.setRange(0.0, 1.0e6)
        self.durationSpin.setDecimals(1)
        self.durationSpin.setSuffix(" s")
        self.durationSpin.setValue(float(acq.get("totalViewTimeSec") or 0.0))
        self.durationSpin.setToolTip("Counting time summed over all views of all detectors "
                                     f"({acq.get('durationSource') or 'not in DICOM'}) - editable")
        self.timingNote = qt.QLabel(self._timingText())
        self.timingNote.setWordWrap(True)
        self.midTimeEdit = _dateTimeEdit()
        setDateTimeEdit(self.midTimeEdit, acq.get("mid"))
        self.midTimeFromDicom = qt.QCheckBox("from DICOM")
        self.midTimeFromDicom.setChecked(bool(acq.get("mid")))
        self.midTimeFromDicom.setToolTip(f"Mid-time from {acq.get('midSource') or 'DICOM'}")
        self.midTimeEdit.setEnabled(not self.midTimeFromDicom.checked)
        self.midTimeFromDicom.connect("toggled(bool)", self._onMidTimeSourceChanged)
        form.addWidget(qt.QLabel("Calibration:"), 0, 0)
        form.addWidget(self.calibrationCombo, 0, 1, 1, 2)
        form.addWidget(self.libraryLabel, 1, 0, 1, 2)
        form.addWidget(self.libraryButton, 1, 2)
        form.addWidget(self.quantNote, 2, 0, 1, 3)
        form.addWidget(qt.QLabel("Total view time:"), 3, 0)
        form.addWidget(self.durationSpin, 3, 1, 1, 2)
        form.addWidget(self.timingNote, 4, 0, 1, 3)
        form.addWidget(qt.QLabel("Scan mid-time:"), 5, 0)
        form.addWidget(self.midTimeEdit, 5, 1)
        form.addWidget(self.midTimeFromDicom, 5, 2)
        self.durationSpin.connect("valueChanged(double)", lambda value: self.timingNote.setText(self._timingText()))

        # SUV: off by default, needs quantification
        self.suvGroup, suvForm = _collapsibleGroup("SUV")
        self.suvGroup.setEnabled(False)
        self.suvGroup.setToolTip("Needs quantification (Bq/ml) first")
        side.addWidget(self.suvGroup)
        self.suvTypeCombo = _compactCombo()
        self.suvTypeCombo.addItems(quant.SUV_TYPES)
        self.syringeSpin = qt.QDoubleSpinBox()
        self.syringeSpin.setRange(0.0, 100000.0)
        self.syringeSpin.setDecimals(2)
        self.syringeSpin.setSuffix(" MBq")
        self.syringeSpin.setValue(float(acq.get("injectedMBq") or 0.0))
        self.syringeTimeEdit = _dateTimeEdit()
        setDateTimeEdit(self.syringeTimeEdit, acq.get("injectionTime") or acq.get("start"))
        self.residualSpin = qt.QDoubleSpinBox()
        self.residualSpin.setRange(0.0, 100000.0)
        self.residualSpin.setDecimals(2)
        self.residualSpin.setSuffix(" MBq")
        self.residualTimeEdit = _dateTimeEdit()
        setDateTimeEdit(self.residualTimeEdit, acq.get("injectionTime") or acq.get("start"))
        self.weightSpin = qt.QDoubleSpinBox()
        self.weightSpin.setRange(0.0, 500.0)
        self.weightSpin.setDecimals(1)
        self.weightSpin.setSuffix(" kg")
        self.weightSpin.setValue(float(acq.get("weightKg") or 0.0))
        self.heightSpin = qt.QDoubleSpinBox()
        self.heightSpin.setRange(0.0, 250.0)
        self.heightSpin.setDecimals(1)
        self.heightSpin.setSuffix(" cm")
        self.heightSpin.setValue(float(acq.get("heightCm") or 0.0))
        self.sexCombo = qt.QComboBox()
        self.sexCombo.addItems(["", "M", "F"])
        self.sexCombo.setCurrentIndex({"M": 1, "F": 2}.get(acq.get("sex", ""), 0))
        rows = [("SUV type:", self.suvTypeCombo, None),
                ("Syringe activity:", self.syringeSpin, self.syringeTimeEdit),
                ("Residual activity:", self.residualSpin, self.residualTimeEdit),
                ("Weight:", self.weightSpin, None),
                ("Height:", self.heightSpin, None),
                ("Sex:", self.sexCombo, None)]
        for row, (label, widget, extra) in enumerate(rows):
            suvForm.addWidget(qt.QLabel(label), row, 0)
            suvForm.addWidget(widget, row, 1, 1, 1 if extra else 2)
            if extra:
                suvForm.addWidget(extra, row, 2)
        self.suvNote = qt.QLabel("<i>Prefilled from DICOM where available - please verify. "
                                 "Activities are decayed to the scan mid-time.</i>")
        self.suvNote.setWordWrap(True)
        suvForm.addWidget(self.suvNote, len(rows), 0, 1, 3)

        self.quantGroup.connect("toggled(bool)", self.onQuantToggled)
        self.calibrationCombo.connect("currentIndexChanged(int)", lambda index: self._applyCalibration())
        self.libraryButton.connect("clicked()", self.onChooseLibrary)
        self._updateLibraryLabel()

    def _timingText(self):
        """Detectors and per-detector / total time, as far as the header shows them."""
        acq = self.acquisition
        total = self.durationSpin.value if hasattr(self, "durationSpin") else (acq.get("totalViewTimeSec") or 0)
        detectors = acq.get("detectors") or 1
        if total <= 0:
            return "<span style='color:#d08000'>Total view time unknown - enter it.</span>"
        if detectors > 1:
            text = (f"<small>{detectors} detectors detected ({acq.get('detectorsEvidence')}): "
                    f"{total / detectors:.1f} s each, {total:.1f} s total.</small>")
        elif "timestamps" in (acq.get("midSource") or ""):
            text = f"<small>1 detector recorded, {total:.1f} s total view time.</small>"
        else:
            text = (f"<small>1 detector recorded ({acq.get('detectorsEvidence')}). If this camera has two "
                    f"heads, the DICOM mid-time is about {total / 4.0 / 60.0:.1f} min late - set it manually."
                    "</small>")
        return text + f"<br><small>Mid-time: {acq.get('midSource') or 'not in DICOM'}.</small>"

    def _onMidTimeSourceChanged(self, fromDicom):
        self.midTimeEdit.setEnabled(not fromDicom)
        if fromDicom:
            setDateTimeEdit(self.midTimeEdit, self.acquisition.get("mid"))

    def _updateLibraryLabel(self):
        path = quant.calibrationLibraryPath()
        self.libraryLabel.setText(f"<small>{path}</small>")
        self.libraryLabel.setToolTip(path)

    def onChooseLibrary(self):
        if quant.chooseCalibrationLibrary(self.dialog):
            self._updateLibraryLabel()
            self._reloadCalibrations()

    def _reloadCalibrations(self):
        """Fill the calibration list; compatible calibrations first, the others marked."""
        path = quant.calibrationLibraryPath()
        try:
            records = quant.loadLibrary(path)
        except Exception as e:
            records = []
            self.log(f"ERROR: could not read calibration library {path}: {e}")
        scored = []
        for record in records:
            errors, _ = quant.compatibility(record, self.acquisition)
            scored.append((bool(errors), record.get("created", ""), record))
        scored.sort(key=lambda item: item[1], reverse=True)  # newest first...
        scored.sort(key=lambda item: item[0])                 # ...usable ones before the rest
        self.calibrations = [record for _, _, record in scored]
        self.calibrationCombo.blockSignals(True)
        self.calibrationCombo.clear()
        for unusable, _, record in scored:
            label = ("[not for this study] " if unusable else "") + quant.describe(record)
            self.calibrationCombo.addItem(label)
            self.calibrationCombo.setItemData(self.calibrationCombo.count - 1, label, qt.Qt.ToolTipRole)
        self.calibrationCombo.blockSignals(False)
        if not records:
            self.quantNote.setText("<span style='color:#d08000'>No calibrations in this library yet - "
                                   "make one in the SPECT Calibration module.</span>")
        self._applyCalibration()

    def selectedCalibration(self):
        index = self.calibrationCombo.currentIndex
        if not self.quantGroup.checked or index < 0 or index >= len(self.calibrations):
            return None
        return self.calibrations[index]

    def onQuantToggled(self, checked):
        self.suvGroup.setEnabled(checked)
        if not checked:
            self.suvGroup.setChecked(False)
        if checked:
            self._reloadCalibrations()
        else:
            self._applyCalibration()

    def _applyCalibration(self):
        """Make AC, scatter and photopeak follow the selected calibration and lock them;
        a calibration made without scatter correction switches scatter correction off."""
        calibration = self.selectedCalibration()
        self.quantLocked = calibration is not None
        if calibration is None:
            if self.quantGroup.checked and self.calibrations:
                self.quantNote.setText("<span style='color:#d08000'>Select a calibration.</span>")
            elif not self.quantGroup.checked:
                self.quantNote.setText("")
            self._updateLockedWidgets()
            return

        errors, warnings = quant.compatibility(calibration, self.acquisition)
        windows = self.energyWindows
        peakIndex = quant.findWindowIndex(windows, calibration.get("photopeak"))
        if peakIndex is not None and peakIndex != self.peakCombo.currentIndex:
            self.peakCombo.setCurrentIndex(peakIndex)  # also resets the default scatter windows

        if self.ctFiles:
            self.acCheckBox.setChecked(True)
        else:
            errors.append("the calibration needs CT attenuation correction, but no CT is selected")

        scatter = calibration.get("scatter", {})
        if scatter.get("method", "None") == "None":
            self.scatterCheckBox.setChecked(False)
        else:
            lower = quant.findWindowIndex(windows, scatter.get("lower")) if scatter.get("lower") else None
            upper = quant.findWindowIndex(windows, scatter.get("upper")) if scatter.get("upper") else None
            self.lowerCombo.setCurrentIndex(0 if lower is None else lower + 1)
            self.upperCombo.setCurrentIndex(0 if upper is None else upper + 1)
            self.scatterCheckBox.setChecked(lower is not None)
            weight = scatter.get("weight")
            self.scatterWeightSpin.setValue(0.0 if weight is None else float(weight))
        self._updateScatterState()

        recon = calibration.get("recon", {})
        if recon.get("iterations") and (recon.get("iterations"), recon.get("subsets")) != \
                (self.itersSpin.value, self.subsetsSpin.value):
            warnings.append(f"calibrated with {recon.get('iterations')} iterations x {recon.get('subsets')} "
                            f"subsets, current {self.itersSpin.value} x {self.subsetsSpin.value}")
        text = (f"{float(calibration.get('cpsPerMBq', 0)):.3f} cps/MBq, AC CT, "
                f"SC {quant.scatterLabel(calibration)}")
        if errors:
            text = "<span style='color:#d04040'>&#9888; Not usable: " + "; ".join(errors) + "</span>"
        elif warnings:
            text += "<br><span style='color:#d08000'>&#9888; " + "; ".join(warnings) + "</span>"
        self.quantNote.setText(text)
        self._updateLockedWidgets()

    def _updateLockedWidgets(self):
        """Enabled state of controls that a calibration or a running reconstruction can lock."""
        if not hasattr(self, "quantGroup"):
            return
        locked = self.quantLocked
        busy = self.busy
        self.acCheckBox.setEnabled(bool(self.ctFiles) and not locked)
        scatterAvailable = self._scatterAvailable()
        for widget in (self.scatterCheckBox, self.lowerCombo, self.upperCombo):
            widget.setEnabled(scatterAvailable and not locked)
        self._updateWeightEnabled()
        self.peakCombo.setEnabled(not busy and not locked)
        self.scalingCombo.setEnabled(not self.quantGroup.checked)
        self.scalingCombo.setToolTip("Quantification works from acquisition counts (x number of views)"
                                     if self.quantGroup.checked else "")

    def _midTime(self):
        return dateTimeFromEdit(self.midTimeEdit)

    def _quantify(self, image, params):
        """Acquisition-counts image -> Bq/ml (and SUV). Returns (image, units, info)."""
        calibration = next((c for c in self.calibrations if c.get("id") == params["quant"]["id"]), None)
        if calibration is None:
            raise ValueError("Quantification is on, but no calibration is selected.")
        errors, warnings = quant.compatibility(calibration, self.acquisition)
        if errors:
            raise ValueError("The calibration does not fit this study: " + "; ".join(errors))
        if not params["attenuationCorrection"]:
            raise ValueError("Quantification needs CT attenuation correction.")
        wantedScatter = calibration.get("scatter", {}).get("method", "None")
        if scatterMethodName(params) != wantedScatter:
            raise ValueError(f"The calibration was made with scatter correction '{wantedScatter}', "
                             f"the current setting is '{scatterMethodName(params)}'.")
        wantedWeight = calibration.get("scatter", {}).get("weight")
        if wantedScatter == "DEW" and not sameScatterWeight(wantedWeight, params.get("scatterWeight")):
            raise ValueError(f"The calibration was made with scatter weight {scatterWeightText(wantedWeight)}, "
                             f"the current setting is {scatterWeightText(params.get('scatterWeight'))}.")
        duration = params["quant"]["duration"]
        if duration <= 0:
            raise ValueError("Enter the counting duration of the acquisition.")
        for warning in warnings:
            self.log(f"WARNING (quantification): {warning}")

        spacing = image.GetSpacing()
        voxelMl = spacing[0] * spacing[1] * spacing[2] / 1000.0
        cf = float(calibration["cpsPerMBq"])
        factor = quant.bqPerMlFactor(duration, cf, voxelMl)
        self.log(f"Quantification: {calibration.get('name')} ({cf:.3f} cps/MBq), total view time {duration:.1f} s, "
                 f"voxel {voxelMl * 1000:.1f} mm3 -> x {factor:.6g} to Bq/ml")
        info = {"Units": "Bq/ml", "CalibrationName": calibration.get("name", ""),
                "CalibrationId": calibration.get("id", ""), "CalibrationCpsPerMBq": f"{cf:.6g}",
                "CalibrationDate": calibration.get("created", ""), "QuantTotalViewTimeSec": f"{duration:.3f}",
                "QuantReferenceTime": quant.toIso(params["quant"]["mid"]), "BqPerMlFactor": f"{factor:.6g}"}
        image = image * float(factor)
        units = "Bq/ml"

        suv = params.get("suv")
        if suv:
            halfLife = float(calibration.get("halfLifeSec") or self.acquisition.get("halfLifeSec") or 0)
            if halfLife <= 0:
                raise ValueError("Isotope half-life unknown - cannot decay-correct the injected activity.")
            mid = params["quant"]["mid"]
            if suv["syringeTime"] > mid:
                raise ValueError("The syringe was measured after the scan mid-time - check the times.")
            injectedMBq = quant.activityAt(mid, halfLife, suv["syringeMBq"], suv["syringeTime"],
                                           suv["residualMBq"], suv["residualTime"])
            suvScale, massKg = quant.suvFactor(injectedMBq * 1e6, suv["weightKg"], suv["heightCm"],
                                               suv["sex"], suv["kind"])
            image = image * float(suvScale)
            units = "SUL" if suv["kind"] == quant.SUV_LBM else "SUVbw"
            self.log(f"SUV: injected {injectedMBq:.2f} MBq at scan mid-time, mass {massKg:.1f} kg "
                     f"({suv['kind']}) -> x {suvScale:.6g}")
            info.update({"Units": units, "SUVType": suv["kind"], "SUVMassKg": f"{massKg:.2f}",
                         "InjectedMBqAtScan": f"{injectedMBq:.3f}", "SyringeMBq": suv["syringeMBq"],
                         "SyringeTime": quant.toIso(suv["syringeTime"]), "ResidualMBq": suv["residualMBq"],
                         "ResidualTime": quant.toIso(suv["residualTime"]), "PatientWeightKg": suv["weightKg"],
                         "PatientHeightCm": suv["heightCm"], "PatientSex": suv["sex"]})
        return image, units, info

    def _acquisitionAttributes(self, params):
        """Camera/isotope/timing data stored on the volume, so the SPECT Calibration module
        (and anyone checking later) has everything without the original DICOM file."""
        import json
        acq = self.acquisition
        windows = {w["index"]: w for w in self.energyWindows}

        def energyRange(index):
            w = windows.get(index)
            return json.dumps([w["low"], w["high"]]) if w and w.get("low") is not None else ""

        duration = self.durationSpin.value if self.durationSpin.value > 0 else (acq.get("totalViewTimeSec") or 0)
        detectors = acq.get("detectors") or 1
        return {
            "Manufacturer": acq.get("manufacturer", ""), "CameraModel": acq.get("model", ""),
            "CameraSerial": acq.get("serial", ""), "Station": acq.get("station", ""),
            "Collimator": acq.get("collimator", ""), "Isotope": acq.get("isotope", ""),
            "IsotopeDicomText": acq.get("isotopeText", ""), "HalfLifeSec": acq.get("halfLifeSec") or "",
            "Views": acq.get("views") or "", "FrameDurationMs": acq.get("frameDurationMs") or "",
            "TotalViewTimeSec": f"{duration:.3f}" if duration else "",
            "TotalViewTimeSource": acq.get("durationSource", ""),
            "Detectors": detectors, "DetectorsEvidence": acq.get("detectorsEvidence", ""),
            "PerDetectorTimeSec": f"{duration / detectors:.3f}" if duration else "",
            "AcquisitionStart": acq.get("start", ""),
            "ScanMidTimeSource": acq.get("midSource", "") if (not hasattr(self, "midTimeFromDicom")
                                                              or self.midTimeFromDicom.checked) else "entered manually",
            "ScanMidTime": quant.toIso(self._midTime()) if hasattr(self, "midTimeEdit") else acq.get("mid", ""),
            "PhotopeakRange": energyRange(params["indexPeak"]),
            "ScatterMethod": scatterMethodName(params),
            "ScatterWeight": scatterWeightText(params.get("scatterWeight"))
                             if scatterMethodName(params) == "DEW" else "",
            "ScatterLowerRange": energyRange(params["indexLower"]) if params["indexLower"] is not None else "",
            "ScatterUpperRange": energyRange(params["indexUpper"]) if params["indexUpper"] is not None else "",
            "ReconIterations": params["iters"], "ReconSubsets": params["subsets"],
        }

    def _resizeDialog(self):
        """Open large enough that the panels need no scrolling, but never larger than the screen."""
        screenWidth, screenHeight = self._availableScreenSize()
        columns, rows = self._gridShape()
        width = columns * self.viewSize + self.SIDE_PANEL_WIDTH + 80
        height = rows * (self.viewSize + 80) + 80
        self.dialog.resize(min(width, screenWidth), min(height, screenHeight))

    # ---------------- CT / scatter state ----------------
    def _updateCtState(self):
        """Reflect the current CT pairing in the settings and the fusion panels."""
        hasCt = bool(self.ctFiles)
        self.acCheckBox.setEnabled(hasCt)
        self.showOnCtCheckBox.setEnabled(hasCt)
        self.ctCombo.setVisible(bool(self.ctCandidates))
        if not hasCt:
            self.acCheckBox.setChecked(False)
            self.acNote.setText("<i>no CT found in this study - pick a CT folder to enable</i>")
            self.acCheckBox.setToolTip("No CT series was found for this SPECT study")
        else:
            if not self.acUserSet:
                self.acCheckBox.setChecked(True)  # a CT is available, so correct by default
            text = f"{len(self.ctFiles)} CT slices, {self.ctMatchType}"
            if self.ctMatchType != "same frame of reference":
                # Not registered by the scanner: attenuation correction may be misaligned
                self.acNote.setText(f"<span style='color:#d08000'>&#9888; {text}</span>")
            else:
                self.acNote.setText(f"<i>{text}</i>")
            if self.ctPreviewFailed:
                self.acNote.setText(self.acNote.text + "<br><span style='color:#d08000'>"
                                    "CT could not be shown in the preview (see log)</span>")
            self.acCheckBox.setToolTip(text)
        if hasattr(self, "quantGroup"):
            self._updateLockedWidgets()
            if self.quantLocked:
                self._applyCalibration()
        showCtPanels = hasCt and not self.ctPreviewFailed
        self.fusionPanel.setVisible(showCtPanels)
        self.ctPanel.setVisible(showCtPanels)
        self.opacitySlider.setVisible(showCtPanels)
        if self.isGated:
            self.gatedFusionPanel.setVisible(showCtPanels)

    def onAcClicked(self, checked):
        self.acUserSet = True

    def _initScatterWindows(self):
        """Scatter needs at least one window besides the photopeak."""
        available = len(self.energyWindows) > 1
        self.scatterCheckBox.setEnabled(available)
        for combo in (self.peakCombo, self.lowerCombo, self.upperCombo):
            combo.setEnabled(available)
        self.scatterWeightSpin.setEnabled(False)
        if not available:
            count = len(self.energyWindows)
            self.scatterNote.setText(f"<i>{count} energy window in this acquisition - "
                                     f"no scatter window to subtract</i>")
            self.scatterCheckBox.setChecked(False)
            return
        # Photopeak: best guess from window names and layout (window 1 is not always the peak)
        peakIndex = defaultPeakIndex(self.energyWindows)
        self.peakCombo.blockSignals(True)
        self.peakCombo.setCurrentIndex(peakIndex)
        self.peakCombo.blockSignals(False)
        self.onPeakWindowChanged(peakIndex)

    def onPeakWindowChanged(self, index):
        """Pick the adjacent windows below (and above) the photopeak as scatter windows."""
        if not self.energyWindows or index < 0 or index >= len(self.energyWindows):
            return
        peak = self.energyWindows[index]
        below, above = adjacentScatterWindows(peak, self.energyWindows)
        lowerIndex = max(below, key=lambda w: w["center"])["index"] if below else None
        upperIndex = min(above, key=lambda w: w["center"])["index"] if above else None
        if lowerIndex is None:
            upperIndex = None  # upper-only is not supported by the scatter estimate
        self.lowerCombo.setCurrentIndex(0 if lowerIndex is None else lowerIndex + 1)
        self.upperCombo.setCurrentIndex(0 if upperIndex is None else upperIndex + 1)
        # Scatter correction is on by default when a usable scatter window exists
        if not self._scatterAvailable():
            self.scatterCheckBox.setChecked(False)
        else:
            self.scatterCheckBox.setChecked(lowerIndex is not None)
        self._updateScatterState()
        # The projection preview shows the photopeak window
        if self.rawProjection is not None:
            self._showRawProjections()

    def scatterProblem(self, params=None):
        params = params or self.currentParams()
        return scatterWindowProblem(self.energyWindows, params["indexPeak"],
                                    params["indexLower"], params["indexUpper"])

    def _scatterAvailable(self):
        """Scatter windows exist (independent of the widget being locked by a calibration)."""
        return len(self.energyWindows) > 1

    def _isDualWindow(self):
        return (self._scatterAvailable() and bool(self.scatterCheckBox.checked)
                and self.lowerCombo.currentIndex > 0 and self.upperCombo.currentIndex <= 0)

    def _updateWeightEnabled(self):
        """The manual weight applies to dual window correction only."""
        locked = getattr(self, "quantLocked", False)
        self.scatterWeightSpin.setEnabled(self._isDualWindow() and not locked)

    def _autoWeight(self):
        """Weight the automatic dual window estimate amounts to, from the window widths."""
        peak = self.peakCombo.currentIndex
        lower = self.lowerCombo.currentIndex - 1
        if not (0 <= peak < len(self.energyWindows)) or not (0 <= lower < len(self.energyWindows)):
            return None
        width = lambda w: (w["high"] - w["low"]) if w.get("low") is not None and w.get("high") is not None else None
        return autoScatterWeight(width(self.energyWindows[peak]), width(self.energyWindows[lower]))

    def _updateScatterState(self):
        if not self._scatterAvailable():
            return
        self._updateWeightEnabled()
        lower = self.lowerCombo.currentIndex - 1
        upper = self.upperCombo.currentIndex - 1
        if not self.scatterCheckBox.checked:
            self.scatterNote.setText("<i>no scatter correction</i>")
            return
        if lower < 0 and upper < 0:
            self.scatterNote.setText("<span style='color:#d08000'>select a lower (and optionally an "
                                     "upper) scatter window</span>")
            return
        problem = self.scatterProblem()
        if problem:
            self.scatterNote.setText(f"<span style='color:#d04040'>&#9888; {problem}</span>")
        elif lower >= 0 and upper >= 0:
            self.scatterNote.setText("<i>triple energy window</i>")
        else:
            if self.scatterWeightSpin.value > 0:
                weightNote = f"weight {self.scatterWeightSpin.value:g} (manual)"
            else:
                auto = self._autoWeight()
                weightNote = f"weight auto = {auto:.2f}" if auto else "weight auto"
            self.scatterNote.setText(f"<i>dual energy window (lower window only), {weightNote}</i>")

    def onCtSeriesChanged(self, index):
        if index < 0 or index >= len(self.ctCandidates):
            return
        candidate = self.ctCandidates[index]
        self._useCtSeries(candidate["files"], candidate["match"], candidate["label"])

    def _useCtSeries(self, files, matchType, label):
        """Switch to another CT: drop cached CT and attenuation data, refresh the views.
        Cached reconstructions are keyed on the CT, so AC ones are redone on the next Reconstruct."""
        self.log(f"CT series: {label}")
        with slicer.util.WaitCursor():
            files = cleanCtFiles(files, self.log)
        self.ctCleaned = True
        self.ctFiles = files
        self.ctMatchType = matchType
        self.ctArray = None
        self.ctGeometryKey = None
        self.ctPreviewFailed = False
        if self.logic is not None:
            self.logic.ctFiles = files
            self.logic.attenuationMap = None
        if "manually" in matchType or "unrelated" in matchType:
            self.log("WARNING: this CT is not registered to the SPECT by the scanner - "
                     "check the fusion view for misalignment before trusting attenuation correction.")
        self._updateCtState()
        if self.reconArray is not None:
            self.updateSliceViews(self.reconSlider.value)

    def onBrowseCt(self):
        """Manual fallback: point at a folder of CT DICOM files."""
        directory = qt.QFileDialog.getExistingDirectory(self.dialog, "Select folder with the CT series")
        if not directory:
            return
        import pydicom
        series = {}
        with slicer.util.WaitCursor():
            for root, _, names in os.walk(directory):
                for name in names:
                    path = os.path.join(root, name)
                    try:
                        ds = pydicom.dcmread(path, stop_before_pixels=True)
                    except Exception:
                        continue
                    if str(getattr(ds, "Modality", "")).upper() != "CT":
                        continue
                    key = str(getattr(ds, "SeriesInstanceUID", ""))
                    entry = series.setdefault(key, {"files": [], "frame": str(getattr(ds, "FrameOfReferenceUID", ""))})
                    entry["files"].append(path)
        if not series:
            self.log(f"No CT files found in {directory}")
            return
        # A folder can hold several CT series (e.g. low-dose AC CT and a diagnostic CT): take the largest
        chosen = max(series.values(), key=lambda e: len(e["files"]))
        if len(series) > 1:
            self.log(f"{len(series)} CT series in this folder - using the largest ({len(chosen['files'])} files).")
        ctFiles = chosen["files"]

        nmFrame = ""
        try:
            nmFrame = str(getattr(pydicom.dcmread(self.inputDcm, stop_before_pixels=True), "FrameOfReferenceUID", ""))
        except Exception:
            pass
        if nmFrame and chosen["frame"] == nmFrame:
            match = "same frame of reference"
        else:
            match = "chosen manually - check alignment"
        label = f"{os.path.basename(directory.rstrip(os.sep))} - {len(ctFiles)} slices ({match})"
        self.ctCandidates.append({"files": ctFiles, "match": match, "label": label, "rank": 3})
        self.ctCombo.blockSignals(True)
        self.ctCombo.addItem(label)
        self.ctCombo.setCurrentIndex(len(self.ctCandidates) - 1)
        self.ctCombo.blockSignals(False)
        self._useCtSeries(ctFiles, match, label)

    # ---------------- panels, cine, wheel ----------------
    def _makePanel(self, gridLayout, title, row, column, withSlider=True):
        container = qt.QWidget()
        box = qt.QVBoxLayout(container)
        box.setContentsMargins(0, 0, 0, 0)
        header = qt.QLabel(f"<b>{title}</b>")
        header.setAlignment(qt.Qt.AlignCenter)
        view = qt.QLabel()
        view.setFixedSize(self.viewSize, self.viewSize)
        view.setAlignment(qt.Qt.AlignCenter)
        view.setStyleSheet("background-color: black; color: gray;")
        slider = qt.QSlider(qt.Qt.Horizontal)
        slider.setEnabled(False)
        info = qt.QLabel("")
        info.setAlignment(qt.Qt.AlignCenter)
        widgets = (header, view, slider, info) if withSlider else (header, view, info)
        for w in widgets:
            box.addWidget(w)
        gridLayout.addWidget(container, row, column)
        return view, slider, info, container

    def _installWheelScroll(self, view, slider, stopsCine=False):
        def step(direction):
            if stopsCine and self.cineCheckBox.checked:
                self.cineCheckBox.setChecked(False)  # manual scrolling takes over
            if not slider.isEnabled():
                return
            value = slider.value + direction
            slider.setValue(max(slider.minimum, min(slider.maximum, value)))
        try:
            wheelFilter = WheelScrollFilter(step)
            view.installEventFilter(wheelFilter)
            self.wheelFilters.append(wheelFilter)  # keep alive
        except Exception:
            logging.warning("Mouse wheel scrolling not available:\n" + traceback.format_exc())

    def onCineToggled(self, checked):
        self.startProjCine() if checked else self.stopProjCine()

    def onFpsChanged(self, fps):
        if self.projCineTimer is not None:
            self.projCineTimer.setInterval(int(1000 / max(1, fps)))

    def startProjCine(self):
        if self.projArray is None or self.projArray.shape[0] < 2:
            return
        if self.projCineTimer is None:
            self.projCineTimer = qt.QTimer()
            self.projCineTimer.connect("timeout()", self.advanceProjCine)
        self.projCineTimer.setInterval(int(1000 / max(1, self.fpsSpin.value)))
        self.projCineTimer.start()

    def stopProjCine(self):
        if self.projCineTimer is not None:
            self.projCineTimer.stop()

    def advanceProjCine(self):
        if self.projArray is None:
            return
        self.projSlider.setValue((self.projSlider.value + 1) % self.projArray.shape[0])

    def onGateCineToggled(self, checked):
        self.startGateCine() if checked else self.stopGateCine()

    def onGateFpsChanged(self, fps):
        if self.gateCineTimer is not None:
            self.gateCineTimer.setInterval(int(1000 / max(1, fps)))

    def startGateCine(self):
        if len(self.gateArrays) < 2:
            return
        if self.gateCineTimer is None:
            self.gateCineTimer = qt.QTimer()
            self.gateCineTimer.connect("timeout()", self.advanceGateCine)
        self.gateCineTimer.setInterval(int(1000 / max(1, self.gateFpsSpin.value)))
        self.gateCineTimer.start()

    def stopGateCine(self):
        if self.gateCineTimer is not None:
            self.gateCineTimer.stop()

    def advanceGateCine(self):
        if not self.gateArrays:
            return
        self.gateIndex = (self.gateIndex + 1) % len(self.gateArrays)
        self.updateGatedViews(self.reconSlider.value)

    def onFilterChanged(self, index):
        name = self.filterCombo.currentText
        self.fwhmSpin.setEnabled(name == "Gaussian")
        self.cutoffSpin.setEnabled(name in ("Butterworth", "Hann"))
        self.orderSpin.setEnabled(name == "Butterworth")

    def onGatedFilterChanged(self, index):
        name = self.gatedFilterCombo.currentText
        self.gatedFwhmSpin.setEnabled(name == "Gaussian")
        self.gatedCutoffSpin.setEnabled(name in ("Butterworth", "Hann"))
        self.gatedOrderSpin.setEnabled(name == "Butterworth")

    def log(self, message):
        logging.info(message)
        self.logBox.appendPlainText(message)
        self.logBox.verticalScrollBar().setValue(self.logBox.verticalScrollBar().maximum)
        slicer.app.processEvents()

    # ---------------- parameters ----------------
    def currentParams(self):
        params = {
            "iters": self.itersSpin.value,
            "subsets": self.subsetsSpin.value,
            "attenuationCorrection": bool(self.acCheckBox.checked) and bool(self.ctFiles),
            "ctId": (self.ctFiles[0], len(self.ctFiles)) if self.ctFiles else None,
            "filter": self.filterCombo.currentText,
            "fwhm": self.fwhmSpin.value,
            "cutoff": self.cutoffSpin.value,
            "order": int(round(self.orderSpin.value)),
            "scaling": self.scalingCombo.currentText,
            "indexPeak": max(0, self.peakCombo.currentIndex),
            "indexLower": None,
            "indexUpper": None,
            "scatterWeight": None,
            "gated": False,
        }
        if self._scatterAvailable() and self.scatterCheckBox.checked:
            lower = self.lowerCombo.currentIndex - 1
            upper = self.upperCombo.currentIndex - 1
            params["indexLower"] = lower if lower >= 0 else None
            params["indexUpper"] = upper if upper >= 0 else None
            if self._isDualWindow() and self.scatterWeightSpin.value > 0:
                params["scatterWeight"] = float(self.scatterWeightSpin.value)
        if self.isGated:
            params.update({
                "gated": bool(self.gatedEnableCheckBox.checked),
                "gatedIters": self.gatedItersSpin.value,
                "gatedSubsets": self.gatedSubsetsSpin.value,
                "gatedFilter": self.gatedFilterCombo.currentText,
                "gatedFwhm": self.gatedFwhmSpin.value,
                "gatedCutoff": self.gatedCutoffSpin.value,
                "gatedOrder": int(round(self.gatedOrderSpin.value)),
            })
        params["quant"] = None
        params["suv"] = None
        if hasattr(self, "quantGroup") and self.quantGroup.checked:
            calibration = self.selectedCalibration()
            params["scaling"] = SCALING_VIEWS  # quantification starts from acquisition counts
            params["quant"] = {"id": calibration.get("id") if calibration else None,
                               "duration": self.durationSpin.value, "mid": self._midTime()}
            if self.suvGroup.checked:
                params["suv"] = {"kind": self.suvTypeCombo.currentText,
                                 "syringeMBq": self.syringeSpin.value,
                                 "syringeTime": dateTimeFromEdit(self.syringeTimeEdit),
                                 "residualMBq": self.residualSpin.value,
                                 "residualTime": dateTimeFromEdit(self.residualTimeEdit),
                                 "weightKg": self.weightSpin.value,
                                 "heightCm": self.heightSpin.value or None,
                                 "sex": self.sexCombo.currentText}
        return params

    @staticmethod
    def scatterArgument(params):
        if params["indexLower"] is None and params["indexUpper"] is None:
            return None
        return (params["indexLower"], params["indexUpper"], params.get("scatterWeight"))

    @staticmethod
    def reconKeyFor(params, gated=False):
        """Everything that changes the OSEM result (not filtering or count scaling).
        The CT identity is part of the key when attenuation correction is on."""
        prefix = "gated" if gated else ""
        iters = params["gatedIters"] if gated else params["iters"]
        subsets = params["gatedSubsets"] if gated else params["subsets"]
        ctId = params["ctId"] if params["attenuationCorrection"] else None
        return (iters, subsets, params["attenuationCorrection"], ctId, params["indexPeak"],
                params["indexLower"], params["indexUpper"], params.get("scatterWeight"), prefix)

    @staticmethod
    def filterParams(params, gated=False):
        if not gated:
            return {k: params[k] for k in ("filter", "fwhm", "cutoff", "order")}
        return {"filter": params["gatedFilter"], "fwhm": params["gatedFwhm"],
                "cutoff": params["gatedCutoff"], "order": params["gatedOrder"]}

    def _setBusy(self, busy):
        self.busy = busy
        if busy:
            self.abortRequested = False
        self.applyButton.setEnabled(not busy)
        self.applyButton.setText("Reconstructing..." if busy else "Reconstruct")
        self.cancelButton.setText("Stop" if busy else "Cancel")
        self.cancelButton.setEnabled(True)
        # The CT and photopeak must not change underneath a running reconstruction
        for widget in (self.ctCombo, self.ctBrowseButton, self.peakCombo):
            widget.setEnabled(not busy)
        for widget in (self.quantGroup, self.suvGroup):
            widget.setEnabled(not busy and (widget is not self.suvGroup or self.quantGroup.checked))
        self._updateLockedWidgets()
        self._updateLoadState()
        if not busy:
            self.progressBar.setRange(0, 1)
            self.progressBar.setValue(0)
            self.progressLabel.setText("")
        # Don't animate while reconstructing: log updates pump the event loop
        if busy:
            self.stopProjCine()
            self.stopGateCine()
        else:
            if self.cineCheckBox.checked:
                self.startProjCine()
            if self.isGated and self.gateCineCheckBox.checked:
                self.startGateCine()

    def requestStop(self):
        if self.busy and not self.abortRequested:
            self.abortRequested = True
            self.cancelButton.setEnabled(False)
            self.log("Stopping after the current subset...")

    def onCancelClicked(self):
        if self.busy:
            self.requestStop()
        else:
            self.dialog.reject()

    def _progress(self, label, done, total, text):
        self.progressBar.setRange(0, total)
        self.progressBar.setValue(done)
        self.progressLabel.setText(text)
        slicer.app.processEvents()

    def _clearOverrideCursors(self):
        """The DICOM browser sets a busy cursor around load(); drop it while our window is open."""
        self.poppedCursors = 0
        while slicer.app.overrideCursor() is not None and self.poppedCursors < 20:
            slicer.app.restoreOverrideCursor()
            self.poppedCursors += 1

    def _restoreOverrideCursors(self):
        for _ in range(self.poppedCursors):
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
        self.poppedCursors = 0

    # ---------------- Run / buttons ----------------
    def run(self):
        """Show modal dialog. Returns (sitkImage, info) or (None, None) if cancelled.
        Per-gate images, if any, are left in self.gateImages."""
        self._clearOverrideCursors()
        try:
            self.log(f"Acquisition: {self.acquisition.get('manufacturer', '')} {self.acquisition.get('model', '')}, "
                     f"{self.acquisition.get('collimator') or 'collimator ?'}, "
                     f"{self.acquisition.get('isotope') or 'isotope ?'} - {describeTiming(self.acquisition)}")
            for note in self.acquisition.get("notes", []):
                self.log(f"Note: {note}")
            for warning in self.acquisition.get("warnings", []):
                self.log(f"WARNING: {warning}")
            for warning in self.headerWarnings:
                self.log(f"WARNING: {warning}")
            if self.isGated:
                self.log(f"Gated acquisition: {self.nGates} gates ({self.gateNumbers}). "
                         "The static image is reconstructed from the summed gate counts.")
                try:
                    import torch
                    if not torch.cuda.is_available():
                        self.log(f"Note: no GPU - reconstructing {self.nGates} gates on the CPU can take "
                                 "several minutes. Press Stop (or Esc) to interrupt.")
                except Exception:
                    pass
            if self.energyWindows:
                self.log(f"{len(self.energyWindows)} energy window(s): "
                         + ", ".join(w["label"] for w in self.energyWindows))
                peak = self.energyWindows[max(0, self.peakCombo.currentIndex)]
                self.log(f"Photopeak window: {peak['label']} (change it under Scatter correction if wrong)")
            if self.ctCandidates:
                self.log(f"{len(self.ctCandidates)} CT series found for this study:")
                for candidate in self.ctCandidates:
                    self.log(f"  - {candidate['label']}")
                self.log(f"Using: {self.ctCandidates[0]['label']} (attenuation correction on)")
                if self.ctFiles and not self.ctCleaned:
                    with slicer.util.WaitCursor():
                        self.ctFiles = cleanCtFiles(self.ctFiles, self.log)
                    self.ctCleaned = True
                    self._updateCtState()
            else:
                self.log("No CT found in this study - attenuation correction disabled. "
                         "Use 'Choose CT folder...' to point at one.")
            self._loadRawProjections()
            qt.QTimer.singleShot(0, self.onApply)  # initial recon with default settings
            result = self.dialog.exec_()
        finally:
            self.stopProjCine()
            self.stopGateCine()
            self._restoreOverrideCursors()
            self.renderer.cleanup()
            if self.logic is not None:
                self.logic.cleanup()
        if result != qt.QDialog.Accepted or self.filteredImage is None:
            self.gateImages = []
            return None, None
        return self.filteredImage, self.info

    def _validate(self, params):
        """Guard rails checked before any reconstruction starts. Returns params (possibly
        with subsets clamped) or raises ValueError with a message for the user."""
        problem = self.scatterProblem(params)
        if problem:
            raise ValueError(f"Scatter correction: {problem}")
        if self.logic is not None:
            nViews = self.logic.nViews
            for spin, key in ((self.subsetsSpin, "subsets"),
                              (getattr(self, "gatedSubsetsSpin", None), "gatedSubsets")):
                if spin is None or key not in params:
                    continue
                spin.setMaximum(nViews)
                if params[key] > nViews:
                    self.log(f"Subsets reduced from {params[key]} to {nViews} (number of projection views).")
                    params[key] = nViews
                    spin.setValue(nViews)
            for iters, subsets, what in ((params["iters"], params["subsets"], "static"),
                                         (params.get("gatedIters"), params.get("gatedSubsets"), "gated")):
                if iters and subsets and iters * subsets > 120:
                    self.log(f"Note: {iters} x {subsets} = {iters * subsets} subset updates ({what}) - "
                             "images will be noisy and slow; 2-4 iterations with 6-10 subsets is usual.")
        return params

    def onApply(self):
        if self.busy:
            return False
        import SimpleITK as sitk
        params = self.currentParams()
        self._setBusy(True)
        try:
            if self.logic is None or self.logic.indexPeak != params["indexPeak"]:
                if self.logic is not None:
                    self.logic.cleanup()
                    self.logic = None
                self.reconImage = None
                self.reconKey = None
                self.gateRawImages = []
                self.gateRawStats = []
                self.gateReconKey = None
                self.logic = SpectReconLogic(self.inputDcm, self.log, self.ctFiles, params["indexPeak"],
                                             progress=self._progress,
                                             shouldAbort=lambda: self.abortRequested)
            if self.logic.ctFiles != self.ctFiles:
                self.logic.ctFiles = self.ctFiles
                self.logic.attenuationMap = None
            params = self._validate(params)

            scatter = self.scatterArgument(params)
            key = self.reconKeyFor(params)
            if self.reconImage is None or key != self.reconKey:
                self.reconImage, self.reconStats = self.logic.reconstruct(
                    params["iters"], params["subsets"], params["attenuationCorrection"], scatter)
                self.reconKey = key
            else:
                self.log("Static OSEM settings unchanged - re-scaling/re-filtering existing reconstruction.")

            scaled, factor, scalingNote = scaleCounts(self.reconImage, self.reconStats, params["scaling"])
            self.log(f"Count scaling: {scalingNote}, factor {factor:.6g}")
            units, quantInfo = "counts", {}
            if params["quant"]:
                scaled, units, quantInfo = self._quantify(scaled, params)
            self.units = units
            self.filteredImage = applyFilter(scaled, self.filterParams(params), self.log)
            stats = self.reconStats
            self.info = {
                "Algorithm": "OSEM",
                "Iterations": params["iters"],
                "Subsets": params["subsets"],
                "AttenuationCorrection": "CT" if params["attenuationCorrection"] else "None",
                "ScatterCorrection": scatterDescription(params),
                "PhotopeakWindow": self.energyWindows[params["indexPeak"]]["label"]
                                   if params["indexPeak"] < len(self.energyWindows) else params["indexPeak"] + 1,
                "PostFilter": filterDescription(self.filterParams(params)),
                "CountScaling": params["scaling"],
                "CountScalingFactor": f"{factor:.6g}",
                "Views": stats["nViews"],
                "PhotopeakCounts": f"{stats['photopeakCounts']:.0f}",
                "ScatterCounts": f"{stats['scatterCounts']:.0f}",
                "DataConsistency": "n/a" if stats["consistency"] is None else f"{stats['consistency']:.3f}",
                "PyTomographyVersion": pytomographyVersion(),
                "SourceFile": self.inputDcm,
                "Units": units,
                # Multiply voxel values by this to get acquisition counts (used by SPECT Calibration)
                "ToAcquisitionCounts": f"{stats['nViews'] / factor:.9g}" if units == "counts" else "n/a",
            }
            self.info.update(self._acquisitionAttributes(params))
            self.info.update(quantInfo)
            if params["attenuationCorrection"]:
                self.info["CTMatch"] = self.ctMatchType
            if self.isGated:
                self.info["StaticFrom"] = f"sum of {self.nGates} gates"
            self._setReconArray(sitk.GetArrayFromImage(self.filteredImage))

            sp = self.filteredImage.GetSpacing()
            self.log(f"Voxel spacing (mm): {sp[0]:.2f} x {sp[1]:.2f} x {sp[2]:.2f}  |  Static preview updated.")

            self._applyGates(params, scatter)
            self.appliedParams = params
            self.lastRunFailed = False
            return True
        except ReconstructionAborted:
            self.appliedParams = None
            self.lastRunFailed = True
            self.log("Stopped by user. Previous results are kept; press Reconstruct to run again.")
            return False
        except Exception as e:
            self.appliedParams = None
            self.lastRunFailed = True
            self.log(f"ERROR: {e}")
            logging.error(traceback.format_exc())
            return False
        finally:
            self._setBusy(False)

    def _applyGates(self, params, scatter):
        import SimpleITK as sitk
        if not (self.isGated and params.get("gated")):
            self.gateImages = []
            self.gateArrays = []
            self.stopGateCine()
            if self.isGated:
                self.gatedInfo.setText("gates not reconstructed")
            return

        gateKey = self.reconKeyFor(params, gated=True)
        if not self.gateRawImages or gateKey != self.gateReconKey:
            # Never hand stale gates to Slicer if this run fails half way
            self.gateImages = []
            images, stats = [], []
            for i, gate in enumerate(self.gateNumbers):
                self.log(f"--- Gate {gate} ({i + 1} / {self.nGates}) ---")
                projectionSet = self.logic.gateSet(gate)
                image, gateStats = self.logic.reconstruct(
                    params["gatedIters"], params["gatedSubsets"],
                    params["attenuationCorrection"], scatter, projectionSet=projectionSet)
                images.append(image)
                stats.append(gateStats)
            self.gateRawImages, self.gateRawStats = images, stats
            self.gateReconKey = gateKey
        else:
            self.log("Gated OSEM settings unchanged - re-scaling/re-filtering existing gate reconstructions.")

        gatedFilter = self.filterParams(params, gated=True)
        self.gateImages = [applyFilter(scaleCounts(image, stats, params["scaling"])[0], gatedFilter, self.log)
                           for image, stats in zip(self.gateRawImages, self.gateRawStats)]
        self.gateArrays = [sitk.GetArrayFromImage(image) for image in self.gateImages]
        self.gateMax = max((display.displayReference(a) for a in self.gateArrays if a.size), default=1.0)
        self.gateIndex = min(self.gateIndex, len(self.gateArrays) - 1)
        self.info["Gates"] = self.nGates
        self.info["GatedIterations"] = params["gatedIters"]
        self.info["GatedSubsets"] = params["gatedSubsets"]
        self.info["GatedPostFilter"] = filterDescription(gatedFilter)
        if params.get("quant"):
            # A gate's counting time is not the acquisition duration, so gates are not converted
            self.info["GatedUnits"] = "acquisition counts (not quantified)"
            self.log("Note: gates stay in acquisition counts; only the static image is quantified.")
        self.updateGatedViews(self.reconSlider.value)
        if self.gateCineCheckBox.checked:
            self.startGateCine()

    def _isUpToDate(self):
        """The preview (what Load would hand to Slicer) was made with the current settings."""
        return (self.filteredImage is not None and self.appliedParams is not None
                and self.appliedParams == self.currentParams())

    def _updateLoadState(self, *args):
        """Load is only possible for a preview that matches the settings on screen."""
        if not hasattr(self, "loadButton"):
            return
        upToDate = self._isUpToDate()
        self.loadButton.setEnabled(upToDate and not self.busy)
        if self.busy:
            text = ""
        elif self.lastRunFailed:
            text = ("<span style='color:#d04040'>&#9888; The last reconstruction did not finish (see the log) - "
                    "press <b>Reconstruct</b> again before loading.</span>")
        elif self.filteredImage is None:
            text = "<i>Press Reconstruct to preview the result.</i>"
        elif not upToDate:
            text = ("<span style='color:#d08000'>&#9888; Settings changed since the last reconstruction - "
                    "press <b>Reconstruct</b> to update the preview before loading.</span>")
        else:
            text = "<span style='color:#2e8b3d'>&#10004; Preview matches the settings - ready to load.</span>"
        self.staleNote.setText(text)

    def _watchSettings(self):
        """Re-check the Load state whenever any setting in the side panel changes."""
        watched = [self.itersSpin, self.subsetsSpin, self.filterCombo, self.fwhmSpin, self.cutoffSpin,
                   self.orderSpin, self.scalingCombo, self.acCheckBox, self.ctCombo, self.scatterCheckBox,
                   self.peakCombo, self.lowerCombo, self.upperCombo, self.scatterWeightSpin,
                   self.quantGroup, self.calibrationCombo,
                   self.durationSpin, self.midTimeEdit, self.midTimeFromDicom, self.suvGroup, self.suvTypeCombo,
                   self.syringeSpin, self.syringeTimeEdit, self.residualSpin, self.residualTimeEdit,
                   self.weightSpin, self.heightSpin, self.sexCombo]
        if self.isGated:
            watched += [self.gatedEnableCheckBox, self.gatedItersSpin, self.gatedSubsetsSpin,
                        self.gatedFilterCombo, self.gatedFwhmSpin, self.gatedCutoffSpin, self.gatedOrderSpin]
        for widget in watched:
            if isinstance(widget, qt.QDateTimeEdit):
                signal = "dateTimeChanged(QDateTime)"
            elif isinstance(widget, (qt.QDoubleSpinBox, ctk.ctkSliderWidget)):
                signal = "valueChanged(double)"
            elif isinstance(widget, qt.QSpinBox):
                signal = "valueChanged(int)"
            elif isinstance(widget, qt.QComboBox):
                signal = "currentIndexChanged(int)"
            else:  # check boxes and checkable group boxes
                signal = "toggled(bool)"
            widget.connect(signal, self._updateLoadState)

    def onLoad(self):
        # Only a preview made with the current settings can be loaded
        if not self._isUpToDate():
            self._updateLoadState()
            return
        self.dialog.accept()

    # ---------------- Views ----------------
    def _petColors(self):
        if self._petTable is None:
            self._petTable = display.petColorTable()
        return self._petTable

    def _opacity(self):
        return self.opacitySlider.value / 100.0

    def _setReconArray(self, arr):
        self.reconArray = arr  # [z, y, x]
        self.reconMax = display.displayReference(arr)
        if arr.size:
            self.log(f"Display window: 0 to {display.WINDOW_FRACTION * self.reconMax:.4g} {self.units} "
                     f"({display.DISPLAY_PERCENTILE:.2f}th percentile {self.reconMax:.4g}, "
                     f"volume max {float(arr.max()):.4g})")
        nz = arr.shape[0]
        current = self.reconSlider.value
        self.reconSlider.setRange(0, nz - 1)
        if not self.reconSlider.isEnabled() or current >= nz:
            current = nz // 2
        self.reconSlider.setEnabled(True)
        self.reconSlider.setValue(current)
        self.updateReconView(current)

    def updateReconView(self, index):
        if self.reconArray is None:
            return
        sl = self.reconArray[index]
        self.reconView.setPixmap(self.renderer.toPixmap(display.colorize(sl, self.reconMax, self._petColors())))
        self.reconInfo.setText(f"Slice {index + 1} / {self.reconArray.shape[0]}   |   slice sum: {sl.sum():.0f}")
        self.updateSliceViews(index)

    def updateSliceViews(self, index):
        self.updateFusionAndCtViews(index)
        self.updateGatedViews(index)

    def _ensureCtArray(self):
        """Resample the CT onto the reconstruction grid (both are in DICOM patient coordinates,
        so an identity transform is the correct resampling for a hybrid SPECT/CT)."""
        if self.filteredImage is None or not self.ctFiles or self.ctPreviewFailed:
            return None
        key = (self.filteredImage.GetSize(), self.filteredImage.GetSpacing(), self.filteredImage.GetOrigin())
        if self.ctArray is not None and key == self.ctGeometryKey:
            return self.ctArray
        import SimpleITK as sitk
        try:
            self.log(f"Resampling CT ({len(self.ctFiles)} slices) onto the reconstruction grid...")
            reader = sitk.ImageSeriesReader()
            files = self.ctFiles if self.ctCleaned else cleanCtFiles(self.ctFiles, self.log)
            reader.SetFileNames(files)
            ctImage = sitk.Cast(reader.Execute(), sitk.sitkFloat32)
            resampled = sitk.Resample(ctImage, self.filteredImage, sitk.Transform(),
                                      sitk.sitkLinear, -1000.0, sitk.sitkFloat32)
            self.ctArray = sitk.GetArrayFromImage(resampled)
            self.ctGeometryKey = key
            return self.ctArray
        except Exception as e:
            # Only the preview is affected: keep the CT for attenuation correction, and say so
            self.log(f"Could not load CT for preview: {e}")
            logging.error(traceback.format_exc())
            self.ctPreviewFailed = True  # don't retry every slice change
            self._updateCtState()
            return None

    def _ctGraySlice(self, index):
        """CT slice mapped to 0-1 with an abdominal window (level 40, width 400)."""
        ct = self._ensureCtArray()
        if ct is None or index >= ct.shape[0]:
            return None, None
        ctSlice = ct[index]
        return display.ctGray(ctSlice), ctSlice

    def updateFusionAndCtViews(self, index):
        if not self.ctFiles or self.ctPreviewFailed or self.reconArray is None:
            return
        ctGray, ctSlice = self._ctGraySlice(index)
        if ctGray is None:
            self.fusionPanel.setVisible(False)
            self.ctPanel.setVisible(False)
            return

        self.ctView.setPixmap(self.renderer.toPixmap(display.grayscale(ctGray, 1.0)))
        self.ctInfo.setText(f"Slice {index + 1} / {self.reconArray.shape[0]}   |   "
                            f"HU range: {ctSlice.min():.0f} to {ctSlice.max():.0f}")

        fused = display.fuse(self.reconArray[index], self.reconMax, ctGray, self._opacity())
        self.fusionView.setPixmap(self.renderer.toPixmap(fused))
        self.fusionInfo.setText(f"Slice {index + 1}   |   opacity {self.opacitySlider.value}%")

    def updateGatedViews(self, index):
        if not self.isGated or not self.gateArrays:
            return
        gateArray = self.gateArrays[self.gateIndex]
        if index >= gateArray.shape[0]:
            return
        gateSlice = gateArray[index]
        self.gatedView.setPixmap(self.renderer.toPixmap(display.colorize(gateSlice, self.gateMax, self._petColors())))
        self.gatedInfo.setText(f"Gate {self.gateIndex + 1} / {len(self.gateArrays)}   |   "
                               f"slice {index + 1} / {gateArray.shape[0]}")
        if self.ctFiles:
            ctGray, _ = self._ctGraySlice(index)
            if ctGray is not None:
                fused = display.fuse(gateSlice, self.gateMax, ctGray, self._opacity())
                self.gatedFusionView.setPixmap(self.renderer.toPixmap(fused))
                self.gatedFusionInfo.setText(f"Gate {self.gateIndex + 1}   |   slice {index + 1}")

    def _loadRawProjections(self):
        """Read the projection frames once; _showRawProjections picks the photopeak window."""
        try:
            import pydicom
            ds = pydicom.dcmread(self.inputDcm)
            frames = np.asarray(ds.pixel_array, dtype=np.float32)
            if frames.ndim == 2:
                frames = frames[np.newaxis]
            n = frames.shape[0]

            def vector(name):
                value = getattr(ds, name, None)
                if value is None:
                    return None
                arr = np.atleast_1d(np.asarray(value))
                return arr if arr.size == n else None

            self.rawProjection = {"frames": frames, "ewv": vector("EnergyWindowVector"),
                                  "det": vector("DetectorVector"), "gates": vector("TimeSlotVector")}
            self._showRawProjections()
        except Exception as e:
            self.projView.setText("Could not read projections")
            logging.error(traceback.format_exc())
            self.logBox.appendPlainText(f"Could not read raw projections for preview: {e}")

    def _showRawProjections(self):
        """Show the frames of the current photopeak window, like the reconstruction."""
        raw = self.rawProjection
        frames = raw["frames"]
        n = frames.shape[0]
        peakWindow = max(0, self.peakCombo.currentIndex) + 1
        sel = (raw["ewv"] == peakWindow) if raw["ewv"] is not None else np.ones(n, dtype=bool)
        if not sel.any():
            sel = np.ones(n, dtype=bool)

        self.projArray = frames[sel]
        self.projDetectors = raw["det"][sel] if raw["det"] is not None else None
        self.projGates = raw["gates"][sel] if raw["gates"] is not None else None
        self.projMax = float(self.projArray.max()) if self.projArray.size else 1.0
        if self.projMax <= 0:
            self.projMax = 1.0
        self.projSlider.setRange(0, self.projArray.shape[0] - 1)
        self.projSlider.setEnabled(True)
        index = min(self.projSlider.value, self.projArray.shape[0] - 1)
        self.projSlider.setValue(index)
        self.updateProjView(index)
        if self.cineCheckBox.checked and not self.busy:
            self.startProjCine()

    def updateProjView(self, index):
        if self.projArray is None:
            return
        self.projView.setPixmap(self.renderer.toPixmap(display.grayscale(self.projArray[index], self.projMax)))
        text = f"Projection {index + 1} / {self.projArray.shape[0]}"
        if self.projDetectors is not None:
            text += f"   |   Detector {int(self.projDetectors[index])}"
        if self.projGates is not None:
            text += f"   |   Gate {int(self.projGates[index])}"
        self.projInfo.setText(text)
