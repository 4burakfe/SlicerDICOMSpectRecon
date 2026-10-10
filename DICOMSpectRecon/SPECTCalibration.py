"""SPECT Calibration - camera calibration factors (cps/MBq) for quantitative SPECT.

Workflow:
    1. Load a phantom of known activity with the DICOM SPECT reconstruction plugin,
       with CT attenuation correction (required) and scatter correction if wanted.
    2. Pick that reconstruction here, choose the VOI (phantom outline from CT with a
       margin, a segment, or the whole image) and enter the activity and times.
    3. Compute: CF = acquisition counts in the VOI / counting duration / activity at mid-scan.
    4. Save it to the calibration library (a JSON file, may be on a shared drive).
The reconstruction window then offers the saved calibrations under "Quantification".
"""

import datetime
import logging
import math
import os
import sys
import traceback

import numpy as np
import qt
import ctk
import vtk
import slicer
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleWidget,
                                           ScriptedLoadableModuleLogic)

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from DICOMSpectReconLib import quantification as quant  # noqa: E402

DATETIME_FORMAT = "yyyy-MM-dd HH:mm:ss"
MBQ_PER_MCI = 37.0


class SPECTCalibration(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "SPECT Calibration"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir"]
        parent.helpText = (
            "Derives a SPECT camera calibration factor (cps/MBq) from a reconstructed phantom of "
            "known activity and stores it in a calibration library (JSON). The DICOM SPECT "
            "reconstruction window uses these calibrations to produce Bq/ml and SUV images. "
            "The phantom must be reconstructed with CT attenuation correction.")
        parent.acknowledgementText = "Reconstruction powered by PyTomography."


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class SPECTCalibrationLogic(ScriptedLoadableModuleLogic):

    @staticmethod
    def reconAttributes(volumeNode):
        """The SPECTRecon.* attributes of a volume, without the prefix."""
        attributes = {}
        for name in volumeNode.GetAttributeNames() or []:
            if name.startswith("SPECTRecon."):
                attributes[name[len("SPECTRecon."):]] = volumeNode.GetAttribute(name)
        return attributes

    @staticmethod
    def wholeImageSum(volumeNode):
        return float(np.asarray(slicer.util.arrayFromVolume(volumeNode), dtype=np.float64).sum())

    @staticmethod
    def segmentSum(volumeNode, segmentationNode, segmentId):
        """Sum of the volume inside a segment, resampled onto the volume's grid."""
        labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        try:
            ids = vtk.vtkStringArray()
            ids.InsertNextValue(segmentId)
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                segmentationNode, ids, labelmap, volumeNode)
            mask = slicer.util.arrayFromVolume(labelmap) > 0
            values = slicer.util.arrayFromVolume(volumeNode)
            if mask.shape != values.shape:
                raise ValueError("The segment could not be resampled onto the SPECT grid.")
            return float(np.asarray(values, dtype=np.float64)[mask].sum()), int(mask.sum())
        finally:
            slicer.mrmlScene.RemoveNode(labelmap)

    @staticmethod
    def ctOutlineSum(volumeNode, ctNode, thresholdHu=-500.0, marginMm=25.0):
        """Sum of the volume inside the CT outline of the phantom (HU > threshold),
        dilated by a margin so counts blurred past the edges are included."""
        import SimpleITK as sitk
        import sitkUtils
        spect = sitkUtils.PullVolumeFromSlicer(volumeNode)
        ct = sitkUtils.PullVolumeFromSlicer(ctNode)
        ctOnSpect = sitk.Resample(sitk.Cast(ct, sitk.sitkFloat32), spect, sitk.Transform(),
                                  sitk.sitkLinear, -1000.0, sitk.sitkFloat32)
        mask = sitk.BinaryThreshold(ctOnSpect, lowerThreshold=float(thresholdHu), upperThreshold=1.0e9,
                                    insideValue=1, outsideValue=0)
        if marginMm > 0:
            radius = [max(1, int(math.ceil(marginMm / s))) for s in spect.GetSpacing()]
            mask = sitk.BinaryDilate(mask, radius, sitk.sitkBall)
        maskArray = sitk.GetArrayFromImage(mask) > 0
        if not maskArray.any():
            raise ValueError("The CT outline is empty on the SPECT grid - does the CT overlap the SPECT?")
        values = sitk.GetArrayFromImage(spect).astype(np.float64)
        return float(values[maskArray].sum()), int(maskArray.sum())


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
def _dateTimeEdit():
    edit = qt.QDateTimeEdit()
    edit.setDisplayFormat(DATETIME_FORMAT)
    edit.setCalendarPopup(True)
    edit.setDateTime(qt.QDateTime.currentDateTime())
    return edit


def _setEdit(edit, isoText):
    if isoText:
        edit.setDateTime(qt.QDateTime.fromString(str(isoText)[:19], "yyyy-MM-ddTHH:mm:ss"))


def _fromEdit(edit):
    return datetime.datetime.strptime(edit.dateTime.toString(DATETIME_FORMAT), "%Y-%m-%d %H:%M:%S")


def _doubleSpin(minimum, maximum, decimals, suffix, value=0.0):
    spin = qt.QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setSuffix(suffix)
    spin.setValue(value)
    return spin


class SPECTCalibrationWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = SPECTCalibrationLogic()
        self.record = None
        layout = self.layout

        # ---- library
        libraryBox = ctk.ctkCollapsibleButton()
        libraryBox.text = "Calibration library"
        layout.addWidget(libraryBox)
        libraryForm = qt.QFormLayout(libraryBox)
        self.libraryLabel = qt.QLabel("")
        self.libraryLabel.setWordWrap(True)
        self.libraryButton = qt.QPushButton("Choose library file...")
        libraryForm.addRow("File:", self.libraryLabel)
        libraryForm.addRow(self.libraryButton)
        self.table = qt.QTableWidget()
        self.tableColumns = ["Name", "Created", "Isotope", "Collimator", "Camera", "SC", "cps/MBq", "Detectors"]
        self.table.setColumnCount(len(self.tableColumns))
        self.table.setHorizontalHeaderLabels(self.tableColumns)
        self.table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(140)
        libraryForm.addRow(self.table)
        buttons = qt.QHBoxLayout()
        self.refreshButton = qt.QPushButton("Refresh")
        self.deleteButton = qt.QPushButton("Delete selected")
        buttons.addWidget(self.refreshButton)
        buttons.addWidget(self.deleteButton)
        libraryForm.addRow(buttons)

        # ---- phantom reconstruction
        phantomBox = ctk.ctkCollapsibleButton()
        phantomBox.text = "Phantom reconstruction"
        layout.addWidget(phantomBox)
        phantomForm = qt.QFormLayout(phantomBox)
        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeSelector.setToolTip("Reconstruction loaded with the DICOM SPECT plugin (CT attenuation correction)")
        self.volumeInfo = qt.QLabel("")
        self.volumeInfo.setWordWrap(True)
        phantomForm.addRow("Volume:", self.volumeSelector)
        phantomForm.addRow(self.volumeInfo)

        # ---- VOI
        voiBox = ctk.ctkCollapsibleButton()
        voiBox.text = "Volume of interest"
        layout.addWidget(voiBox)
        voiForm = qt.QFormLayout(voiBox)
        self.voiCtRadio = qt.QRadioButton("Phantom outline from CT, plus a margin")
        self.voiSegmentRadio = qt.QRadioButton("Segment")
        self.voiWholeRadio = qt.QRadioButton("Whole image")
        self.voiCtRadio.setChecked(True)
        self.voiWholeRadio.setToolTip("Includes positive OSEM noise in empty air - can bias low-count phantoms upward")
        self.ctSelector = slicer.qMRMLNodeComboBox()
        self.ctSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.ctSelector.addEnabled = False
        self.ctSelector.removeEnabled = False
        self.ctSelector.noneEnabled = True
        self.ctSelector.setMRMLScene(slicer.mrmlScene)
        self.thresholdSpin = _doubleSpin(-1000.0, 1000.0, 0, " HU", -500.0)
        self.marginSpin = _doubleSpin(0.0, 100.0, 1, " mm", 25.0)
        self.marginSpin.setToolTip("Counts blur past the phantom edge by about two resolution widths")
        self.segmentSelector = slicer.qMRMLSegmentSelectorWidget()
        self.segmentSelector.setMRMLScene(slicer.mrmlScene)
        voiForm.addRow(self.voiCtRadio)
        voiForm.addRow("   CT:", self.ctSelector)
        voiForm.addRow("   Threshold:", self.thresholdSpin)
        voiForm.addRow("   Margin:", self.marginSpin)
        voiForm.addRow(self.voiSegmentRadio)
        voiForm.addRow("   Segment:", self.segmentSelector)
        voiForm.addRow(self.voiWholeRadio)

        # ---- activity and times
        activityBox = ctk.ctkCollapsibleButton()
        activityBox.text = "Activity and times"
        layout.addWidget(activityBox)
        activityForm = qt.QFormLayout(activityBox)
        self.isotopeCombo = qt.QComboBox()
        self.isotopeCombo.addItems(list(quant.ISOTOPES.keys()) + [quant.OTHER_ISOTOPE])
        self.halfLifeSpin = _doubleSpin(0.0, 1.0e6, 4, " h", quant.ISOTOPES["Tc-99m"]["halfLife"] / 3600.0)
        self.activitySpin = _doubleSpin(0.0, 1.0e6, 3, "")
        self.activityUnitCombo = qt.QComboBox()
        self.activityUnitCombo.addItems(["MBq", "mCi"])
        activityRow = qt.QHBoxLayout()
        activityRow.addWidget(self.activitySpin, 1)
        activityRow.addWidget(self.activityUnitCombo)
        self.measuredTimeEdit = _dateTimeEdit()
        self.residualSpin = _doubleSpin(0.0, 1.0e6, 3, "")
        self.residualTimeEdit = _dateTimeEdit()
        self.durationSpin = _doubleSpin(0.0, 1.0e6, 1, " s")
        self.durationSpin.setToolTip("Counting time summed over all views of all detectors (from DICOM, editable)")
        self.timingNote = qt.QLabel("")
        self.timingNote.setWordWrap(True)
        self.midTimeEdit = _dateTimeEdit()
        self.midFromDicom = qt.QCheckBox("from DICOM")
        self.midFromDicom.setChecked(True)
        midRow = qt.QHBoxLayout()
        midRow.addWidget(self.midTimeEdit, 1)
        midRow.addWidget(self.midFromDicom)
        activityForm.addRow("Isotope:", self.isotopeCombo)
        activityForm.addRow("Half-life:", self.halfLifeSpin)
        activityForm.addRow("Activity (dose calibrator):", activityRow)
        activityForm.addRow("Measured at:", self.measuredTimeEdit)
        activityForm.addRow("Residual (same unit):", self.residualSpin)
        activityForm.addRow("Residual measured at:", self.residualTimeEdit)
        activityForm.addRow("Total view time:", self.durationSpin)
        activityForm.addRow(self.timingNote)
        activityForm.addRow("Scan mid-time:", midRow)

        # ---- compute / save
        resultBox = ctk.ctkCollapsibleButton()
        resultBox.text = "Calibration"
        layout.addWidget(resultBox)
        resultForm = qt.QFormLayout(resultBox)
        self.computeButton = qt.QPushButton("Compute calibration factor")
        self.results = qt.QTextEdit()
        self.results.setReadOnly(True)
        self.results.setMinimumHeight(160)
        self.nameEdit = qt.QLineEdit()
        self.nameEdit.setPlaceholderText("e.g. Symbia T2 LEHR Tc-99m 2026-09")
        self.notesEdit = qt.QLineEdit()
        self.saveButton = qt.QPushButton("Save to library")
        self.saveButton.setEnabled(False)
        resultForm.addRow(self.computeButton)
        resultForm.addRow(self.results)
        resultForm.addRow("Name:", self.nameEdit)
        resultForm.addRow("Notes:", self.notesEdit)
        resultForm.addRow(self.saveButton)
        layout.addStretch(1)

        # ---- connections
        self.libraryButton.connect("clicked()", self.onChooseLibrary)
        self.refreshButton.connect("clicked()", self.refreshTable)
        self.deleteButton.connect("clicked()", self.onDelete)
        self.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onVolumeChanged)
        self.isotopeCombo.connect("currentIndexChanged(int)", self.onIsotopeChanged)
        self.midFromDicom.connect("toggled(bool)", self.onMidSourceChanged)
        self.computeButton.connect("clicked()", self.onCompute)
        self.saveButton.connect("clicked()", self.onSave)
        for widget in (self.activitySpin, self.residualSpin, self.durationSpin, self.halfLifeSpin):
            widget.connect("valueChanged(double)", self._invalidate)
        for radio in (self.voiCtRadio, self.voiSegmentRadio, self.voiWholeRadio):
            radio.connect("toggled(bool)", self._invalidate)

        self._updateLibraryLabel()
        self.refreshTable()
        self.onVolumeChanged(self.volumeSelector.currentNode())

    # ---------------- library ----------------
    def _updateLibraryLabel(self):
        self.libraryLabel.setText(quant.calibrationLibraryPath())

    def onChooseLibrary(self):
        if quant.chooseCalibrationLibrary(slicer.util.mainWindow()):
            self._updateLibraryLabel()
            self.refreshTable()

    def refreshTable(self):
        path = quant.calibrationLibraryPath()
        try:
            self.records = quant.loadLibrary(path)
        except Exception as e:
            self.records = []
            slicer.util.errorDisplay(f"Could not read the calibration library:\n{path}\n\n{e}")
        self.records.sort(key=lambda r: r.get("created", ""), reverse=True)
        self.table.setRowCount(len(self.records))
        for row, record in enumerate(self.records):
            camera = record.get("camera", {})
            detectors = record.get("detectors")
            values = [record.get("name", ""), str(record.get("created", ""))[:16].replace("T", " "),
                      record.get("isotope", ""), record.get("collimator", ""),
                      f"{camera.get('model', '')} {camera.get('serial', '')}".strip(),
                      record.get("scatter", {}).get("method", "None"),
                      f"{float(record.get('cpsPerMBq', 0)):.4f}",
                      str(detectors) if detectors else "?"]
            for column, value in enumerate(values):
                self.table.setItem(row, column, qt.QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()

    def onDelete(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if not rows:
            return
        names = ", ".join(self.records[r].get("name", "") for r in rows)
        if not slicer.util.confirmYesNoDisplay(f"Delete {len(rows)} calibration(s) from the library?\n\n{names}"):
            return
        path = quant.calibrationLibraryPath()
        for row in rows:
            quant.deleteCalibration(path, self.records[row]["id"])
        self.refreshTable()

    # ---------------- inputs ----------------
    def onVolumeChanged(self, node):
        self._invalidate()
        if node is None:
            self.volumeInfo.setText("<i>Select the phantom reconstruction.</i>")
            return
        attributes = self.logic.reconAttributes(node)
        if not attributes:
            self.volumeInfo.setText("<span style='color:#d04040'>This volume was not reconstructed with the "
                                    "DICOM SPECT plugin.</span>")
            return
        ac = attributes.get("AttenuationCorrection", "None")
        parts = [f"AC: {ac}", f"SC: {attributes.get('ScatterMethod', attributes.get('ScatterCorrection', '?'))}",
                 f"{attributes.get('Isotope') or 'isotope ?'}", f"{attributes.get('Collimator') or 'collimator ?'}",
                 f"{attributes.get('CameraModel', '')} {attributes.get('CameraSerial', '')}".strip(),
                 f"{attributes.get('Detectors', '?')} detector(s)", f"units: {attributes.get('Units', 'counts')}"]
        text = ", ".join(p for p in parts if p)
        if ac != "CT":
            text = f"<span style='color:#d04040'>&#9888; Not usable (needs CT attenuation correction). {text}</span>"
        self.volumeInfo.setText(text)

        isotope = attributes.get("Isotope", "")
        index = self.isotopeCombo.findText(isotope) if isotope else -1
        if index >= 0:
            self.isotopeCombo.setCurrentIndex(index)
        halfLife = quant._number(attributes.get("HalfLifeSec"))
        if halfLife:
            self.halfLifeSpin.setValue(halfLife / 3600.0)
        duration = quant._number(attributes.get("TotalViewTimeSec"))
        if duration:
            self.durationSpin.setValue(duration)
        detectors = int(quant._number(attributes.get("Detectors"), 1) or 1)
        evidence = attributes.get("DetectorsEvidence", "")
        if not duration:
            timing = ("<span style='color:#d04040'>No total view time on this volume - reconstruct it again "
                      "with this version.</span>")
        elif detectors > 1:
            timing = (f"{detectors} detectors detected ({evidence}): {duration / detectors:.1f} s each, "
                      f"{duration:.1f} s total.")
        else:
            timing = f"1 detector recorded ({evidence})."
        midSource = attributes.get("ScanMidTimeSource", "")
        self.timingNote.setText(f"<small>{timing}<br>Mid-time: {midSource or 'from DICOM'}.</small>")
        self._dicomMid = attributes.get("ScanMidTime") or attributes.get("AcquisitionStart") or ""
        if self.midFromDicom.checked:
            _setEdit(self.midTimeEdit, self._dicomMid)
        start = attributes.get("AcquisitionStart")
        if start:
            _setEdit(self.measuredTimeEdit, start)
            _setEdit(self.residualTimeEdit, start)
        # Suggest a CT loaded from DICOM (e.g. the one the plugin showed the SPECT on)
        if self.ctSelector.currentNode() is None:
            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
            for candidate in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
                item = shNode.GetItemByDataNode(candidate)
                if candidate is not node and item and \
                        (shNode.GetItemAttribute(item, "DICOM.Modality") or "").upper() == "CT":
                    self.ctSelector.setCurrentNode(candidate)
                    break

    def onIsotopeChanged(self, index):
        self._invalidate()
        name = self.isotopeCombo.currentText
        halfLife = quant.tableHalfLife(name)
        if halfLife:
            self.halfLifeSpin.setValue(halfLife / 3600.0)

    def onMidSourceChanged(self, fromDicom):
        self.midTimeEdit.setEnabled(not fromDicom)
        if fromDicom and getattr(self, "_dicomMid", ""):
            _setEdit(self.midTimeEdit, self._dicomMid)
        self._invalidate()

    def _invalidate(self, *args):
        self.record = None
        self.saveButton.setEnabled(False)

    # ---------------- compute / save ----------------
    def _voiSums(self, node):
        imageSum = self.logic.wholeImageSum(node)
        if self.voiWholeRadio.checked:
            return imageSum, imageSum, "whole image"
        if self.voiSegmentRadio.checked:
            segmentation = self.segmentSelector.currentNode()
            segmentId = self.segmentSelector.currentSegmentID()
            if segmentation is None or not segmentId:
                raise ValueError("Select a segment.")
            voiSum, voxels = self.logic.segmentSum(node, segmentation, segmentId)
            name = segmentation.GetSegmentation().GetSegment(segmentId).GetName()
            return voiSum, imageSum, f"segment '{name}' ({voxels} voxels)"
        ct = self.ctSelector.currentNode()
        if ct is None:
            raise ValueError("Select the CT for the phantom outline (or choose another VOI type).")
        voiSum, voxels = self.logic.ctOutlineSum(node, ct, self.thresholdSpin.value, self.marginSpin.value)
        return voiSum, imageSum, (f"CT outline > {self.thresholdSpin.value:.0f} HU + "
                                  f"{self.marginSpin.value:.0f} mm ({voxels} voxels)")

    def onCompute(self):
        self._invalidate()
        node = self.volumeSelector.currentNode()
        if node is None:
            self.results.setHtml("<b>Select the phantom reconstruction.</b>")
            return
        try:
            with slicer.util.WaitCursor():
                attributes = self.logic.reconAttributes(node)
                voiSum, imageSum, voiText = self._voiSums(node)
                unit = MBQ_PER_MCI if self.activityUnitCombo.currentText == "mCi" else 1.0
                record, messages = quant.evaluatePhantom(
                    attributes, voiSum, imageSum,
                    activityMBq=self.activitySpin.value * unit,
                    measuredTime=_fromEdit(self.measuredTimeEdit),
                    midTime=_fromEdit(self.midTimeEdit),
                    durationSec=self.durationSpin.value,
                    halfLifeSec=self.halfLifeSpin.value * 3600.0,
                    isotope=self.isotopeCombo.currentText,
                    residualMBq=self.residualSpin.value * unit,
                    residualTime=_fromEdit(self.residualTimeEdit),
                    voiDescription=voiText)
                if record is not None:
                    try:
                        drift = quant.driftMessage(record, quant.loadLibrary(quant.calibrationLibraryPath()))
                        if drift:
                            messages.append(drift)
                    except Exception:
                        pass
        except Exception as e:
            logging.error(traceback.format_exc())
            self.results.setHtml(f"<span style='color:#d04040'><b>Error:</b> {e}</span>")
            return

        colors = {quant.ERROR: "#d04040", quant.WARNING: "#d08000", quant.INFO: ""}
        lines = [f"<b>VOI:</b> {voiText}"]
        for level, text in messages:
            style = f" style='color:{colors[level]}'" if colors[level] else ""
            prefix = {"error": "&#10006; ", "warning": "&#9888; ", "info": ""}[level]
            lines.append(f"<span{style}>{prefix}{text}</span>")
        if record is not None:
            lines.insert(0, f"<h3>{record['cpsPerMBq']:.4f} cps/MBq</h3>")
            if not self.nameEdit.text:
                camera = record["camera"].get("model") or "camera"
                self.nameEdit.setText(f"{camera} {record.get('collimator') or ''} {record['isotope']} "
                                      f"{datetime.date.today().isoformat()}".replace("  ", " "))
        self.results.setHtml("<br>".join(lines))
        self.record = record
        self.saveButton.setEnabled(record is not None)

    def onSave(self):
        if self.record is None:
            return
        name = self.nameEdit.text.strip()
        if not name:
            slicer.util.warningDisplay("Enter a name for the calibration.")
            return
        record = dict(self.record)
        record["name"] = name
        record["notes"] = self.notesEdit.text
        path = quant.calibrationLibraryPath()
        try:
            saved = quant.addCalibration(path, record)
        except Exception as e:
            slicer.util.errorDisplay(f"Could not save to the calibration library:\n{path}\n\n{e}")
            return
        slicer.util.infoDisplay(f"Saved '{saved['name']}' ({saved['cpsPerMBq']:.4f} cps/MBq) to\n{path}")
        self._invalidate()
        self.refreshTable()
