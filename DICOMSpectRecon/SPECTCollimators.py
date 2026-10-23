"""SPECT Collimators - the collimator library used for resolution recovery.

Each entry is a parallel-hole collimator (hole diameter, septa, hole length) with the intrinsic
resolution of the camera, and the words used to recognise it in the DICOM header (manufacturer,
camera model, collimator names). The reconstruction window matches the study to an entry ("Auto") or
lets the user pick one, and models the collimator-detector blur in OSEM / BSREM from it.

The library starts as a copy of the collimators PyTomography ships (from SIMIND); entries can be
added, edited and deleted here, and Restore defaults puts the built-in list back.
"""

import logging
import os
import sys
import traceback

import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleWidget,
                                           ScriptedLoadableModuleLogic)

_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from DICOMSpectReconLib import collimators  # noqa: E402

# (field, header, tooltip); the last column is computed
COLUMNS = [
    ("code", "Code", "Unique name of the collimator (shown in the reconstruction window)"),
    ("description", "Description", ""),
    ("manufacturer", "Manufacturer", "Matched (case insensitive) in the DICOM Manufacturer"),
    ("models", "Camera models", "Comma separated words matched in the DICOM camera model name; "
                                "empty matches any camera of the manufacturer"),
    ("names", "DICOM names", "Comma separated collimator names as the camera writes them in the header "
                             "(Collimator/Grid Name), e.g. LEHR, LEHRS"),
    ("holeDiameterMm", "Hole (mm)", "Hole diameter (flat to flat for hexagonal holes)"),
    ("septaMm", "Septa (mm)", "Septal thickness"),
    ("holeLengthMm", "Length (mm)", "Hole length (collimator thickness)"),
    ("intrinsicMm", "Intrinsic (mm)", "Intrinsic resolution of the detector at 140 keV (FWHM); "
                                      "about 3.8 mm for NaI cameras, 2.5 mm for CZT"),
]
RESOLUTION_HEADER = "System @10 cm (mm)"
RESOLUTION_TOOLTIP = ("System resolution (FWHM) at 10 cm, 140 keV, from the PSF model - compare with the "
                      "collimator datasheet")


class SPECTCollimators(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        parent.title = "SPECT Collimators"
        parent.categories = ["Nuclear Medicine"]
        parent.dependencies = []
        parent.contributors = ["Burak Demir"]
        parent.helpText = (
            "Collimator library for SPECT resolution recovery. The DICOM SPECT reconstruction window "
            "matches the study to a collimator here (from the manufacturer, camera model and collimator "
            "name in the DICOM header) and models its depth dependent blur in OSEM and BSREM. "
            "Add or edit collimators, then press Save. Restore defaults puts back the built-in list.")
        parent.acknowledgementText = "Built-in collimators from PyTomography (SIMIND collimator data base)."


class SPECTCollimatorsLogic(ScriptedLoadableModuleLogic):
    pass


class SPECTCollimatorsWidget(ScriptedLoadableModuleWidget):

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.dirty = False
        self.filling = False
        layout = self.layout

        libraryBox = ctk.ctkCollapsibleButton()
        libraryBox.text = "Collimator library"
        layout.addWidget(libraryBox)
        form = qt.QVBoxLayout(libraryBox)
        self.pathLabel = qt.QLabel("")
        self.pathLabel.setWordWrap(True)
        self.pathLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        form.addWidget(self.pathLabel)
        self.searchEdit = qt.QLineEdit()
        self.searchEdit.setPlaceholderText("Filter (code, manufacturer, model, name)")
        self.searchEdit.connect("textChanged(QString)", lambda text: self._applyFilter())
        form.addWidget(self.searchEdit)

        self.table = qt.QTableWidget()
        self.table.setColumnCount(len(COLUMNS) + 1)
        self.table.setHorizontalHeaderLabels([c[1] for c in COLUMNS] + [RESOLUTION_HEADER])
        for column, (_, _, tooltip) in enumerate(COLUMNS):
            if tooltip:
                self.table.horizontalHeaderItem(column).setToolTip(tooltip)
        self.table.horizontalHeaderItem(len(COLUMNS)).setToolTip(RESOLUTION_TOOLTIP)
        self.table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(320)
        self.table.connect("itemChanged(QTableWidgetItem*)", self.onItemChanged)
        form.addWidget(self.table)

        buttons = qt.QHBoxLayout()
        self.addButton = qt.QPushButton("Add")
        self.addButton.setToolTip("Add an empty collimator (fill in the code and the numbers)")
        self.duplicateButton = qt.QPushButton("Duplicate")
        self.duplicateButton.setToolTip("Copy the selected collimator, to start a new one from it")
        self.deleteButton = qt.QPushButton("Delete")
        self.deleteButton.setToolTip("Delete the selected collimators")
        for button, handler in ((self.addButton, self.onAdd), (self.duplicateButton, self.onDuplicate),
                                (self.deleteButton, self.onDelete)):
            button.connect("clicked()", handler)
            buttons.addWidget(button)
        form.addLayout(buttons)
        buttons = qt.QHBoxLayout()
        self.saveButton = qt.QPushButton("Save")
        self.saveButton.setToolTip("Write the library; reconstruction windows opened afterwards use it")
        self.reloadButton = qt.QPushButton("Reload")
        self.reloadButton.setToolTip("Discard unsaved edits and read the library file again")
        self.restoreButton = qt.QPushButton("Restore defaults")
        self.restoreButton.setToolTip("Replace the library with the built-in collimators "
                                      "(your own entries and edits are lost)")
        for button, handler in ((self.saveButton, self.onSave), (self.reloadButton, self.onReload),
                                (self.restoreButton, self.onRestoreDefaults)):
            button.connect("clicked()", handler)
            buttons.addWidget(button)
        form.addLayout(buttons)
        self.statusLabel = qt.QLabel("")
        self.statusLabel.setWordWrap(True)
        form.addWidget(self.statusLabel)

        # Check which entry a DICOM header would get
        matchBox = ctk.ctkCollapsibleButton()
        matchBox.text = "Test header matching"
        matchBox.collapsed = True
        layout.addWidget(matchBox)
        matchForm = qt.QFormLayout(matchBox)
        self.manufacturerEdit = qt.QLineEdit("GE MEDICAL SYSTEMS")
        self.modelEdit = qt.QLineEdit("NM/CT 860")
        self.collimatorEdit = qt.QLineEdit("LEHRS")
        self.energySpin = qt.QDoubleSpinBox()
        self.energySpin.setRange(20.0, 600.0)
        self.energySpin.setValue(140.5)
        self.energySpin.setSuffix(" keV")
        self.matchLabel = qt.QLabel("")
        self.matchLabel.setWordWrap(True)
        matchForm.addRow("Manufacturer:", self.manufacturerEdit)
        matchForm.addRow("Camera model:", self.modelEdit)
        matchForm.addRow("Collimator name:", self.collimatorEdit)
        matchForm.addRow("Photopeak:", self.energySpin)
        matchForm.addRow(self.matchLabel)
        for edit in (self.manufacturerEdit, self.modelEdit, self.collimatorEdit):
            edit.connect("textChanged(QString)", lambda text: self._updateMatch())
        self.energySpin.connect("valueChanged(double)", lambda value: self._updateMatch())

        layout.addStretch(1)
        self.onReload(confirm=False)

    # ---------------- table ----------------
    def _fill(self, entries):
        self.filling = True
        try:
            self.table.setRowCount(0)
            for entry in entries:
                self._appendRow(entry)
            self.table.resizeColumnsToContents()
        finally:
            self.filling = False
        self._applyFilter()
        self._updateMatch()

    def _appendRow(self, entry):
        row = self.table.rowCount
        self.table.insertRow(row)
        for column, (field, _, _) in enumerate(COLUMNS):
            value = entry.get(field, "")
            text = f"{value:g}" if isinstance(value, float) else str(value)
            self.table.setItem(row, column, qt.QTableWidgetItem(text))
        resolution = qt.QTableWidgetItem("")
        resolution.setFlags(qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled)
        resolution.setToolTip(RESOLUTION_TOOLTIP)
        self.table.setItem(row, len(COLUMNS), resolution)
        self._updateResolution(row)
        return row

    def _rowEntry(self, row):
        entry = {}
        for column, (field, _, _) in enumerate(COLUMNS):
            item = self.table.item(row, column)
            entry[field] = item.text().strip() if item else ""
        return entry

    def _entries(self):
        """All rows as clean entries; raises ValueError naming the first bad row."""
        entries = []
        for row in range(self.table.rowCount):
            try:
                entries.append(collimators.cleanEntry(self._rowEntry(row)))
            except ValueError as e:
                raise ValueError(f"Row {row + 1}: {e}")
        return entries

    def _updateResolution(self, row):
        item = self.table.item(row, len(COLUMNS))
        if item is None:
            return
        try:
            entry = collimators.cleanEntry(self._rowEntry(row))
            item.setText(f"{collimators.systemResolutionMm(entry):.2f}")
            item.setToolTip(RESOLUTION_TOOLTIP)
        except Exception as e:
            item.setText("invalid")
            item.setToolTip(str(e))

    def onItemChanged(self, item):
        if self.filling or item.column() >= len(COLUMNS):
            return
        self.filling = True
        try:
            self._updateResolution(item.row())
        finally:
            self.filling = False
        self._setDirty(True)
        self._updateMatch()

    def _applyFilter(self):
        words = self.searchEdit.text.strip().upper().split()
        for row in range(self.table.rowCount):
            text = " ".join(self._rowEntry(row).values()).upper()
            self.table.setRowHidden(row, bool(words) and not all(w in text for w in words))

    def _selectedRows(self):
        return sorted({index.row() for index in self.table.selectionModel().selectedRows()})

    def _setDirty(self, dirty):
        self.dirty = dirty
        self.statusLabel.setText("<span style='color:#d08000'>Unsaved changes - press Save.</span>"
                                 if dirty else "")

    # ---------------- buttons ----------------
    def onAdd(self):
        self.searchEdit.setText("")
        self.filling = True
        try:
            row = self._appendRow({"code": "", "intrinsicMm": collimators.DEFAULT_INTRINSIC_MM})
        finally:
            self.filling = False
        self.table.scrollToItem(self.table.item(row, 0))
        self.table.setCurrentCell(row, 0)
        self.table.editItem(self.table.item(row, 0))
        self._setDirty(True)

    def onDuplicate(self):
        rows = self._selectedRows()
        if not rows:
            self.statusLabel.setText("Select a collimator to duplicate.")
            return
        entry = self._rowEntry(rows[0])
        codes = {self._rowEntry(r)["code"] for r in range(self.table.rowCount)}
        code, n = entry["code"] + " copy", 2
        while code in codes:
            code, n = f"{entry['code']} copy {n}", n + 1
        entry["code"] = code
        self.searchEdit.setText("")
        self.filling = True
        try:
            row = self._appendRow(entry)
        finally:
            self.filling = False
        self.table.scrollToItem(self.table.item(row, 0))
        self.table.setCurrentCell(row, 0)
        self._setDirty(True)

    def onDelete(self):
        rows = self._selectedRows()
        if not rows:
            self.statusLabel.setText("Select the collimators to delete.")
            return
        codes = ", ".join(self._rowEntry(r)["code"] or "(no code)" for r in rows)
        if not slicer.util.confirmYesNoDisplay(f"Delete {codes}?\n\nThe library file changes when you press Save."):
            return
        for row in reversed(rows):
            self.table.removeRow(row)
        self._setDirty(True)
        self._updateMatch()

    def onSave(self):
        try:
            path = collimators.saveLibrary(self._entries())
        except Exception as e:
            slicer.util.errorDisplay(f"Collimator library not saved:\n{e}")
            return
        self._setDirty(False)
        self.statusLabel.setText(f"Saved {self.table.rowCount} collimators. Reconstruction windows opened "
                                 "from now on use them.")
        logging.info(f"Collimator library saved: {path}")

    def onReload(self, confirm=True):
        if confirm and self.dirty and not slicer.util.confirmYesNoDisplay("Discard the unsaved changes?"):
            return
        path = collimators.libraryPath()
        try:
            entries = collimators.loadLibrary(path)
            source = path if os.path.exists(path) else "built-in collimators (no library file saved yet)"
        except Exception as e:
            logging.warning("Could not read the collimator library:\n" + traceback.format_exc())
            slicer.util.errorDisplay(f"Could not read the collimator library {path}:\n{e}\n\n"
                                     "Showing the built-in collimators; Save replaces the broken file.")
            entries = collimators.builtinCollimators()
            source = "built-in collimators"
        self.pathLabel.setText(f"File: {path}<br>Showing: {source}")
        self._fill(entries)
        self._setDirty(False)
        if not entries:
            self.statusLabel.setText("<span style='color:#d04040'>No collimators - is PyTomography installed? "
                                     "Add them by hand, or Restore defaults after installing it.</span>")

    def onRestoreDefaults(self):
        builtin = collimators.builtinCollimators()
        if not builtin:
            slicer.util.errorDisplay("The built-in collimators could not be read (is PyTomography installed?).")
            return
        if not slicer.util.confirmYesNoDisplay(
                f"Replace the library with the {len(builtin)} built-in collimators?\n\n"
                "Your own collimators and edits are lost."):
            return
        try:
            collimators.restoreDefaults()
        except Exception as e:
            slicer.util.errorDisplay(f"Could not write the collimator library:\n{e}")
            return
        self.onReload(confirm=False)
        self.statusLabel.setText(f"Built-in collimators restored ({len(builtin)}).")

    # ---------------- matching test ----------------
    def _updateMatch(self):
        if not hasattr(self, "matchLabel"):
            return
        entries = []
        for row in range(self.table.rowCount):
            try:
                entries.append(collimators.cleanEntry(self._rowEntry(row)))
            except ValueError:
                continue
        entry = collimators.matchCollimator(entries, self.manufacturerEdit.text, self.modelEdit.text,
                                            self.collimatorEdit.text)
        if entry is None:
            self.matchLabel.setText("<span style='color:#d04040'>No match - add the header's collimator name "
                                    "to an entry's DICOM names (and the camera model words).</span>")
            return
        energy = self.energySpin.value
        try:
            slope, intercept, intrinsic = collimators.psfParameters(entry, energy)
            fwhm = collimators.FWHM_PER_SIGMA * 10.0
            text = (f"Match: <b>{entry['code']}</b><br>At {energy:g} keV: FWHM "
                    f"{fwhm * (intercept ** 2 + intrinsic ** 2) ** 0.5:.1f} mm at the face, "
                    f"{collimators.systemResolutionMm(entry, 100.0, energy):.1f} mm at 10 cm, "
                    f"{collimators.systemResolutionMm(entry, 200.0, energy):.1f} mm at 20 cm")
        except Exception as e:
            text = f"Match: <b>{entry['code']}</b><br><span style='color:#d04040'>{e}</span>"
        self.matchLabel.setText(text)
