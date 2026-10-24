"""
DICOM SPECT Reconstruction for 3D Slicer

Recognizes NM SPECT projection series (ImageType contains TOMO or GATED TOMO)
in the DICOM browser. On Load, opens a preview window:
    left   = gated reconstruction in cine, and its CT fusion (gated studies only)
    middle = static reconstruction (summed gate counts when gated), and its CT fusion
    right  = raw projections in cine, and the CT resampled to the SPECT grid
    side   = OSEM iterations/subsets (separately for static and gated), BSREM iterations/
             subsets/prior, or FBP filter window/cutoff, CT attenuation
             correction, dual/triple energy window scatter correction, output count
             scaling, and post-filters

Reconstruction logic is taken from the standalone PyTomography app
(OSEM, save_dcm -> SimpleITK read, filters). BSREM is OSEM-like with a relative difference prior and a decreasing
relaxation, started from one OSEM iteration. FBP is the ramp filter with a window, scatter
subtracted from the projections and first order Chang attenuation correction; it is scaled to the
same native units as OSEM. Output count scaling is optional:
    native      = PyTomography's OSEM units (expected counts per voxel per projection view)
    views       = native x number of views: counts each voxel contributes over the whole
                  acquisition (default; keeps the effect of attenuation/scatter correction)
    primary     = total image counts made equal to the detected primary counts
                  (photopeak minus scatter estimate) - the old standalone behaviour,
                  corrected for scatter; it cancels the global effect of attenuation correction
None of these is a quantitative (Bq/ml) calibration; that needs a measured system sensitivity.

Quantification (optional, off by default): with a calibration made in the SPECT Calibration
module (SPECTCalibration.py), the static image is converted to Bq/ml, and optionally to SUV:
    Bq/ml = acquisition counts / (T x CF) x 1e6 / voxel volume (ml)
where T is the total view time (counting time summed over all views of all detectors, so it
does not depend on how many detectors acquired at once - many exports store a dual-head
acquisition as one merged detector) and CF the calibration factor in cps/MBq.
Gates stay in acquisition counts.

Layout: this file registers the module and the DICOM plugin; SPECTCalibration.py is the
calibration module; the work is done in the DICOMSpectReconLib package (see its __init__.py).

Install: from the Extensions Manager (DICOMSpectRecon extension), or without building: add this
folder (the inner DICOMSpectRecon folder that holds this file, SPECTCalibration.py and the
DICOMSpectReconLib package) to Edit > Application Settings > Modules > Additional module paths,
then restart.
"""

import os
import sys
import logging
import traceback

import qt
import slicer
from DICOMLib import DICOMPlugin, DICOMLoadable

# The helper package lives next to this file; make sure it is importable when the
# folder is used through "Additional module paths" as well as in an extension build
_MODULE_DIR = os.path.dirname(os.path.abspath(globals().get("__file__", "") or "."))
if globals().get("__file__") and _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)

from DICOMSpectReconLib.reconstruction import pytomographyVersion  # noqa: E402


# NM projection file -> (CT file list, how it was matched), filled in during examine()
_CT_FOR_NM_FILE = {}


def showAndCenterVolume(volumeNodeID, ctNodeID=None):
    """Make the new volume the active layer everywhere and center the views on it.
    With a CT present, show the SPECT as a colored overlay on the CT."""
    try:
        volumeNode = slicer.mrmlScene.GetNodeByID(volumeNodeID)
        if not volumeNode:
            return
        ctNode = slicer.mrmlScene.GetNodeByID(ctNodeID) if ctNodeID else None

        volumeNode.CreateDefaultDisplayNodes()
        displayNode = volumeNode.GetDisplayNode()
        if displayNode:
            displayNode.SetInterpolate(True)  # smooth display of the coarse SPECT grid
            colorNode = slicer.mrmlScene.GetFirstNodeByName("PET-DICOM")
            if colorNode:
                displayNode.SetAndObserveColorNodeID(colorNode.GetID())
                

        appLogic = slicer.app.applicationLogic()
        selectionNode = appLogic.GetSelectionNode()
        if ctNode:
            selectionNode.SetActiveVolumeID(ctNode.GetID())
            selectionNode.SetSecondaryVolumeID(volumeNode.GetID())
            appLogic.PropagateVolumeSelection(0)
            slicer.util.setSliceViewerLayers(background=ctNode, foreground=volumeNode,
                                             foregroundOpacity=0.5, fit=True)
        else:
            selectionNode.SetActiveVolumeID(volumeNode.GetID())
            selectionNode.SetSecondaryVolumeID(None)
            appLogic.PropagateVolumeSelection(0)
        appLogic.FitSliceToAll()

        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        center = [(bounds[0] + bounds[1]) / 2.0, (bounds[2] + bounds[3]) / 2.0, (bounds[4] + bounds[5]) / 2.0]
        slicer.modules.markups.logic().JumpSlicesToLocation(center[0], center[1], center[2], True)

        layoutManager = slicer.app.layoutManager()
        if layoutManager and layoutManager.threeDWidget(0):
            layoutManager.threeDWidget(0).threeDView().resetFocalPoint()
    except Exception:
        logging.warning("Could not center views on the reconstructed volume:\n" + traceback.format_exc())


# ---------------------------------------------------------------------------
# Module registration (hidden module, only used by the DICOM browser)
# ---------------------------------------------------------------------------
class DICOMSpectRecon:
    def __init__(self, parent):
        parent.title = "DICOM SPECT Reconstruction"
        parent.categories = ["Developer Tools.DICOM Plugins"]
        parent.contributors = ["Burak Demir"]
        parent.helpText = (
            "Loads SPECT projection data from DICOM and reconstructs it with OSEM, "
            "BSREM or FBP (PyTomography), with an interactive preview before loading."
        )
        parent.acknowledgementText = "Reconstruction powered by PyTomography."
        parent.hidden = True

        try:
            slicer.modules.dicomPlugins
        except AttributeError:
            slicer.modules.dicomPlugins = {}
        slicer.modules.dicomPlugins["DICOMSpectRecon"] = DICOMSpectReconPluginClass


# ---------------------------------------------------------------------------
# DICOM plugin: examine + load
# ---------------------------------------------------------------------------
class DICOMSpectReconPluginClass(DICOMPlugin):
    def __init__(self):
        super().__init__()
        self.loadType = "SPECT Projections (OSEM/BSREM/FBP Recon)"
        self.tags["modality"] = "0008,0060"
        self.tags["imageType"] = "0008,0008"
        self.tags["seriesDescription"] = "0008,103e"
        self.tags["seriesNumber"] = "0020,0011"
        self.tags["studyInstanceUID"] = "0020,000d"
        self.tags["seriesInstanceUID"] = "0020,000e"
        self.tags["frameOfReferenceUID"] = "0020,0052"

    def examineForImport(self, fileLists):
        # This is the method the DICOM browser calls; examine() is only kept for compatibility
        return self.examine(fileLists)

    def examine(self, fileLists):
        # Collect every CT series the user selected, so it can be paired with the SPECT
        ctSeries = []
        for files in fileLists:
            if not files:
                continue
            modality = (slicer.dicomDatabase.fileValue(files[0], self.tags["modality"]) or "").strip().upper()
            if modality == "CT":
                ctSeries.append({
                    "files": list(files),
                    "frameOfReference": slicer.dicomDatabase.fileValue(files[0], self.tags["frameOfReferenceUID"]) or "",
                    "study": slicer.dicomDatabase.fileValue(files[0], self.tags["studyInstanceUID"]) or "",
                })

        loadables = []
        for files in fileLists:
            loadables += self.examineFiles(files, ctSeries)
        return loadables

    def isSpectProjectionFile(self, filePath):
        db = slicer.dicomDatabase
        modality = (db.fileValue(filePath, self.tags["modality"]) or "").strip().upper()
        if modality != "NM":
            return False
        imageType = db.fileValue(filePath, self.tags["imageType"]) or ""
        parts = [p.strip().upper() for p in imageType.split("\\")]
        # 'TOMO' or 'GATED TOMO'; 'RECON TOMO' is an already reconstructed volume
        return "TOMO" in parts or "GATED TOMO" in parts

    def ctSeriesFromDatabase(self, nmFile):
        """Find CT series in the same study, whether or not the user selected them.
        Relying on multi-selection is unreliable: the browser may examine one series
        at a time and caches loadables per file list."""
        db = slicer.dicomDatabase
        found = []
        try:
            studyUID = db.fileValue(nmFile, self.tags["studyInstanceUID"]) or ""
            if not studyUID:
                return found
            for seriesUID in db.seriesForStudy(studyUID):
                files = list(db.filesForSeries(seriesUID))
                if not files:
                    continue
                if (db.fileValue(files[0], self.tags["modality"]) or "").strip().upper() != "CT":
                    continue
                found.append({
                    "files": files,
                    "frameOfReference": db.fileValue(files[0], self.tags["frameOfReferenceUID"]) or "",
                    "study": db.fileValue(files[0], self.tags["studyInstanceUID"]) or "",
                })
        except Exception:
            logging.warning("Could not query the DICOM database for CT series:\n" + traceback.format_exc())
        return found

    def ctCandidates(self, nmFile, ctSeries):
        """All CT series that could belong to this SPECT, best match first.
        Same Frame of Reference means the two are spatially registered (hybrid SPECT/CT);
        same Study is a weaker fallback and is flagged to the user."""
        candidates = list(ctSeries)
        known = {s["files"][0] for s in candidates if s["files"]}
        for series in self.ctSeriesFromDatabase(nmFile):
            if series["files"] and series["files"][0] not in known:
                candidates.append(series)

        db = slicer.dicomDatabase
        nmFrame = db.fileValue(nmFile, self.tags["frameOfReferenceUID"]) or ""
        nmStudy = db.fileValue(nmFile, self.tags["studyInstanceUID"]) or ""

        described = []
        for series in candidates:
            files = series["files"]
            if not files:
                continue
            if nmFrame and series["frameOfReference"] == nmFrame:
                match, rank = "same frame of reference", 0
            elif nmStudy and series["study"] == nmStudy:
                match, rank = "same study - check alignment", 1
            else:
                match, rank = "unrelated study - check alignment", 2
            desc = db.fileValue(files[0], self.tags["seriesDescription"]) or "CT"
            number = db.fileValue(files[0], self.tags["seriesNumber"]) or ""
            label = f"{('#' + number + ' ') if number else ''}{desc} - {len(files)} slices ({match})"
            described.append({"files": files, "match": match, "label": label, "rank": rank})

        described.sort(key=lambda s: (s["rank"], -len(s["files"])))
        logging.info(f"SPECT recon plugin: {len(described)} CT series found for {os.path.basename(nmFile)}")
        return described

    def examineFiles(self, files, ctSeries=None):
        spectFiles = [f for f in files if self.isSpectProjectionFile(f)]
        loadables = []
        for i, f in enumerate(spectFiles):
            desc = slicer.dicomDatabase.fileValue(f, self.tags["seriesDescription"]) or "SPECT"
            candidates = self.ctCandidates(f, ctSeries or [])
            # Keep the pairing outside the loadable: Qt-wrapped loadables can lose Python attributes
            _CT_FOR_NM_FILE[f] = candidates

            loadable = DICOMLoadable()
            loadable.files = [f]
            loadable.name = f"{desc} OSEM" + (f" ({i + 1})" if len(spectFiles) > 1 else "")
            tooltip = "SPECT projections - reconstructed with OSEM, BSREM or FBP (PyTomography) on load"
            if candidates:
                tooltip += (f"\nCT available for attenuation correction: {len(candidates)} series, "
                            f"best match {candidates[0]['label']}")
            else:
                tooltip += "\nSelect the CT series as well to enable attenuation correction"
            loadable.tooltip = tooltip
            loadable.selected = True
            # Above 1.0: the DICOM browser keeps only the highest confidence loadables of each series
            # (DICOMUtils.selectHighestConfidenceLoadables, grouped by the series of their first file).
            # With two NM series selected, MultiVolumeImporter lumps them into one multi-series
            # loadable of confidence 1.0, which would otherwise deselect the reconstruction of one of
            # them; the scalar volume plugin offers less than that.
            loadable.confidence = 1.05
            loadables.append(loadable)
        return loadables

    def load(self, loadable):
        if not ensureDependencies():
            return None
        _reloadLibInDeveloperMode()
        from DICOMSpectReconLib.dialog import SpectReconDialog

        ctCandidates = _CT_FOR_NM_FILE.get(loadable.files[0])
        if not ctCandidates:
            # examine() may not have re-run (the browser caches loadables), so look again now
            ctCandidates = self.ctCandidates(loadable.files[0], [])
        dialog = SpectReconDialog(loadable.files[0], loadable.name, ctCandidates)
        image, info = dialog.run()
        if image is None:
            logging.info("SPECT reconstruction cancelled by user.")
            return None

        import sitkUtils
        suffix = "AC" if info.get("AttenuationCorrection") == "CT" else "NC"
        if info.get("ScatterCorrection", "None") != "None":
            suffix += " SC"
        if info.get("ResolutionRecovery", "None") not in ("None", "", None):
            suffix += " RR"
        units = info.get("Units", "counts")
        unitSuffix = {"Bq/ml": " Bqml", "SUVbw": " SUVbw", "SUL": " SUL"}.get(units, "")
        baseName = loadable.name
        algorithm = info.get("Algorithm", "OSEM")
        if algorithm != "OSEM" and " OSEM" in baseName:
            baseName = baseName.replace(" OSEM", f" {algorithm}", 1)
        nodeName = slicer.mrmlScene.GenerateUniqueName(f"{baseName} {suffix}{unitSuffix}")
        volumeNode = sitkUtils.PushVolumeToSlicer(image, name=nodeName)

        # Keep reconstruction settings with the volume (traceability for dosimetry)
        for key, value in info.items():
            volumeNode.SetAttribute(f"SPECTRecon.{key}", str(value))
        setVoxelUnits(volumeNode, units)

        try:
            self.addSeriesInSubjectHierarchy(loadable, volumeNode)
        except Exception:
            logging.warning("Could not place volume in subject hierarchy:\n" + traceback.format_exc())

        gateImages = getattr(dialog, "gateImages", [])
        if gateImages:
            # Each gate carries its own (gated) recon settings and counts, never quantified
            self.loadGatesAsSequence(gateImages, f"{loadable.name} {suffix} gated", dialog.gateInfos,
                                     getattr(dialog, "gateNumbers", None), volumeNode)

        # Show and center after the DICOM module has finished its own loading.
        # With a CT, show the SPECT on top of it (the CT is loaded if it is not in the scene yet).
        volumeNodeID = volumeNode.GetID()
        fusionCtFiles = dialog.ctFiles if (dialog.ctFiles and dialog.showOnCtCheckBox.checked) else None

        def finishLoading():
            ctNodeID = None
            if fusionCtFiles:
                ctNode = loadCtForFusion(fusionCtFiles)
                ctNodeID = ctNode.GetID() if ctNode else None
            showAndCenterVolume(volumeNodeID, ctNodeID)

        qt.QTimer.singleShot(0, finishLoading)

        return volumeNode

    def loadGatesAsSequence(self, gateImages, name, gateInfos, gateNumbers=None, staticNode=None):
        """Put the per-gate reconstructions in a sequence so Slicer can play them.
        gateInfos holds the SPECTRecon.* attributes of each gate, in gate order."""
        try:
            import sitkUtils
            gateNumbers = list(gateNumbers or range(1, len(gateImages) + 1))
            baseName = slicer.mrmlScene.GenerateUniqueName(name)
            sequenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", baseName)
            sequenceNode.SetIndexName("Gate")
            sequenceNode.SetIndexUnit("")
            for image, gate, info in zip(gateImages, gateNumbers, gateInfos):
                gateNode = sitkUtils.PushVolumeToSlicer(image, name=f"{baseName} G{gate}")
                for key, value in info.items():
                    gateNode.SetAttribute(f"SPECTRecon.{key}", str(value))
                gateNode.SetAttribute("SPECTRecon.Gate", str(gate))
                sequenceNode.SetDataNodeAtValue(gateNode, str(gate))
                slicer.mrmlScene.RemoveNode(gateNode)

            browserNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", f"{baseName} browser")
            slicer.modules.sequences.logic().AddSynchronizedNode(sequenceNode, None, browserNode)
            # Renamed from Master to Primary in Slicer 5.x; support both
            if hasattr(browserNode, "SetAndObservePrimarySequenceNodeID"):
                browserNode.SetAndObservePrimarySequenceNodeID(sequenceNode.GetID())
            else:
                browserNode.SetAndObserveMasterSequenceNodeID(sequenceNode.GetID())

            proxyNode = browserNode.GetProxyNode(sequenceNode)
            if proxyNode:
                proxyNode.SetName(f"{baseName} (current gate)")
                proxyNode.CreateDefaultDisplayNodes()
                colorNode = slicer.mrmlScene.GetFirstNodeByName("PET-DICOM")
                if colorNode and proxyNode.GetDisplayNode():
                    proxyNode.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())

            # Put the gated data next to the static volume in the patient/study tree
            if staticNode is not None:
                shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
                parentItem = shNode.GetItemParent(shNode.GetItemByDataNode(staticNode))
                if parentItem:
                    for node in (sequenceNode, browserNode, proxyNode):
                        if node:
                            item = shNode.GetItemByDataNode(node)
                            if not item:
                                item = shNode.CreateItem(parentItem, node)
                            shNode.SetItemParent(item, parentItem)

            logging.info(f"Loaded {len(gateImages)} gates as sequence '{baseName}'.")
            return sequenceNode
        except Exception:
            logging.warning("Could not create a gate sequence:\n" + traceback.format_exc())
            return None


def setVoxelUnits(volumeNode, units):
    """Record the voxel unit the way Slicer does for PET (UCUM coded entry), so it shows in
    Data Probe and survives saving; plain counts are left without a unit."""
    codes = {"Bq/ml": ("Bq/ml", "UCUM", "Becquerels/milliliter"),
             "SUVbw": ("{SUVbw}g/ml", "UCUM", "Standardized Uptake Value body weight"),
             "SUL": ("{SUVlbm}g/ml", "UCUM", "Standardized Uptake Value lean body mass")}
    if units not in codes:
        return
    try:
        entry = slicer.vtkCodedEntry()
        entry.SetValueSchemeMeaning(*codes[units])
        volumeNode.SetVoxelValueUnits(entry)
    except Exception:
        logging.warning("Could not set voxel value units:\n" + traceback.format_exc())


def _existingCtNode(ctFiles):
    """A volume already in the scene that was loaded from this CT series, if any."""
    try:
        uid = slicer.dicomDatabase.instanceForFile(ctFiles[0])
    except Exception:
        uid = ""
    if not uid:
        return None
    for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
        if uid in (node.GetAttribute("DICOM.instanceUIDs") or "").split():
            return node
    return None


def loadCtForFusion(ctFiles):
    """Return a volume node for the CT, loading it through Slicer's own DICOM plugin
    (so it lands in the subject hierarchy with CT window presets) unless it is already loaded."""
    try:
        node = _existingCtNode(ctFiles)
        if node:
            return node
        plugins = getattr(slicer.modules, "dicomPlugins", {})
        if "DICOMScalarVolumePlugin" in plugins:
            try:
                plugin = plugins["DICOMScalarVolumePlugin"]()
                loadables = [l for l in plugin.examineFiles(list(ctFiles)) if l.files]
                if loadables:
                    best = max(loadables, key=lambda l: l.confidence)
                    node = plugin.load(best)
                    if node:
                        try:
                            shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
                            if not shNode.GetItemUID(shNode.GetItemByDataNode(node), "DICOM"):
                                plugin.addSeriesInSubjectHierarchy(best, node)
                        except Exception:
                            pass
                        return node
            except Exception:
                logging.warning("DICOM scalar volume plugin could not load the CT, "
                                "falling back to SimpleITK:\n" + traceback.format_exc())
        # Files chosen manually may not be in the DICOM database
        import SimpleITK as sitk
        import sitkUtils
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(list(ctFiles))
        return sitkUtils.PushVolumeToSlicer(reader.Execute(), name=slicer.mrmlScene.GenerateUniqueName("CT"))
    except Exception:
        logging.warning("Could not load the CT for fusion display:\n" + traceback.format_exc())
        return None



def _reloadLibInDeveloperMode():
    """With Slicer's developer mode on, pick up edits to the helper package on the next
    load without restarting Slicer (the module's own Reload does not reach it)."""
    try:
        if not slicer.util.settingsValue("Developer/DeveloperMode", False, converter=slicer.util.toBool):
            return
        import importlib
        import sys
        # Dependency order; reload each one before importing the next, so a module is never
        # first imported against a stale (not yet reloaded) dependency
        for name in ("nmdata", "quantification", "reconstruction", "display", "dialog"):
            fullName = f"DICOMSpectReconLib.{name}"
            if fullName in sys.modules:
                importlib.reload(sys.modules[fullName])
            else:
                importlib.import_module(fullName)
    except Exception:
        logging.warning("Could not reload DICOMSpectReconLib:\n" + traceback.format_exc())


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def ensureDependencies():
    """PyTorch comes only from Slicer's PyTorch extension (PyTorchUtils), which picks the
    right CPU/CUDA build for this computer; this plugin never pip-installs torch itself.
    PyTomography is then installed with torch pinned to the installed build, so pip
    cannot replace it with a generic wheel while resolving PyTomography's requirements."""
    try:
        if not _ensureTorch():
            return False
        try:
            import pytomography  # noqa: F401
        except ImportError:
            if not slicer.util.confirmOkCancelDisplay(
                    "PyTomography is required for SPECT reconstruction but is not installed.\n\n"
                    "Install it now? Your PyTorch installation will not be changed."):
                return False
            with slicer.util.WaitCursor():
                _installPytomographyKeepingTorch()
            import pytomography  # noqa: F401

        # No version check: the version in use is logged and stored on every loaded volume
        logging.info(f"SPECT recon plugin: using PyTomography {pytomographyVersion()}")
        return True
    except Exception as e:
        slicer.util.errorDisplay(f"Failed to install dependencies: {e}", detailedText=traceback.format_exc())
        return False


def _ensureTorch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        pass

    try:
        import PyTorchUtils
    except ImportError:
        # The PyTorch extension is not installed: send the user there instead of installing torch here
        message = ("SPECT reconstruction needs PyTorch, which is provided by Slicer's 'PyTorch' extension.\n\n"
                   "Install the 'PyTorch' extension from the Extensions Manager, restart Slicer, "
                   "then load the SPECT series again.")
        if hasattr(slicer.app, "openExtensionsManagerDialog"):
            if slicer.util.confirmOkCancelDisplay(message + "\n\nOpen the Extensions Manager now?"):
                slicer.app.openExtensionsManagerDialog()
        else:
            slicer.util.infoDisplay(message)
        return False

    if not slicer.util.confirmOkCancelDisplay(
            "SPECT reconstruction needs PyTorch, which is not installed yet.\n\n"
            "Install it now with the PyTorch extension? It selects the CPU or CUDA build for "
            "this computer and can take several minutes."):
        return False
    with slicer.util.WaitCursor():
        torch = PyTorchUtils.PyTorchUtilsLogic().installTorch(askConfirmation=False)
    if torch is None:
        slicer.util.errorDisplay("PyTorch installation failed. You can install it from the PyTorch module "
                                 "(Utilities > PyTorch) and then load the SPECT series again.")
        return False
    return True


def _installPytomographyKeepingTorch():
    """pip install pytomography with a constraint that holds torch at the installed build."""
    import importlib.metadata as metadata
    import tempfile

    torchVersion = metadata.version("torch")
    with tempfile.TemporaryDirectory() as tempDir:
        constraints = os.path.join(tempDir, "keep-torch.txt")
        with open(constraints, "w") as f:
            f.write(f"torch=={torchVersion}\n")
        # List form: the constraints path is passed as one argument even if it contains spaces
        slicer.util.pip_install(["pytomography", "--constraint", constraints])

    after = metadata.version("torch")
    if after != torchVersion:
        slicer.util.warningDisplay(f"PyTorch changed from {torchVersion} to {after} while installing "
                                   "PyTomography. Reinstall it from the PyTorch module if GPU support is lost.")
