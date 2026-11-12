"""Headless Slicer test of the multi-bed loadable (DICOM browser examine + plugin load + dialog recon).

  Slicer --no-splash --no-main-window --disable-settings --python-script slicer_multibed_test.py CODE DATA OUT

CODE is the DICOMSpectRecon folder, DATA the output of make_phantom.py DATA 2. A temporary DICOM
database is used. The dialog is not shown: its exec is replaced by one Reconstruct + Load with the
default settings. Results go to OUT/slicer_multibed.txt and OUT/*.nrrd."""
import os, sys, glob, json, traceback
import numpy as np
import qt, slicer

code, data, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
lines = []
def report(*args):
    text = " ".join(str(a) for a in args)
    lines.append(text)
    print(text)

try:
    sys.path.insert(0, code)
    import DICOMSpectRecon as plugin
    from DICOMSpectReconLib import dialog as dialogModule
    from DICOMLib import DICOMUtils
    plugin.ensureDependencies = lambda: True          # never install anything here
    report("PyTomography", plugin.pytomographyVersion())

    # Replace the modal exec: one Reconstruct (as the dialog's first timer does) and accept
    original = dialogModule.SpectReconDialog.run
    logs = []
    class Headless:
        """The dialog's QDialog, but exec runs one Reconstruct and accepts."""
        def __init__(self, owner, dialog):
            self._owner, self._dialog = owner, dialog
        def __getattr__(self, name):
            return getattr(self._dialog, name)
        def exec_(self):
            ok = self._owner.onApply()
            report(f"onApply returned {ok}")
            return qt.QDialog.Accepted

    def run(self):
        self.dialog = Headless(self, self.dialog)
        singleShot = qt.QTimer.singleShot
        qt.QTimer.singleShot = lambda *a, **k: None
        oldLog = self.log
        self.log = lambda m: (logs.append(m), oldLog(m))
        try:
            return original(self)
        finally:
            qt.QTimer.singleShot = singleShot
    dialogModule.SpectReconDialog.run = run

    with DICOMUtils.TemporaryDICOMDatabase(os.path.join(out, "db")) as db:
        DICOMUtils.importDicom(data, db)
        pluginObject = plugin.DICOMSpectReconPluginClass()
        series = {}
        for study in db.studiesForPatient(db.patients()[0]):
            for s in db.seriesForStudy(study):
                files = list(db.filesForSeries(s))
                series[db.fileValue(files[0], "0008,103e")] = files
        report("series:", {k: len(v) for k, v in series.items()})
        bed1, bed2 = series["PHANTOM bed 1"], series["PHANTOM bed 2"]
        ct = [v for k, v in series.items() if not k.startswith("PHANTOM bed")][0]

        def show(loadables, title):
            report(title)
            for l in loadables:
                report(f"   {l.name!r}: {len(l.files)} file(s) "
                       f"[{', '.join(db.fileValue(f, '0008,103e') for f in l.files)}], "
                       f"confidence {l.confidence}, selected {l.selected}")

        show(pluginObject.examineForImport([bed1]), "examine (bed 1 series only):")
        show(pluginObject.examineForImport([bed2]), "examine (bed 2 series only):")
        loadables = pluginObject.examineForImport([bed1, bed2, ct])
        show(loadables, "examine (bed 1, bed 2 and CT selected):")
        DICOMUtils.selectHighestConfidenceLoadables({pluginObject: loadables})
        show(loadables, "after the browser's highest-confidence selection:")

        loaded = []
        for l in loadables:
            if l.selected:
                node = pluginObject.load(l)
                report(f"load {l.name!r}: {'node ' + node.GetName() if node else 'skipped/none'}")
                if node:
                    loaded.append(node)
        for m in logs:
            if any(k in m for k in ("Multi-bed", "Bed ", "bed:", "overlap", "Stitch", "WARNING", "ERROR", "consistency", "Voxel spacing")):
                report("   log |", m)
        for node in loaded:
            arr = slicer.util.arrayFromVolume(node)
            report(f"volume {node.GetName()}: dims {node.GetImageData().GetDimensions()}, spacing "
                   f"{np.round(node.GetSpacing(), 3).tolist()}, origin {np.round(node.GetOrigin(), 2).tolist()}, "
                   f"sum {arr.sum():.6g}, max {arr.max():.4g}")
            for key in ("Beds", "BedFiles", "BedTimingFactors", "BedStitching", "Algorithm", "AttenuationCorrection",
                        "ScatterCorrection", "CountScaling", "CountScalingFactor", "PhotopeakCounts", "DataConsistency",
                        "Units", "SourceFile"):
                value = node.GetAttribute(f"SPECTRecon.{key}")
                if value is not None:
                    report(f"   SPECTRecon.{key} = {value if 'File' not in key else [os.path.basename(v) for v in json.loads(value)] if key == 'BedFiles' else os.path.basename(value)}")
            slicer.util.saveNode(node, os.path.join(out, node.GetName().replace(" ", "_") + ".nrrd"))
except Exception:
    report("FAILED:\n" + traceback.format_exc())
finally:
    open(os.path.join(out, "slicer_multibed.txt"), "w").write("\n".join(lines) + "\n")
    slicer.util.exit(0)
