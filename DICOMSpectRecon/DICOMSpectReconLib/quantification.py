"""Quantitative SPECT: calibration factors (cps/MBq), activity concentration (Bq/ml) and SUV.

Unit convention (see also the module docstring of DICOMSpectRecon.py):
    acquisition counts   the "x number of views" image: each voxel holds the counts it
                         contributed over the whole acquisition, summed over all views
    T                    total view time in seconds: the counting time summed over ALL views of
                         ALL detectors (views x frame duration, or the vendor's per-view durations).
                         Independent of how many detectors acquired at the same time, so merged
                         dual-head exports (stored as one 'detector') need no special handling.
    CF (cps/MBq)         CF = sum(acquisition counts in VOI) / T / activity at mid-scan (MBq)
                         Deliberately not split per detector: detectors can see different
                         attenuation (e.g. from the bed), so a per-detector value is not reliable.
    Bq/ml                acquisition counts / (T x CF) x 1e6 / voxel volume (ml)
    SUV                  Bq/ml x body mass (g) / injected activity at mid-scan (Bq)
The number of detectors only sets the wall-clock mid-scan time used for decay correction.

No Slicer dependency: everything here can be unit-tested with plain Python + numpy.
"""

import datetime
import json
import math
import os
import re
import tempfile
import uuid

LIBRARY_FORMAT_VERSION = 1
# Time basis of calibration factors. Calibrations saved with another basis are refused.
TIME_BASIS = "total-view-time"

# Half-lives in seconds. The DICOM header value (RadionuclideHalfLife) is preferred when present;
# these are used when it is missing and to cross-check it.
_DAY = 86400.0
_HOUR = 3600.0
ISOTOPES = {
    "Tc-99m": {"halfLife": 6.0067 * _HOUR, "element": ("technetium", "tc"), "mass": "99"},
    "I-123": {"halfLife": 13.2235 * _HOUR, "element": ("iodine", "i"), "mass": "123"},
    "I-131": {"halfLife": 8.0252 * _DAY, "element": ("iodine", "i"), "mass": "131"},
    "In-111": {"halfLife": 2.8047 * _DAY, "element": ("indium", "in"), "mass": "111"},
    "Lu-177": {"halfLife": 6.6443 * _DAY, "element": ("lutetium", "lu"), "mass": "177"},
    "Tl-201": {"halfLife": 3.0421 * _DAY, "element": ("thallium", "tl"), "mass": "201"},
    "Ga-67": {"halfLife": 3.2617 * _DAY, "element": ("gallium", "ga"), "mass": "67"},
    "Y-90": {"halfLife": 2.6684 * _DAY, "element": ("yttrium", "y"), "mass": "90"},
    "Ho-166": {"halfLife": 26.824 * _HOUR, "element": ("holmium", "ho"), "mass": "166"},
    "Sm-153": {"halfLife": 46.284 * _HOUR, "element": ("samarium", "sm"), "mass": "153"},
}
OTHER_ISOTOPE = "Other"

# Guard rail thresholds
HALF_LIFE_MISMATCH = 0.01            # DICOM vs table half-life, relative
WINDOW_TOLERANCE_KEV = 1.0           # energy window edges must agree this well
COUNT_RATE_WARNING_CPS = 40000.0     # photopeak count rate per head above which dead time may bias
SENSITIVITY_RANGE = (0.5, 500.0)     # cps/MBq per total view time; far outside means a unit or input error
VOI_FRACTION_WARNING = 0.9           # VOI should hold nearly all image counts for a whole-phantom VOI
DRIFT_WARNING = 0.10                 # change against the previous calibration of the same setup


# ---------------------------------------------------------------------------
# Isotopes and decay
# ---------------------------------------------------------------------------
def identifyIsotope(text):
    """Canonical isotope name from free text such as '^99m^Technetium', 'Tc-99m',
    'Lutetium^177^' or '177Lu'. None when not recognised."""
    if not text:
        return None
    lowered = str(text).lower()
    letters = re.sub(r"[^a-z]", " ", lowered)
    words = set(letters.split())
    numbers = set(re.findall(r"\d+", lowered))
    for name, entry in ISOTOPES.items():
        elementName, symbol = entry["element"]
        if entry["mass"] not in numbers:
            continue
        if elementName in lowered or symbol in words or (symbol + "m") in words:
            return name
    return None


def tableHalfLife(isotope):
    entry = ISOTOPES.get(isotope)
    return entry["halfLife"] if entry else None


def decayFactor(seconds, halfLifeSec):
    """Fraction remaining after `seconds` (negative seconds give a factor above 1)."""
    return math.exp(-math.log(2.0) * seconds / halfLifeSec)


def activityAt(targetTime, halfLifeSec, measuredMBq, measuredTime, residualMBq=0.0, residualTime=None):
    """Activity delivered (syringe minus residual), each decayed from its own measurement time."""
    activity = measuredMBq * decayFactor((targetTime - measuredTime).total_seconds(), halfLifeSec)
    if residualMBq:
        residualTime = residualTime or measuredTime
        activity -= residualMBq * decayFactor((targetTime - residualTime).total_seconds(), halfLifeSec)
    return activity


# ---------------------------------------------------------------------------
# Date/time helpers (DICOM DA/TM/DT and ISO strings)
# ---------------------------------------------------------------------------
def parseDicomDateTime(date=None, time=None, dt=None):
    """datetime from DICOM DA + TM, or from a DT value. Time zone suffixes are ignored
    (all times of one study come from the same scanner clock). None if unparseable."""
    try:
        if dt:
            text = re.split(r"[+-]", str(dt).strip())[0]
            date, time = text[:8], text[8:]
        if not date:
            return None
        date = str(date).strip().replace("-", "").replace(".", "")
        time = str(time or "").strip().replace(":", "")
        hh = int(time[0:2]) if len(time) >= 2 else 0
        mm = int(time[2:4]) if len(time) >= 4 else 0
        ss = int(time[4:6]) if len(time) >= 6 else 0
        micro = 0
        if "." in time:
            fraction = time.split(".", 1)[1]
            micro = int((fraction + "000000")[:6]) if fraction.isdigit() else 0
        return datetime.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), hh, mm, ss, micro)
    except (ValueError, TypeError, IndexError):
        return None


def toIso(value):
    return value.strftime("%Y-%m-%dT%H:%M:%S") if value else ""


def fromIso(text):
    if not text:
        return None
    try:
        return datetime.datetime.strptime(str(text)[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Calibration and conversion
# ---------------------------------------------------------------------------
def sensitivity(voiAcquisitionCounts, durationSec, activityMBq):
    """Calibration factor CF in cps/MBq, with durationSec the total view time."""
    if durationSec <= 0:
        raise ValueError("Acquisition duration must be positive.")
    if activityMBq <= 0:
        raise ValueError("Phantom activity at mid-scan must be positive.")
    if voiAcquisitionCounts <= 0:
        raise ValueError("The VOI holds no counts.")
    return voiAcquisitionCounts / durationSec / activityMBq


def bqPerMlFactor(durationSec, cpsPerMBq, voxelVolumeMl):
    """Multiply an acquisition-counts image by this to get Bq/ml."""
    if durationSec <= 0 or cpsPerMBq <= 0 or voxelVolumeMl <= 0:
        raise ValueError("Duration, calibration factor and voxel volume must be positive.")
    return 1.0e6 / (durationSec * cpsPerMBq * voxelVolumeMl)


def leanBodyMassJanmahasatian(weightKg, heightCm, sex):
    """Lean body mass (kg), Janmahasatian 2005 - the formula PERCIST uses for SUL."""
    if weightKg <= 0 or heightCm <= 0:
        raise ValueError("Weight and height must be positive for lean body mass.")
    bmi = weightKg / (heightCm / 100.0) ** 2
    sex = (sex or "").strip().upper()[:1]
    if sex == "M":
        return 9270.0 * weightKg / (6680.0 + 216.0 * bmi)
    if sex == "F":
        return 9270.0 * weightKg / (8780.0 + 244.0 * bmi)
    raise ValueError("Sex (M/F) is needed for lean body mass.")


SUV_BW = "SUVbw (body weight)"
SUV_LBM = "SUL (lean body mass, Janmahasatian)"
SUV_TYPES = [SUV_BW, SUV_LBM]


def suvFactor(injectedBqAtScan, weightKg, heightCm=None, sex=None, kind=SUV_BW):
    """Multiply a Bq/ml image by this to get SUV (g/ml, tissue density 1 g/ml).
    Returns (factor, massKg used)."""
    if injectedBqAtScan <= 0:
        raise ValueError("Injected activity at scan time must be positive.")
    if weightKg is None or weightKg <= 0:
        raise ValueError("Body weight must be positive.")
    massKg = leanBodyMassJanmahasatian(weightKg, heightCm or 0, sex) if kind == SUV_LBM else weightKg
    return massKg * 1000.0 / injectedBqAtScan, massKg


# ---------------------------------------------------------------------------
# Calibration library (JSON file, possibly shared between workstations)
# ---------------------------------------------------------------------------
def loadLibrary(path):
    """List of calibration records; empty list when the file does not exist yet."""
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("calibrations", []) if isinstance(data, dict) else []
    return [r for r in records if isinstance(r, dict) and r.get("id")]


def _writeLibrary(path, records):
    """Atomic write: a crash or a second workstation never sees a half-written file."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    handle, tempPath = tempfile.mkstemp(prefix=".spectcal_", suffix=".json", dir=folder)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as f:
            json.dump({"format": "SPECTCalibrationLibrary", "version": LIBRARY_FORMAT_VERSION,
                       "calibrations": records}, f, indent=2)
        os.replace(tempPath, path)
    except Exception:
        if os.path.exists(tempPath):
            os.remove(tempPath)
        raise


def addCalibration(path, record):
    """Re-read the file and append, so calibrations saved meanwhile by others are kept."""
    record = dict(record)
    record.setdefault("id", uuid.uuid4().hex)
    record.setdefault("created", toIso(datetime.datetime.now()))
    records = loadLibrary(path)
    records.append(record)
    _writeLibrary(path, records)
    return record


def deleteCalibration(path, calibrationId):
    records = [r for r in loadLibrary(path) if r.get("id") != calibrationId]
    _writeLibrary(path, records)


def windowsMatch(a, b, tolerance=WINDOW_TOLERANCE_KEV):
    """Energy windows given as [low, high] keV."""
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a[0]) - float(b[0])) <= tolerance and abs(float(a[1]) - float(b[1])) <= tolerance


def setupKey(record):
    """What makes two calibrations comparable (for drift): same camera, collimator,
    isotope, photopeak and scatter method (with its weight)."""
    camera = record.get("camera", {})
    photopeak = record.get("photopeak") or [0, 0]
    return (camera.get("model", ""), camera.get("serial", ""), record.get("collimator", ""),
            record.get("isotope", ""), round(float(photopeak[0])), round(float(photopeak[1])),
            scatterLabel(record))


def previousCalibration(records, record):
    """Most recent earlier calibration of the same setup, or None."""
    key = setupKey(record)
    same = [r for r in records if setupKey(r) == key and r.get("id") != record.get("id")]
    same.sort(key=lambda r: r.get("created", ""))
    return same[-1] if same else None


def compatibility(calibration, acquisition):
    """(errors, warnings) for applying `calibration` to an acquisition described by
    readAcquisitionInfo(). Errors make the calibration unusable for this study."""
    errors, warnings = [], []
    camera = calibration.get("camera", {})

    def differs(a, b):
        return bool(a) and bool(b) and str(a).strip().lower() != str(b).strip().lower()

    if differs(calibration.get("isotope"), acquisition.get("isotope")):
        errors.append(f"isotope {calibration.get('isotope')} vs {acquisition.get('isotope')}")
    if differs(camera.get("model"), acquisition.get("model")):
        errors.append(f"camera model {camera.get('model')} vs {acquisition.get('model')}")
    if differs(camera.get("serial"), acquisition.get("serial")):
        errors.append(f"camera serial {camera.get('serial')} vs {acquisition.get('serial')}")
    if differs(calibration.get("collimator"), acquisition.get("collimator")):
        errors.append(f"collimator {calibration.get('collimator')} vs {acquisition.get('collimator')}")
    if calibration.get("timeBasis") != TIME_BASIS:
        errors.append("made with an older version of the plugin (different time convention) - recalibrate")
    if not acquisition.get("isotope"):
        warnings.append("isotope not recorded in this study - check it matches")
    if not acquisition.get("collimator"):
        warnings.append("collimator not recorded in this study - check it matches")

    windows = acquisition.get("windows", [])
    if not any(windowsMatch(calibration.get("photopeak"), [w["low"], w["high"]]) for w in windows
               if w.get("low") is not None):
        errors.append(f"no energy window matching the calibration photopeak {calibration.get('photopeak')} keV")
    scatter = calibration.get("scatter", {})
    for side in ("lower", "upper"):
        wanted = scatter.get(side)
        if wanted and not any(windowsMatch(wanted, [w["low"], w["high"]]) for w in windows
                              if w.get("low") is not None):
            errors.append(f"no {side} scatter window matching {wanted} keV")
    return errors, warnings


def findWindowIndex(windows, energyRange):
    for w in windows:
        if w.get("low") is not None and windowsMatch(energyRange, [w["low"], w["high"]]):
            return w["index"]
    return None


def scatterLabel(record):
    """Scatter method of a calibration, with the dual window weight: 'DEW k=1.1', 'DEW auto', 'TEW'.
    Calibrations made before the weight existed used the automatic weight."""
    scatter = record.get("scatter", {})
    method = scatter.get("method", "None")
    if method != "DEW":
        return method
    weight = scatter.get("weight")
    return "DEW auto" if weight is None else f"DEW k={float(weight):g}"


def describe(record):
    """One-line label for lists."""
    scatter = scatterLabel(record)
    created = str(record.get("created", ""))[:10]
    return (f"{record.get('name', 'unnamed')} - {record.get('isotope', '?')}, "
            f"{record.get('collimator', '?')}, SC {scatter}, "
            f"{float(record.get('cpsPerMBq', 0)):.2f} cps/MBq ({created})")


# ---------------------------------------------------------------------------
# Where the library lives (a Slicer setting; the file itself can be on a shared drive)
# ---------------------------------------------------------------------------
LIBRARY_SETTINGS_KEY = "SPECTRecon/CalibrationLibraryPath"


def calibrationLibraryPath():
    import qt
    import slicer
    path = qt.QSettings().value(LIBRARY_SETTINGS_KEY) or ""
    if not path:
        path = os.path.join(os.path.dirname(slicer.app.slicerUserSettingsFilePath), "SPECTCalibrations.json")
    return str(path)


def setCalibrationLibraryPath(path):
    import qt
    qt.QSettings().setValue(LIBRARY_SETTINGS_KEY, str(path))


def chooseCalibrationLibrary(parent=None):
    """File dialog to pick an existing library or name a new one. Returns the path or None."""
    import qt
    dialog = qt.QFileDialog(parent, "SPECT calibration library (JSON)", calibrationLibraryPath(),
                            "Calibration library (*.json)")
    dialog.setAcceptMode(qt.QFileDialog.AcceptSave)          # allows naming a new file
    dialog.setOption(qt.QFileDialog.DontConfirmOverwrite, True)  # picking an existing one is normal
    dialog.setDefaultSuffix("json")
    if not dialog.exec_():
        return None
    files = dialog.selectedFiles()
    if not files:
        return None
    path = str(files[0])
    setCalibrationLibraryPath(path)
    return path


# ---------------------------------------------------------------------------
# Phantom evaluation (used by the SPECT Calibration module)
# ---------------------------------------------------------------------------
ERROR, WARNING, INFO = "error", "warning", "info"


def _number(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _range(text):
    try:
        values = json.loads(text) if text else None
        return [float(values[0]), float(values[1])] if values else None
    except (ValueError, TypeError, IndexError):
        return None


def evaluatePhantom(attributes, voiSum, imageSum, activityMBq, measuredTime, midTime, durationSec,
                    halfLifeSec, isotope, residualMBq=0.0, residualTime=None, voiDescription="",
                    previous=None):
    """Calibration record from a reconstructed phantom of known activity.

    attributes  the volume's SPECTRecon.* attributes, without the prefix
    voiSum      sum of voxel values in the VOI (in the volume's own units)
    imageSum    sum over the whole image (to check the VOI holds the counts)
    Returns (record or None, messages) where messages are (level, text) tuples;
    record is None when any message is an error."""
    messages = []

    def add(level, text):
        messages.append((level, text))

    if attributes.get("AttenuationCorrection") != "CT":
        add(ERROR, "The phantom must be reconstructed with CT attenuation correction (NAC is not accepted).")
    if attributes.get("Units", "counts") != "counts":
        add(ERROR, f"The volume is already quantified ({attributes.get('Units')}); use a reconstruction in counts.")
    if attributes.get("Gate"):
        add(ERROR, "This is a single gate; use the static reconstruction.")
    toCounts = _number(attributes.get("ToAcquisitionCounts"))
    if not toCounts or toCounts <= 0:
        add(ERROR, "The volume does not record its count scaling (reconstruct it again with this version).")
    photopeak = _range(attributes.get("PhotopeakRange"))
    if not photopeak:
        add(ERROR, "The photopeak energy window is not recorded on the volume.")
    if not attributes.get("TotalViewTimeSec"):
        add(ERROR, "The volume does not record its total view time (reconstruct it again with this version).")
    detectors = int(_number(attributes.get("Detectors"), 1) or 1)
    if not isotope:
        add(ERROR, "Select the isotope.")
    if not halfLifeSec or halfLifeSec <= 0:
        add(ERROR, "Half-life must be positive.")
    if not durationSec or durationSec <= 0:
        add(ERROR, "Total view time must be positive.")
    if activityMBq <= 0:
        add(ERROR, "Phantom activity must be positive.")
    if measuredTime and midTime and measuredTime > midTime:
        add(WARNING, "The activity was measured after the scan - check the times.")
    if residualMBq and residualTime and midTime and residualTime > midTime:
        add(INFO, "Residual measured after the scan (normal); it is decayed back to the scan mid-time.")
    if any(level == ERROR for level, _ in messages):
        return None, messages

    activityAtMid = activityAt(midTime, halfLifeSec, activityMBq, measuredTime, residualMBq, residualTime)
    if activityAtMid <= 0:
        add(ERROR, "Activity at the scan mid-time is not positive (residual larger than the syringe?).")
        return None, messages
    elapsedHours = (midTime - measuredTime).total_seconds() / 3600.0
    add(INFO, f"Activity at scan mid-time: {activityAtMid:.3f} MBq "
              f"({elapsedHours:+.2f} h from measurement, decay factor "
              f"{decayFactor(elapsedHours * 3600.0, halfLifeSec):.4f}).")

    voiCounts = voiSum * toCounts
    try:
        cf = sensitivity(voiCounts, durationSec, activityAtMid)
    except ValueError as e:
        add(ERROR, str(e))
        return None, messages
    add(INFO, f"VOI acquisition counts {voiCounts:.4g} over a total view time of {durationSec:.1f} s -> "
              f"{cf:.4f} cps/MBq.")
    evidence = attributes.get("DetectorsEvidence", "")
    if detectors > 1:
        add(INFO, f"{detectors} detectors detected ({evidence}): {durationSec / detectors:.1f} s each, "
                  f"{durationSec:.1f} s total view time. The factor uses the total, so it does not depend "
                  "on the detector count.")
    else:
        add(INFO, f"1 detector recorded ({evidence}). The factor uses the total view time, so this only "
                  "affects the scan mid-time - check it if the camera has two heads.")

    # Guard rails
    if not (SENSITIVITY_RANGE[0] <= cf <= SENSITIVITY_RANGE[1]):
        add(WARNING, f"{cf:.3g} cps/MBq is outside the plausible range {SENSITIVITY_RANGE} - "
                     "check activity units, times and the total view time.")
    fraction = voiSum / imageSum if imageSum > 0 else None
    if fraction is not None and fraction < VOI_FRACTION_WARNING and "whole image" not in voiDescription.lower():
        add(WARNING, f"The VOI holds only {100 * fraction:.1f}% of the image counts - enlarge the margin "
                     "unless there is activity outside the phantom.")
    photopeakCounts = _number(attributes.get("PhotopeakCounts"))
    countRate = None
    if photopeakCounts:
        countRate = photopeakCounts / durationSec  # counts per detector-second = rate seen by a detector
        add(INFO, f"Photopeak count rate {countRate:.0f} cps per detector.")
        if countRate > COUNT_RATE_WARNING_CPS:
            add(WARNING, f"Count rate above {COUNT_RATE_WARNING_CPS:.0f} cps per detector - dead time may "
                         "lower the calibration; use less activity.")
    consistency = _number(attributes.get("DataConsistency"))
    if consistency is not None and abs(consistency - 1.0) > 0.15:
        add(WARNING, f"Data consistency {consistency:.3f} - is the phantom fully inside the field of view?")

    scatterMethod = attributes.get("ScatterMethod", "None") or "None"
    scatterWeight = _number(attributes.get("ScatterWeight")) if scatterMethod == "DEW" else None
    record = {
        "isotope": isotope,
        "halfLifeSec": halfLifeSec,
        "camera": {"manufacturer": attributes.get("Manufacturer", ""), "model": attributes.get("CameraModel", ""),
                   "serial": attributes.get("CameraSerial", ""), "station": attributes.get("Station", "")},
        "collimator": attributes.get("Collimator", ""),
        "timeBasis": TIME_BASIS,
        "detectors": detectors,
        "detectorsEvidence": evidence,
        "photopeak": photopeak,
        "scatter": {"method": scatterMethod,
                    "lower": _range(attributes.get("ScatterLowerRange")),
                    "upper": _range(attributes.get("ScatterUpperRange")),
                    "weight": scatterWeight},
        "attenuationCorrection": "CT",
        "recon": {"iterations": int(_number(attributes.get("ReconIterations"), 0) or 0) or None,
                  "subsets": int(_number(attributes.get("ReconSubsets"), 0) or 0) or None,
                  "postFilter": attributes.get("PostFilter", ""),
                  "pytomography": attributes.get("PyTomographyVersion", "")},
        "cpsPerMBq": cf,
        "phantom": {"activityMBq": activityMBq, "measuredTime": toIso(measuredTime),
                    "residualMBq": residualMBq, "residualTime": toIso(residualTime),
                    "scanMidTime": toIso(midTime), "totalViewTimeSec": durationSec,
                    "activityAtMidMBq": activityAtMid, "voi": voiDescription,
                    "voiAcquisitionCounts": voiCounts, "voiFractionOfImage": fraction,
                    "countRatePerHeadCps": countRate, "dataConsistency": consistency,
                    "sourceFile": attributes.get("SourceFile", "")},
    }
    if previous:
        drift = driftMessage(record, [previous])
        if drift:
            add(*drift)
    if scatterMethod == "None":
        add(INFO, "No scatter correction: this calibration will only be offered for reconstructions without it.")
    return record, messages


def driftMessage(record, records):
    """(level, text) comparing a new calibration with the latest one of the same setup, or None."""
    previous = previousCalibration(records, record)
    if not previous or float(previous.get("cpsPerMBq", 0)) <= 0:
        return None
    drift = float(record["cpsPerMBq"]) / float(previous["cpsPerMBq"]) - 1.0
    level = WARNING if abs(drift) > DRIFT_WARNING else INFO
    return level, (f"{100 * drift:+.1f}% against the previous calibration of this setup "
                   f"('{previous.get('name')}', {str(previous.get('created', ''))[:10]}).")
