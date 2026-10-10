"""NM projection data: header inspection (energy windows, gating), energy window
defaults and validation, CT series cleanup, and writing gated data as plain NM files.

No Slicer dependency: needs only numpy and pydicom (imported when used).
"""

import logging
import traceback

import numpy as np


# ---------------------------------------------------------------------------
# NM header inspection (energy windows, gating)
# ---------------------------------------------------------------------------
def readNmInfo(filePath):
    """Energy windows and gating information from an NM projection file."""
    info = {"windows": [], "nGates": 1, "gateNumbers": [], "warnings": []}
    try:
        import pydicom
        ds = pydicom.dcmread(filePath, stop_before_pixels=True)
        nFrames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        for i, item in enumerate(getattr(ds, "EnergyWindowInformationSequence", []) or []):
            name = str(getattr(item, "EnergyWindowName", "") or f"Window {i + 1}")
            low = high = None
            ranges = getattr(item, "EnergyWindowRangeSequence", None)
            if ranges:
                low = float(getattr(ranges[0], "EnergyWindowLowerLimit", 0) or 0)
                high = float(getattr(ranges[0], "EnergyWindowUpperLimit", 0) or 0)
            center = (low + high) / 2.0 if (low is not None and high is not None) else None
            label = name if center is None else f"{name} ({low:.0f}-{high:.0f} keV)"
            info["windows"].append({"index": i, "name": name, "low": low, "high": high,
                                    "center": center, "label": label})
        # Gating is taken from the per-frame TimeSlotVector (the gates actually present),
        # not from NumberOfTimeSlots, which can count gates that hold no frames
        declared = int(getattr(ds, "NumberOfTimeSlots", 1) or 1)
        timeSlots = getattr(ds, "TimeSlotVector", None)
        gateNumbers = []
        if timeSlots is not None:
            try:
                slots = np.atleast_1d(np.asarray(timeSlots, dtype=int))
                if slots.size == nFrames:
                    gateNumbers = sorted(set(int(g) for g in slots))
                else:
                    info["warnings"].append(
                        f"TimeSlotVector has {slots.size} entries for {nFrames} frames - gating ignored.")
            except Exception:
                info["warnings"].append("TimeSlotVector could not be read - gating ignored.")
        elif declared > 1:
            info["warnings"].append(
                f"Header declares {declared} time slots but has no TimeSlotVector - treated as not gated.")
        if len(gateNumbers) > 1:
            if declared > 1 and declared != len(gateNumbers):
                info["warnings"].append(
                    f"Header declares {declared} time slots, frames exist for {len(gateNumbers)}: "
                    f"using gates {gateNumbers}.")
            info["gateNumbers"] = gateNumbers
            info["nGates"] = len(gateNumbers)
    except Exception:
        logging.warning("Could not read NM header info:\n" + traceback.format_exc())
    return info


# ---------------------------------------------------------------------------
# Energy window helpers (defaults and validation)
# ---------------------------------------------------------------------------
def _windowWidth(window):
    if window.get("low") is None or window.get("high") is None:
        return None
    return window["high"] - window["low"]


def _edgeTolerance(peak):
    return max(2.0, 0.05 * (peak["center"] or 0.0))


def adjacentScatterWindows(peak, windows):
    """Windows directly below / above the photopeak (touching it, or within a small gap).
    A distant window - e.g. the 113 keV photopeak when 208 keV is the peak - is not a
    scatter window and is never suggested."""
    if peak.get("center") is None:
        return [], []
    tol = _edgeTolerance(peak)
    gap = 0.15 * peak["center"]
    below, above = [], []
    for w in windows:
        if w["index"] == peak["index"] or w.get("center") is None:
            continue
        if w["center"] < peak["center"] and peak["low"] - gap <= w["high"] <= peak["low"] + tol:
            below.append(w)
        elif w["center"] > peak["center"] and peak["high"] - tol <= w["low"] <= peak["high"] + gap:
            above.append(w)
    return below, above


def defaultPeakIndex(windows):
    """Best guess at the photopeak window: named like one (EM, PEAK), not named like a
    scatter window (SC, LOWER, UPPER), with scatter windows on both sides, and wide."""
    usable = [w for w in windows if w.get("center") is not None]
    if not usable:
        return 0

    def score(w):
        name = str(w["name"]).upper()
        s = 0
        if name.startswith("EM") or "PEAK" in name or name.startswith("PP"):
            s += 10
        if "SC" in name or "LOWER" in name or "UPPER" in name:
            s -= 10
        below, above = adjacentScatterWindows(w, windows)
        s += 2 * (bool(below) + bool(above))
        return (s, _windowWidth(w) or 0.0)

    return max(usable, key=score)["index"]


def scatterWindowProblem(windows, indexPeak, indexLower, indexUpper):
    """None if the window choice is usable for energy window scatter correction,
    otherwise a short explanation."""
    if indexLower is None and indexUpper is None:
        return None
    if indexLower is None:
        return ("Pick a lower scatter window: PyTomography's energy window scatter estimate "
                "needs one (upper-only is not supported).")
    byIndex = {w["index"]: w for w in windows}
    peak = byIndex.get(indexPeak)
    if peak is None:
        return "Photopeak window not found in the header."
    for label, idx in (("Lower", indexLower), ("Upper", indexUpper)):
        if idx is None:
            continue
        if idx == indexPeak:
            return f"{label} scatter window is the photopeak window itself."
        w = byIndex.get(idx)
        if w is None:
            return f"{label} scatter window not found in the header."
        width = _windowWidth(w)
        if width is not None and width <= 0:
            return f"{label} scatter window has zero width in the header."
        if peak.get("center") is not None and w.get("center") is not None:
            tol = _edgeTolerance(peak)
            if label == "Lower" and (w["center"] >= peak["center"] or w["high"] > peak["low"] + tol):
                return "Lower scatter window is not below the photopeak window (or overlaps it)."
            if label == "Upper" and (w["center"] <= peak["center"] or w["low"] < peak["high"] - tol):
                return "Upper scatter window is not above the photopeak window (or overlaps it)."
    peakWidth = _windowWidth(peak)
    if peakWidth is not None and peakWidth <= 0:
        return "Photopeak window has zero width in the header."
    return None


# ---------------------------------------------------------------------------
# CT series cleanup
# ---------------------------------------------------------------------------
def cleanCtFiles(files, log=logging.info):
    """Keep only the axial image slices of a CT series, sorted along the slice normal.
    Drops localizers/scouts, slices with another orientation or matrix size, and
    duplicate positions - any of these breaks the series reader and the mu-map."""
    import pydicom
    entries = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue
        imageType = [str(v).upper() for v in (getattr(ds, "ImageType", []) or [])]
        if "LOCALIZER" in imageType:
            continue
        position = getattr(ds, "ImagePositionPatient", None)
        orientation = getattr(ds, "ImageOrientationPatient", None)
        if position is None or orientation is None:
            continue
        orientation = tuple(round(float(v), 3) for v in orientation)
        entries.append({"file": f, "position": [float(v) for v in position],
                        "orientation": orientation,
                        "matrix": (int(getattr(ds, "Rows", 0)), int(getattr(ds, "Columns", 0)))})
    if not entries:
        return list(files)

    # The most common orientation and matrix size defines the series
    def mostCommon(key):
        values = [e[key] for e in entries]
        return max(set(values), key=values.count)

    orientation, matrix = mostCommon("orientation"), mostCommon("matrix")
    kept = [e for e in entries if e["orientation"] == orientation and e["matrix"] == matrix]
    normal = np.cross(np.array(orientation[:3]), np.array(orientation[3:]))
    for e in kept:
        e["along"] = float(np.dot(normal, e["position"]))
    kept.sort(key=lambda e: e["along"])

    unique, lastAlong = [], None
    for e in kept:
        if lastAlong is not None and abs(e["along"] - lastAlong) < 1e-3:
            continue
        unique.append(e)
        lastAlong = e["along"]

    dropped = len(files) - len(unique)
    if dropped:
        log(f"CT: using {len(unique)} of {len(files)} files "
            f"(dropped {dropped} localizer/odd-orientation/duplicate files).")
    return [e["file"] for e in unique]


_FRAME_VECTOR_KEYS = ["EnergyWindowVector", "DetectorVector", "RotationVector",
                      "AngularPositionVector", "TimeSlotVector", "PhaseVector", "SliceVector"]


def _frameVectors(ds, nFrames):
    vectors = {}
    for key in _FRAME_VECTOR_KEYS:
        value = getattr(ds, key, None)
        if value is None:
            continue
        arr = np.atleast_1d(np.asarray(value, dtype=int))
        if arr.size == nFrames:
            vectors[key] = arr
    return vectors


def _writeUngatedCopy(ds, vectors, keepIndices, newFrames, outPath):
    """Write an NM file that holds newFrames and no gating tags, so PyTomography
    reads it as an ordinary (non-gated) tomographic acquisition."""
    import pydicom

    for key, arr in vectors.items():
        if key == "TimeSlotVector":
            continue
        setattr(ds, key, [int(v) for v in arr[keepIndices]])

    for key in ("TimeSlotVector", "NumberOfTimeSlots", "TimeSlotInformationSequence",
                "TriggerVector", "NumberOfTriggers", "RRIntervalVector", "NumberOfRRIntervals"):
        if key in ds:
            delattr(ds, key)

    pointer = getattr(ds, "FrameIncrementPointer", None)
    if pointer is not None:
        timeSlotTag = pydicom.tag.Tag(0x0054, 0x0070)
        try:
            kept = [t for t in (pointer if isinstance(pointer, (list, pydicom.multival.MultiValue))
                                else [pointer]) if pydicom.tag.Tag(t) != timeSlotTag]
            ds.FrameIncrementPointer = kept
        except Exception:
            pass

    imageType = list(getattr(ds, "ImageType", []) or [])
    ds.ImageType = ["TOMO" if str(v).upper() == "GATED TOMO" else v for v in imageType]

    ds.NumberOfFrames = int(newFrames.shape[0])
    ds.PixelData = np.ascontiguousarray(newFrames).tobytes()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(outPath, write_like_original=False)


def writeGateFile(sourceFile, outPath, gateNumber):
    """One time slot of a gated acquisition, written as a standalone NM file."""
    import pydicom
    ds = pydicom.dcmread(sourceFile)
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress()
    frames = ds.pixel_array
    if frames.ndim == 2:
        frames = frames[np.newaxis]
    vectors = _frameVectors(ds, frames.shape[0])
    timeSlots = vectors.get("TimeSlotVector")
    if timeSlots is None:
        raise ValueError("No TimeSlotVector in this file - it is not gated.")
    keep = np.where(timeSlots == gateNumber)[0]
    if keep.size == 0:
        raise ValueError(f"No frames for gate {gateNumber}.")
    _writeUngatedCopy(ds, vectors, keep, frames[keep], outPath)
    return outPath


def writeSummedGatesFile(sourceFile, outPath, log=print):
    """Sum the planar counts of all time slots into one non-gated NM file.
    Returns the factor the counts were scaled by (1.0 unless the sum overflowed)."""
    import pydicom
    ds = pydicom.dcmread(sourceFile)
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress()
    frames = ds.pixel_array
    if frames.ndim == 2:
        frames = frames[np.newaxis]
    originalDtype = frames.dtype
    vectors = _frameVectors(ds, frames.shape[0])
    timeSlots = vectors.get("TimeSlotVector")
    if timeSlots is None:
        raise ValueError("No TimeSlotVector in this file - it is not gated.")

    gates = sorted(set(int(g) for g in timeSlots))
    template = np.where(timeSlots == gates[0])[0]
    summed = np.zeros((template.size,) + frames.shape[1:], dtype=np.float64)
    for gate in gates:
        idx = np.where(timeSlots == gate)[0]
        if idx.size != template.size:
            raise ValueError(f"Gate {gate} has {idx.size} frames, gate {gates[0]} has {template.size}.")
        summed += frames[idx].astype(np.float64)
    log(f"Summed {len(gates)} gates into {template.size} projection frames "
        f"(total counts {summed.sum():.0f}).")

    scale = 1.0
    maxStorable = float(2 ** int(getattr(ds, "BitsStored", 16)) - 1)
    if summed.max() > maxStorable:
        scale = maxStorable / float(summed.max())
        summed = summed * scale
        log(f"  -> Summed counts exceed the pixel range; scaled by {scale:.6f} and undone after reconstruction.")

    newFrames = np.rint(summed).astype(originalDtype)
    _writeUngatedCopy(ds, vectors, template, newFrames, outPath)
    return scale


# ---------------------------------------------------------------------------
# Acquisition metadata for quantification
# ---------------------------------------------------------------------------
def _first(sequence):
    try:
        return sequence[0] if sequence else None
    except Exception:
        return None


def _text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        value = "\\".join(str(v) for v in value)
    return str(value).strip()


def _privateElement(ds, group, creator, offset):
    """Value of a private element identified by its creator string (e.g. 'GEMS_GENIE_1')
    and its offset in the creator's block; None when absent. Works on items of sequences too."""
    try:
        import pydicom
        for blockElement in range(0x10, 0x100):
            tag = pydicom.tag.Tag(group, blockElement)
            if tag not in ds:
                continue
            value = ds[tag].value
            if isinstance(value, bytes):
                value = value.decode("latin1", "ignore")
            if str(value).strip().rstrip("\x00").strip() == creator:
                target = pydicom.tag.Tag(group, (blockElement << 8) | offset)
                return ds[target].value if target in ds else None
    except Exception:
        return None
    return None


def _asList(value):
    if value is None:
        return []
    if isinstance(value, (bytes, str)):
        text = value.decode("latin1", "ignore") if isinstance(value, bytes) else value
        return [v for v in text.strip("\x00 ").split("\\")]
    try:
        return list(value)
    except TypeError:
        return [value]


def _uint32Array(value, expected):
    """GE per-view arrays are UL; in implicit-VR files they arrive as raw bytes."""
    import struct
    if isinstance(value, bytes):
        if len(value) != 4 * expected:
            return None
        return list(struct.unpack("<%dI" % expected, value))
    values = _asList(value)
    try:
        return [float(v) for v in values] if len(values) == expected else None
    except (TypeError, ValueError):
        return None


def _gePerViewDurationsMs(ds, views, frameMs):
    """GE (Infinia/Xeleris 'QUASAR_INTERNAL_USE' block): actual duration of every view in ms.
    Used only when there is one value per view and they agree with ActualFrameDuration."""
    values = _uint32Array(_privateElement(ds, 0x0009, "QUASAR_INTERNAL_USE", 0x03), views)
    if not values or not frameMs:
        return None
    median = sorted(values)[len(values) // 2]
    if abs(median - frameMs) > 0.2 * frameMs or min(values) <= 0:
        return None
    return values


def _medisoViewTimes(rotation, views):
    """Mediso ('MEDISO-1' block in the rotation item): start and end date-time of every view."""
    from .quantification import parseDicomDateTime
    if rotation is None:
        return None
    starts = _asList(_privateElement(rotation, 0x0009, "MEDISO-1", 0x27))
    ends = _asList(_privateElement(rotation, 0x0009, "MEDISO-1", 0x28))
    if len(starts) != views or len(ends) != views:
        return None
    starts = [parseDicomDateTime(dt=str(v)) for v in starts]
    ends = [parseDicomDateTime(dt=str(v)) for v in ends]
    if any(v is None for v in starts + ends):
        return None
    return starts, ends


def _geCameraSensitivity(ds):
    """GE Xeleris 'Camera Sensitivity Factor and Isotope info per Energy Set', if the export
    carries one: list of values per energy set (0 = not calibrated on the GE side)."""
    try:
        import pydicom
        sequence = None
        for element in ds:
            if element.tag.group == 0x0033 and element.VR == "SQ":
                for item in element.value:
                    if str(_privateElement(item, 0x0033, "GEMS_XELPRV_01", 0x11) or "").strip() == "CameraSensitivity":
                        sequence = item
                        break
            if sequence is not None:
                break
        if sequence is None:
            return None
        values = _asList(_privateElement(sequence, 0x0033, "GEMS_XELPRV_01", 0x1E))
        labels = _asList(_privateElement(sequence, 0x0033, "GEMS_XELPRV_01", 0x22))
        values = [float(v) for v in values]
        result, position = {}, 0
        for name, count in zip(labels[0::2], labels[1::2]):
            count = int(count)
            result[name.strip()] = values[position:position + count]
            position += count
        return result.get("CameraSensitivity")
    except Exception:
        return None


def readAcquisitionInfo(filePath):
    """Camera, collimator, isotope, timing and patient data needed for quantification.

    Timing convention: 'totalViewTimeSec' is the counting time summed over all views of all
    detectors (views x frame duration, or the vendor's per-view durations). The calibration uses
    it directly, so it does not depend on knowing how many detectors acquired at the same time -
    exports often store a dual-head acquisition as one merged 'detector'. The detector count is
    only used for the wall-clock mid-time (decay correction) and for information.
    Missing items are left empty and listed in 'warnings'."""
    import datetime
    import pydicom
    from .quantification import (identifyIsotope, parseDicomDateTime, tableHalfLife, toIso,
                                 HALF_LIFE_MISMATCH)

    info = {"warnings": [], "notes": []}
    ds = pydicom.dcmread(filePath, stop_before_pixels=True)
    n = int(getattr(ds, "NumberOfFrames", 1) or 1)

    info["manufacturer"] = _text(getattr(ds, "Manufacturer", ""))
    info["model"] = _text(getattr(ds, "ManufacturerModelName", ""))
    info["serial"] = _text(getattr(ds, "DeviceSerialNumber", ""))
    info["station"] = _text(getattr(ds, "StationName", ""))

    detectorItems = list(getattr(ds, "DetectorInformationSequence", None) or [])
    detector = _first(detectorItems)
    collimator = ""
    if detector is not None:
        collimator = _text(getattr(detector, "CollimatorGridName", "")) or \
                     _text(getattr(detector, "CollimatorType", ""))
    info["collimator"] = collimator

    # ---- isotope and administered activity
    radiopharm = _first(getattr(ds, "RadiopharmaceuticalInformationSequence", None))
    isotopeText, dicomHalfLife, injectedBq, injectionTime, radiopharmaceutical = "", None, None, None, ""
    if radiopharm is not None:
        code = _first(getattr(radiopharm, "RadionuclideCodeSequence", None))
        if code is not None:
            isotopeText = _text(getattr(code, "CodeMeaning", "")) or _text(getattr(code, "CodeValue", ""))
        radiopharmaceutical = _text(getattr(radiopharm, "Radiopharmaceutical", ""))
        try:
            dicomHalfLife = float(getattr(radiopharm, "RadionuclideHalfLife", 0) or 0) or None
        except (TypeError, ValueError):
            dicomHalfLife = None
        try:
            injectedBq = float(getattr(radiopharm, "RadionuclideTotalDose", 0) or 0) or None
        except (TypeError, ValueError):
            injectedBq = None
        injectionTime = parseDicomDateTime(dt=getattr(radiopharm, "RadiopharmaceuticalStartDateTime", None)) or \
            parseDicomDateTime(getattr(ds, "SeriesDate", None) or getattr(ds, "StudyDate", None),
                               getattr(radiopharm, "RadiopharmaceuticalStartTime", None))
    # Fallbacks when the standard code is empty: vendor fields and window/series names
    windowNames = [_text(getattr(w, "EnergyWindowName", ""))
                   for w in (getattr(ds, "EnergyWindowInformationSequence", None) or [])]
    candidates = [("radionuclide code", isotopeText), ("radiopharmaceutical", radiopharmaceutical),
                  ("GE private isotope", _text(_privateElement(ds, 0x0011, "GEMS_GENIE_1", 0x0D))),
                  ("energy window name", " ".join(windowNames)),
                  ("series description", _text(getattr(ds, "SeriesDescription", "")))]
    isotope, isotopeSource = "", ""
    for source, text in candidates:
        found = identifyIsotope(text)
        if found:
            isotope, isotopeSource = found, source
            break
    info["isotopeText"] = isotopeText or radiopharmaceutical or \
        _text(_privateElement(ds, 0x0011, "GEMS_GENIE_1", 0x0D))
    info["isotope"] = isotope
    info["isotopeSource"] = isotopeSource
    if isotope and isotopeSource not in ("radionuclide code", "radiopharmaceutical"):
        info["notes"].append(f"Isotope {isotope} taken from the {isotopeSource} (standard field empty).")
    tableValue = tableHalfLife(isotope) if isotope else None
    if dicomHalfLife and tableValue and abs(dicomHalfLife - tableValue) / tableValue > HALF_LIFE_MISMATCH:
        info["warnings"].append(f"DICOM half-life {dicomHalfLife:.0f} s differs from the {isotope} "
                                f"table value {tableValue:.0f} s - check the isotope.")
    info["halfLifeSec"] = dicomHalfLife or tableValue
    info["injectedMBq"] = injectedBq / 1e6 if injectedBq else None
    info["injectionTime"] = toIso(injectionTime)
    if not isotope:
        info["warnings"].append("Isotope could not be identified from the DICOM header.")

    # ---- views and frame duration
    def vector(name):
        value = getattr(ds, name, None)
        if value is None:
            return None
        try:
            values = [int(v) for v in value]
        except TypeError:
            values = [int(value)]
        return values if len(values) == n else None

    windowsVector = vector("EnergyWindowVector")
    slotsVector = vector("TimeSlotVector")
    detectorVector = vector("DetectorVector")
    selected = [True] * n
    if windowsVector:
        selected = [s and w == windowsVector[0] for s, w in zip(selected, windowsVector)]
    if slotsVector:
        selected = [s and t == slotsVector[0] for s, t in zip(selected, slotsVector)]
    views = sum(selected)
    rotation = _first(getattr(ds, "RotationInformationSequence", None))
    frameMs = None
    for source in (rotation, ds):
        if source is None:
            continue
        try:
            frameMs = float(getattr(source, "ActualFrameDuration", 0) or 0) or None
        except (TypeError, ValueError):
            frameMs = None
        if frameMs:
            break
    info["views"] = views
    info["frameDurationMs"] = frameMs

    perViewMs = _gePerViewDurationsMs(ds, views, frameMs)
    medisoTimes = _medisoViewTimes(rotation, views)
    if perViewMs:
        totalViewTime = sum(perViewMs) / 1000.0
        info["durationSource"] = "GE per-view durations"
    elif frameMs:
        totalViewTime = views * frameMs / 1000.0
        info["durationSource"] = f"{views} views x {frameMs / 1000.0:.3f} s"
    else:
        totalViewTime = None
        info["durationSource"] = ""
        info["warnings"].append("Frame duration missing - enter the total view time manually.")
    info["totalViewTimeSec"] = totalViewTime
    if medisoTimes and frameMs:
        elapsed = [(e - s).total_seconds() for s, e in zip(*medisoTimes)]
        mean = sum(elapsed) / len(elapsed)
        if abs(mean * 1000.0 - frameMs) > 0.01 * frameMs:
            info["notes"].append(f"Mediso view timestamps give {mean:.3f} s per view, the frame duration "
                                 f"{frameMs / 1000.0:.3f} s ({100 * (mean * 1000.0 / frameMs - 1):+.1f}%); "
                                 "the frame duration is used for counting time.")
    if slotsVector and len(set(slotsVector)) > 1:
        info["warnings"].append("Gated acquisition: check that the frame duration covers the whole "
                                "view (all gates) before quantifying.")

    # ---- how many detectors acquired at the same time (information + mid-time only)
    evidence = []
    if detectorVector and len(set(detectorVector)) > 1:
        evidence.append((len(set(detectorVector)), "per-view detector labels"))
    if len(detectorItems) > 1:
        evidence.append((len(detectorItems), f"detector information sequence has {len(detectorItems)} entries"))
    if perViewMs and views % 2 == 0 and perViewMs[:views // 2] == perViewMs[views // 2:]:
        evidence.append((2, "GE per-view durations repeat for both halves of the views"))
    if medisoTimes:
        distinct = len(set(medisoTimes[0]))
        if distinct and views % distinct == 0 and views // distinct > 1:
            evidence.append((views // distinct, f"Mediso view start times: {distinct} stops for {views} views"))
    try:
        declared = int(getattr(ds, "NumberOfDetectors", 1) or 1)
    except (TypeError, ValueError):
        declared = 1
    if declared > 1:
        evidence.append((declared, "NumberOfDetectors"))
    detectors = max([count for count, _ in evidence], default=1)
    info["detectors"] = detectors
    info["detectorsEvidence"] = "; ".join(text for _, text in evidence) if evidence else \
        "no second detector in the header; merged dual-head exports often look like this"
    info["perDetectorTimeSec"] = totalViewTime / detectors if totalViewTime else None

    # ---- wall-clock timing: start and mid-scan (decay correction)
    start = parseDicomDateTime(dt=getattr(ds, "AcquisitionDateTime", None)) or \
        parseDicomDateTime(getattr(ds, "AcquisitionDate", None) or getattr(ds, "SeriesDate", None)
                           or getattr(ds, "StudyDate", None),
                           getattr(ds, "AcquisitionTime", None) or getattr(ds, "SeriesTime", None)
                           or getattr(ds, "StudyTime", None))
    if medisoTimes:
        first, last = min(medisoTimes[0]), max(medisoTimes[1])
        info["start"] = toIso(first)
        info["mid"] = toIso(first + (last - first) / 2)
        info["midSource"] = "Mediso per-view timestamps (first view start to last view end)"
    else:
        info["start"] = toIso(start)
        if start and info["perDetectorTimeSec"]:
            info["mid"] = toIso(start + datetime.timedelta(seconds=info["perDetectorTimeSec"] / 2.0))
            info["midSource"] = (f"acquisition start + half the per-detector time "
                                 f"({detectors} detector{'s' if detectors > 1 else ''})")
        else:
            info["mid"] = info["start"]
            info["midSource"] = "acquisition start"
    if not info["start"]:
        info["warnings"].append("Acquisition start time missing - enter the imaging time manually.")

    # ---- GE's own calibration, if the export carries one
    geSensitivity = _geCameraSensitivity(ds)
    info["geSensitivity"] = geSensitivity
    if geSensitivity is not None:
        if any(float(v) for v in geSensitivity):
            info["notes"].append("GE Xeleris camera sensitivity stored in this file (per energy set, GE units): "
                                 + ", ".join(f"{float(v):.4g}" for v in geSensitivity))
        else:
            info["notes"].append("GE Xeleris camera sensitivity field present but not set (0).")

    # ---- patient
    try:
        info["weightKg"] = float(getattr(ds, "PatientWeight", 0) or 0) or None
    except (TypeError, ValueError):
        info["weightKg"] = None
    try:
        size = float(getattr(ds, "PatientSize", 0) or 0)
        info["heightCm"] = size * 100.0 if 0 < size < 3 else (size or None)
    except (TypeError, ValueError):
        info["heightCm"] = None
    info["sex"] = _text(getattr(ds, "PatientSex", ""))[:1].upper()

    info["windows"] = readNmInfo(filePath)["windows"]
    return info


def describeTiming(acq):
    """One line for logs and dialogs, e.g. '2 detectors (evidence) - 612.5 s each, total view time 1225.0 s'."""
    total = acq.get("totalViewTimeSec")
    detectors = acq.get("detectors") or 1
    if not total:
        return "total view time unknown"
    text = f"total view time {total:.1f} s ({acq.get('durationSource', '')})"
    if detectors > 1:
        text = f"{detectors} detectors, {total / detectors:.1f} s each - " + text
    else:
        text = "1 detector recorded - " + text
    return text
