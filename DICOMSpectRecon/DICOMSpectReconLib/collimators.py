"""Collimator library for resolution recovery (collimator-detector response, PSF) in SPECT.

Every entry is a parallel-hole collimator: hole diameter, hole length and septa, the intrinsic
resolution of the camera at 140 keV, and the keys used to recognise it from the DICOM header
(manufacturer, words of the camera model, collimator names). The built-in entries come from the
collimator data base PyTomography ships (taken from the SIMIND Monte Carlo program); the user's
library is a JSON file that starts as a copy of them and can be edited in the SPECT Collimators
module.

The PSF is PyTomography's Gaussian model (as get_psfmeta_from_scanner_params): at a distance r from
the collimator face, sigma(r) = sqrt(((d / L_eff) r + d)^2 + intrinsic^2) x FWHM-to-sigma, with
L_eff = L - 2 / mu(lead, photopeak energy) and the intrinsic resolution scaled with E^(-1/2).

No Slicer dependency except libraryPath(); PyTomography is imported when used.
"""

import json
import math
import os
import re

LIBRARY_FILE_NAME = "SPECTCollimators.json"
LIBRARY_VERSION = 1
DEFAULT_INTRINSIC_MM = 3.8   # NaI(Tl) 3/8" cameras at 140 keV (FWHM)
CZT_INTRINSIC_MM = 2.5       # pixelated CZT detectors
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

FIELDS = ["code", "description", "manufacturer", "models", "names",
          "holeDiameterMm", "septaMm", "holeLengthMm", "intrinsicMm"]
NUMERIC_FIELDS = ("holeDiameterMm", "septaMm", "holeLengthMm", "intrinsicMm")

# Code prefix of the SIMIND data base -> (manufacturer, camera model words). Matching is on these words
# (case insensitive) in the DICOM Manufacturer and ManufacturerModelName.
PREFIXES = {
    "GZ": ("GE", "CZT"),
    "G8": ("GE", "870, 860, 850, 830"),
    "GT": ("GE", "Ventri"),
    "GO": ("GE", "Optima, 640, 630"),
    "GI": ("GE", "Infinia, Hawkeye, 670, Brivo"),
    "GV": ("GE", "Millennium VG, VG"),
    "GM": ("GE", "Millennium MG, MPR, MG"),
    "SY": ("Siemens", "Symbia, Intevo"),
    "SE": ("Siemens", "E.CAM, ECAM"),
    "SI": ("Siemens", ""),
    "PB": ("Philips", "BrightView"),
    "PR": ("Picker", "PRISM"),
    "MA": ("Mediso", "AnyScan"),
    "ME": ("Mediso", "Spirit, Nucline"),
}

# Collimator names as cameras write them in the DICOM header, for a data base code suffix
NAME_ALIASES = {
    "LHRS": ["LEHRS"],
    "LUHR": ["LEUHR"],
    "ELEG": ["ELEGP"],
    "LEUR": ["LEUHR"],
    "LEUHR": ["LEUR"],
    "LEAP": ["LEGP"],
    "ME": ["MELP", "MEGP"],
    "HE": ["HEGP"],
}


def _normalised(text):
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def _words(text):
    return [w.strip() for w in str(text or "").split(",") if w.strip()]


# ---------------------------------------------------------------------------
# Built-in entries
# ---------------------------------------------------------------------------
def _pytomographyDataDir():
    import pytomography
    return os.path.join(os.path.dirname(os.path.abspath(pytomography.__file__)), "data")


def builtinCollimators():
    """Parallel-hole collimators of PyTomography's collimator data base (cm there, mm here)."""
    try:
        path = os.path.join(_pytomographyDataDir(), "collim.col")
        with open(path) as f:
            lines = f.readlines()
    except Exception:
        return []
    entries, seen = [], set()
    for line in lines:
        if not line.startswith("*"):
            continue
        parts = line[1:].split()
        if len(parts) < 6:
            continue
        code = parts[0]
        try:
            diameter, septa, length = (10.0 * float(v) for v in parts[1:4])
            colType = int(float(parts[5]))
        except ValueError:
            continue
        if colType != 0 or code in seen or "-" not in code:
            continue  # fan beam, pinhole and duplicate records
        seen.add(code)
        prefix, suffix = code.split("-", 1)
        manufacturer, models = PREFIXES.get(prefix, ("", ""))
        names = [suffix] + [a for a in NAME_ALIASES.get(suffix, []) if a != suffix]
        description = " ".join(parts[13:]).strip(' "') if len(parts) > 13 else ""
        entries.append({
            "code": code, "description": description, "manufacturer": manufacturer, "models": models,
            "names": ", ".join(names), "holeDiameterMm": round(diameter, 3), "septaMm": round(septa, 3),
            "holeLengthMm": round(length, 3),
            "intrinsicMm": CZT_INTRINSIC_MM if "CZT" in models.upper() else DEFAULT_INTRINSIC_MM,
        })
    return entries


# ---------------------------------------------------------------------------
# The user's library (JSON)
# ---------------------------------------------------------------------------
def libraryPath():
    import slicer
    return os.path.join(os.path.dirname(slicer.app.slicerUserSettingsFilePath), LIBRARY_FILE_NAME)


def cleanEntry(entry):
    """An entry with every field, numbers as floats; raises ValueError for an unusable one."""
    clean = {key: entry.get(key, "") for key in FIELDS}
    clean["code"] = str(clean["code"]).strip()
    if not clean["code"]:
        raise ValueError("A collimator needs a code (name).")
    for key in NUMERIC_FIELDS:
        try:
            clean[key] = float(clean[key])
        except (TypeError, ValueError):
            raise ValueError(f"{clean['code']}: {key} is not a number.")
    if clean["holeDiameterMm"] <= 0 or clean["holeLengthMm"] <= 0:
        raise ValueError(f"{clean['code']}: hole diameter and length must be above 0.")
    if clean["intrinsicMm"] < 0 or clean["septaMm"] < 0:
        raise ValueError(f"{clean['code']}: septa and intrinsic resolution cannot be negative.")
    for key in ("description", "manufacturer", "models", "names"):
        clean[key] = str(clean[key] or "").strip()
    return clean


def loadLibrary(path=None):
    """The user's collimators, or the built-in ones while there is no library file yet."""
    path = path or libraryPath()
    if not os.path.exists(path):
        return builtinCollimators()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entries = []
    for entry in data.get("collimators", []):
        try:
            entries.append(cleanEntry(entry))
        except ValueError:
            continue
    return entries


def saveLibrary(entries, path=None):
    path = path or libraryPath()
    clean = [cleanEntry(e) for e in entries]
    codes = [e["code"] for e in clean]
    duplicates = sorted({c for c in codes if codes.count(c) > 1})
    if duplicates:
        raise ValueError("Collimator codes must be unique: " + ", ".join(duplicates))
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": LIBRARY_VERSION, "collimators": clean}, f, indent=2)
    os.replace(tmp, path)
    return path


def restoreDefaults(path=None):
    """Replace the library with the built-in entries (user entries and edits are lost)."""
    return saveLibrary(builtinCollimators(), path)


def findEntry(entries, code):
    return next((e for e in entries if e["code"] == code), None)


# ---------------------------------------------------------------------------
# Matching the DICOM header
# ---------------------------------------------------------------------------
def matchCollimator(entries, manufacturer, model, collimatorName):
    """Best library entry for a camera and the collimator name in its DICOM header, or None.
    The collimator name must match one of the entry's names; the manufacturer and camera model
    words decide between entries with the same name (an entry without model words matches any model
    of its manufacturer, after the entries whose model words match)."""
    name = _normalised(collimatorName)
    if not name:
        return None
    maker = str(manufacturer or "").upper()
    camera = str(model or "").upper()
    best, bestScore = None, -1
    for entry in entries:
        if name not in {_normalised(n) for n in _words(entry["names"])}:
            continue
        entryMaker = entry["manufacturer"].upper()
        if entryMaker and maker and entryMaker not in maker:
            continue
        models = [m.upper() for m in _words(entry["models"])]
        if models and not any(m in camera for m in models):
            continue
        score = (2 if models else 0) + (1 if entryMaker and maker else 0)
        if score > bestScore:
            best, bestScore = entry, score
    return best


# ---------------------------------------------------------------------------
# PSF
# ---------------------------------------------------------------------------
def leadMu(energyKeV):
    """Linear attenuation of lead (1/cm) at an energy, from PyTomography's NIST data."""
    from pytomography.utils.nist_data import get_mu_from_spectrum_interp
    path = os.path.join(_pytomographyDataDir(), "NIST_attenuation_data", "lead.csv")
    return float(get_mu_from_spectrum_interp(path, float(energyKeV)))


def psfParameters(entry, energyKeV):
    """(slope, intercept, intrinsic) of sigma(r) = sqrt((slope r + intercept)^2 + intrinsic^2), in cm."""
    if not energyKeV or energyKeV <= 0:
        raise ValueError("Resolution recovery needs the photopeak energy.")
    d = entry["holeDiameterMm"] / 10.0
    effectiveLength = entry["holeLengthMm"] / 10.0 - 2.0 / leadMu(energyKeV)
    if effectiveLength <= 0:
        raise ValueError(f"{entry['code']}: the holes are too short for {energyKeV:g} keV "
                         "(septal penetration) - check the collimator.")
    intrinsic = entry["intrinsicMm"] / 10.0 * (float(energyKeV) / 140.0) ** -0.5
    return d / effectiveLength / FWHM_PER_SIGMA, d / FWHM_PER_SIGMA, intrinsic / FWHM_PER_SIGMA


def systemResolutionMm(entry, distanceMm=100.0, energyKeV=140.0):
    """System resolution (FWHM, mm) at a distance from the collimator face, as datasheets give it
    (usually at 10 cm, 140 keV), to compare an entry with its datasheet."""
    slope, intercept, intrinsic = psfParameters(entry, energyKeV)
    sigma = math.sqrt((slope * distanceMm / 10.0 + intercept) ** 2 + intrinsic ** 2)
    return 10.0 * sigma * FWHM_PER_SIGMA


def psfTransform(entry, energyKeV):
    """PyTomography transform that models the collimator-detector response of an entry."""
    import numpy as np
    from pytomography.metadata.SPECT import SPECTPSFMeta
    from pytomography.transforms.SPECT import SPECTPSFTransform
    params = list(psfParameters(entry, energyKeV))
    meta = SPECTPSFMeta(sigma_fit_params=params,
                        sigma_fit=lambda r, a, b, c: np.sqrt((a * r + b) ** 2 + c ** 2),
                        min_sigmas=3, shape="gaussian")
    return SPECTPSFTransform(meta)


def describe(entry, energyKeV=None):
    text = (f"{entry['code']} (hole {entry['holeDiameterMm']:g} mm x {entry['holeLengthMm']:g} mm, "
            f"intrinsic {entry['intrinsicMm']:g} mm at 140 keV")
    if energyKeV:
        text += f", {float(energyKeV):g} keV"
    return text + ")"
