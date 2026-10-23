"""Patient motion detection and correction on the SPECT projections (before reconstruction).

Every projection frame has a slot: its position within its (energy window, time slot) group, in
file order (detector 1 views, then detector 2 views). This is the order PyTomography reads them in
(parse_projection_dataset), and slot p of every window and gate is the same camera position, so a
correction is one (dx, dz) shift in pixels per slot, applied to all windows and gates.

Detection (photopeak window, gates summed):
- Axial (dz): the axial count profile of every slot (the linogram) is matched to the median profile
  by cross-correlation; an up/down movement is a step in the profile position.
  The shifts are smoothed with a median filter in acquisition time (per detector), so single noisy
  views are not shifted.
- Lateral (dx): the transaxial centre of mass of a parallel-hole projection follows a sinusoid over
  the angles, x_c = A cos(theta) + B sin(theta) + C. A sideways movement moves the views acquired
  after it onto another sinusoid: the acquisition is split in time where that happens, and the views
  are moved back onto the sinusoid of the longest segment.
Shifts below a threshold are set to zero.

No Slicer dependency; pydicom is imported when files are read or written.
"""

import numpy as np

DEFAULT_MAX_SHIFT = 8.0     # pixels searched for the axial shift
DEFAULT_SMOOTHING = 5       # views in the median filter (acquisition time, per detector)
DEFAULT_THRESHOLD = 0.5     # pixels; smaller shifts are left alone


# ---------------------------------------------------------------------------
# Frame bookkeeping
# ---------------------------------------------------------------------------
def frameSlots(nFrames, energyWindows=None, timeSlots=None):
    """Slot (0-based position within the frame's energy window / time slot group) of every frame."""
    ewv = np.ones(nFrames, dtype=int) if energyWindows is None else np.asarray(energyWindows, dtype=int)
    tsv = np.ones(nFrames, dtype=int) if timeSlots is None else np.asarray(timeSlots, dtype=int)
    slots = np.zeros(nFrames, dtype=int)
    counters = {}
    for i in range(nFrames):
        key = (int(ewv[i]), int(tsv[i]))
        slots[i] = counters.get(key, 0)
        counters[key] = slots[i] + 1
    return slots


def slotGeometry(ds, nSlots):
    """(angles in degrees, detector of each slot) as PyTomography computes them; evenly spread
    angles over 360 degrees and one detector when the header does not say."""
    try:
        detectors = np.asarray(ds.DetectorVector, dtype=int)
        uniqueDetectors = list(np.unique(detectors))
        rotation = ds.RotationInformationSequence[0]
        nAngles = int(rotation.NumberOfFramesInRotation)
        step = float(rotation.AngularStep)
        direction = str(getattr(rotation, "RotationDirection", "CW")).upper()
        angles, slotDetectors = [], []
        for detector in uniqueDetectors:
            try:
                start = float(ds.DetectorInformationSequence[detector - 1].StartAngle)
            except Exception:
                start = float(rotation.StartAngle)
            sign = 1.0 if direction in ("CC", "CCW") else -1.0
            angles.extend(start + sign * step * np.arange(nAngles))
            slotDetectors.extend([detector] * nAngles)
        if len(angles) == nSlots:
            return (np.asarray(angles) + 180.0) % 360.0, np.asarray(slotDetectors, dtype=int)
    except Exception:
        pass
    return np.arange(nSlots) * 360.0 / max(1, nSlots), np.ones(nSlots, dtype=int)


def slotStack(frames, slots, energyWindows=None, peakWindow=None):
    """[slots, rows, cols] of one energy window with the time slots (gates) summed."""
    frames = np.asarray(frames, dtype=np.float32)
    keep = np.ones(frames.shape[0], dtype=bool)
    if energyWindows is not None and peakWindow is not None:
        keep = np.asarray(energyWindows, dtype=int) == int(peakWindow)
        if not keep.any():
            keep[:] = True
    nSlots = int(np.max(slots[keep])) + 1
    stack = np.zeros((nSlots,) + frames.shape[1:], dtype=np.float32)
    for i in np.where(keep)[0]:
        stack[slots[i]] += frames[i]
    return stack


def timeIndex(slotDetectors):
    """Acquisition time step of every slot: its position among the slots of its detector
    (the detectors acquire at the same time)."""
    index = np.zeros(len(slotDetectors), dtype=int)
    counters = {}
    for i, detector in enumerate(slotDetectors):
        index[i] = counters.get(int(detector), 0)
        counters[int(detector)] = index[i] + 1
    return index


# ---------------------------------------------------------------------------
# Shifting
# ---------------------------------------------------------------------------
def shiftImage(image, dz, dx):
    """Image moved by dz rows and dx columns (sub-pixel, linear interpolation, zero outside)."""
    out = np.asarray(image, dtype=np.float64)
    for axis, shift in ((0, dz), (1, dx)):
        if abs(shift) < 1e-6:
            continue
        n = out.shape[axis]
        whole = int(np.floor(shift))
        frac = shift - whole
        moved = np.zeros_like(out)
        for offset, weight in ((whole, 1.0 - frac), (whole + 1, frac)):
            if weight == 0.0 or abs(offset) >= n:
                continue
            src = [slice(None)] * out.ndim
            dst = [slice(None)] * out.ndim
            if offset >= 0:
                src[axis], dst[axis] = slice(0, n - offset), slice(offset, n)
            else:
                src[axis], dst[axis] = slice(-offset, n), slice(0, n + offset)
            moved[tuple(dst)] += weight * out[tuple(src)]
        out = moved
    return out


def applyShifts(frames, slots, dx, dz):
    """Every frame shifted by the (dx, dz) of its slot."""
    frames = np.asarray(frames)
    out = np.empty(frames.shape, dtype=np.float64)
    for i in range(frames.shape[0]):
        out[i] = shiftImage(frames[i], dz[slots[i]], dx[slots[i]])
    return out


def correctedStack(stack, dx, dz):
    return np.stack([shiftImage(stack[s], dz[s], dx[s]) for s in range(stack.shape[0])]).astype(np.float32)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _subpixelShift(profile, reference, maxShift):
    """Shift s (pixels) with profile(i) ~ reference(i - s), by cross-correlation and a parabola
    through the peak."""
    p = profile - profile.mean()
    r = reference - reference.mean()
    n = len(p)
    lags = np.arange(-int(maxShift), int(maxShift) + 1)
    scores = np.array([np.dot(p[max(0, k):n + min(0, k)], r[max(0, -k):n - max(0, k)]) for k in lags])
    best = int(np.argmax(scores))
    shift = float(lags[best])
    if 0 < best < len(lags) - 1:
        a, b, c = scores[best - 1], scores[best], scores[best + 1]
        denominator = a - 2 * b + c
        if denominator < 0:
            shift += 0.5 * (a - c) / denominator
    return shift


def _medianFilter(values, width):
    if width <= 1 or len(values) < 3:
        return np.asarray(values, dtype=float)
    half = width // 2
    padded = np.pad(np.asarray(values, dtype=float), half, mode="edge")
    return np.array([np.median(padded[i:i + width]) for i in range(len(values))])


def _smoothInTime(values, slotDetectors, width):
    """Median filter along acquisition time, separately for every detector."""
    out = np.asarray(values, dtype=float).copy()
    for detector in np.unique(slotDetectors):
        idx = np.where(slotDetectors == detector)[0]
        out[idx] = _medianFilter(out[idx], width)
    return out


def axialShifts(stack, maxShift=DEFAULT_MAX_SHIFT):
    """Measured axial position (pixels) of every slot's count profile against the median profile."""
    profiles = stack.sum(axis=2)
    reference = np.median(profiles, axis=0)
    shifts = np.array([_subpixelShift(p, reference, maxShift) for p in profiles])
    return shifts - np.median(shifts)


def transaxialCentres(stack):
    """Transaxial centre of mass (column, pixels) of every slot, above 10 % of its profile maximum."""
    columns = np.arange(stack.shape[2], dtype=float)
    centres = np.full(stack.shape[0], np.nan)
    for s in range(stack.shape[0]):
        profile = stack[s].sum(axis=0).astype(float)
        profile = np.clip(profile - 0.1 * profile.max(), 0.0, None)  # ignore the low background
        total = profile.sum()
        if total > 0:
            centres[s] = (profile * columns).sum() / total
    return centres


def _segmentFit(centres, theta, times, breaks):
    """Least squares fit of x_c = A_k cos + B_k sin + C (A, B per time segment, C shared: the
    rotation axis does not move). Returns (sum of squared residuals, fitted values, segment of
    every slot)."""
    edges = [0] + sorted(breaks) + [int(times.max()) + 1]
    segment = np.searchsorted(edges, times, side="right") - 1
    nSeg = len(edges) - 1
    design = np.zeros((len(centres), 2 * nSeg + 1))
    for k in range(nSeg):
        mask = segment == k
        design[mask, 2 * k] = np.cos(theta[mask])
        design[mask, 2 * k + 1] = np.sin(theta[mask])
    design[:, -1] = 1.0
    use = np.isfinite(centres)
    coefficients, *_ = np.linalg.lstsq(design[use], centres[use], rcond=None)
    fitted = design @ coefficients
    sse = float(np.sum((centres[use] - fitted[use]) ** 2))
    return sse, coefficients, segment


def lateralCorrection(centres, angles, times, slotDetectors=None, minSegment=6, maxBreaks=3, sensitivity=25.0):
    """Correction dx (pixels) for every slot from a sideways movement: the acquisition is split in
    time where the centre of mass jumps to another sinusoid (binary segmentation, a split is kept
    when it lowers the residual by more than sensitivity x the noise variance), and the views of
    every segment are moved onto the sinusoid of the longest segment."""
    theta = np.radians(angles)
    times = np.asarray(times, dtype=int)
    nTimes = int(times.max()) + 1
    use = np.isfinite(centres)
    if use.sum() < 8 or nTimes < 2 * minSegment:
        return np.zeros(len(centres))
    sse, coefficients, segment = _segmentFit(centres, theta, times, [])
    # noise from the differences of neighbouring views of one detector (insensitive to the steps)
    residual = centres - _evaluate(coefficients, theta, segment)
    detectors = np.ones(len(centres), dtype=int) if slotDetectors is None else np.asarray(slotDetectors)
    diffs = []
    for detector in np.unique(detectors):
        idx = np.where(detectors == detector)[0]
        values = residual[idx[np.argsort(times[idx], kind="stable")]]
        diffs.extend(np.diff(values[np.isfinite(values)]))
    diffs = np.asarray(diffs)
    sigma2 = (np.median(np.abs(diffs)) * 1.4826) ** 2 / 2.0 + 1e-6
    breaks = []
    for _ in range(maxBreaks):
        best = None
        for b in range(minSegment, nTimes - minSegment + 1):
            if any(abs(b - other) < minSegment for other in breaks):
                continue
            candidate = _segmentFit(centres, theta, times, breaks + [b])
            if best is None or candidate[0] < best[1][0]:
                best = (b, candidate)
        if best is None or sse - best[1][0] < sensitivity * sigma2:
            break
        breaks.append(best[0])
        sse, coefficients, segment = best[1]
    if not breaks:
        return np.zeros(len(centres))
    counts = np.bincount(segment)
    reference = int(np.argmax(counts))
    A, B = coefficients[2 * reference], coefficients[2 * reference + 1]
    target = A * np.cos(theta) + B * np.sin(theta) + coefficients[-1]
    return target - _evaluate(coefficients, theta, segment)


def _evaluate(coefficients, theta, segment):
    return (coefficients[2 * segment] * np.cos(theta) + coefficients[2 * segment + 1] * np.sin(theta)
            + coefficients[-1])


def detectMotion(stack, angles, slotDetectors, maxShift=DEFAULT_MAX_SHIFT, smoothing=DEFAULT_SMOOTHING,
                 threshold=DEFAULT_THRESHOLD, axial=True, lateral=True):
    """Correction (dx, dz) in pixels for every slot: minus the smoothed measured shifts."""
    n = stack.shape[0]
    dz = np.zeros(n)
    dx = np.zeros(n)
    if axial:
        dz = -_smoothInTime(axialShifts(stack, maxShift), slotDetectors, smoothing)
    if lateral:
        dx = lateralCorrection(transaxialCentres(stack), angles, timeIndex(slotDetectors), slotDetectors)
    dz[np.abs(dz) < threshold] = 0.0
    dx[np.abs(dx) < threshold] = 0.0
    return np.round(dx, 2), np.round(dz, 2)


def describe(dx, dz):
    moved = int(np.count_nonzero((np.abs(dx) > 0) | (np.abs(dz) > 0)))
    if not moved:
        return "None"
    return (f"{moved} of {len(dx)} views shifted (max |dx| {np.max(np.abs(dx)):.1f} px, "
            f"max |dz| {np.max(np.abs(dz)):.1f} px)")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def writeCorrectedFile(sourceFile, outPath, dx, dz):
    """Copy of an NM projection file with every frame shifted by its slot's correction (all
    energy windows and gates). Counts are rounded back to the stored integer type."""
    import pydicom
    ds = pydicom.dcmread(sourceFile)
    if ds.file_meta.TransferSyntaxUID.is_compressed:
        ds.decompress()
    frames = ds.pixel_array
    if frames.ndim == 2:
        frames = frames[np.newaxis]
    n = frames.shape[0]

    def vector(name):
        value = getattr(ds, name, None)
        arr = None if value is None else np.atleast_1d(np.asarray(value, dtype=int))
        return arr if arr is not None and arr.size == n else None

    slots = frameSlots(n, vector("EnergyWindowVector"), vector("TimeSlotVector"))
    if int(slots.max()) + 1 != len(dx):
        raise ValueError(f"The motion correction has {len(dx)} views, the file {int(slots.max()) + 1}.")
    shifted = applyShifts(frames, slots, dx, dz)
    maxStorable = float(2 ** int(getattr(ds, "BitsStored", 16)) - 1)
    shifted = np.clip(np.rint(shifted), 0, maxStorable).astype(frames.dtype)
    ds.PixelData = np.ascontiguousarray(shifted).tobytes()
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(outPath, write_like_original=False)
    return outPath
