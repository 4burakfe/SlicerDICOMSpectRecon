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

Written for this extension, on the ideas of: R.L. Eisner et al., Use of cross-correlation function to
detect patient motion during SPECT imaging, J Nucl Med 1987;28:97-101 (axial); W.J. Geckle et al.,
Correction for patient and organ movement in SPECT, J Nucl Med 1988;29:441-450 (tracking the centre
of the activity over the projections and shifting them); see J.A. Cooper et al., J Nucl Med
1993;34:1341-1348 for a comparison of detection methods.

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
    """(angles in degrees, detector of each slot, clockwise) as PyTomography computes the angles;
    evenly spread angles over 360 degrees and one detector when the header does not say."""
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
            return ((np.asarray(angles) + 180.0) % 360.0, np.asarray(slotDetectors, dtype=int),
                    direction not in ("CC", "CCW"))
    except Exception:
        pass
    return np.arange(nSlots) * 360.0 / max(1, nSlots), np.ones(nSlots, dtype=int), False


def displayOrder(angles, clockwise=True):
    """Slots in the order the camera sweeps the angles (for sinograms and linograms): along the
    rotation direction, starting after the largest angular gap (a 180 degree orbit) or, for a full
    circle, at the first acquired view (slot 0), so the first view is at the top and the curves run
    on without a jump."""
    angles = np.asarray(angles, dtype=float) % 360.0
    n = len(angles)
    if n < 2:
        return np.arange(n)
    key = (-angles if clockwise else angles) % 360.0  # grows along the rotation
    order = np.argsort(key, kind="stable")
    sortedKeys = key[order]
    gaps = np.diff(np.r_[sortedKeys, sortedKeys[0] + 360.0])
    largest = int(np.argmax(gaps))
    if gaps[largest] > 1.5 * np.median(gaps):
        start = (largest + 1) % n
    else:
        start = int(np.where(order == 0)[0][0])
    return np.roll(order, -start)


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


# ---------------------------------------------------------------------------
# Fast 2D reconstruction of the axial slice at one projection row, for the live preview
# ---------------------------------------------------------------------------
SLICE_MAX_BINS = 128
SLICE_MAX_HALF_SLAB = 4       # rows on each side reconstructed for the axial part of the filters


class SliceReconstructor:
    """2D reconstruction of single projection rows (1 pixel thick axial slices) with a parallel-hole
    line-integral projector (bilinear, matching PyTomography's orientation): OSEM, BSREM with a
    relative difference prior, or FBP. No attenuation, scatter or resolution recovery, so it is
    quick enough to follow the motion correction live. Wide projections are binned to at most
    SLICE_MAX_BINS pixels."""

    def __init__(self, angles, nBins, subsets=1):
        self.binning = int(np.ceil(nBins / SLICE_MAX_BINS))
        n = int(np.ceil(nBins / self.binning))
        self.n = n
        nAngles = len(angles)
        subsets = max(1, min(int(subsets), nAngles))
        groups = [np.arange(k, nAngles, subsets) for k in range(subsets)]
        # angles reordered so every subset is one contiguous block (slicing then copies nothing)
        self.perm = np.concatenate(groups)
        ends = np.cumsum([len(g) for g in groups])
        self.blocks = list(zip(np.r_[0, ends[:-1]], ends))
        theta = np.deg2rad(np.asarray(angles, dtype=np.float64)[self.perm])
        centre = (n - 1) / 2.0
        u = np.arange(n) - centre
        t = np.arange(n) - centre
        cos, sin = np.cos(theta)[:, None, None], np.sin(theta)[:, None, None]
        x = u[None, :, None] * cos - t[None, None, :] * sin + centre
        y = centre - (u[None, :, None] * sin + t[None, None, :] * cos)
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
        x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
        indices, weights = [], []
        for dy, dx, w in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx),
                          (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
            yy, xx = y0 + dy, x0 + dx
            inside = (yy >= 0) & (yy < n) & (xx >= 0) & (xx < n)
            indices.append(np.where(inside, yy * n + xx, 0).astype(np.int32))
            weights.append(np.where(inside, w, 0.0).astype(np.float32))
        self.index = np.stack(indices, axis=-1)    # [angle (subset order), bin, depth, 4]
        self.weight = np.stack(weights, axis=-1)
        self.sensitivity = [np.maximum(self._back(np.ones((e - b, n), np.float32), b, e), 1e-6)
                            for b, e in self.blocks]
        self.sensitivityAll = np.maximum(sum(self.sensitivity), 1e-6)
        yy, xx = np.mgrid[0:n, 0:n] - centre
        self.fov = (xx ** 2 + yy ** 2 <= (n / 2.0) ** 2).astype(np.float32)

    def _forward(self, image, begin, end):
        return (image.ravel()[self.index[begin:end]] * self.weight[begin:end]).sum(axis=(2, 3))

    def _back(self, projection, begin, end):
        weights = self.weight[begin:end] * projection[:, :, None, None]
        out = np.bincount(self.index[begin:end].ravel(), weights=weights.ravel(), minlength=self.n * self.n)
        return out.reshape(self.n, self.n).astype(np.float32)

    def binned(self, sinogram):
        """[angle, bin] in the caller's angle order -> binned, in subset order."""
        sinogram = np.asarray(sinogram, dtype=np.float32)
        if self.binning > 1:
            pad = self.n * self.binning - sinogram.shape[1]
            sinogram = np.pad(sinogram, ((0, 0), (0, pad)))
            sinogram = sinogram.reshape(sinogram.shape[0], self.n, self.binning).sum(axis=2)
        return sinogram[self.perm]

    def osem(self, sinogram, iterations, image=None):
        data = self.binned(sinogram)
        image = self.fov.copy() if image is None else image.copy()
        for _ in range(int(iterations)):
            for (begin, end), sensitivity in zip(self.blocks, self.sensitivity):
                estimate = np.maximum(self._forward(image, begin, end), 1e-6)
                image *= self._back(data[begin:end] / estimate, begin, end) / sensitivity
                image *= self.fov
        return image

    def bsrem(self, sinogram, iterations, beta, gamma, warmStart=1, relaxationIterations=10.0):
        """BSREM as PyTomography runs it: an OSEM starting image, then per subset
        x += alpha(n) x / (w H^T 1) (gradient of the subset likelihood - w beta grad R), with
        alpha(n) = 1 / (1 + n / relaxationIterations) and w the subset's share of the views."""
        data = self.binned(sinogram)
        image = self.osem(sinogram, warmStart)
        nAngles = data.shape[0]
        for iteration in range(int(iterations)):
            alpha = 1.0 / (1.0 + iteration / relaxationIterations)
            for (begin, end), sensitivity in zip(self.blocks, self.sensitivity):
                share = (end - begin) / nAngles
                estimate = np.maximum(self._forward(image, begin, end), 1e-6)
                gradient = self._back(data[begin:end] / estimate, begin, end) - sensitivity
                if beta > 0:
                    gradient -= share * beta * rdpGradient(image, gamma)
                image = np.maximum(image + alpha * image / (share * self.sensitivityAll) * gradient, 0.0)
                image *= self.fov
        return image.astype(np.float32)

    def backprojectAll(self, filtered):
        """(pi / views) x backprojection of filtered [angle, bin] (subset order) data, for FBP."""
        return self._back(np.asarray(filtered, dtype=np.float32), 0, filtered.shape[0]) \
            * (np.pi / filtered.shape[0]) * self.fov


def rdpGradient(image, gamma):
    """Gradient of the relative difference prior over the 8 in-plane neighbours (weights 1/distance):
    sum_k w (x_j - x_k)(gamma |x_j - x_k| + x_j + 3 x_k) / (x_j + x_k + gamma |x_j - x_k|)^2."""
    padded = np.pad(image, 1, mode="edge")
    n0, n1 = image.shape
    out = np.zeros_like(image, dtype=np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbour = padded[1 + dy:1 + dy + n0, 1 + dx:1 + dx + n1]
            diff = image - neighbour
            denominator = image + neighbour + gamma * np.abs(diff) + 1e-9
            out += (diff * (gamma * np.abs(diff) + image + 3 * neighbour) / denominator ** 2) / np.hypot(dy, dx)
    return out.astype(np.float32)


def sliceHalfSlab(params, pixelMm):
    """Rows on each side of the line needed for the axial extent of the filters."""
    pixel = float(pixelMm or 4.0)
    half = 0
    name = params.get("filter")
    if name == "Gaussian":
        half = int(np.ceil(float(params.get("fwhm", 0.0)) / pixel))
    elif name in ("Butterworth", "Hann"):
        half = 3
    if params.get("algorithm") == "FBP" and params.get("fbp", {}).get("window") not in (None, "Ramp"):
        half = max(half, 3)
    return min(half, SLICE_MAX_HALF_SLAB)


def reconstructSliceAt(recon, stack, row, params, pixelMm, log=lambda message: None):
    """Axial slice at projection row `row` of stack [view, row, bin] with the main window's
    algorithm, iterations, subsets, prior or FBP window and post-filter. Neighbouring rows are
    reconstructed too, as far as the filters reach along the axis, and the filters run in 3D."""
    from .reconstruction import (ALGORITHM_BSREM, ALGORITHM_FBP, BSREM_RELAXATION_ITERATIONS,
                                 BSREM_WARM_START_ITERATIONS, applyFilter, cutoffFraction,
                                 filterProjectionsFBP)
    pixel = float(pixelMm or 4.0)
    nRows = stack.shape[1]
    half = sliceHalfSlab(params, pixel)
    rows = list(range(max(0, row - half), min(nRows, row + half + 1)))
    algorithm = params.get("algorithm")
    if algorithm == ALGORITHM_FBP:
        fbp = params["fbp"]
        pixelR, pixelZ = pixel * recon.binning / 10.0, pixel / 10.0   # cm
        fraction, _ = cutoffFraction(float(fbp.get("cutoff", 1.0)), 10.0 * pixelR)
        data = np.stack([recon.binned(stack[:, r, :]) for r in rows], axis=2)   # [angle, bin, row]
        filtered = filterProjectionsFBP(data, fbp["window"], fraction, int(fbp.get("order", 10)), pixelR, pixelZ)
        slab = [np.maximum(recon.backprojectAll(filtered[:, :, k]), 0.0) for k in range(len(rows))]
    elif algorithm == ALGORITHM_BSREM:
        bsrem = params["bsrem"]
        slab = [recon.bsrem(stack[:, r, :], bsrem["iters"], float(bsrem["beta"]), float(bsrem.get("gamma", 2.0)),
                            BSREM_WARM_START_ITERATIONS, BSREM_RELAXATION_ITERATIONS) for r in rows]
    else:
        slab = [recon.osem(stack[:, r, :], params["iters"]) for r in rows]
    volume = np.stack(slab).astype(np.float32)   # [row, y, x]
    if params.get("filter") in ("Gaussian", "Butterworth", "Hann"):
        import SimpleITK as sitk
        image = sitk.GetImageFromArray(volume)
        image.SetSpacing((pixel * recon.binning, pixel * recon.binning, pixel))
        volume = sitk.GetArrayFromImage(applyFilter(image, params, log))
    return volume[rows.index(row)]
