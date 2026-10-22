"""OSEM, BSREM and FBP reconstruction with PyTomography, output count scaling and post-filters.

No Slicer dependency. torch, pytomography and SimpleITK are imported when used, so
importing this module is cheap (it is imported when Slicer starts).
"""

import os
import shutil
import tempfile

import numpy as np

from .nmdata import prepareProjectionFile, readNmInfo, writeGateFile, writeSummedGatesFile


# Output count scaling modes (see module docstring)
SCALING_VIEWS = "Acquisition counts (x number of views)"
SCALING_NATIVE = "Native OSEM units (counts per view)"
SCALING_PRIMARY = "Match detected primary counts"
SCALING_MODES = [SCALING_VIEWS, SCALING_NATIVE, SCALING_PRIMARY]

# Guard rail thresholds
CONSISTENCY_TOLERANCE = 0.15   # |forward model / measured - 1| above this is flagged
SCATTER_FRACTION_WARNING = 0.6  # scatter estimate above this fraction of the photopeak is flagged
CT_COVERAGE_WARNING = 0.9      # fraction of SPECT slices that should contain tissue in the mu-map
MU_MAX_PLAUSIBLE = 0.6         # /cm; far above bone at SPECT energies

ALGORITHM_OSEM = "OSEM"
ALGORITHM_BSREM = "BSREM"
ALGORITHM_FBP = "FBP"
ALGORITHMS = [ALGORITHM_OSEM, ALGORITHM_BSREM, ALGORITHM_FBP]
ITERATIVE_ALGORITHMS = (ALGORITHM_OSEM, ALGORITHM_BSREM)

# BSREM: OSEM iterations used as the starting image (as in Q.Clear), and the relaxation sequence
# alpha(n) = 1 / (1 + n / BSREM_RELAXATION_ITERATIONS), which makes the iterations converge
BSREM_WARM_START_ITERATIONS = 1
BSREM_RELAXATION_ITERATIONS = 10.0

# FBP windows multiplied onto the ramp. Each is a function of f and the cutoff fc, both as a
# fraction of the Nyquist frequency.
FBP_WINDOWS = ["Ramp", "Shepp-Logan", "Cosine", "Hamming", "Hann", "Butterworth"]
FBP_WINDOWS_WITH_ORDER = ("Butterworth",)


class ReconstructionAborted(Exception):
    """Raised from the OSEM callback when the user presses Stop."""


def pytomographyVersion():
    try:
        from importlib.metadata import version
        return version("pytomography")
    except Exception:
        try:
            import pytomography
            return getattr(pytomography, "__version__", "unknown")
        except Exception:
            return "unknown"


# ---------------------------------------------------------------------------
# Reconstruction logic (from the original standalone app)
# ---------------------------------------------------------------------------
class ProjectionSet:
    """One reconstructable dataset: the static acquisition, the summed gates, or one gate."""

    def __init__(self, fileNM, indexPeak, log=print, countScale=1.0, label="static"):
        from pytomography.io.SPECT import dicom as spect_dicom
        self.fileNM = fileNM
        self.label = label
        self.countScale = countScale  # undoes any scaling applied when writing the file
        self.log = log
        self.object_meta, self.proj_meta = spect_dicom.get_metadata(fileNM, index_peak=indexPeak)
        allProjections = spect_dicom.get_projections(fileNM)
        self.allProjections = allProjections
        if allProjections.ndim >= 4:
            if indexPeak >= allProjections.shape[0]:
                raise ValueError(f"Photopeak window {indexPeak + 1} does not exist "
                                 f"({allProjections.shape[0]} energy windows in the data).")
            self.projections = allProjections[indexPeak]
        else:
            self.projections = allProjections

    @property
    def nViews(self):
        try:
            return int(np.asarray(self.proj_meta.angles).size)
        except Exception:
            return int(self.projections.shape[0])

    def scatterEstimate(self, indexPeak, indexLower, indexUpper, weight=None):
        """weight None: PyTomography's width-normalised estimate. weight k (dual window only):
        k x lower window counts, the convention of vendor DEW settings (e.g. GE "scatter weight")."""
        if weight is not None:
            if indexUpper is not None:
                raise ValueError("A manual scatter weight is only used with dual energy window correction.")
            return self.allProjections[indexLower] * float(weight)
        from pytomography.io.SPECT import dicom as spect_dicom
        return spect_dicom.get_energy_window_scatter_estimate_projections(
            self.fileNM, self.allProjections, indexPeak, indexLower, indexUpper)


class _OsemMonitor:
    """PyTomography callback: reports progress after every subset and aborts on request.
    Duck-typed (run/finalize) so it works across PyTomography 3.x callback signatures."""

    def __init__(self, label, nIters, nSubsets, progress=None, shouldAbort=None, done=0, total=None):
        self.label = label
        self.offset = done  # subset updates already reported (BSREM warm start)
        self.overall = total
        self.total = max(1, nIters * nSubsets)
        self.nSubsets = max(1, nSubsets)
        self.nIters = nIters
        self.progress = progress
        self.shouldAbort = shouldAbort

    def run(self, object, n_iter=0, n_subset=0, *args, **kwargs):
        if self.progress is not None:
            done = n_iter * self.nSubsets + n_subset + 1
            self.progress(self.label, self.offset + done, self.overall or self.total,
                          f"{self.label}: iteration {n_iter + 1}/{self.nIters}, "
                          f"subset {n_subset + 1}/{self.nSubsets}")
        if self.shouldAbort is not None and self.shouldAbort():
            raise ReconstructionAborted()
        return object

    def finalize(self, *args, **kwargs):
        return None


def fbpWindow(name, f, cutoff, order=10):
    """Window H(f) applied with the FBP ramp; f and cutoff as fractions of the Nyquist frequency.
    All but Butterworth are zero above the cutoff; Ramp is the plain ramp cut off there."""
    f = np.asarray(f, dtype=np.float64)
    cutoff = max(float(cutoff), 1e-6)
    x = np.minimum(f / cutoff, 1.0)
    inside = f <= cutoff
    if name == "Butterworth":
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + (f / cutoff) ** (2 * int(order)))
    if name == "Ramp":
        window = np.ones_like(f)
    elif name == "Shepp-Logan":
        window = np.sinc(x / 2.0)  # sin(pi f / 2fc) / (pi f / 2fc)
    elif name == "Cosine":
        window = np.cos(np.pi * x / 2.0)
    elif name == "Hamming":
        window = 0.54 + 0.46 * np.cos(np.pi * x)
    elif name == "Hann":
        window = 0.5 * (1.0 + np.cos(np.pi * x))
    else:
        raise ValueError(f"Unknown FBP window '{name}'.")
    return np.where(inside, window, 0.0)


def _rampResponse(n):
    """DFT of the band-limited spatial ramp (Ram-Lak, Kak & Slaney eq. 3.61) on n samples, unit
    pixel. Unlike sampling |f| directly it has the right DC value, so FBP keeps the total counts."""
    k = np.arange(n)
    k = np.minimum(k, n - k)  # circular distance
    h = np.zeros(n)
    h[0] = 0.25
    odd = (k % 2) == 1
    h[odd] = -1.0 / (np.pi * k[odd]) ** 2
    return np.real(np.fft.fft(h))


def filterProjectionsFBP(projections, window, cutoff, order, pixelR, pixelZ):
    """Ramp-filter SPECT projections [views, r, z] along r and apply the window in 2D (r and z,
    the way camera software applies a Butterworth/Hann pre-filter). cutoff is a fraction of the
    radial Nyquist frequency; frequencies are physical so the window is isotropic when the pixels
    are not square. Zero padding to at least twice the size avoids wrap-around."""
    arr = np.asarray(projections, dtype=np.float64)
    _, nr, nz = arr.shape
    padR = int(2 ** np.ceil(np.log2(max(2 * nr, 2))))
    padZ = int(2 ** np.ceil(np.log2(max(2 * nz, 2))))
    nyquist = 1.0 / (2.0 * float(pixelR))
    fr = np.fft.fftfreq(padR, d=float(pixelR)) / nyquist
    fz = np.fft.fftfreq(padZ, d=float(pixelZ)) / nyquist
    f = np.sqrt(fr[:, None] ** 2 + fz[None, :] ** 2)
    response = _rampResponse(padR)[:, None] * fbpWindow(window, f, cutoff, order)
    spectrum = np.fft.fft2(arr, s=(padR, padZ), axes=(1, 2)) * response[None, :, :]
    return np.real(np.fft.ifft2(spectrum, axes=(1, 2)))[:, :nr, :nz]


def filteredBackProjection(projections, systemMatrix, window, cutoff, order, pixelR, pixelZ):
    """FBP: (pi / number of views) x backprojection of the filtered projections. The system matrix
    must model only the geometry (no attenuation or PSF). Units are the same as OSEM's: an image
    whose forward projection reproduces the projections (counts per voxel per view)."""
    import torch
    device = projections.device if hasattr(projections, "device") else None
    data = projections.detach().cpu().numpy() if hasattr(projections, "detach") else np.asarray(projections)
    filtered = filterProjectionsFBP(data, window, cutoff, order, pixelR, pixelZ)
    filtered = torch.as_tensor(filtered, dtype=torch.float32)
    if device is not None:
        filtered = filtered.to(device)
    nViews = int(filtered.shape[0])
    return systemMatrix.backward(filtered) * (np.pi / nViews)


def changCorrection(attenuatedMatrix, geometricMatrix, maximum=20.0):
    """First order Chang attenuation correction factors for FBP: 1 / (mean attenuation factor over
    the views) = A^T 1 / A_mu^T 1, with A_mu the system matrix with attenuation. Capped, so voxels
    that no view sees unattenuated (outside the camera orbit) cannot blow up."""
    import torch
    with torch.no_grad():
        plain = geometricMatrix.compute_normalization_factor()
        attenuated = attenuatedMatrix.compute_normalization_factor()
    factor = plain / torch.clamp(attenuated, min=1e-6)
    factor = torch.where(plain > 0, factor, torch.ones_like(factor))
    return torch.clamp(factor, 1.0, float(maximum))


class SpectReconLogic:
    def __init__(self, inputDcm, log=print, ctFiles=None, indexPeak=0,
                 progress=None, shouldAbort=None):
        import torch

        self.inputDcm = inputDcm
        self.log = log
        self.ctFiles = ctFiles
        self.indexPeak = indexPeak
        self.progress = progress
        self.shouldAbort = shouldAbort
        self.attenuationMap = None
        self.tempDir = tempfile.mkdtemp(prefix="spect_gates_")
        self.gateSets = {}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.log(f"Using device: {device}  |  PyTomography {pytomographyVersion()}")

        try:
            inputDcm = prepareProjectionFile(inputDcm, self.tempDir, self.log)
        except Exception as e:
            self.log(f"(Could not check the detector radii: {e})")
        self.inputDcm = inputDcm
        nmInfo = readNmInfo(inputDcm)
        self.gateNumbers = nmInfo["gateNumbers"]
        self.nGates = nmInfo["nGates"]
        self.isGated = self.nGates > 1

        try:
            self._loadData(inputDcm, indexPeak)
        except Exception:
            self.cleanup()  # don't leave the temporary gate files behind
            raise

    def _loadData(self, inputDcm, indexPeak):
        # 1. Load Data (done once, reused for every Apply)
        self.log("Extracting metadata and projections from DICOM...")
        if self.isGated:
            self.log(f"Gated acquisition with {self.nGates} time slots.")
            staticFile = os.path.join(self.tempDir, "summed_gates.dcm")
            scale = writeSummedGatesFile(inputDcm, staticFile, self.log)
            self.staticSet = ProjectionSet(staticFile, indexPeak, self.log,
                                           countScale=1.0 / scale, label="summed gates")
        else:
            self.staticSet = ProjectionSet(inputDcm, indexPeak, self.log, label="static")

        self.object_meta = self.staticSet.object_meta
        self.proj_meta = self.staticSet.proj_meta
        self.projections = self.staticSet.projections
        self.nViews = self.staticSet.nViews
        self.logFingerprint()

        totalCounts = float(self.projections.sum().item())
        if not np.isfinite(totalCounts) or totalCounts <= 0:
            raise ValueError("The photopeak window of this acquisition contains no counts - "
                             "check the photopeak window selection.")

    def cleanup(self):
        shutil.rmtree(self.tempDir, ignore_errors=True)

    def gateSet(self, gateNumber):
        """ProjectionSet for one gate (written out and loaded on first use)."""
        if gateNumber in self.gateSets:
            return self.gateSets[gateNumber]
        path = os.path.join(self.tempDir, f"gate_{gateNumber:02d}.dcm")
        writeGateFile(self.inputDcm, path, gateNumber)
        projectionSet = ProjectionSet(path, self.indexPeak, self.log, label=f"gate {gateNumber}")
        self.gateSets[gateNumber] = projectionSet
        return projectionSet

    def logFingerprint(self):
        """Print what is actually being reconstructed, to compare with the standalone app."""
        self.log(f"File: {self.inputDcm}")
        try:
            angles = np.asarray(self.proj_meta.angles, dtype=float).ravel()
            radii = np.asarray(self.proj_meta.radii, dtype=float).ravel()
            self.log(f"Projections shape: {tuple(self.projections.shape)}, total counts: {self.projections.sum().item():.0f}")
            self.log(f"Angles: n={angles.size}, first={angles[0]:.2f}, last={angles[-1]:.2f}, "
                     f"step={(angles[1] - angles[0]) if angles.size > 1 else 0:.2f}")
            self.log(f"Radii (cm): min={radii.min():.2f}, max={radii.max():.2f}")
            self.log(f"Object shape: {tuple(self.object_meta.shape)}, voxel (cm): {tuple(self.object_meta.dr)}")
        except Exception as e:
            self.log(f"(Could not print full fingerprint: {e})")

    def getAttenuationMap(self):
        """Attenuation map resampled onto the SPECT object grid (cached).
        The gates share the static geometry, so one map serves all reconstructions."""
        if self.attenuationMap is not None:
            return self.attenuationMap
        from pytomography.io.SPECT import dicom as spect_dicom
        if not self.ctFiles:
            raise ValueError("Attenuation correction is on, but no CT series is selected.")
        self.log(f"Building attenuation map from {len(self.ctFiles)} CT slices...")
        amap = spect_dicom.get_attenuation_map_from_CT_slices(
            self.ctFiles, self.staticSet.fileNM, index_peak=self.indexPeak)
        self.log(f"  -> Attenuation map shape {tuple(amap.shape)}, "
                 f"mean mu {amap.mean().item():.4f} /cm, max mu {amap.max().item():.4f} /cm")
        self.checkAttenuationMap(amap)
        self.attenuationMap = amap
        return amap

    def checkAttenuationMap(self, amap):
        """Guard rails: the CT has to overlap the SPECT field, and mu values have to be plausible."""
        arr = amap.detach().cpu().numpy() if hasattr(amap, "detach") else np.asarray(amap)
        if not np.all(np.isfinite(arr)):
            raise ValueError("The attenuation map contains invalid values (NaN/inf) - check the CT series.")
        if arr.ndim < 3 or float(arr.max()) <= 0.01:
            raise ValueError("The CT does not overlap the SPECT field of view (the attenuation map is empty). "
                             "Pick another CT series or switch attenuation correction off.")
        # PyTomography object arrays are [x, y, z] and projections [views, r, z]: z is axial
        # on the same grid. Every axial slice that holds counts should hold tissue in the mu-map.
        sliceMax = arr.reshape(-1, arr.shape[-1]).max(axis=0)
        try:
            projections = self.staticSet.projections
            projections = projections.detach().cpu().numpy() if hasattr(projections, "detach") else np.asarray(projections)
            profile = projections.reshape(-1, projections.shape[-1]).sum(axis=0)
            active = profile > 0.01 * float(profile.max())
            if active.size != sliceMax.size or not active.any():
                active = np.ones(sliceMax.size, dtype=bool)
        except Exception:
            active = np.ones(sliceMax.size, dtype=bool)
        coverage = float(np.mean(sliceMax[active] > 0.05))
        if coverage < CT_COVERAGE_WARNING:
            self.log(f"WARNING: only {100 * coverage:.0f}% of the SPECT slices with counts have tissue in the CT. "
                     "The CT probably does not cover the whole SPECT field - those slices are "
                     "not attenuation corrected.")
        if float(arr.max()) > MU_MAX_PLAUSIBLE:
            self.log(f"WARNING: maximum mu {float(arr.max()):.2f} /cm is implausibly high "
                     "(metal, contrast or a CT that is not in HU?).")

    def reconstruct(self, iters, subsets, useAttenuationCorrection=False,
                    scatter=None, projectionSet=None, algorithm=ALGORITHM_OSEM, fbp=None, bsrem=None):
        """OSEM or FBP reconstruction in PyTomography's native units (counts per voxel per view).
        scatter: None, or (indexLower, indexUpper[, weight]) with indexUpper possibly None (dual
        window) and weight None (automatic) or a manual dual window scatter weight.
        bsrem (BSREM only): {"beta", "gamma"} of the relative difference prior; iters and subsets are
        the BSREM ones.
        fbp (FBP only): {"window", "cutoff" (cycles/cm), "order"}. FBP subtracts the scatter estimate
        from the projections and corrects attenuation with the first order Chang method.
        Returns (SimpleITK float image, stats); count scaling is applied later with scaleCounts()."""
        import torch
        import SimpleITK as sitk
        from pytomography.io.SPECT import dicom as spect_dicom
        from pytomography.projectors.SPECT import SPECTSystemMatrix
        from pytomography.likelihoods import PoissonLogLikelihood
        from pytomography.algorithms import OSEM

        projectionSet = projectionSet or self.staticSet
        base_temp_dir = None
        try:
            nViews = projectionSet.nViews
            if algorithm not in ALGORITHMS:
                raise ValueError(f"Unknown reconstruction algorithm '{algorithm}'.")
            if algorithm in ITERATIVE_ALGORITHMS:
                if subsets < 1 or subsets > nViews:
                    raise ValueError(f"{subsets} subsets requested, but there are only {nViews} projection views.")
                if iters < 1:
                    raise ValueError(f"At least one {algorithm} iteration is needed.")
                if algorithm == ALGORITHM_BSREM and (not bsrem or float(bsrem.get("beta", -1)) < 0):
                    raise ValueError("BSREM needs a prior strength (beta) of 0 or more.")
            elif not fbp or fbp.get("window") not in FBP_WINDOWS:
                raise ValueError("FBP needs a filter window (" + ", ".join(FBP_WINDOWS) + ").")

            obj2obj_transforms = []
            if useAttenuationCorrection:
                from pytomography.transforms.SPECT import SPECTAttenuationTransform
                obj2obj_transforms.append(SPECTAttenuationTransform(self.getAttenuationMap()))

            photopeakCounts = float(projectionSet.projections.sum().item())
            additive_term = None
            scatterCounts = 0.0
            if scatter is not None:
                indexLower, indexUpper, weight = (tuple(scatter) + (None,))[:3]
                if indexLower is None:
                    raise ValueError("Scatter correction needs a lower energy window.")
                method = "dual" if indexUpper is None else "triple"
                weightNote = "" if weight is None else f", weight {float(weight):g} x lower window"
                self.log(f"Estimating scatter ({method} energy window{weightNote}) for {projectionSet.label}...")
                additive_term = projectionSet.scatterEstimate(self.indexPeak, indexLower, indexUpper, weight)
                if not bool(torch.isfinite(additive_term).all()):
                    raise ValueError("The scatter estimate contains invalid values - check the energy windows.")
                scatterCounts = float(additive_term.sum().item())
                fraction = scatterCounts / max(1e-9, photopeakCounts)
                self.log(f"  -> Scatter counts: {scatterCounts:.0f} ({100.0 * fraction:.1f}% of photopeak)")
                if scatterCounts <= 0:
                    self.log("WARNING: the scatter windows hold no counts - scatter correction has no effect.")
                elif fraction > SCATTER_FRACTION_WARNING:
                    self.log(f"WARNING: estimated scatter is {100.0 * fraction:.0f}% of the photopeak - "
                             "check that the scatter windows are the right ones.")

            # 2. Build System Matrix
            self.log("Building System Matrix"
                     + (" (with attenuation correction)..." if useAttenuationCorrection else "..."))
            system_matrix = SPECTSystemMatrix(
                obj2obj_transforms=obj2obj_transforms,
                proj2proj_transforms=[],
                object_meta=projectionSet.object_meta,
                proj_meta=projectionSet.proj_meta
            )

            # 3. Reconstruction
            if algorithm == ALGORITHM_FBP:
                reconstructed_object = self._reconstructFBP(projectionSet, system_matrix, additive_term,
                                                            useAttenuationCorrection, fbp)
            elif algorithm == ALGORITHM_BSREM:
                reconstructed_object = self._reconstructBSREM(projectionSet, system_matrix, additive_term,
                                                              iters, subsets, bsrem)
            else:
                self.log(f"Running OSEM on {projectionSet.label} (Iterations: {iters}, Subsets: {subsets})...")
                if subsets > 1 and nViews % subsets != 0:
                    self.log(f"  -> Note: {nViews} views do not divide evenly into {subsets} subsets.")
                likelihood = PoissonLogLikelihood(system_matrix, projectionSet.projections,
                                                  additive_term=additive_term)
                reconstruction_algorithm = OSEM(likelihood)
                monitor = _OsemMonitor(projectionSet.label, iters, subsets, self.progress, self.shouldAbort)
                reconstructed_object = reconstruction_algorithm(n_iters=iters, n_subsets=subsets, callback=monitor)
                self.log("OSEM reconstruction complete.")

            if not bool(torch.isfinite(reconstructed_object).all()):
                raise ValueError("The reconstruction contains invalid values (NaN/inf).")
            if float(reconstructed_object.max().item()) <= 0:
                raise ValueError("The reconstruction is empty (all voxels zero).")

            # Guard rail: the fitted model should reproduce the measured counts
            consistency = None
            try:
                with torch.no_grad():
                    modelCounts = float(system_matrix.forward(reconstructed_object).sum().item()) + scatterCounts
                consistency = modelCounts / max(1e-9, photopeakCounts)
                self.log(f"  -> Data consistency (model / measured counts): {consistency:.3f}")
                if abs(consistency - 1.0) > CONSISTENCY_TOLERANCE:
                    self.log("WARNING: the reconstruction does not reproduce the measured counts well "
                             "(activity outside the field of view, too few iterations, or a geometry problem).")
            except Exception as e:
                self.log(f"  -> (Data consistency check skipped: {e})")

            # Undo any scaling applied when the projections were written (summed gates)
            countScale = float(projectionSet.countScale)
            if countScale != 1.0:
                reconstructed_object = reconstructed_object * countScale

            stats = {
                "label": projectionSet.label,
                "nViews": nViews,
                "photopeakCounts": photopeakCounts * countScale,
                "scatterCounts": scatterCounts * countScale,
                "nativeSum": float(reconstructed_object.sum().item()),
                "consistency": consistency,
            }

            # 4. Save to temporary DICOM to establish spatial formatting
            self.log("Exporting to spatial 3D DICOM format...")
            base_temp_dir = tempfile.mkdtemp(prefix="recon_base_")
            target_dicom_dir = os.path.join(base_temp_dir, "dicom_output")
            spect_dicom.save_dcm(
                save_path=target_dicom_dir,
                object=reconstructed_object,
                file_NM=projectionSet.fileNM,
                recon_name="temp_3d_recon"
            )

            # 5. Read 3D DICOM back with SimpleITK (correct geometry)
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(target_dicom_dir)
            if not dicom_names:
                raise ValueError("Failed to generate intermediate DICOM slices.")
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            return sitk.Cast(image, sitk.sitkFloat32), stats
        finally:
            if base_temp_dir and os.path.exists(base_temp_dir):
                shutil.rmtree(base_temp_dir, ignore_errors=True)


    def _reconstructBSREM(self, projectionSet, system_matrix, additive_term, iters, subsets, bsrem):
        """BSREM with a relative difference prior, started from OSEM. The decreasing relaxation makes it
        converge, so beta (not the iteration count) sets the noise; no post-filter is needed."""
        from pytomography.likelihoods import PoissonLogLikelihood
        from pytomography.algorithms import BSREM, OSEM
        from pytomography.priors import RelativeDifferencePrior

        beta, gamma = float(bsrem["beta"]), float(bsrem.get("gamma", 2.0))
        nViews = projectionSet.nViews
        self.log(f"Running BSREM on {projectionSet.label} (Iterations: {iters}, Subsets: {subsets}, "
                 f"relative difference prior beta {beta:g}, gamma {gamma:g})...")
        if subsets > 1 and nViews % subsets != 0:
            self.log(f"  -> Note: {nViews} views do not divide evenly into {subsets} subsets.")
        likelihood = PoissonLogLikelihood(system_matrix, projectionSet.projections, additive_term=additive_term)
        warm = BSREM_WARM_START_ITERATIONS
        total = (warm + iters) * subsets
        self.log(f"  -> Starting image: {warm} OSEM iteration(s) x {subsets} subsets")
        monitor = _OsemMonitor(f"{projectionSet.label} (OSEM start)", warm, subsets,
                               self.progress, self.shouldAbort, total=total)
        initial = OSEM(likelihood)(n_iters=warm, n_subsets=subsets, callback=monitor)
        prior = RelativeDifferencePrior(beta=beta, gamma=gamma) if beta > 0 else None
        algorithm = BSREM(likelihood, object_initial=initial.clone(), prior=prior,
                          relaxation_sequence=lambda n: 1.0 / (1.0 + n / BSREM_RELAXATION_ITERATIONS))
        monitor = _OsemMonitor(projectionSet.label, iters, subsets, self.progress, self.shouldAbort,
                               done=warm * subsets, total=total)
        image = algorithm(n_iters=iters, n_subsets=subsets, callback=monitor)
        self.log("BSREM reconstruction complete.")
        return image

    def _reconstructFBP(self, projectionSet, system_matrix, additive_term, useAttenuationCorrection, fbp):
        """Filtered backprojection of one projection set; negative voxels are set to 0 (the image is
        stored as unsigned DICOM, and negative counts have no meaning for the display or the VOIs)."""
        import torch
        from pytomography.projectors.SPECT import SPECTSystemMatrix

        window, cutoffCm, order = fbp["window"], float(fbp.get("cutoff", 1.0)), int(fbp.get("order", 10))
        pixelR, pixelZ = (float(d) for d in projectionSet.proj_meta.dr[:2])  # cm
        fraction, clipped = cutoffFraction(cutoffCm, 10.0 * pixelR)
        text = f"{window} window, cutoff {cutoffCm:.2f} cycles/cm = {fraction:.2f} x Nyquist"
        if clipped:
            text += " (limited to the Nyquist frequency)"
        if window in FBP_WINDOWS_WITH_ORDER:
            text += f", order {order}"
        self.log(f"Running FBP on {projectionSet.label} ({text})...")
        if self.progress is not None:
            self.progress(projectionSet.label, 0, 1, f"{projectionSet.label}: filtered backprojection")
        if self.shouldAbort is not None and self.shouldAbort():
            raise ReconstructionAborted()

        projections = projectionSet.projections
        if additive_term is not None:
            # FBP has no noise model: subtract the scatter estimate from the data
            projections = torch.clamp(projections - additive_term, min=0)
        geometric = system_matrix
        if useAttenuationCorrection:
            geometric = SPECTSystemMatrix(obj2obj_transforms=[], proj2proj_transforms=[],
                                          object_meta=projectionSet.object_meta,
                                          proj_meta=projectionSet.proj_meta)
        with torch.no_grad():
            image = filteredBackProjection(projections, geometric, window, fraction, order, pixelR, pixelZ)
            if useAttenuationCorrection:
                self.log("  -> Chang (first order) attenuation correction from the CT attenuation map")
                image = image * changCorrection(system_matrix, geometric)
        negative = float(-image[image < 0].sum().item())
        positive = float(image[image > 0].sum().item())
        image = torch.clamp(image, min=0)
        if positive > 0:
            self.log(f"  -> Negative FBP values set to 0 ({100.0 * negative / positive:.1f}% of the positive counts)")
        if self.progress is not None:
            self.progress(projectionSet.label, 1, 1, f"{projectionSet.label}: filtered backprojection done")
        self.log("FBP reconstruction complete.")
        return image


def countScalingFactor(stats, mode):
    """Factor that turns a native OSEM image into the requested count scale, and a note."""
    if mode == SCALING_NATIVE:
        return 1.0, "native OSEM units (counts per voxel per view)"
    if mode == SCALING_PRIMARY:
        primary = stats["photopeakCounts"] - stats["scatterCounts"]
        if stats["nativeSum"] <= 0 or primary <= 0:
            raise ValueError("Cannot match primary counts: the reconstruction or the "
                             "scatter-corrected photopeak holds no counts.")
        return primary / stats["nativeSum"], f"total = detected primary counts ({primary:.0f})"
    return float(stats["nViews"]), f"x {stats['nViews']} views (counts over the whole acquisition)"


def scaleCounts(image, stats, mode):
    factor, note = countScalingFactor(stats, mode)
    return (image * float(factor) if factor != 1.0 else image), factor, note


FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))  # 2.355


def _frequencyFilter3D(image, response):
    """Apply a radial low-pass response H(f) in 3D, f as a fraction of the in-plane Nyquist
    frequency. Frequencies are physical (cycles/mm), so the filter is isotropic even when the
    slice spacing differs from the pixel size. H(0) = 1, so total counts are preserved
    (apart from clipping ringing negatives to 0)."""
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(image).astype(np.float64)  # [z, y, x]
    nz, ny, nx = arr.shape
    sx, sy, sz = image.GetSpacing()
    nyquist = 1.0 / (2.0 * sx)  # cycles/mm
    fz = np.fft.fftfreq(nz, d=sz) / nyquist
    fy = np.fft.fftfreq(ny, d=sy) / nyquist
    fx = np.fft.rfftfreq(nx, d=sx) / nyquist
    f = np.sqrt(fz[:, None, None] ** 2 + fy[None, :, None] ** 2 + fx[None, None, :] ** 2)
    out = np.fft.irfftn(np.fft.rfftn(arr) * response(f), s=arr.shape, axes=(0, 1, 2))
    out = np.clip(out, 0, None).astype(np.float32)
    result = sitk.GetImageFromArray(out)
    result.CopyInformation(image)
    return result


def butterworth3D(image, cutoff, order):
    """3D Butterworth low-pass: H(f) = 1 / (1 + (f/fc)^(2n))."""
    def response(f):
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + (f / cutoff) ** (2 * order))
    return _frequencyFilter3D(image, response)


def hann3D(image, cutoff):
    """3D Hann low-pass: H(f) = 0.5 x (1 + cos(pi f/fc)) below the cutoff, 0 above it."""
    def response(f):
        return np.where(f < cutoff, 0.5 * (1.0 + np.cos(np.pi * np.minimum(f / cutoff, 1.0))), 0.0)
    return _frequencyFilter3D(image, response)


def nyquistCyclesPerCm(pixelMm):
    """Nyquist frequency of a pixel size: 1 / (2 x pixel size), in cycles/cm."""
    return 10.0 / (2.0 * float(pixelMm))


def cutoffFraction(cutoffCm, pixelMm):
    """(cutoff as a fraction of the Nyquist frequency, clipped to 1, True when it was clipped) of a
    cutoff given in cycles/cm."""
    fraction = float(cutoffCm) / nyquistCyclesPerCm(pixelMm)
    return min(fraction, 1.0), fraction > 1.0


def applyFilter(image, params, log):
    """Post-filter. Butterworth and Hann cutoffs are in cycles/cm, so the same setting smooths the
    same on every pixel size; they are turned into a Nyquist fraction of this image's pixels."""
    import SimpleITK as sitk
    name = params["filter"]
    if name == "Gaussian":
        sigma = params["fwhm"] / FWHM_PER_SIGMA
        log(f"Applying Gaussian Filter (FWHM: {params['fwhm']:g} mm, sigma {sigma:.2f} mm)...")
        return sitk.SmoothingRecursiveGaussian(image, sigma)
    if name in ("Butterworth", "Hann"):
        pixel = image.GetSpacing()[0]
        fraction, clipped = cutoffFraction(params["cutoff"], pixel)
        text = (f"cutoff: {params['cutoff']:.2f} cycles/cm = {fraction:.2f} x Nyquist "
                f"(Nyquist {nyquistCyclesPerCm(pixel):.2f} cycles/cm at {pixel:.2f} mm pixels)")
        if clipped:
            text += ", limited to the Nyquist frequency"
        if name == "Butterworth":
            log(f"Applying Butterworth Filter ({text}, order: {params['order']})...")
            return butterworth3D(image, fraction, params["order"])
        log(f"Applying Hann Filter ({text})...")
        return hann3D(image, fraction)
    log("No post-filter applied.")
    return image


def scatterDescription(params):
    lower, upper = params.get("indexLower"), params.get("indexUpper")
    if lower is None and upper is None:
        return "None"
    if lower is not None and upper is not None:
        return f"Triple energy window (lower {lower + 1}, upper {upper + 1})"
    if lower is not None:
        weight = params.get("scatterWeight")
        weightText = "auto" if weight is None else f"{float(weight):g}"
        return f"Dual energy window (lower {lower + 1}, weight {weightText})"
    return "None (upper-only window is not supported)"


def autoScatterWeight(peakWidth, lowerWidth):
    """Scatter weight k (scatter = k x lower window counts) that PyTomography's automatic dual
    window estimate amounts to: the triple window formula with an empty upper window,
    0.5 x photopeak width / lower window width."""
    if not peakWidth or not lowerWidth or lowerWidth <= 0:
        return None
    return 0.5 * float(peakWidth) / float(lowerWidth)


def bsremDescription(bsrem):
    return f"relative difference prior beta={float(bsrem['beta']):g}, gamma={float(bsrem['gamma']):g}"


def fbpDescription(fbp):
    text = f"{fbp['window']} cutoff={float(fbp['cutoff']):.2f} cycles/cm"
    if fbp["window"] in FBP_WINDOWS_WITH_ORDER:
        text += f", order={int(fbp['order'])}"
    return text


def filterDescription(params):
    if params["filter"] == "Gaussian":
        return f"Gaussian FWHM={params['fwhm']:g} mm"
    if params["filter"] == "Butterworth":
        return f"Butterworth cutoff={params['cutoff']:.2f} cycles/cm, order={params['order']}"
    if params["filter"] == "Hann":
        return f"Hann cutoff={params['cutoff']:.2f} cycles/cm"
    return "None"
