"""Reconstruction on GPUs with little memory: the projector is run on axial slabs, its caches kept on the host.

A 256 x 256 SPECT reconstruction with PyTomography's SPECT system matrix needs about 4-5 GB of GPU memory (the
object is padded to 363 x 363 x 256 for the rotations, and each view's projection holds several such objects at once,
next to the cached rotation transposes and one H_m^T 1 image per subset). On a 2 GB laptop GPU this does not fail on
Windows: the driver spills into system memory and the reconstruction slows to a crawl.

The parallel-hole projector works plane by plane in z: the rotation is about z and attenuation is summed along x, so
every z plane of the projections depends on the same z plane of the object only. The collimator-detector response
(resolution recovery) blurs along z as well, but only within its kernel, so a slab of planes plus a margin of half the
largest z kernel on each side gives that slab's planes exactly. SlabSPECTSystemMatrix therefore projects and back
projects one slab at a time with ordinary PyTomography system matrices of the slab's size, and keeps only the margin
free part of each. The images are the same as those of the whole-object projector, to float rounding.

It also keeps on the host (RAM) what the reconstruction would otherwise keep on the GPU for its whole run: the H_m^T 1
image of every subset (moved to the GPU when its subset is used) and, optionally, the sparse transposed rotation of
every view that back projection uses (moved to the GPU for each view).

planMemory() chooses, from the memory free on the GPU, between the whole-object projector, slabs, and the CPU.
"""

import numpy as np

MEMORY_AUTO = "Automatic"
MEMORY_GPU = "GPU, full speed"
MEMORY_LOW = "GPU, low memory"
MEMORY_CPU = "CPU"
MEMORY_MODES = [MEMORY_AUTO, MEMORY_GPU, MEMORY_LOW, MEMORY_CPU]

# GPU memory left free for the display, other programs and estimate errors, and the share of the free memory planned
GPU_RESERVE_BYTES = 0.15e9
GPU_FREE_FRACTION = 0.95
# PyTorch's CUDA cache above the arrays (PyTomography 4 estimates 0.05 GB + 35 % of all of them; arrays held for the
# whole run are allocated once and fragment the cache much less, so they get 10 %)
CUDA_CACHE_BYTES = 0.05e9
CUDA_CACHE_FRACTION = 0.35
CUDA_CACHE_FRACTION_HELD = 0.10
# Fewest planes in a slab (PyTomography's rotation squeezes a one-plane object to 2D)
MIN_SLAB_PLANES = 4
# Slab counts tried, fewest (fastest) first
SLAB_COUNTS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
# Below this many planes per slab the slabs cost more in launches than the CPU would; use the CPU instead
CPU_BELOW_SLAB_PLANES = 4
# Most work the slabs may take, as a multiple of the whole object's (each slab also projects its PSF margin planes:
# at 256^3 with resolution recovery the margin is ~18 planes a side, so 64 slabs would do ~10 times the work for
# hardly less memory than 16)
MAX_SLAB_WORK = 4.0


def slabWork(nz, slabs, halo):
    """Planes the slabs project, as a multiple of nz."""
    return sum(e1 - e0 for _, _, e0, e1 in slabRanges(nz, slabs, halo)) / float(nz)


class MemoryPlan:
    """How a reconstruction runs: device ('cuda', 'cpu', ...), slabs (0 = whole object), whether the H_m^T 1 images and
    the rotation transposes stay on the host, the estimated GPU bytes and a note for the log."""

    def __init__(self, device, slabs=0, hostNorms=False, hostTransposes=False, gpuBytes=None, freeBytes=None, note=""):
        self.device = device
        self.slabs = int(slabs)
        self.hostNorms = bool(hostNorms)
        self.hostTransposes = bool(hostTransposes)
        self.gpuBytes = gpuBytes
        self.freeBytes = freeBytes
        self.note = note
        self.work = None

    @property
    def lowMemory(self):
        return self.slabs > 0

    def describe(self):
        if self.device == "cpu":
            text = f"CPU, {self.slabs} axial slabs" if self.slabs else "CPU"
        elif self.slabs:
            where = "host" if self.hostTransposes else "GPU"
            text = (f"GPU, low memory: {self.slabs} axial slabs, H_m^T 1 images on the host, "
                    f"rotation transposes on the {where}")
            if self.work and self.work > 1.05:
                text += f", {self.work:.1f}x the projections of the whole object (PSF margins)"
        else:
            text = f"GPU ({self.device}), whole object"
        if self.gpuBytes is not None and self.freeBytes is not None:
            text += f" (estimated {self.gpuBytes / 1e9:.2f} GB of {self.freeBytes / 1e9:.2f} GB free)"
        elif self.gpuBytes is not None:
            text += f" (estimated {self.gpuBytes / 1e9:.2f} GB)"
        if self.note:
            text += f" - {self.note}"
        return text


# ---------------------------------------------------------------------------
# Memory estimate and plan
# ---------------------------------------------------------------------------
def padSize(width):
    """PyTomography's compute_pad_size: the padding on each side that keeps a rotated object in its grid."""
    return int(np.ceil((np.sqrt(2) * width - width) / 2))


def transposeBytes(H, W):
    """PyTomography's rotation_transpose_bytes for float32: at most four weights and int32 indices per pixel."""
    return 4 * H * W * (4 + 4) + 4 * (H * W + 1)


def slabRanges(nz, slabs, halo):
    """[(z0, z1, e0, e1)]: slab planes z0..z1-1 of nz, computed on planes e0..e1-1 (the slab and halo planes on each
    side, inside the object). Slabs are as even as possible and at least MIN_SLAB_PLANES planes."""
    slabs = max(1, min(int(slabs), nz // MIN_SLAB_PLANES if nz >= MIN_SLAB_PLANES else 1))
    edges = np.linspace(0, nz, slabs + 1).round().astype(int)
    ranges = []
    for z0, z1 in zip(edges[:-1], edges[1:]):
        if z1 <= z0:
            continue
        e0, e1 = max(0, int(z0) - halo), min(nz, int(z1) + halo)
        if e1 - e0 < 2:  # the rotation needs two planes
            e0, e1 = max(0, e1 - 2), max(e1, min(nz, e0 + 2))
        ranges.append((int(z0), int(z1), int(e0), int(e1)))
    return ranges


def estimateGpuBytes(objectShape, nViews, projShape, nSubsets, attenuation, psf, halo=0, slabs=0,
                     hostNorms=False, hostTransposes=False, bsrem=False):
    """GPU memory (bytes) a reconstruction needs on top of what is already there (the CUDA context, the projections
    and the attenuation map are loaded before it starts), after PyTomography 4's own estimate of the SPECT system
    matrix: the image and the images of an update, H_m^T 1 of every subset, the cached rotation grids and transposes,
    a subset's projections and padded objects, and one view's rotated object with the transforms' temporaries (six
    padded objects for attenuation without a cache, three for a Gaussian PSF); then PyTorch's cache above that."""
    Lx, Ly, Lz = (int(v) for v in objectShape)
    p = padSize(Ly)
    Px, Py = Lx + 2 * p, Ly + 2 * p
    image = 4.0 * Lx * Ly * Lz
    n = max(1, int(nSubsets))
    views = -(-int(nViews) // n)
    Lr, Lpz = int(projShape[-2]), int(projShape[-1])
    viewPadded = 4.0 * (Lr + 2 * padSize(Lr)) * Lpz
    if slabs:
        planes = max(e1 - e0 for _, _, e0, e1 in slabRanges(Lz, slabs, halo))
    else:
        planes = Lz
    padded = 4.0 * Px * Py * planes
    held = image + 3 * image                                    # the image and the images of an update
    held += 2 * image if hostNorms else n * image                 # H_m^T 1 (on the host: one at a time, and BSREM's sum)
    if bsrem:
        held += 2 * image                                         # prior gradient and the relaxation step
    grids = int(nViews) * Px * Py * 2 * 4                         # the attenuation map is rotated on the same grids
    held += grids
    if not hostTransposes:
        held += int(nViews) * transposeBytes(Px, Py)
    else:
        held += transposeBytes(Px, Py)                            # the one in use
    subset = max(5 * views * viewPadded, int(nViews) * 4.0 * Lr * Lpz + 2 * views * viewPadded) + 3 * padded
    temporaries = max(6 * padded if attenuation else 0, 3 * padded if psf else 0, 2 * padded)
    chunk = padded + max(temporaries, padded)
    transient = subset + chunk
    return (held * (1 + CUDA_CACHE_FRACTION_HELD) + transient * (1 + CUDA_CACHE_FRACTION) + CUDA_CACHE_BYTES)


def freeGpuBytes(device=None):
    """(free, total) bytes on the CUDA device, or (None, None) when there is no CUDA GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        free, total = torch.cuda.mem_get_info(device)
        return float(free), float(total)
    except Exception:
        return None, None


def planMemory(mode, objectShape, nViews, projShape, nSubsets, attenuation=False, psf=False, halo=0, bsrem=False,
               gpuDevice=None, freeBytes=None):
    plan = _planMemory(mode, objectShape, nViews, projShape, nSubsets, attenuation, psf, halo, bsrem, gpuDevice,
                       freeBytes)
    if plan.slabs:
        plan.work = slabWork(int(objectShape[2]), plan.slabs, halo)
    return plan


def _planMemory(mode, objectShape, nViews, projShape, nSubsets, attenuation=False, psf=False, halo=0, bsrem=False,
                gpuDevice=None, freeBytes=None):
    """MemoryPlan for the memory mode (MEMORY_MODES). Automatic: the whole object on the GPU when it fits in the memory
    free now, else the fewest slabs that fit (rotation transposes on the GPU if they still fit, else on the host), else
    the CPU. 'GPU, low memory' always uses slabs (as many as needed, at least 4). freeBytes overrides the free memory
    (for tests)."""
    import torch
    gpuDevice = gpuDevice or ("cuda" if torch.cuda.is_available() else
                              "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                              else "cpu")
    if mode == MEMORY_LOW and gpuDevice == "cpu":
        return MemoryPlan("cpu", slabs=4, note="no GPU: slabs on the CPU, to use less RAM")
    if mode == MEMORY_CPU or gpuDevice == "cpu":
        note = "" if mode == MEMORY_CPU else "no GPU"
        return MemoryPlan("cpu", note=note)
    args = dict(objectShape=objectShape, nViews=nViews, projShape=projShape, nSubsets=nSubsets,
                attenuation=attenuation, psf=psf, halo=halo, bsrem=bsrem)
    whole = estimateGpuBytes(**args)
    if torch.device(gpuDevice).type != "cuda":
        # Apple's MPS (unified memory) has no free-memory query: whole object unless low memory is asked for
        if mode == MEMORY_LOW:
            return MemoryPlan(gpuDevice, slabs=4, hostNorms=True, hostTransposes=True)
        return MemoryPlan(gpuDevice, gpuBytes=whole)
    if freeBytes is None:
        freeBytes, _ = freeGpuBytes(gpuDevice)
    budget = None if freeBytes is None else max(0.0, GPU_FREE_FRACTION * freeBytes - GPU_RESERVE_BYTES)
    if mode == MEMORY_GPU:
        note = ""
        if budget is not None and whole > budget:
            note = "WARNING: more than the GPU has free, expect it to be very slow; Automatic would use slabs"
        return MemoryPlan(gpuDevice, gpuBytes=whole, freeBytes=freeBytes, note=note)
    if mode != MEMORY_LOW and (budget is None or whole <= budget):
        return MemoryPlan(gpuDevice, gpuBytes=whole, freeBytes=freeBytes)
    Lz = int(objectShape[2])
    counts = [s for s in SLAB_COUNTS if (mode != MEMORY_LOW or s >= 4) and Lz // s >= CPU_BELOW_SLAB_PLANES]
    counts = [s for s in counts if slabWork(Lz, s, halo) <= MAX_SLAB_WORK] or counts[:1] or [1]
    for hostTransposes in (False, True):
        for slabs in counts:
            need = estimateGpuBytes(**args, slabs=slabs, hostNorms=True, hostTransposes=hostTransposes)
            if budget is None or need <= budget:
                return MemoryPlan(gpuDevice, slabs=slabs, hostNorms=True, hostTransposes=hostTransposes,
                                  gpuBytes=need, freeBytes=freeBytes)
    # Nothing fits the budget. The fewest slabs that fit in the memory free (without the reserve; a little over it,
    # Windows spills into system memory, slower but still faster than the CPU) still go on the GPU
    slabs = counts[-1]
    for count in counts:
        if estimateGpuBytes(**args, slabs=count, hostNorms=True, hostTransposes=True) <= freeBytes:
            slabs = count
            break
    need = estimateGpuBytes(**args, slabs=slabs, hostNorms=True, hostTransposes=True)
    if mode == MEMORY_LOW or need <= freeBytes:
        return MemoryPlan(gpuDevice, slabs=slabs, hostNorms=True, hostTransposes=True, gpuBytes=need,
                          freeBytes=freeBytes, note="close to the GPU memory free; may be slower")
    return MemoryPlan("cpu", freeBytes=freeBytes, note="not enough GPU memory free even for thin slabs")


# ---------------------------------------------------------------------------
# Slab system matrix
# ---------------------------------------------------------------------------
def psfHalo(psfTransform):
    """Half the largest z kernel of a configured SPECTPSFTransform (0 without z blurring)."""
    halo = 0
    for layer in getattr(psfTransform, "layers", {}).values():
        layerZ = getattr(layer, "layer_z", None)
        if layerZ is not None:
            halo = max(halo, int(layerZ.kernel_size[0]) // 2)
    return halo


def _hostTransposeRotation(store):
    """A RotationTransform whose backward() (the exact transpose of the rotation) keeps the sparse transposes in
    ``store`` on the host and moves each one to the GPU when it is used, instead of keeping every view's on the GPU.
    Each transpose is built on the device the back projection runs on, as PyTomography builds it there (its weights
    follow that device's arithmetic), so the result is the same."""
    from pytomography.transforms.shared import RotationTransform
    from pytomography.utils import spatial

    class HostTransposeRotation(RotationTransform):
        def backward(self, object, angles):
            x = object.permute(2, 0, 1).unsqueeze(0)
            B, C, H, W = x.shape
            key = (round(float(angles), 6), H, W, str(x.device), x.dtype, self.mode)
            transpose = store.get(key)
            if transpose is None:
                built = spatial._rotation_transpose(float(angles), H, W, x.device, x.dtype, self.mode)
                spatial._ROTATION_TRANSPOSES.pop(key, None)
                transpose = built.to("cpu")
                del built
                store[key] = transpose
            transpose = transpose.to(x.device)
            y = transpose @ x.permute(2, 3, 0, 1).reshape(H * W, B * C)
            return y.reshape(H, W, B, C).permute(2, 3, 0, 1)[0].permute(1, 2, 0)

    return HostTransposeRotation()


def makeSlabSystemMatrix(objectMeta, projMeta, attenuationMap=None, psfFactory=None, slabs=4, hostNorms=True,
                         hostTransposes=False, log=None):
    """A system matrix with the same forward() and backward() as PyTomography's SPECTSystemMatrix with an attenuation
    transform of attenuationMap (if given) followed by psfFactory()'s transform (if given), computed slab by slab."""
    import copy
    import torch
    import pytomography
    from pytomography.projectors.SPECT import SPECTSystemMatrix
    from pytomography.transforms.SPECT import SPECTAttenuationTransform

    halo = 0
    if psfFactory is not None:
        probe = psfFactory()
        probe.configure(objectMeta, projMeta)
        halo = psfHalo(probe)
        del probe

    class SlabSPECTSystemMatrix(SPECTSystemMatrix):
        """SPECTSystemMatrix computed on axial slabs (see the module docstring)."""

        def __init__(self):
            super().__init__(obj2obj_transforms=[], proj2proj_transforms=[], object_meta=objectMeta,
                             proj_meta=projMeta)
            self.halo = halo
            self.hostNorms = hostNorms
            self.hostTransposes = hostTransposes
            self.ranges = slabRanges(objectMeta.shape[2], slabs, halo)
            self.transposeStore = {}
            self.subMatrices = []
            mu = None
            if attenuationMap is not None:
                mu = attenuationMap.to(pytomography.device)
            for z0, z1, e0, e1 in self.ranges:
                om = copy.copy(objectMeta)
                om.shape = (objectMeta.shape[0], objectMeta.shape[1], e1 - e0)
                pm = copy.copy(projMeta)
                pm.shape = (projMeta.shape[0], projMeta.shape[1], e1 - e0)
                pm.compute_padded_shape()
                transforms = []
                if mu is not None:
                    # a view of the map (no copy): the transform pads and rotates it into new tensors
                    transforms.append(SPECTAttenuationTransform(mu[:, :, e0:e1]))
                if psfFactory is not None:
                    transforms.append(psfFactory())
                sub = SPECTSystemMatrix(obj2obj_transforms=transforms, proj2proj_transforms=[],
                                        object_meta=om, proj_meta=pm)
                if hostTransposes and torch.device(pytomography.device).type != "cpu":
                    sub.rotation_transform = _hostTransposeRotation(self.transposeStore)
                self.subMatrices.append(sub)
            if log is not None:
                log(f"  -> {len(self.ranges)} axial slabs of up to {max(z1 - z0 for z0, z1, _, _ in self.ranges)} "
                    f"planes, {halo} margin plane(s) on each side")

        def set_n_subsets(self, n_subsets):
            super().set_n_subsets(n_subsets)
            for sub in self.subMatrices:
                sub.subset_indices_array = self.subset_indices_array

        def forward(self, object, subset_idx=None):
            object = object.to(pytomography.device)
            nAngles = len(self._host_angle_indices(subset_idx))
            proj = torch.zeros((nAngles, projMeta.shape[1], projMeta.shape[2]), device=pytomography.device)
            for (z0, z1, e0, e1), sub in zip(self.ranges, self.subMatrices):
                part = sub.forward(object[:, :, e0:e1], subset_idx)
                proj[:, :, z0:z1] = part[:, :, z0 - e0:z1 - e0]
                del part
            return proj

        def backward(self, proj, subset_idx=None):
            proj = proj.to(pytomography.device)
            object = torch.zeros(tuple(objectMeta.shape), device=pytomography.device)
            for (z0, z1, e0, e1), sub in zip(self.ranges, self.subMatrices):
                part = sub.backward(proj[:, :, e0:e1], subset_idx)
                object[:, :, z0:z1] = part[:, :, z0 - e0:z1 - e0]
                del part
            return object

        def compute_normalization_factor(self, subset_idx=None):
            norm = super().compute_normalization_factor(subset_idx)
            return norm.cpu() if self.hostNorms else norm

        def _fbp(self, projections, filter=None, **kwargs):
            if tuple(projections.shape) != tuple(projMeta.shape):
                raise ValueError(f"projections of shape {tuple(projections.shape)} do not match the system matrix, "
                                 f"{tuple(projMeta.shape)}")
            object = torch.zeros(tuple(objectMeta.shape), device=pytomography.device)
            for (z0, z1, e0, e1), sub in zip(self.ranges, self.subMatrices):
                part = sub._fbp(projections[:, :, e0:e1], filter, **kwargs)
                object[:, :, z0:z1] = part[:, :, z0 - e0:z1 - e0]
                del part
            return object

    return SlabSPECTSystemMatrix()


def hostNormLikelihood(systemMatrix, projections, additive_term=None):
    """PoissonLogLikelihood whose sum of the H_m^T 1 images (BSREM's preconditioner) is made once on the host, instead
    of stacking every subset's image on the GPU at every update."""
    import torch
    import pytomography
    from pytomography.likelihoods import PoissonLogLikelihood

    class HostNormLikelihood(PoissonLogLikelihood):
        _normSum = None
        _normSumKey = None

        def _get_normBP(self, subset_idx, return_sum=False):
            if return_sum and getattr(self, "n_subsets", 1) >= 2:
                key = id(self.norm_BPs)
                if self._normSumKey != key:
                    self._normSum = torch.stack([n.cpu() for n in self.norm_BPs]).sum(axis=0)
                    self._normSumKey = key
                return self._normSum.to(pytomography.device)
            return super()._get_normBP(subset_idx, return_sum)

    return HostNormLikelihood(systemMatrix, projections, additive_term=additive_term)


def releaseGpuCaches():
    """Frees PyTomography's GPU caches of rotation transposes (the largest) and PyTorch's cached blocks."""
    try:
        from pytomography.utils import spatial
        spatial._ROTATION_TRANSPOSES.clear()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class useDevice:
    """Context manager: PyTomography computes on ``device`` inside the block (e.g. the CPU on a CUDA machine)."""

    def __init__(self, device):
        self.device = device
        self.previous = None

    def __enter__(self):
        import pytomography
        self.previous = pytomography.device
        if self.device is not None:
            import torch
            pytomography.set_device(torch.device(self.device))
        return self

    def __exit__(self, *exc):
        import pytomography
        pytomography.set_device(self.previous)
        return False
