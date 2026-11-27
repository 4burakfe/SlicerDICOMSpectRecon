"""Low GPU memory check: a synthetic L x L x L SPECT study (attenuation, PSF, scatter) reconstructed with OSEM by the
whole-object projector and by the slab projector of DICOMSpectReconLib.lowmemory; prints the plan Automatic chooses,
the time and peak GPU memory of each run, and the largest difference between the images.

    PythonSlicer lowmemory_check.py CODE [L] [VIEWS] [SUBSETS] [ITERS] [MODES]

CODE: the DICOMSpectRecon folder (holding DICOMSpectReconLib). Defaults: 256 120 8 1. MODES: comma separated,
from auto, whole, low, cpu (default auto,low,whole: whole last, since on a small GPU it may crawl)."""
import functools
import os
import sys
import time
import warnings

warnings.simplefilter("ignore")
print = functools.partial(print, flush=True)
code = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 256
NV = int(sys.argv[3]) if len(sys.argv) > 3 else 120
SUBSETS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
ITERS = int(sys.argv[5]) if len(sys.argv) > 5 else 1
MODES = (sys.argv[6] if len(sys.argv) > 6 else "auto,low,whole").split(",")
sys.path.insert(0, code)

import numpy as np
import torch
import pytomography
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta, SPECTPSFMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.algorithms import OSEM
from DICOMSpectReconLib import lowmemory as LM

cuda = torch.cuda.is_available()
print(f"PyTomography {pytomography.__version__}, torch {torch.__version__}, device {pytomography.device}"
      + (f" ({torch.cuda.get_device_name(0)}, {torch.cuda.mem_get_info()[1] / 1e9:.2f} GB)" if cuda else ""))
pix = 2.2 * 256 / L
om = SPECTObjectMeta((pix, pix, pix), (L, L, L))
angles = np.linspace(0, 360, NV, endpoint=False)
radii = 230.0 + 40.0 * np.abs(np.sin(np.deg2rad(angles)))          # body contour orbit
pm = SPECTProjMeta((L, L), (pix, pix), angles, radii)
# LEHR-like PSF in mm: sigma(d) = (0.0285 d + 0.71) as in collimators.py for 1.5/35 mm holes at 140 keV
psfMeta = SPECTPSFMeta((0.0285 * 2.0 / 2.355, 3.6 / 2.355), sigma_fit=lambda r, a, b: a * r + b)
psfFactory = lambda: SPECTPSFTransform(psfMeta)

g = torch.Generator().manual_seed(1)
x, y, z = torch.meshgrid(*(torch.linspace(-1, 1, L),) * 3, indexing="ij")
body = ((x / 0.7) ** 2 + (y / 0.5) ** 2 <= 1) & (z.abs() < 0.9)
mu = (body.float() * 0.0153).to(pytomography.device)               # water, 1/mm
truth = body.float() * 1.0
for cx, cy, cz, r, v in ((0.2, 0.1, 0.0, 0.12, 8.0), (-0.3, -0.1, 0.4, 0.08, 6.0), (0.0, 0.25, -0.5, 0.06, 10.0)):
    truth += v * (((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) <= r * r).float()
truth = truth.to(pytomography.device)


def peak(fn):
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    out = fn()
    if cuda:
        torch.cuda.synchronize()
    return out, time.time() - t, (torch.cuda.max_memory_reserved() / 1e9 if cuda else float("nan"))


# the data are made with slabs (forward projection only, the plan of 'GPU, low memory'), so that it fits on any GPU
probe = psfFactory()
probe.configure(om, pm)
halo = LM.psfHalo(probe)
makerPlan = LM.planMemory(LM.MEMORY_LOW, (L, L, L), NV, (NV, L, L), SUBSETS, True, True, halo=halo)
print(f"Making the data: {makerPlan.describe()}")
t0 = time.time()
maker = LM.makeSlabSystemMatrix(om, pm, mu, psfFactory, slabs=makerPlan.slabs, hostNorms=True, hostTransposes=True)
with torch.no_grad():
    clean = maker.forward(truth).cpu()
del maker, truth
scale = 2e7 / float(clean.sum())
projections = torch.poisson(clean * scale, generator=g).to(pytomography.device)
scatter = torch.full_like(projections, 0.2 * float(projections.mean()))
del clean
LM.releaseGpuCaches()
print(f"Projections {tuple(projections.shape)}, {float(projections.sum()):.3g} counts (made in {time.time() - t0:.0f} s); "
      f"OSEM {ITERS} x {SUBSETS}")
plan = LM.planMemory(LM.MEMORY_AUTO, (L, L, L), NV, tuple(projections.shape), SUBSETS, attenuation=True, psf=True,
                     halo=halo)
print("Automatic plan:", plan.describe())

class Progress:
    """OSEM callback: prints the time of every subset."""
    def __init__(self):
        self.t = time.time()

    def run(self, object, n_iter=0, n_subset=0, *args, **kwargs):
        now = time.time()
        print(f"       iteration {n_iter + 1}, subset {n_subset + 1}/{SUBSETS}: {now - self.t:.1f} s")
        self.t = now
        return object

    def finalize(self, *args, **kwargs):
        return None


images = {}
for mode in MODES:
    if mode == "auto":
        p = plan
    elif mode == "whole":
        p = LM.MemoryPlan(str(pytomography.device))
    elif mode == "low":
        p = LM.planMemory(LM.MEMORY_LOW, (L, L, L), NV, tuple(projections.shape), SUBSETS, True, True, halo=halo)
    else:
        p = LM.MemoryPlan("cpu")
    LM.releaseGpuCaches()
    print(f"{mode}: {p.describe()}")

    def run():
        with LM.useDevice("cpu" if p.device == "cpu" else None):
            proj = projections.cpu() if p.device == "cpu" else projections
            add = scatter.cpu() if p.device == "cpu" else scatter
            amap = mu.cpu() if p.device == "cpu" else mu
            if p.lowMemory:
                sm = LM.makeSlabSystemMatrix(om, pm, amap, psfFactory, slabs=p.slabs, hostNorms=p.hostNorms,
                                             hostTransposes=p.hostTransposes, log=print)
                likelihood = LM.hostNormLikelihood(sm, proj, additive_term=add)
            else:
                sm = SPECTSystemMatrix([SPECTAttenuationTransform(amap), psfFactory()], [], om, pm)
                likelihood = PoissonLogLikelihood(sm, proj, additive_term=add)
            return OSEM(likelihood)(n_iters=ITERS, n_subsets=SUBSETS, callback=Progress()).cpu()

    image, seconds, gb = peak(run)
    images[mode] = image
    print(f"{mode:6s} {p.describe()}\n       {seconds:7.1f} s, peak GPU reserved {gb:.2f} GB, sum {float(image.sum()):.6g}")

names = list(images)
for other in names[1:]:
    a, b = images[names[0]], images[other]
    print(f"max |{names[0]} - {other}| / max = {float((a - b).abs().max() / a.abs().max()):.2e}, "
          f"sums differ by {abs(float(a.sum() - b.sum())) / float(a.sum()):.2e}")
