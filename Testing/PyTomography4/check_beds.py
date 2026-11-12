"""Multi-bed check on the phantom of make_phantom.py OUT 2: bed detection, timing normalisation and
stitching of the plugin code in argv[1] (the DICOMSpectRecon folder), data in argv[2].

check_beds.py CODE DATA [OUT]: prints the numbers; with OUT, writes the stitched and per-bed images."""
import os, sys, glob, warnings
warnings.simplefilter("ignore")
code, data = sys.argv[1:3]
out = sys.argv[3] if len(sys.argv) > 3 else None
sys.path.insert(0, code)
import numpy as np, SimpleITK as sitk
from DICOMSpectReconLib import nmdata, reconstruction as R

beds = sorted(glob.glob(os.path.join(data, "nm_bed*.dcm")))
ct = sorted(glob.glob(os.path.join(data, "ct", "*.dcm")))
truth = np.load(os.path.join(data, "truth.npz"))
log = lambda m: print("   |", m) if ("WARNING" in m or "Bed " in m or "overlap" in m or "Stitched" in m) else None

# 1. detection: from either bed, the other one is found, inferior bed first; a repeat at the same height is dropped
print("detection from bed 1:", [os.path.basename(f) for f in nmdata.findBeds(beds[0], beds)])
print("detection from bed 2:", [os.path.basename(f) for f in nmdata.findBeds(beds[1], beds)])
print("single-bed file alone:", [os.path.basename(f) for f in nmdata.findBeds(beds[0], [beds[0]])])
for f in beds:
    i = nmdata.bedInfo(f)
    print(f"  {os.path.basename(f)}: centre z {i['z']:.1f} mm, axial FOV {i['fov']:.1f} mm")

# 2. timing factors vs the phantom's truth (factor = 1 / relative counts)
factors, notes = R.bedTimingFactors(beds, beds[0], print)
for f, rel in zip(beds, truth["rel"]):
    print(f"timing factor {os.path.basename(f)}: {factors[f]:.6f}  (truth {1 / rel:.6f})  {notes[f]}")

# 3. reconstruct: the reference bed alone, and both beds stitched (OSEM 4x6, CT AC, DEW scatter)
kw = dict(iters=4, subsets=6, useAttenuationCorrection=True, scatter=(1, None))
single = R.SpectReconLogic(beds[0], log=lambda m: None, ctFiles=ct, indexPeak=0)
img1, st1 = single.reconstruct(**kw)
single.cleanup()
multi = R.MultiBedLogic(beds, log=log, ctFiles=ct, indexPeak=0)
img, st = multi.reconstruct(**kw)
print("stats:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()})
print("reference slices in stitched image:", multi.referenceSlices)
bedImages = [b for b in multi.beds]
per = []
for b in multi.beds:
    i, s = b.reconstruct(**kw)
    per.append((i, s))
multi.cleanup()

A = sitk.GetArrayFromImage(img).astype(np.float64)        # [z, y, x]
B1 = sitk.GetArrayFromImage(img1).astype(np.float64)
o1, n = multi.referenceSlices
print(f"stitched: size {img.GetSize()}, origin {np.round(img.GetOrigin(), 2).tolist()}, sum {A.sum():.6g}")
print(f"bed 1 alone: size {img1.GetSize()}, origin {np.round(img1.GetOrigin(), 2).tolist()}, sum {B1.sum():.6g}")
print(f"per-bed native sums after timing factors: {[round(s['nativeSum'], 1) for _, s in per]}, total {sum(s['nativeSum'] for _, s in per):.6g}")

# stitched == bed 1 alone where only bed 1 contributes (bed 1 is the reference, factor 1)
offsets = st["bedOffsets"]
lowOff = min(offsets)
upper = [k for k in range(A.shape[0]) if k >= offsets[0] + 1 and k <= offsets[0] + n - 2 and k > offsets[1] + n - 2]
diff = np.abs(A[upper] - B1[[k - offsets[0] for k in upper]]).max()
print(f"bed-1-only slices {upper[0]}..{upper[-1]}: max |stitched - bed 1 alone| = {diff:.3g}")

# expected stitched sum from the weights
w = R.bedStitchingWeights(sorted(offsets), n)
order = np.argsort(offsets)
exp = 0.0
for wk, i in zip(w, order):
    arr = sitk.GetArrayFromImage(per[i][0]).astype(np.float64)
    exp += (arr * wk[:, None, None]).sum()
print(f"stitched sum vs weighted per-bed sum: {A.sum():.6g} vs {exp:.6g} (ratio {A.sum() / exp:.6f})")
ov = [k for k in range(A.shape[0]) if sum(1 for o in offsets if o + 1 <= k <= o + n - 2) == 2]
print(f"overlap (blended) slices: {ov[0]}..{ov[-1]} ({len(ov)} slices)")
for k in ov[:: max(1, len(ov) // 5)]:
    print(f"   slice {k}: weights " + ", ".join(f"bed{j + 1} {w[list(order).index(j)][k - offsets[j]]:.3f}" for j in range(2)))

# 4. seam continuity: mean of a uniform background cylinder (body, away from spheres and bone) per slice
spacing, origin = np.array(img.GetSpacing()), np.array(img.GetOrigin())
zz, yy, xx = np.indices(A.shape)
X = origin[0] + xx * spacing[0]; Y = origin[1] + yy * spacing[1]; Z = origin[2] + zz * spacing[2]
bg = ((X / 140.0) ** 2 + (Y / 100.0) ** 2 <= 0.7 ** 2) & (X ** 2 + (Y - 55.0) ** 2 > 30.0 ** 2)   # body, not the bone
for (cx, cy, cz, r, v) in truth["spheres"]:
    bg &= (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2 > (r + 15) ** 2
prof = np.array([A[k][bg[k]].mean() if bg[k].any() else np.nan for k in range(A.shape[0])])
lo, hi = ov[0], ov[-1]
below = prof[lo - 6:lo]; inside = prof[lo:hi + 1]; above = prof[hi + 1:hi + 7]
print(f"background mean per slice: 6 below overlap {np.nanmean(below):.4f}, in overlap {np.nanmean(inside):.4f}, "
      f"6 above {np.nanmean(above):.4f}")
print("   profile around the overlap (slice: value):",
      ", ".join(f"{k}:{prof[k]:.3f}" for k in range(lo - 4, hi + 5)))
step = np.abs(np.diff(prof[lo - 6:hi + 7]))
print(f"   largest slice-to-slice change around the overlap {step.max():.4f} "
      f"(typical elsewhere in bed 1 {np.nanmedian(np.abs(np.diff(prof[offsets[0] + 20:offsets[0] + 40]))):.4f})")
# the beds' own background levels in the overlap (each alone, before blending)
for j, (im, _) in enumerate(per):
    arr = sitk.GetArrayFromImage(im).astype(np.float64)
    ks = [k - offsets[j] for k in ov]
    vals = [arr[kk][bg[k]].mean() for k, kk in zip(ov, ks)]
    print(f"   bed {j + 1} alone, background in the overlap slices: {np.mean(vals):.4f}")

# 5. spheres: mean in 0.6 r, relative to background
def sphereMean(arr, img, c, r):
    sp, org = np.array(img.GetSpacing()), np.array(img.GetOrigin())
    z, y, x = np.indices(arr.shape)
    d = (org[0] + x * sp[0] - c[0]) ** 2 + (org[1] + y * sp[1] - c[1]) ** 2 + (org[2] + z * sp[2] - c[2]) ** 2
    m = d <= (0.6 * r) ** 2
    return float(arr[m].mean()) if m.any() else float("nan")
bgLevel = np.nanmean(prof[ov[0] - 20:ov[-1] + 20])
print(f"background level near the overlap: {bgLevel:.4f}")
for i, (cx, cy, cz, r, v) in enumerate(truth["spheres"]):
    c = (cx, cy, cz)
    row = f"sphere {i} at z {cz:.1f} (truth {v:g}): stitched {sphereMean(A, img, c, r) / bgLevel:.3f}"
    for j, (im, _) in enumerate(per):
        arr = sitk.GetArrayFromImage(im).astype(np.float64)
        val = sphereMean(arr, im, c, r)
        row += f" | bed {j + 1} alone {val / bgLevel:.3f}" if np.isfinite(val) else f" | bed {j + 1} -"
    print(row)

# 6. count scaling: Match detected primary counts on the stitched image
factor, note = R.countScalingFactor(st, R.SCALING_PRIMARY)
print(f"primary scaling: factor {factor:.6g}, {note}; stitched total after scaling {A.sum() * factor:.6g}")
if out:
    os.makedirs(out, exist_ok=True)
    sitk.WriteImage(img, os.path.join(out, "stitched.nrrd"))
    for j, (im, _) in enumerate(per):
        sitk.WriteImage(im, os.path.join(out, f"bed{j + 1}.nrrd"))
    print("written", out)
