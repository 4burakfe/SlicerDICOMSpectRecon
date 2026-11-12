"""Compare reconstructions of two runs (argv[2], argv[3]) of the phantom in argv[1], in patient mm."""
import os, sys, json
import numpy as np, SimpleITK as sitk
data, a, b = sys.argv[1:4]
truth = np.load(os.path.join(data, "truth.npz"))
spheres = truth["spheres"]

def voi(image, c, r):
    arr = sitk.GetArrayFromImage(image)          # [z, y, x]
    idx = np.indices(arr.shape)
    pts = np.stack([idx[2], idx[1], idx[0]], -1).reshape(-1, 3).astype(float)
    D = np.array(image.GetDirection()).reshape(3, 3) * np.array(image.GetSpacing())
    xyz = pts @ D.T + np.array(image.GetOrigin())
    inside = np.linalg.norm(xyz - c, axis=1) <= r
    return float(arr.reshape(-1)[inside].mean()), xyz, arr.reshape(-1)

def describe(path):
    im = sitk.ReadImage(path)
    out = {"origin": np.round(im.GetOrigin(), 2).tolist(), "spacing": np.round(im.GetSpacing(), 3).tolist(),
           "size": im.GetSize(), "sum": float(sitk.GetArrayFromImage(im).sum())}
    bg, xyz, flat = voi(im, np.array([-60.0, -30.0, -110.0]), 15)
    out["bg"] = bg
    for i, (cx, cy, cz, r, v) in enumerate(spheres):
        m, _, _ = voi(im, np.array([cx, cy, cz]), r * 0.6)
        out[f"s{i}"] = m / bg if bg > 0 else np.nan          # contrast vs background (truth: v)
        near = np.linalg.norm(xyz - [cx, cy, cz], axis=1) <= 2 * r
        w = flat[near]
        if v > 0 and w.sum() > 0:
            out[f"s{i}_com_err_mm"] = float(np.linalg.norm((xyz[near] * w[:, None]).sum(0) / w.sum() - [cx, cy, cz]))
    return im, out

stats_a = json.load(open(os.path.join(a, "stats.json")))
stats_b = json.load(open(os.path.join(b, "stats.json")))
print("truth contrasts:", [float(s[4]) for s in spheres])
for name in stats_a:
    ia, da = describe(os.path.join(a, name + ".nrrd"))
    ib, db = describe(os.path.join(b, name + ".nrrd"))
    rb = sitk.Resample(ib, ia, sitk.Transform(), sitk.sitkLinear, 0.0)
    A, B = sitk.GetArrayFromImage(ia), sitk.GetArrayFromImage(rb)
    nrmsd = float(np.sqrt(((A - B) ** 2).mean()) / A.mean())
    print(f"\n== {name}: geometry equal={da['origin'] == db['origin'] and da['spacing'] == db['spacing']}"
          f"  NRMSD(4 vs 3)={nrmsd:.3f}  time {stats_a[name]['seconds']}s vs {stats_b[name]['seconds']}s")
    for key in da:
        if key in ("size",):
            continue
        va, vb = da[key], db.get(key)
        if isinstance(va, float) and isinstance(vb, float):
            print(f"   {key:14s} A={va:10.4g}  B={vb:10.4g}  A/B={va / vb if vb else float('nan'):.4f}")
        else:
            print(f"   {key:14s} A={va}  B={vb}")
    for key in ("photopeakCounts", "scatterCounts", "nativeSum", "consistency"):
        va, vb = stats_a[name].get(key), stats_b[name].get(key)
        print(f"   {key:14s} A={va}  B={vb}")
