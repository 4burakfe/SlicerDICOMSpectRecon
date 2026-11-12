"""Run the plugin's reconstruction logic (code in argv[1]) on the phantom (argv[2]); images to argv[3]."""
import os, sys, json, time, warnings
warnings.simplefilter("ignore")
code, data, out = sys.argv[1:4]
sys.path.insert(0, code)
os.makedirs(out, exist_ok=True)
import SimpleITK as sitk
from DICOMSpectReconLib import reconstruction as R, collimators as C

ct = sorted(os.path.join(data, "ct", f) for f in os.listdir(os.path.join(data, "ct")))
logic = R.SpectReconLogic(os.path.join(data, "nm.dcm"), log=lambda m: None, ctFiles=ct, indexPeak=0)
entry = C.findEntry(C.builtinCollimators(), "G8-LEHR")
print("PyTomography", R.pytomographyVersion(), "| G8-LEHR", entry and {k: entry[k] for k in ("holeDiameterMm", "holeLengthMm", "septaMm")},
      "| builtin collimators:", len(C.builtinCollimators()))
psf = {"entry": entry, "energyKeV": 140.5}
cases = {
    "osem_nc": dict(iters=4, subsets=6),
    "osem_ac_dew": dict(iters=4, subsets=6, useAttenuationCorrection=True, scatter=(1, None)),
    "osem_ac_dewk": dict(iters=4, subsets=6, useAttenuationCorrection=True, scatter=(1, None, 1.1)),
    "osem_ac_tew_rr": dict(iters=8, subsets=6, useAttenuationCorrection=True, scatter=(1, 2), psf=psf),
    "bsrem_ac_dew": dict(iters=10, subsets=6, useAttenuationCorrection=True, scatter=(1, None),
                         algorithm=R.ALGORITHM_BSREM, bsrem={"beta": 0.3, "gamma": 2.0}),
    "fbp_bw": dict(iters=1, subsets=1, algorithm=R.ALGORITHM_FBP, fbp={"window": "Butterworth", "cutoff": 0.5, "order": 10}),
    "fbp_ac_dew_hann": dict(iters=1, subsets=1, useAttenuationCorrection=True, scatter=(1, None),
                            algorithm=R.ALGORITHM_FBP, fbp={"window": "Hann", "cutoff": 0.8, "order": 10}),
}
res = {}
for name, kw in cases.items():
    t = time.time()
    image, stats = logic.reconstruct(**kw)
    dt = time.time() - t
    sitk.WriteImage(image, os.path.join(out, name + ".nrrd"))
    stats = {k: v for k, v in stats.items() if isinstance(v, (int, float, str)) or v is None}
    stats["seconds"] = round(dt, 2)
    res[name] = stats
    print(f"{name:18s} {dt:6.1f} s  sum {stats['nativeSum']:.4g}  consistency {stats['consistency']}")
json.dump(res, open(os.path.join(out, "stats.json"), "w"), indent=1)
logic.cleanup()
