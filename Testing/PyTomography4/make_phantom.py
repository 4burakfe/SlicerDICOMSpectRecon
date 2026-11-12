"""Synthetic GE-like dual-head Tc-99m SPECT/CT study (needs PyTomography 4 to forward project).
Writes out/nm.dcm (3 energy windows: peak 126-154, lower 114-126, upper 154-166 keV) and out/ct/*.dcm,
and out/truth.npz (activity per mm^3 in LPS mm, sphere definitions).

make_phantom.py OUT [BEDS]: with BEDS > 1, a multi-bed study instead: out/nm_bed1.dcm ... (bed 1 the most
superior, each next bed 3/4 of the axial field of view lower, so neighbours overlap by 1/4), one CT over
all beds. Later beds are scanned longer per view and later (decay), and their counts follow: the plugin
has to normalise them to bed 1 before stitching. truth.npz then also holds the beds' timing factors."""
import os, sys, datetime
import numpy as np, pydicom, torch
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

OUT = sys.argv[1]
os.makedirs(os.path.join(OUT, "ct"), exist_ok=True)
rng = np.random.default_rng(1)
study, frame = generate_uid(), generate_uid()
ct_series = generate_uid()
L, PIX, NV = 64, 4.42, 30          # bins, mm, views per head
ZTOP = 100.0
BEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
STEP = 0.75 * L * PIX                                  # axial bed step (mm): neighbours overlap by L/4 rows
ZTOPS = [ZTOP - b * STEP for b in range(BEDS)]
FRAME_MS = [20000 + 5000 * b for b in range(BEDS)]     # per-view time grows from bed to bed
HALF_LIFE = 6.0067 * 3600.0                            # Tc-99m, s
STARTS = [0.0]                                         # bed start times, s after 12:00:00
for b in range(1, BEDS):
    STARTS.append(STARTS[-1] + NV * FRAME_MS[b - 1] / 1000.0 + 60.0)
MIDS = [s + NV * f / 2000.0 for s, f in zip(STARTS, FRAME_MS)]   # both heads acquire at once
# counts of each bed relative to bed 1 for the same activity: counting time and decay
REL = [(f / FRAME_MS[0]) * np.exp(-np.log(2) * (m - MIDS[0]) / HALF_LIFE) for f, m in zip(FRAME_MS, MIDS)]

def clock(seconds):
    t = datetime.datetime(2026, 11, 12, 12, 0, 0) + datetime.timedelta(seconds=seconds)
    return t.strftime("%H%M%S.%f")[:13]

def base(modality, sop_class):
    meta = pydicom.dataset.FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.Manufacturer = modality, "GE MEDICAL SYSTEMS"
    ds.PatientName, ds.PatientID, ds.PatientPosition = "PHANTOM", "P1", "HFS"
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = study, generate_uid(), frame
    ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = "20261112"
    ds.StudyTime = ds.SeriesTime = ds.AcquisitionTime = "120000"
    ds.StudyID, ds.SeriesNumber, ds.AccessionNumber = "1", 1, "1"
    ds.StudyDescription = ds.SeriesDescription = "PHANTOM"
    ds.PatientBirthDate, ds.PatientSex = "19700101", "O"
    return ds

# ---------- NM header ----------
def nm(pixels=None, ztop=ZTOP, frame_ms=20000, start=None, bed=None):
    ds = base("NM", "1.2.840.10008.5.1.4.1.1.20")
    if bed is not None:
        ds.SeriesNumber, ds.SeriesDescription = bed + 1, f"PHANTOM bed {bed + 1}"
    if start is not None:
        ds.SeriesTime = ds.AcquisitionTime = start
    ds.ManufacturerModelName = "Discovery NM/CT 670"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "TOMO", "EMISSION"]
    ds.Rows = ds.Columns = L
    ds.PixelSpacing = [PIX, PIX]
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    nW, nD = 3, 2
    n = nW * nD * NV
    ds.NumberOfFrames = n
    ds.EnergyWindowVector = [w + 1 for w in range(nW) for d in range(nD) for v in range(NV)]
    ds.DetectorVector = [d + 1 for w in range(nW) for d in range(nD) for v in range(NV)]
    ds.RotationVector = [1] * n
    ds.AngularViewVector = [v + 1 for w in range(nW) for d in range(nD) for v in range(NV)]
    ds.NumberOfEnergyWindows, ds.NumberOfDetectors, ds.NumberOfRotations = nW, nD, 1
    wins = []
    for name, lo, hi in (("Tc99m PEAK", 126.0, 154.0), ("SC LOWER", 114.0, 126.0), ("SC UPPER", 154.0, 166.0)):
        w = Dataset(); r = Dataset()
        r.EnergyWindowLowerLimit, r.EnergyWindowUpperLimit = lo, hi
        w.EnergyWindowRangeSequence = [r]; w.EnergyWindowName = name
        wins.append(w)
    ds.EnergyWindowInformationSequence = wins
    rad = Dataset(); code = Dataset(); code.CodeMeaning = "99m Technetium"
    rad.RadionuclideCodeSequence = [code]
    ds.RadiopharmaceuticalInformationSequence = [rad]
    rot = Dataset()
    rot.StartAngle, rot.AngularStep, rot.RotationDirection = 0.0, 6.0, "CW"
    rot.NumberOfFramesInRotation, rot.ActualFrameDuration = NV, frame_ms
    rot.ScanArc, rot.TableHeight = 360, 0.0
    rot.RadialPosition = [250.0 + 20 * np.sin(np.radians(6 * v)) for v in range(NV)]
    ds.RotationInformationSequence = [rot]
    dets = []
    for d in range(nD):
        it = Dataset()
        it.StartAngle = 0.0 + 180.0 * d
        it.RadialPosition = rot.RadialPosition
        it.ImageOrientationPatient = [1, 0, 0, 0, 0, -1]
        it.ImagePositionPatient = [-(L - 1) / 2 * PIX, -(L - 1) / 2 * PIX, ztop]
        it.CollimatorGridName, it.CollimatorType = "LEHR", "PARA"
        dets.append(it)
    ds.DetectorInformationSequence = dets
    ds.PixelData = (np.zeros((n, L, L), np.uint16) if pixels is None else pixels).tobytes()
    return ds

# ---------- CT ----------
NCT, DCT, TCT = 128, 3.0, 3.0
xs = (np.arange(NCT) - (NCT - 1) / 2) * DCT
ZBOT = min(ZTOPS) - 64 * PIX
zs = np.arange(ZBOT + 20, ZTOP + 10, TCT)
def body(x, y):
    return (x / 140.0) ** 2 + (y / 100.0) ** 2 <= 1
SPHERES = [((40.0, 0.0, -40.0), 20.0, 8.0), ((-45.0, 10.0, -60.0), 12.0, 8.0), ((0.0, -50.0, -20.0), 15.0, 0.0)]
if BEDS > 1:   # one sphere in the overlap of beds 1 and 2, one in bed 2 only
    SPHERES += [((30.0, -10.0, ZTOP - 63 * PIX + (64 * PIX - STEP) / 2), 15.0, 6.0), ((-30.0, 20.0, ZTOPS[1] - 170.0), 15.0, 8.0)]
BONE = ((0.0, 55.0), 15.0)  # (x, y), radius: a bone rod along z
for k, z in enumerate(zs):
    ds = base("CT", "1.2.840.10008.5.1.4.1.1.2")
    ds.SeriesInstanceUID = ct_series
    ds.ManufacturerModelName, ds.KVP = "Discovery NM/CT 670", 120
    ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.Rows = ds.Columns = NCT
    ds.PixelSpacing = [DCT, DCT]; ds.SliceThickness = TCT
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.ImagePositionPatient = [float(xs[0]), float(xs[0]), float(z)]
    ds.InstanceNumber = k + 1
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 1
    ds.RescaleSlope, ds.RescaleIntercept, ds.RescaleType = 1, -1024, "HU"
    X, Y = np.meshgrid(xs, xs, indexing="xy")      # rows along y, columns along x
    hu = np.where(body(X, Y), 0.0, -1000.0)
    hu[(X - BONE[0][0]) ** 2 + (Y - BONE[0][1]) ** 2 <= BONE[1] ** 2] = 1000.0
    ds.PixelData = (hu + 1024).astype(np.int16).tobytes()
    ds.save_as(os.path.join(OUT, "ct", f"ct{k:03d}.dcm"), enforce_file_format=True)

# ---------- activity and projections (PyTomography 4) ----------
from pytomography.io.SPECT import dicom as d4
from pytomography.io import attenuation_map
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.io.SPECT import collimator_psf

psf = collimator_psf("G8-LEHR", 140.5, intrinsic_fwhm_140keV=3.8)
ct_files = sorted(os.path.join(OUT, "ct", f) for f in os.listdir(os.path.join(OUT, "ct")))
from scipy.ndimage import gaussian_filter
scale, written, truth = None, [], {}
for b, ztop in enumerate(ZTOPS):
    tmp = os.path.join(OUT, "_empty.dcm")
    nm(ztop=ztop).save_as(tmp, enforce_file_format=True)
    geom = d4._geometry(pydicom.dcmread(tmp), tmp)
    object_meta, proj_meta = d4._metadata(geom)
    A = object_meta.affine
    I, J, K = np.meshgrid(*[np.arange(s) for s in object_meta.shape], indexing="ij")
    P = np.einsum("ab,bijk->aijk", A[:3, :3], np.stack([I, J, K])) + A[:3, 3][:, None, None, None]
    x, y, z = P
    act = np.where(body(x, y) & (z < ZTOP - 15) & (z > ZBOT + 30), 1.0, 0.0)
    for (cx, cy, cz), r, val in SPHERES:
        act[(x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r ** 2] = val
    mu = attenuation_map(ct_files, object_meta, 140.5)
    H = SPECTSystemMatrix([SPECTAttenuationTransform(mu), SPECTPSFTransform(psf)], [], object_meta, proj_meta)
    primary = H.forward(torch.tensor(act, dtype=torch.float32)).cpu().numpy()
    if scale is None:
        scale = 2.0e6 / primary.sum()     # expected primary counts in the photopeak (bed 1)
    primary *= scale * REL[b]
    scatter = gaussian_filter(primary, (0, 6, 6)) * 0.35
    k = 0.5 * 28.0 / 12.0                 # DEW weight that PyTomography's automatic estimate amounts to
    lower = scatter / k
    upper = np.full_like(primary, 0.2 * REL[b])
    proj = np.stack([primary + scatter, lower, upper])                 # [W, θ, r, z] in β order
    noisy = rng.poisson(proj).astype(np.float64)

    # back to frames: inverse of read_dicom
    views = noisy[:, None]                                              # [W, S, θ, r, z]
    flip = geom["flip_r"]
    views = views.copy(); views[:, :, flip] = views[:, :, flip, ::-1, :]
    views = np.swapaxes(views, -1, -2)                                  # [W, S, θ, z, r]
    if geom["flip_z"]:
        views = views[..., ::-1, :]
    pixels = np.zeros((geom["frames"].max() + 1, L, L), np.uint16)
    for w in range(3):
        for t in range(views.shape[2]):
            pixels[geom["frames"][w, 0, t]] = views[w, 0, t]
    name = "nm.dcm" if BEDS == 1 else f"nm_bed{b + 1}.dcm"
    if BEDS == 1:
        nm(pixels).save_as(os.path.join(OUT, name), enforce_file_format=True)
    else:
        nm(pixels, ztop, FRAME_MS[b], clock(STARTS[b]), b).save_as(os.path.join(OUT, name), enforce_file_format=True)
    os.remove(tmp)
    written.append((name, primary.sum(), scatter.sum(), noisy[0].sum()))
    if b == 0:
        truth = dict(act=act, affine=A, primary=primary.sum(), voxel_mm3=float(np.prod(object_meta.dr)))
np.savez(os.path.join(OUT, "truth.npz"), spheres=np.array([[*c, r, v] for c, r, v in SPHERES]),
         rel=np.array(REL), ztops=np.array(ZTOPS), **truth)
for name, p, s, n in written:
    print("written", os.path.join(OUT, name) if BEDS > 1 else OUT, "primary", p, "scatter", s, "check readback",
          float(d4.read_dicom(os.path.join(OUT, name))[0][0].sum()), float(n))
if BEDS > 1:
    print("bed timing factors (counts relative to bed 1):", [round(float(r), 6) for r in REL])
