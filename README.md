# DICOM SPECT Reconstruction (3D Slicer)

Reconstructs SPECT studies straight from the raw NM tomographic projections in Slicer's DICOM browser.
The projection series (ImageType TOMO or GATED TOMO) shows up as a loadable, and Load opens a
reconstruction window with a live preview before anything is added to the scene.

Author: Burak Demir, MD, FEBNM

> **Built on [PyTomography](https://github.com/qurit/PyTomography).** The reconstruction engine of this
> extension - the SPECT system model, attenuation and scatter modelling, and OSEM - is PyTomography,
> developed by the [Quantitative Radiomolecular Imaging and Therapy (Qurit) lab](https://www.qurit.ca/).
> This extension adds the Slicer side: DICOM browsing, CT matching, the preview window and quantification.
> If you use it in your work, please also cite PyTomography:
>
> L. Polson, R. Fedrigo, C. Li, M. Sabouri, O. Dzikunu, S. Ahamed, N. Karakatsanis, S. Kurkowska,
> P. Sheikhzadeh, P. Esquinas, A. Rahmim, C. Uribe. *PyTomography: A Python Library for Medical Image
> Reconstruction.* SoftwareX. Preprint: [arXiv:2309.01977](https://arxiv.org/abs/2309.01977).

![Reconstruction window](screenshot2.jpg)

## Features

- **OSEM** reconstruction with [PyTomography](https://github.com/qurit/PyTomography), with separate
  iterations/subsets for the static and the gated images.
- **BSREM** (block sequential regularised EM, a Bayesian penalised-likelihood method, Q.Clear-like)
  with a relative difference prior: OSEM-like subset updates with a decreasing step, started from one
  OSEM iteration, so it converges and the prior strength beta, not the iteration count, sets the noise
  level. Defaults: 4 iterations, beta 0.2 and gamma 2 for the static image, beta 0.4 and gamma 2 for the
  gates, with the Butterworth post-filter left on.
- **FBP** (filtered backprojection) as an alternative: ramp filter with a Ramp, Shepp-Logan, Cosine,
  Hamming, Hann or Butterworth window and a cutoff in cycles/cm (applied in 2D on the projections, like
  camera software). Scatter is subtracted from the projections and attenuation is corrected with the
  first order Chang method from the CT. The image is in the same units as OSEM; negative values are set
  to 0.
- **Resolution recovery** (OSEM and BSREM): the depth dependent collimator-detector blur is modelled in
  the system matrix (Gaussian PSF, sigma(d) = sqrt((d_hole / L_eff x d + d_hole)^2 + intrinsic^2), with
  L_eff = L - 2 / mu_lead at the photopeak energy), as in vendor resolution recovery. The collimator is
  matched from the DICOM header (manufacturer, camera model, collimator name; e.g. a GE NM/CT 850/860/870
  "LEHRS" gets G8-LHRS: 1.43 mm holes, 0.13 mm septa, 32 mm long) or picked by hand. The collimator
  dimensions come from the collimator data base of the SIMIND Monte Carlo program [1], as shipped with
  PyTomography. It raises contrast
  recovery of small hot spots and needs more iterations to converge; edge ringing (Gibbs) comes with it.
  While it is on, the iterations are multiplied by an "RR iterations" factor (x2 for OSEM, static and
  gated; x2 for BSREM) without changing the iteration boxes; the real count is logged and stored.
- **Motion correction** (orange *Motion correction...* button): a separate window shows a fast
  reconstruction of the axial slice at the yellow line (2D, with the main window's algorithm,
  iterations, subsets, prior or FBP window and post-filter, without attenuation, scatter or resolution
  recovery; computed in the background after 1 s without changes), the projections (in cine), the
  summed sinogram and the linogram on the top row and the motion corrected ones below, with a plot of
  the shift of every view. *Detect* finds axial (up/down) movement by cross-correlating every view's
  axial count profile with the median profile [2], and sideways movement from the transaxial centre of
  mass of the views [3], which follows a sinusoid over the angles until the patient moves (the
  acquisition is split in time where it jumps to another sinusoid, and the views are moved back onto
  the longest one); the views are shifted before reconstruction, as in [3]. See [4] for a comparison of
  such detection methods. The detection and correction were written for this extension; they are not
  a copy of a published or vendor implementation. Shifts can also be added by hand to a single time
  step (the default) or a range of steps, on all detectors or one, and are applied live; changing a
  step stops the cine and shows that view. The mouse wheel steps through the projections, or moves the
  line over the other views. The shifts are applied to every energy window and gate before
  reconstruction; gated studies are detected on the summed gates, so only patient motion (not the
  heart's own motion between gates) is corrected.
- **CT attenuation correction**: a CT of the same study (same frame of reference) is found
  automatically, or picked from a folder; it is also shown fused with the SPECT.
- **Scatter correction** with dual or triple energy windows, chosen from the acquisition's windows.
  The dual window weight is automatic (0.5 x photopeak width / lower window width) or can be set
  by hand, e.g. 1.1 to match GE Xeleris' "Scatter weight for Tc99m".
- **Gated studies**: gated reconstruction in cine plus the summed static image; gates are loaded as a
  sequence. The ECG gating information of the acquisition is kept on the volumes (`SPECTRecon.Gating*`
  attributes): accepted R-R window, accepted and rejected beats, heart rate / mean R-R (estimated from the
  frame time when the scanner leaves it out), frame width, framing and beat rejection settings, and the
  per-view values when the scanner stores one entry per projection view (Mediso), plus the photopeak
  projection counts of every gate (`SPECTRecon.GatePhotopeakCounts`), used by the Gated QC step of the
  Myocardial Perfusion SPECT extension.
- **Post-filters** (Butterworth, Hann, Gaussian with FWHM in mm) and output count scaling.
- **Quantification (optional)**: with a calibration factor (cps/MBq) made in the **SPECT Calibration**
  module, the static image is converted to Bq/ml and optionally to SUV:
  `Bq/ml = acquisition counts / (T x CF) x 1e6 / voxel volume (ml)`, where T is the total view time.
  Calibrations are kept in a JSON library that can live on a shared drive.

![Preview](screenshot.jpg)

## Use

1. Import the study into the DICOM database and select the NM SPECT series.
2. Load opens the reconstruction window: projections in cine, reconstruction and CT fusion previews,
   and the settings (OSEM, BSREM or FBP, attenuation, scatter, filter, scaling, quantification).
3. Press **Reconstruct**, check the preview, then **Load into Slicer**.

Next to the raw projections the window shows a **summed sinogram** (all projection rows summed along the
axis for every view, in the order of the rotation, first view at the top; a sideways patient movement
shows as a jump in the sine curves) and a **linogram** (the axial count profile of every view, views left
to right; an up/down movement shows as a step), in PET-DICOM colours and stretched to the square panel,
with the gates of a gated study summed. They sit in a row below the previews, or in a column on the right
when a CT is shown. A yellow line over the rotating projections (drag it with the left mouse button)
follows the axial slice, and the slice follows it; the projection, linogram and reconstruction windows
are scaled to the maximum on that line. On every preview,
drag with the **middle mouse button** to change the display window (left/right: width, up/down: level);
a middle double-click resets it.

### SPECT Calibration module

1. Reconstruct a phantom of known activity with CT attenuation correction (and scatter correction if wanted).
2. In *Nuclear Medicine > SPECT Calibration* pick that reconstruction, choose the VOI (phantom outline
   from CT with a margin, a segment, or the whole image) and enter the activity and times.
3. Compute and save the calibration factor; the reconstruction window then offers it under
   *Quantification*.

### SPECT Collimators module

*Nuclear Medicine > SPECT Collimators* edits the collimator library used for resolution recovery (a JSON
file next to Slicer's settings). It starts with the parallel-hole collimators that PyTomography ships
(from the SIMIND data base [1]) and shows each one's system resolution at 10 cm / 140 keV to compare with the
datasheet. Add, Duplicate, Delete and edit the cells, then **Save**; **Restore defaults** puts back the
built-in list. *Test header matching* shows which collimator a header (manufacturer, model, collimator
name) gets. Reconstruction windows read the library when they open.

## Requirements

PyTorch comes from Slicer's **PyTorch** extension (installed as a dependency, the module offers to install
the right CPU/CUDA build); PyTomography is installed on first use with PyTorch kept at that build.
A GPU makes reconstruction much faster.

## Layout

```
CMakeLists.txt                 extension (depends on the PyTorch extension)
DICOMSpectRecon/
  DICOMSpectRecon.py           hidden module that registers the DICOM plugin
  SPECTCalibration.py          calibration module (Nuclear Medicine category)
  SPECTCollimators.py          collimator library module (Nuclear Medicine category)
  DICOMSpectReconLib/
    nmdata.py                  NM header, energy windows, gating (incl. ECG R-R / beats), CT matching
    reconstruction.py          PyTomography OSEM, BSREM and FBP, count scaling, post-filters
    quantification.py          calibration factors, Bq/ml, SUV, calibration library
    collimators.py             collimator library, header matching, resolution recovery PSF
    motion.py                  motion detection and projection shifts
    motiondialog.py            motion correction window
    display.py                 in-memory previews
    dialog.py                  reconstruction window
  Resources/Icons/
```

Without building, add the inner `DICOMSpectRecon` folder to *Edit > Application Settings > Modules >
Additional module paths* and restart Slicer.

## Acknowledgements

Reconstruction powered by [PyTomography](https://github.com/qurit/PyTomography) (see the citation at the
top of this page).

## References

1. M. Ljungberg, S.-E. Strand. A Monte Carlo program for the simulation of scintillation camera
   characteristics. *Computer Methods and Programs in Biomedicine* 1989;29(4):257-272.
   [SIMIND](https://www.msf.lu.se/en/research/simind-monte-carlo-program) (source of the collimator
   data base).
2. R.L. Eisner, T. Noever, D. Nowak, W. Carlson, D. Dunn, J. Oates, K. Cloninger, H.A. Liberman,
   R.E. Patterson. Use of cross-correlation function to detect patient motion during SPECT imaging.
   *J Nucl Med* 1987;28(1):97-101. [jnm.snmjournals.org/content/28/1/97](https://jnm.snmjournals.org/content/28/1/97)
3. W.J. Geckle, T.L. Frank, J.M. Links, L.C. Becker. Correction for patient and organ movement in SPECT:
   application to exercise thallium-201 cardiac imaging. *J Nucl Med* 1988;29(4):441-450.
   [jnm.snmjournals.org/content/29/4/441](https://jnm.snmjournals.org/content/29/4/441)
4. J.A. Cooper, P.H. Neumann, B.K. McCandless. Detection of patient motion during tomographic myocardial
   perfusion imaging. *J Nucl Med* 1993;34(8):1341-1348.
   [jnm.snmjournals.org/content/34/8/1341](https://jnm.snmjournals.org/content/34/8/1341)
