"""Helper package of the DICOMSpectRecon and SPECTCalibration modules.

    nmdata          NM header, energy windows, gating, CT cleanup, acquisition metadata (numpy/pydicom only)
    reconstruction  PyTomography OSEM, count scaling, post-filters (no Slicer)
    quantification  calibration factors (cps/MBq), Bq/ml, SUV, calibration library JSON (no Slicer)
    display         numpy -> in-memory preview images
    dialog          the preview/settings window
"""
