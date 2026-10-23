#-----------------------------------------------------------------------------
# DICOM SPECT reconstruction: hidden module that registers the DICOM plugin, and its helper package
set(MODULE_NAME DICOMSpectRecon)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/nmdata.py
  ${MODULE_NAME}Lib/reconstruction.py
  ${MODULE_NAME}Lib/quantification.py
  ${MODULE_NAME}Lib/collimators.py
  ${MODULE_NAME}Lib/display.py
  ${MODULE_NAME}Lib/dialog.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  )

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  )

#-----------------------------------------------------------------------------
# SPECT Calibration: camera calibration factors (uses the helper package above)
slicerMacroBuildScriptedModule(
  NAME SPECTCalibration
  SCRIPTS SPECTCalibration.py
  RESOURCES Resources/Icons/SPECTCalibration.png
  )

#-----------------------------------------------------------------------------
# SPECT Collimators: collimator library for resolution recovery (uses the helper package above)
slicerMacroBuildScriptedModule(
  NAME SPECTCollimators
  SCRIPTS SPECTCollimators.py
  RESOURCES Resources/Icons/SPECTCollimators.png
  )
