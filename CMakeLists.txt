cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(DICOMSpectRecon)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/4burakfe/SlicerDICOMSpectRecon")
set(EXTENSION_CATEGORY "Nuclear Medicine")
set(EXTENSION_CONTRIBUTORS "Burak Demir")
set(EXTENSION_DESCRIPTION "SPECT reconstruction from NM tomographic projections in the DICOM browser: OSEM with CT attenuation and energy-window scatter correction (PyTomography), gated studies, interactive preview, and Bq/ml and SUV quantification with camera calibration.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/4burakfe/SlicerDICOMSpectRecon/main/DICOMSpectRecon/Resources/Icons/DICOMSpectRecon.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/4burakfe/SlicerDICOMSpectRecon/main/screenshot.jpg https://raw.githubusercontent.com/4burakfe/SlicerDICOMSpectRecon/main/screenshot2.jpg")
set(EXTENSION_DEPENDS PyTorch)

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(DICOMSpectRecon)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
