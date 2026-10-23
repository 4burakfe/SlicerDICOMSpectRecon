"""Preview rendering: numpy slices -> RGB arrays -> QPixmap, entirely in memory.

Slices are turned into 8-bit RGB with numpy, wrapped in a binary PPM header and
decoded by Qt's built-in PPM reader (QImage.fromData). There is no disk access and
no compression, and no Qt pixmap cache that could return a stale image.
If that path does not work in a given Slicer/PythonQt build, the renderer falls back
once, permanently, to the previous route through PNG files in a temporary folder.
"""

import logging
import os
import shutil
import tempfile
import traceback

import numpy as np
import qt

# Previews are windowed 0 to WINDOW_FRACTION x the DISPLAY_PERCENTILE of the
# non-zero voxels, so a single hot focus does not darken everything else
DISPLAY_PERCENTILE = 99.75
WINDOW_FRACTION = 1.0

# CT preview window (abdomen)
CT_LEVEL = 40.0
CT_WIDTH = 400.0


# ---------------------------------------------------------------------------
# Intensity -> colour
# ---------------------------------------------------------------------------
def displayReference(arr, percentile=DISPLAY_PERCENTILE):
    """Intensity the display window is scaled against: a high percentile of the
    non-zero voxels rather than the maximum, so outlier voxels do not set the scale."""
    if arr is None or not arr.size:
        return 1.0
    values = arr[arr > 0]
    if not values.size:
        return float(arr.max()) or 1.0
    reference = float(np.percentile(values, percentile))
    if reference <= 0:
        reference = float(arr.max()) or 1.0
    return reference


def hotTable():
    """Black -> red -> yellow -> white ramp, 256 x RGB uint8."""
    x = np.linspace(0, 1, 256)
    table = np.stack([np.clip(3 * x, 0, 1), np.clip(3 * x - 1, 0, 1), np.clip(3 * x - 2, 0, 1)], axis=-1)
    return (table * 255).astype(np.uint8)


def colorTableFromNode(node):
    """256-entry RGB table sampled from a Slicer colour node (procedural or lookup table)."""
    function = None
    if hasattr(node, "GetColorTransferFunction"):
        function = node.GetColorTransferFunction()
    if function is None:
        function = node.GetLookupTable()
    if function is None:
        return None
    low, high = function.GetRange()
    table = np.zeros((256, 3), dtype=np.uint8)
    rgb = [0.0, 0.0, 0.0]
    for i in range(256):
        function.GetColor(low + (high - low) * i / 255.0, rgb)
        table[i] = [int(round(255 * c)) for c in rgb]
    return table


def petColorTable():
    """Slicer's PET-DICOM colours, so the preview matches the loaded volume; hot ramp if unavailable."""
    try:
        import slicer
        node = slicer.mrmlScene.GetFirstNodeByName("PET-DICOM")
        if node is not None:
            table = colorTableFromNode(node)
            if table is not None:
                return table
    except Exception:
        logging.warning("Could not read the PET-DICOM colour node:\n" + traceback.format_exc())
    return hotTable()


def _normalized(activitySlice, vmax, windowFraction=WINDOW_FRACTION, lower=0.0):
    """0..1 between lower x vmax and windowFraction x vmax."""
    scale = vmax if vmax > 0 else 1.0
    top, bottom = scale * windowFraction, scale * lower
    return np.clip((activitySlice - bottom) / max(top - bottom, 1e-12 * scale), 0.0, 1.0)


def colorize(activitySlice, vmax, table, windowFraction=WINDOW_FRACTION, lower=0.0):
    """Activity slice -> RGB uint8 [h, w, 3] through a colour table."""
    indices = (_normalized(activitySlice, vmax, windowFraction, lower) * 255.0).astype(np.uint8)
    return table[indices]


def grayscale(arr, vmax, windowFraction=1.0, lower=0.0):
    """Scalar slice scaled 0..vmax (or the given window of it) -> grey RGB uint8 [h, w, 3]."""
    gray = (_normalized(arr, vmax, windowFraction, lower) * 255.0).astype(np.uint8)
    return np.repeat(gray[..., np.newaxis], 3, axis=-1)


def ctGray(ctSlice, level=CT_LEVEL, width=CT_WIDTH):
    """CT slice in HU -> 0..1 with the given window."""
    return np.clip((ctSlice - (level - width / 2.0)) / width, 0.0, 1.0)


def fuse(activitySlice, vmax, ctGraySlice, opacity, windowFraction=WINDOW_FRACTION, lower=0.0):
    """Hot colour activity over grey CT, opacity weighted by activity -> RGB uint8."""
    activity = _normalized(activitySlice, vmax, windowFraction, lower)
    hot = np.stack([np.clip(3 * activity, 0, 1),
                    np.clip(3 * activity - 1, 0, 1),
                    np.clip(3 * activity - 2, 0, 1)], axis=-1)
    alpha = float(opacity) * activity[..., np.newaxis]
    base = np.repeat(ctGraySlice[..., np.newaxis], 3, axis=-1)
    return np.clip((base * (1.0 - alpha) + hot * alpha) * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Bicubic resampling of the scalar slices, before the colours
# ---------------------------------------------------------------------------
_cubicCache = {}


def _cubicWeights(n, m, a=-0.5):
    """[m, n] matrix resampling n samples to m with Keys' cubic convolution (bicubic, a = -0.5),
    pixel centres aligned and edges repeated. Rows sum to 1, so flat regions keep their value."""
    key = (n, m)
    if key not in _cubicCache:
        x = (np.arange(m) + 0.5) * n / m - 0.5
        base = np.floor(x).astype(int)
        weights = np.zeros((m, n))
        rows = np.arange(m)
        for k in range(-1, 3):
            d = np.abs(x - (base + k))
            w = np.where(d <= 1, (a + 2) * d ** 3 - (a + 3) * d ** 2 + 1,
                         np.where(d < 2, a * d ** 3 - 5 * a * d ** 2 + 8 * a * d - 4 * a, 0.0))
            np.add.at(weights, (rows, np.clip(base + k, 0, n - 1)), w)
        _cubicCache[key] = weights
    return _cubicCache[key]


def fitSize(shape, size):
    """(height, width) of a [h, w] slice scaled to fit size x size, keeping its aspect ratio."""
    h, w = shape[0], shape[1]
    scale = float(size) / max(h, w)
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def resampleCubicTo(arr, height, width, minimum=None, maximum=None):
    """Scalar array resampled bicubically to exactly height x width (aspect not kept), e.g. a
    sinogram with few angles stretched to the panel."""
    arr = np.asarray(arr, dtype=np.float64)
    out = _cubicWeights(arr.shape[0], height) @ arr @ _cubicWeights(arr.shape[1], width).T
    if minimum is not None or maximum is not None:
        out = np.clip(out, minimum, maximum)
    return out.astype(np.float32)


def resampleCubic(arr, size, minimum=None, maximum=None):
    """Scalar slice resampled to fit size x size with bicubic interpolation, so the preview is
    interpolated in the data and only then coloured (Qt's smooth scaling would blend the colours).
    The overshoot of the cubic kernel is clipped to [minimum, maximum] when given."""
    arr = np.asarray(arr, dtype=np.float64)
    height, width = fitSize(arr.shape, size)
    if (height, width) == arr.shape[:2]:
        out = arr
    else:
        out = _cubicWeights(arr.shape[0], height) @ arr @ _cubicWeights(arr.shape[1], width).T
    if minimum is not None or maximum is not None:
        out = np.clip(out, minimum, maximum)
    return out.astype(np.float32)


LINE_COLOUR = (255, 230, 0)


def drawLine(rgb, row, rows, colour=LINE_COLOUR):
    """Horizontal reference line over row (of rows) of a fitted RGB image, in place, with a small
    handle in the middle to grab it by."""
    height, width = rgb.shape[0], rgb.shape[1]
    y = int(min(height - 1, max(0, round((row + 0.5) * height / rows - 0.5))))
    rgb[y, :] = colour
    half = max(3, width // 40)
    centre = width // 2
    rgb[max(0, y - 3):y + 4, max(0, centre - half):centre + half + 1] = colour
    return rgb


# ---------------------------------------------------------------------------
# RGB array -> QPixmap
# ---------------------------------------------------------------------------
def ppmBytes(rgb):
    """Binary PPM (P6) encoding of an RGB uint8 array whose row 0 is the top row."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[0], rgb.shape[1]
    return b"P6\n%d %d\n255\n" % (w, h) + rgb.tobytes()


class PixmapRenderer:
    """Turns RGB arrays into QPixmaps of a fixed display size."""

    def __init__(self, size):
        self.size = size
        self.useFiles = not self._inMemoryWorks()
        self.tempDir = None
        if self.useFiles:
            logging.warning("SPECT preview: in-memory image decoding unavailable, using temporary PNG files.")

    # --- in-memory path ---
    @staticmethod
    def _decode(rgb):
        return qt.QImage.fromData(ppmBytes(rgb), "PPM")

    def _inMemoryWorks(self):
        """Decode a 2x2 test image and check size, orientation and channel order."""
        try:
            test = np.array([[[255, 0, 0], [0, 255, 0]],
                             [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
            image = self._decode(test)
            if image is None or image.isNull() or image.width() != 2 or image.height() != 2:
                return False
            return (int(image.pixel(0, 0)) & 0xFFFFFF) == 0xFF0000 and \
                   (int(image.pixel(1, 1)) & 0xFFFFFF) == 0xFFFFFF and \
                   (int(image.pixel(0, 1)) & 0xFFFFFF) == 0x0000FF
        except Exception:
            logging.warning("SPECT preview: in-memory image test failed:\n" + traceback.format_exc())
            return False

    # --- fallback path (previous behaviour) ---
    def _viaFile(self, rgb):
        import vtk
        import vtk.util.numpy_support as numpy_support
        if self.tempDir is None:
            self.tempDir = tempfile.mkdtemp(prefix="spectrecon_preview_")
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h, w = rgb.shape[0], rgb.shape[1]
        vtkImg = vtk.vtkImageData()
        vtkImg.SetDimensions(w, h, 1)
        # VTK images are bottom-up, flip so row 0 is shown at the top
        flipped = np.ascontiguousarray(rgb[::-1])
        scalars = numpy_support.numpy_to_vtk(flipped.reshape(-1, 3), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtkImg.GetPointData().SetScalars(scalars)
        path = os.path.join(self.tempDir, "preview.png")
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputData(vtkImg)
        writer.Write()
        # QImage (not QPixmap(path)) to avoid Qt's pixmap cache returning a stale image
        return qt.QImage(path)

    def toPixmap(self, rgb):
        image = None
        if not self.useFiles:
            image = self._decode(rgb)
            if image is None or image.isNull():
                logging.warning("SPECT preview: in-memory decoding failed, switching to temporary PNG files.")
                self.useFiles = True
                image = None
        if image is None:
            image = self._viaFile(rgb)
        pixmap = qt.QPixmap.fromImage(image)
        return pixmap.scaled(self.size, self.size, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation)

    def cleanup(self):
        if self.tempDir:
            shutil.rmtree(self.tempDir, ignore_errors=True)
            self.tempDir = None
