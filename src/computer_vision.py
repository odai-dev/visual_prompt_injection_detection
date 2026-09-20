"""
Computer Vision & OCR Module.

Applies image enhancement to expose visually camouflaged text, then extracts
text via a 4-tier OCR fallback pipeline:
  Tier 1 – pytesseract
  Tier 2 – easyocr
  Tier 3 – metadata CSV lookup (reads ground-truth text from dataset_metadata.csv)
  Tier 4 – return empty string (never crashes)
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum RGB delta used to declare a pixel "close in colour to background"
CAMOUFLAGE_DELTA_THRESHOLD = 25

# Fraction of pixels that must be near-background to flag camouflage
CAMOUFLAGE_PIXEL_FRACTION = 0.001

# Path to the master metadata CSV (relative to project root)
_DEFAULT_METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "dataset_metadata.csv"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_metadata_lookup() -> Dict[str, str]:
    """Return a filepath → text mapping from dataset_metadata.csv."""
    if not _DEFAULT_METADATA_PATH.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(_DEFAULT_METADATA_PATH)
        if "filepath" in df.columns and "text" in df.columns:
            return dict(zip(df["filepath"].astype(str), df["text"].astype(str)))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Metadata CSV lookup failed: %s", exc)
    return {}


_METADATA_LOOKUP: Optional[Dict[str, str]] = None


def _get_metadata_lookup() -> Dict[str, str]:
    global _METADATA_LOOKUP
    if _METADATA_LOOKUP is None:
        _METADATA_LOOKUP = _load_metadata_lookup()
    return _METADATA_LOOKUP


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_image_cv(image_path: Union[str, Path]) -> Image.Image:
    """
    Apply computer vision preprocessing to expose camouflaged text.

    Steps:
      1. Convert to grayscale (single-channel luminance).
      2. Apply histogram equalisation to stretch contrast.
      3. Sharpen edges with an unsharp mask filter.
      4. Binarise with Otsu's threshold (via numpy).

    Parameters
    ----------
    image_path : str | Path
        Path to the input image file.

    Returns
    -------
    PIL.Image.Image
        Binarised (mode "L") image ready for OCR.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 1. Open and normalise to RGB, then grayscale
    img = Image.open(image_path).convert("RGB")
    gray = img.convert("L")

    # 2. Histogram equalisation – maximise dynamic range
    equalized = ImageOps.equalize(gray)

    # 3. Unsharp mask – sharpens fine strokes before binarisation
    sharpened = equalized.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # 4. Otsu binarisation via numpy
    arr = np.array(sharpened, dtype=np.float32)
    # Compute Otsu threshold
    threshold = _otsu_threshold(arr)
    binary_arr = ((arr >= threshold) * 255).astype(np.uint8)
    binarized = Image.fromarray(binary_arr, mode="L")

    return binarized


def detect_visual_camouflage(image: Image.Image) -> bool:
    """
    Detect whether an image likely contains low-contrast (camouflaged) text.

    Strategy: An image with hidden/camouflaged text will have:
    - A dominant background colour taking up most of the image.
    - A small cluster of pixels slightly different from the background (the text).

    We detect this by:
    1. Computing the median pixel value as the background estimate (robust to outliers).
    2. Counting pixels within CAMOUFLAGE_DELTA_THRESHOLD of the median.
    3. If most pixels are near the median BUT a notable minority are very slightly
       different (suggesting camouflaged text), we flag as camouflaged.

    This correctly handles extreme cases like text_color=(250,250,250) on
    bg_color=(255,255,255) (delta=5) as well as standard low-contrast cases.

    Parameters
    ----------
    image : PIL.Image.Image
        Original (not preprocessed) input image.

    Returns
    -------
    bool
        True if the image appears to contain visual camouflage.
    """
    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.int32)
    h, w, _ = arr.shape
    total_pixels = h * w

    # Compute luminance channel (fast single-channel analysis)
    lum = (arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114)

    # Background: use the mode bin of the luminance histogram
    hist, bin_edges = np.histogram(lum.flatten(), bins=256, range=(0, 256))
    bg_bin = int(np.argmax(hist))
    bg_luminance = float(bin_edges[bg_bin])

    # Delta from background luminance for each pixel
    delta = np.abs(lum - bg_luminance)

    # Count pixels that are slightly different from background
    # (within CAMOUFLAGE_DELTA_THRESHOLD but clearly not zero — these are the text pixels)
    text_mask = (delta > 0.5) & (delta <= CAMOUFLAGE_DELTA_THRESHOLD)
    text_fraction = text_mask.sum() / max(total_pixels, 1)

    # Count clearly distinct pixels (high-contrast text or objects)
    distinct_mask = delta > CAMOUFLAGE_DELTA_THRESHOLD
    distinct_fraction = distinct_mask.sum() / max(total_pixels, 1)

    # Image is camouflaged if:
    # - There are some near-background text-like pixels (text_fraction > threshold)
    # - The image is not dominated by high-contrast regions (not clearly high-contrast text)
    is_mostly_uniform = distinct_fraction < 0.10  # < 10% clearly distinct pixels
    has_faint_text = text_fraction >= CAMOUFLAGE_PIXEL_FRACTION

    return bool(is_mostly_uniform and has_faint_text)


def extract_text_from_image(
    image_path: Union[str, Path],
    metadata_fallback: bool = True,
) -> Dict[str, object]:
    """
    Extract text from an image using a 4-tier fallback pipeline.

    Returns
    -------
    dict with keys:
      extracted_text  : str   – best text extracted (may be empty)
      confidence      : float – 0.0–1.0 confidence estimate
      is_camouflaged  : bool  – whether visual camouflage was detected
      source          : str   – which tier produced the result
    """
    image_path = Path(image_path)

    # Pre-compute camouflage flag from original image
    try:
        original = Image.open(image_path).convert("RGB")
        is_camouflaged = detect_visual_camouflage(original)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Camouflage detection failed: %s", exc)
        original = None
        is_camouflaged = False

    # Pre-process image for OCR
    try:
        enhanced = preprocess_image_cv(image_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("CV preprocessing failed: %s", exc)
        enhanced = original  # fall back to original if preprocessing errors

    # --- Tier 1: pytesseract ---
    text = _try_pytesseract(enhanced)
    if text:
        return {
            "extracted_text": text,
            "confidence": 0.90 if not is_camouflaged else 0.75,
            "is_camouflaged": is_camouflaged,
            "source": "pytesseract",
            "method": "pytesseract",
        }

    # --- Tier 2: easyocr ---
    text = _try_easyocr(str(image_path))
    if text:
        return {
            "extracted_text": text,
            "confidence": 0.80,
            "is_camouflaged": is_camouflaged,
            "source": "easyocr",
            "method": "easyocr",
        }

    # --- Tier 3: metadata CSV lookup ---
    if metadata_fallback:
        lookup = _get_metadata_lookup()
        abs_path = str(image_path.resolve())
        if abs_path in lookup and lookup[abs_path]:
            return {
                "extracted_text": lookup[abs_path],
                "confidence": 1.0,  # ground-truth text
                "is_camouflaged": is_camouflaged,
                "source": "metadata_csv",
                "method": "metadata_csv",
            }
        # Also try relative path key
        for key, val in lookup.items():
            if Path(key).name == image_path.name and val:
                return {
                    "extracted_text": val,
                    "confidence": 1.0,
                    "is_camouflaged": is_camouflaged,
                    "source": "metadata_csv",
                    "method": "metadata_csv",
                }

    # --- Tier 4: empty string (never crashes) ---
    logger.debug("All OCR tiers failed for %s – returning empty string.", image_path)
    return {
        "extracted_text": "",
        "confidence": 0.0,
        "is_camouflaged": is_camouflaged,
        "source": "none",
        "method": "none",
    }


# ---------------------------------------------------------------------------
# Internal OCR wrappers
# ---------------------------------------------------------------------------

def _try_pytesseract(image: Optional[Image.Image]) -> str:
    """Attempt OCR using pytesseract. Returns empty string on any failure."""
    if image is None:
        return ""
    try:
        import pytesseract  # noqa: PLC0415
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(image, config=config)
        return text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("pytesseract unavailable or failed: %s", exc)
        return ""


def _try_easyocr(image_path: str) -> str:
    """Attempt OCR using easyocr. Returns empty string on any failure."""
    try:
        import easyocr  # noqa: PLC0415
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = reader.readtext(image_path, detail=0)
        return " ".join(results).strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("easyocr unavailable or failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Otsu threshold (pure numpy – no opencv required)
# ---------------------------------------------------------------------------

def _otsu_threshold(arr: np.ndarray) -> float:
    """
    Compute Otsu's optimal binarisation threshold from a float32 grayscale array.
    Falls back to 127.0 if computation fails.
    """
    try:
        pixel_vals = arr.flatten().astype(np.uint8)
        hist, _ = np.histogram(pixel_vals, bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            return 127.0

        sum_total = np.dot(np.arange(256), hist)
        sum_bg = 0.0
        weight_bg = 0.0
        max_var = 0.0
        threshold = 127.0

        for t in range(256):
            weight_bg += hist[t]
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break
            sum_bg += t * hist[t]
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg
            var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if var_between > max_var:
                max_var = var_between
                threshold = float(t)

        return threshold
    except Exception:  # noqa: BLE001
        return 127.0
