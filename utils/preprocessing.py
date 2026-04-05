"""
utils/preprocessing.py
──────────────────────
ForgeNet-X — Image Preprocessing Module  (Windows-compatible)

Pipeline
────────
  Step 1  Load original image
  Step 2  Grayscale conversion
  Step 3  Gaussian blur  (noise removal)
  Step 4  Otsu binarisation  (ink = 255, background = 0)
  Step 5  Morphological opening  (remove specks)
  Step 6  Resize to TARGET_WIDTH  (aspect-preserving)
  Step 7  Normalise to [0, 1] float

Visual confirmation
───────────────────
  save_pipeline_visuals()  writes 5 labelled PNG files so every stage
  can be shown side-by-side in the PDF report.
  Each image carries a white banner across the top with the step name,
  a short description, and (for binary stages) a pixel-count overlay.
"""

import os
import cv2
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_WIDTH  = 1024
BLUR_KERNEL   = (3, 3)
MORPH_KERNEL  = np.ones((2, 2), np.uint8)

# Banner drawn on each visual
BANNER_H      = 38          # px  height of the top label banner
BANNER_COLOR  = (26, 26, 46)   # BGR  dark navy
TEXT_COLOR    = (255, 255, 255)
FONT          = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> dict:
    """
    Full preprocessing pipeline.

    Returns
    -------
    dict  {original, gray, denoised, binary, normalized, resized,
           height, width, path}
    """
    # 1. Load
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # 2. Grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # 3. Denoise
    denoised = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)

    # 4. Binarise  (ink = 255)
    _, binary = cv2.threshold(denoised, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 5. Morphological clean
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, MORPH_KERNEL)

    # 6. Resize
    h, w    = binary.shape
    scale   = TARGET_WIDTH / w
    new_h   = int(h * scale)
    resized = cv2.resize(binary, (TARGET_WIDTH, new_h),
                         interpolation=cv2.INTER_AREA)

    # 7. Normalise
    normalized = resized.astype(np.float32) / 255.0

    return {
        "original"   : original,
        "gray"       : gray,
        "denoised"   : denoised,
        "binary"     : binary,
        "normalized" : normalized,
        "resized"    : resized,
        "height"     : new_h,
        "width"      : TARGET_WIDTH,
        "path"       : str(image_path),
    }


def save_pipeline_visuals(processed: dict, output_dir: str,
                          base_name: str) -> dict:
    """
    Save one annotated PNG per preprocessing stage.

    Each file has a coloured top banner with:
      • Step number  +  stage name
      • One-line description
      • Image dimensions  +  (for binary stages) ink-pixel percentage

    Parameters
    ----------
    processed   : dict  —  output of preprocess_image()
    output_dir  : str   —  folder to write files into
    base_name   : str   —  stem used for filenames  (e.g. the upload UUID)

    Returns
    -------
    dict  {stage_key: absolute_file_path, …}
        Keys: "original", "gray", "denoised", "binary", "resized"
    """
    os.makedirs(output_dir, exist_ok=True)
    stem  = os.path.splitext(base_name)[0]
    paths = {}

    # ── Stage definitions ─────────────────────────────────────────────────────
    # (key, display_name, short_description, accent_BGR, source_key, is_binary)
    stages = [
        ("original",  "Step 1 — Original Image",
         "Raw input loaded from disk  •  BGR colour space",
         (52, 152, 219),   "original",  False),

        ("gray",      "Step 2 — Grayscale Conversion",
         "cv2.COLOR_BGR2GRAY  •  Single-channel luminance",
         (39, 174, 96),    "gray",      False),

        ("denoised",  "Step 3 — Gaussian Blur  (Denoising)",
         f"GaussianBlur kernel {BLUR_KERNEL}  •  Removes salt-and-pepper noise",
         (241, 196, 15),   "denoised",  False),

        ("binary",    "Step 4 — Otsu Binarisation  +  Morphological Clean",
         "THRESH_BINARY_INV + THRESH_OTSU  •  Ink = white (255) / BG = black (0)",
         (231, 76, 60),    "binary",    True),

        ("resized",   "Step 5 — Resized Binary  (Working Image)",
         f"Aspect-preserving resize to {TARGET_WIDTH} px wide  •  Used for segmentation",
         (155, 89, 182),   "resized",   True),
    ]

    for key, title, desc, accent, src_key, is_binary in stages:
        img = processed[src_key]

        # Convert to 3-channel BGR for annotation
        if img.ndim == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()

        # Scale down if very large (keep PDF manageable)
        vis = _scale_to_max_width(vis, 1024)

        h, w = vis.shape[:2]

        # Build top banner
        banner = np.zeros((BANNER_H, w, 3), dtype=np.uint8)
        banner[:] = BANNER_COLOR
        # Accent left bar (4 px wide)
        banner[:, :4] = accent

        # Title text
        cv2.putText(banner, title, (12, 24),
                    FONT, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

        # Description  +  stats in smaller font
        stats = f"  |  {w}×{h} px"
        if is_binary:
            ink_pct = round(np.sum(img > 0) / img.size * 100, 1)
            stats += f"  |  Ink pixels: {ink_pct}%"
        cv2.putText(banner, desc + stats, (12, 36),
                    FONT, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

        annotated = np.vstack([banner, vis])

        fname = f"pp_{key}_{stem}.png"
        fpath = os.path.join(output_dir, fname)
        cv2.imwrite(fpath, annotated)
        paths[key] = fpath

    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Supporting utilities
# ──────────────────────────────────────────────────────────────────────────────

def deskew(image: np.ndarray) -> np.ndarray:
    """Correct rotational skew using min-area bounding rect of ink pixels."""
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    h, w = image.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def extract_writing_style(processed: dict) -> dict:
    """Return stroke-width, slant angle, and line-spacing estimates."""
    binary = processed["resized"]

    dist    = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    nonzero = dist[dist > 0]
    avg_stroke = float(np.mean(nonzero)) * 2 if len(nonzero) else 0.0

    edges  = cv2.Canny(binary, 50, 150)
    lines  = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            deg = np.degrees(theta) - 90
            if -45 < deg < 45:
                angles.append(deg)
    slant = float(np.mean(angles)) if angles else 0.0

    h_proj      = np.sum(binary, axis=1)
    baseline    = np.max(h_proj) * 0.1
    in_text     = h_proj > baseline
    transitions = np.diff(in_text.astype(int))
    gap_starts  = np.where(transitions == -1)[0]
    gap_ends    = np.where(transitions ==  1)[0]
    if len(gap_starts) > 1 and len(gap_ends) > 1:
        spacings     = gap_ends[1:len(gap_starts)] - gap_starts[:len(gap_ends)-1]
        line_spacing = float(np.mean(spacings)) if len(spacings) else 0.0
    else:
        line_spacing = 0.0

    return {
        "avg_stroke_width"   : round(avg_stroke,   2),
        "estimated_slant_deg": round(slant,         2),
        "line_spacing_px"    : round(line_spacing,  2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _scale_to_max_width(img: np.ndarray, max_w: int) -> np.ndarray:
    """Downscale image if wider than max_w, preserving aspect ratio."""
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
