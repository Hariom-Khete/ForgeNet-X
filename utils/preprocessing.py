"""
utils/preprocessing.py
──────────────────────
ForgeNet-X — Image Preprocessing Module  (Windows-compatible)

WHY OTSU FAILS ON HANDWRITING IMAGES
──────────────────────────────────────
Otsu's method is a *global* algorithm: it picks one single threshold
for the entire image by maximising between-class variance.  This works
well only when the image histogram is cleanly bimodal (two sharp peaks —
one for background, one for ink).  Real handwriting photos break all of
Otsu's assumptions:

  1. UNEVEN ILLUMINATION — phone photos have shadows in corners and
     brighter centres.  Otsu picks a single midpoint that is too dark
     for bright areas → paper texture gets classified as ink.

  2. PAPER TEXTURE — ruled lines, watermarks, and surface grain have
     grey values that sit between background and ink.  Otsu's single
     threshold drags them into the ink class.

  3. SMALL INK FRACTION — ink pixels are a minority (~5–15 % of the
     image).  Otsu is biased toward the dominant (background) class,
     so the boundary is set too low → noise lifts above it.

  References
  ──────────
  • Otsu (1979) — "A threshold selection method from gray-level
    histograms", IEEE Trans. Systems, Man, and Cybernetics 9(1).
  • Sauvola & Pietikäinen (2000) — "Adaptive document image
    binarization", Pattern Recognition 33(2) 225–236.
  • Shafait et al. (2008) — "Efficient implementation of local adaptive
    thresholding using integral images", SPIE Doc. Recog. & Retr. XV.
  • Sezgin & Sankur (2004) — "Survey over image thresholding techniques
    and quantitative performance evaluation", J. Electronic Imaging 13(1).

SOLUTION: LAYERED ADAPTIVE PIPELINE
──────────────────────────────────────
Stage 1  Bilateral filter       — edge-preserving denoising (keeps ink
                                  edges sharp while smoothing paper texture)
Stage 2  CLAHE                  — contrast-limited adaptive histogram
                                  equalisation: normalises local contrast
                                  so bright/dark zones become comparable
Stage 3  Adaptive Gaussian      — local threshold per pixel using a
   threshold (primary)            Gaussian-weighted neighbourhood mean
                                  (cv2.ADAPTIVE_THRESH_GAUSSIAN_C).
                                  Each pixel gets its own threshold so
                                  shadow/lighting variation is neutralised.
Stage 4  Otsu on CLAHE image    — secondary global threshold applied to
   (secondary, blended)           the contrast-normalised image; AND-ed
                                  with Stage 3 so only pixels that *both*
                                  methods agree are ink are kept → reduces
                                  false positives from either method alone.
Stage 5  Morphological clean    — opening (remove isolated specks) then
                                  closing (connect broken strokes)
Stage 6  Connected-component    — remove any surviving blob whose area is
   area filter                    below MIN_BLOB_AREA px².  Paper grain
                                  produces tiny single-pixel or 2-pixel
                                  blobs; real ink strokes are much larger.
Stage 7  Resize                 — aspect-preserving scale to TARGET_WIDTH
Stage 8  Normalise              — float32 [0, 1]
"""

import os
import cv2
import numpy as np


# ── Tuning knobs ──────────────────────────────────────────────────────────────
TARGET_WIDTH    = 1024

# Bilateral filter — preserves ink edges while smoothing paper grain
BILATERAL_D         = 9       # diameter of pixel neighbourhood
BILATERAL_SIGMA_COL = 75      # colour sigma (higher = more colour smoothing)
BILATERAL_SIGMA_SPC = 75      # space sigma  (higher = farther pixels mix)

# CLAHE — contrast normalisation
CLAHE_CLIP_LIMIT    = 2.0     # higher → more contrast enhancement (max ~4.0)
CLAHE_TILE_GRID     = (8, 8)  # grid size for local histogram equalisation

# Adaptive threshold
ADAPT_BLOCK_SIZE    = 31      # neighbourhood window (must be odd).
                              # Larger = handles wider lighting gradients.
                              # Smaller = more sensitive to local detail.
ADAPT_C             = 10      # constant subtracted from mean.
                              # Higher = stricter (fewer false positives).
                              # Lower  = more sensitive (fewer missed strokes).

# Morphological kernels
MORPH_OPEN_K  = np.ones((2, 2), np.uint8)   # remove specks
MORPH_CLOSE_K = np.ones((2, 2), np.uint8)   # close tiny gaps in strokes

# Connected-component noise filter
MIN_BLOB_AREA  = 20   # px² — blobs smaller than this are paper/noise, not ink

# Visual annotation
BANNER_H     = 44
BANNER_BG    = (26, 26, 46)
TEXT_WHITE   = (255, 255, 255)
TEXT_GREY    = (170, 170, 170)
FONT         = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> dict:
    """
    Full adaptive binarisation pipeline.

    Returns
    -------
    dict with keys:
        original    – BGR image as loaded
        gray        – single-channel luminance
        bilateral   – edge-preserving denoised grayscale
        clahe       – contrast-normalised grayscale
        adaptive    – adaptive-threshold binary (ink=255)
        otsu_mask   – Otsu binary on CLAHE image (ink=255)
        binary      – final combined + cleaned binary
        normalized  – float32 [0,1] version of binary
        resized     – binary scaled to TARGET_WIDTH
        height      – resized image height
        width       – TARGET_WIDTH
        path        – input path string
        otsu_value  – the Otsu threshold value Otsu found (for reporting)
    """
    # ── Stage 1: Load ─────────────────────────────────────────────────────────
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # ── Stage 2: Grayscale ────────────────────────────────────────────────────
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # ── Stage 3: Bilateral filter (edge-preserving denoising) ─────────────────
    # Unlike Gaussian blur which blurs ink edges too, bilateral filter
    # preserves high-contrast edges (ink strokes) while averaging out
    # low-contrast texture (paper grain, watermarks).
    bilateral = cv2.bilateralFilter(gray,
                                    BILATERAL_D,
                                    BILATERAL_SIGMA_COL,
                                    BILATERAL_SIGMA_SPC)

    # ── Stage 4: CLAHE (contrast-limited adaptive histogram equalisation) ──────
    # Divides the image into tiles and equalises each tile's histogram
    # independently.  This makes dark-shadow zones and bright-highlight zones
    # comparable before thresholding — something global normalisation cannot do.
    clahe_op = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                                tileGridSize=CLAHE_TILE_GRID)
    clahe = clahe_op.apply(bilateral)

    # ── Stage 5a: Adaptive Gaussian threshold (primary binariser) ─────────────
    # Computes a LOCAL threshold for every pixel based on the Gaussian-weighted
    # mean of its ADAPT_BLOCK_SIZE × ADAPT_BLOCK_SIZE neighbourhood minus ADAPT_C.
    # This completely neutralises global lighting variation.
    adaptive = cv2.adaptiveThreshold(
        clahe, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPT_BLOCK_SIZE,
        ADAPT_C
    )

    # ── Stage 5b: Otsu on CLAHE (secondary, for AND-blending) ─────────────────
    # Run Otsu on the CLAHE-normalised image (not the raw gray), so it sees a
    # better-distributed histogram.  Used only as a second vote — pixels must
    # pass BOTH adaptive AND Otsu to survive.  This removes adaptive's occasional
    # false positives in very uniform bright regions.
    otsu_val, otsu_mask = cv2.threshold(
        clahe, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # ── Stage 6: AND-blend — keep only consensus ink pixels ───────────────────
    # A pixel must be called "ink" by BOTH methods.
    # Adaptive alone → misses nothing but adds paper texture noise.
    # Otsu alone      → misses nothing in uniform regions but fails in shadows.
    # AND             → tight intersection eliminates most false positives.
    combined = cv2.bitwise_and(adaptive, otsu_mask)

    # ── Stage 7: Morphological clean ──────────────────────────────────────────
    # Opening: erode then dilate — removes isolated single/double-pixel specks
    # that survived the AND (usually paper grain on uniform bright areas).
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  MORPH_OPEN_K)
    # Closing: dilate then erode — reconnects slightly broken ink strokes.
    binary = cv2.morphologyEx(opened,   cv2.MORPH_CLOSE, MORPH_CLOSE_K)

    # ── Stage 8: Connected-component area filter ───────────────────────────────
    # Any surviving blob with area < MIN_BLOB_AREA px² is paper noise, not ink.
    # Real handwriting strokes are always larger.
    binary = _remove_small_blobs(binary, MIN_BLOB_AREA)

    # ── Stage 9: Resize ───────────────────────────────────────────────────────
    h, w    = binary.shape
    scale   = TARGET_WIDTH / w
    new_h   = int(h * scale)
    resized = cv2.resize(binary, (TARGET_WIDTH, new_h),
                         interpolation=cv2.INTER_AREA)
    # Re-binarise after resize interpolation artefacts
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    # ── Stage 10: Normalise ───────────────────────────────────────────────────
    normalized = resized.astype(np.float32) / 255.0

    return {
        "original"  : original,
        "gray"      : gray,
        "bilateral" : bilateral,
        "clahe"     : clahe,
        "adaptive"  : adaptive,
        "otsu_mask" : otsu_mask,
        "binary"    : binary,
        "normalized": normalized,
        "resized"   : resized,
        "height"    : new_h,
        "width"     : TARGET_WIDTH,
        "path"      : str(image_path),
        "otsu_value": int(otsu_val),
    }


def save_pipeline_visuals(processed: dict, output_dir: str,
                          base_name: str) -> dict:
    """
    Save one annotated PNG per preprocessing stage.

    Returns
    -------
    dict  { stage_key: absolute_file_path }
    """
    os.makedirs(output_dir, exist_ok=True)
    stem  = os.path.splitext(base_name)[0]
    paths = {}

    # (key, display_title, description, accent_BGR, is_binary)
    stages = [
        ("original",
         "Stage 1 — Original Image",
         "Raw input loaded from disk  •  BGR colour space  •  No processing",
         (52, 152, 219), False),

        ("gray",
         "Stage 2 — Grayscale Conversion",
         "cv2.COLOR_BGR2GRAY  •  Single-channel luminance",
         (39, 174, 96), False),

        ("bilateral",
         "Stage 3 — Bilateral Filter  (Edge-Preserving Denoise)",
         f"bilateralFilter d={BILATERAL_D} σ_color={BILATERAL_SIGMA_COL} σ_space={BILATERAL_SIGMA_SPC}"
         "  •  Smooths paper texture, keeps ink edges sharp",
         (241, 196, 15), False),

        ("clahe",
         "Stage 4 — CLAHE  (Contrast-Limited Adaptive Histogram Equalisation)",
         f"clipLimit={CLAHE_CLIP_LIMIT}  tileGrid={CLAHE_TILE_GRID}"
         "  •  Normalises local contrast — neutralises shadow/highlight zones",
         (230, 126, 34), False),

        ("adaptive",
         "Stage 5a — Adaptive Gaussian Threshold  (Primary Binariser)",
         f"blockSize={ADAPT_BLOCK_SIZE}  C={ADAPT_C}  BINARY_INV"
         "  •  Per-pixel local threshold — immune to global lighting variation",
         (231, 76, 60), True),

        ("otsu_mask",
         f"Stage 5b — Otsu Threshold on CLAHE  (Secondary Vote)  "
         f"[T={processed.get('otsu_value','?')}]",
         "THRESH_BINARY_INV + THRESH_OTSU on contrast-normalised image"
         "  •  Used only as 2nd vote; AND-blended with adaptive",
         (155, 89, 182), True),

        ("binary",
         "Stage 6 — Final Binary  (AND-blend + Morph + Blob Filter)",
         f"AND(adaptive, otsu)  →  morphOpen  →  morphClose"
         f"  →  blob area filter (min {MIN_BLOB_AREA}px²)",
         (26, 188, 156), True),

        ("resized",
         "Stage 7 — Resized Working Image",
         f"Aspect-preserving resize to {TARGET_WIDTH}px wide"
         "  •  Input for all downstream segmentation",
         (236, 95, 128), True),
    ]

    for key, title, desc, accent, is_binary in stages:
        img = processed.get(key)
        if img is None:
            continue

        # Convert to 3-channel BGR for annotation
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        vis = _scale_to_max_width(vis, 1024)
        h, w = vis.shape[:2]

        # Build banner
        banner = np.zeros((BANNER_H, w, 3), dtype=np.uint8)
        banner[:] = BANNER_BG
        banner[:, :5] = accent   # left accent stripe

        cv2.putText(banner, title,
                    (12, 20), FONT, 0.50, TEXT_WHITE, 1, cv2.LINE_AA)

        stats = f"  |  {w}×{h}px"
        if is_binary:
            ink_pct = round(float(np.sum(img > 0)) / img.size * 100, 2)
            stats  += f"  |  ink={ink_pct}%"
        cv2.putText(banner, desc + stats,
                    (12, 38), FONT, 0.34, TEXT_GREY, 1, cv2.LINE_AA)

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
    """Stroke-width, slant, and line-spacing estimates from the final binary."""
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

def _remove_small_blobs(binary: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove connected components whose area is below `min_area` pixels.
    These are paper-grain / noise blobs that survived morphological cleaning.
    Uses cv2.connectedComponentsWithStats for efficiency.
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    clean = np.zeros_like(binary)
    for label in range(1, n_labels):   # label 0 = background
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == label] = 255
    return clean


def _scale_to_max_width(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
