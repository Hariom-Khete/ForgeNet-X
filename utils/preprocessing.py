"""ForgeNet-X — Image Preprocessing"""

import os
import cv2
import numpy as np


# ── Tuning ────────────────────────────────────────────────────────────────────
TARGET_WIDTH        = 1024
BILATERAL_D         = 9
BILATERAL_SIGMA_COL = 75
BILATERAL_SIGMA_SPC = 75
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_GRID     = (8, 8)
ADAPT_BLOCK_SIZE    = 31
ADAPT_C             = 10
MORPH_OPEN_K        = np.ones((2, 2), np.uint8)
MORPH_CLOSE_K       = np.ones((1, 1), np.uint8)  # 1×1 avoids bridging adjacent characters
MIN_BLOB_AREA       = 20

# ── Visual annotation ─────────────────────────────────────────────────────────
BANNER_H   = 44
BANNER_BG  = (26, 26, 46)
TEXT_WHITE = (255, 255, 255)
TEXT_GREY  = (170, 170, 170)
FONT       = cv2.FONT_HERSHEY_SIMPLEX


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> dict:
    """Full adaptive binarisation pipeline. Returns dict of stage images."""
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray      = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, BILATERAL_D, BILATERAL_SIGMA_COL, BILATERAL_SIGMA_SPC)

    clahe_op  = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    clahe     = clahe_op.apply(bilateral)

    adaptive  = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPT_BLOCK_SIZE, ADAPT_C
    )

    otsu_val, otsu_mask = cv2.threshold(
        clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    combined = cv2.bitwise_and(adaptive, otsu_mask)
    opened   = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  MORPH_OPEN_K)
    binary   = cv2.morphologyEx(opened,   cv2.MORPH_CLOSE, MORPH_CLOSE_K)
    binary   = _remove_small_blobs(binary, MIN_BLOB_AREA)

    h, w    = binary.shape
    scale   = TARGET_WIDTH / w
    new_h   = int(h * scale)
    resized = cv2.resize(binary, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

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


def save_pipeline_visuals(processed: dict, output_dir: str, base_name: str) -> dict:
    """Save one annotated PNG per preprocessing stage."""
    os.makedirs(output_dir, exist_ok=True)
    stem   = os.path.splitext(base_name)[0]
    paths  = {}

    stages = [
        ("original",  "Stage 1 — Original Image",
         "Raw input  •  BGR colour  •  No processing", (52, 152, 219), False),
        ("gray",      "Stage 2 — Grayscale",
         "cv2.COLOR_BGR2GRAY  •  Single-channel luminance", (39, 174, 96), False),
        ("bilateral", "Stage 3 — Bilateral Filter",
         f"d={BILATERAL_D} σ_color={BILATERAL_SIGMA_COL} σ_space={BILATERAL_SIGMA_SPC}", (241, 196, 15), False),
        ("clahe",     "Stage 4 — CLAHE",
         f"clipLimit={CLAHE_CLIP_LIMIT}  tileGrid={CLAHE_TILE_GRID}", (230, 126, 34), False),
        ("adaptive",  "Stage 5a — Adaptive Gaussian Threshold",
         f"blockSize={ADAPT_BLOCK_SIZE}  C={ADAPT_C}", (231, 76, 60), True),
        ("otsu_mask", f"Stage 5b — Otsu on CLAHE  [T={processed.get('otsu_value','?')}]",
         "Secondary vote  •  AND-blended with adaptive", (155, 89, 182), True),
        ("binary",    "Stage 6 — Final Binary",
         f"AND + morphOpen + morphClose + blob filter (≥{MIN_BLOB_AREA}px²)", (26, 188, 156), True),
        ("resized",   "Stage 7 — Resized Working Image",
         f"Aspect-preserving resize to {TARGET_WIDTH}px", (236, 95, 128), True),
    ]

    for key, title, desc, accent, is_binary in stages:
        img = processed.get(key)
        if img is None:
            continue

        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        vis = _scale_to_max_width(vis, 1024)
        h, w = vis.shape[:2]

        banner = np.zeros((BANNER_H, w, 3), dtype=np.uint8)
        banner[:] = BANNER_BG
        banner[:, :5] = accent
        cv2.putText(banner, title, (12, 20), FONT, 0.50, TEXT_WHITE, 1, cv2.LINE_AA)

        stats = f"  |  {w}×{h}px"
        if is_binary:
            ink_pct = round(float(np.sum(img > 0)) / img.size * 100, 2)
            stats  += f"  |  ink={ink_pct}%"
        cv2.putText(banner, desc + stats, (12, 38), FONT, 0.34, TEXT_GREY, 1, cv2.LINE_AA)

        fpath = os.path.join(output_dir, f"pp_{key}_{stem}.png")
        cv2.imwrite(fpath, np.vstack([banner, vis]))
        paths[key] = fpath

    return paths


# ── Utilities ─────────────────────────────────────────────────────────────────

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
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def extract_writing_style(processed: dict) -> dict:
    """Stroke-width, slant, and line-spacing estimates from the binary."""
    binary = processed["resized"]

    dist      = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    nonzero   = dist[dist > 0]
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

    h_proj   = np.sum(binary, axis=1)
    baseline = np.max(h_proj) * 0.1
    in_text  = h_proj > baseline
    trans    = np.diff(in_text.astype(int))
    gap_s    = np.where(trans == -1)[0]
    gap_e    = np.where(trans ==  1)[0]
    if len(gap_s) > 1 and len(gap_e) > 1:
        spacings     = gap_e[1:len(gap_s)] - gap_s[:len(gap_e)-1]
        line_spacing = float(np.mean(spacings)) if len(spacings) else 0.0
    else:
        line_spacing = 0.0

    return {
        "avg_stroke_width"   : round(avg_stroke,  2),
        "estimated_slant_deg": round(slant,        2),
        "line_spacing_px"    : round(line_spacing, 2),
    }


def _remove_small_blobs(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area px²."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label] = 255
    return clean


def _scale_to_max_width(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
