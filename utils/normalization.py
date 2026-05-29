"""
utils/normalization.py
ForgeNet-X — Character Resizing + Normalization System

Times New Roman is used as a proportional REFERENCE MODEL only — the author's
handwriting stroke style is fully preserved.  A blend factor controls how
strongly TNR proportions influence the final sizes:

    blend = 0.00  →  pure author ratios (only overall x-height is normalised)
    blend = 0.35  →  recommended: gentle nudge toward TNR, style preserved
    blend = 1.00  →  strict TNR proportions (individuality erased)

Public API
----------
extract_handwriting_metrics(manifest, char_dir) -> hw_metrics
    Measure the author's typographic heights from the segmented manifest.

compute_normalization_targets(hw_metrics, target_x_height_px, blend) -> targets
    Blend author ratios with TNR ratios; return pixel heights per category.

normalize_character(img, label, targets) -> (norm_img, baseline_y)
    Resize one character image; return a canvas where baseline_y (int, from
    top) marks the writing baseline row.

build_normalized_atlas(char_dir, blend, target_x_height_px) -> (atlas, targets, hw_metrics)
    Load all chars, measure metrics, normalise every character.
    atlas = { label: (norm_img, baseline_y) }

save_normalization_preview(atlas, targets, output_path)
    Write a debug grid image showing all characters on a shared baseline.
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

# ── Times New Roman proportional reference (x_height = 1.0) ──────────────────
#
# All values are ratios relative to x-height.
# Sources: measured from TNR glyphs at 1000-unit UPM (standard Type-1 metrics).
# ─────────────────────────────────────────────────────────────────────────────
TNR: dict[str, float] = {
    # Main height zones
    "x_height":        1.000,   # a c e m n o r s u v w x z
    "cap_height":      1.636,   # A–Z
    "ascender_height": 1.818,   # b d f h i j k l t
    "descender_below": 0.454,   # below-baseline depth for g j p q y
    # Punctuation heights (above baseline, unless noted)
    "period_height":   0.160,   # .
    "comma_height":    0.280,   # ,  (body + tail combined)
    "colon_height":    0.850,   # :  ;
    "excl_height":     1.450,   # !  ?
    "paren_height":    1.350,   # (  )
    "dash_height":     0.090,   # -  (hyphen thickness)
    "dash_center":     0.500,   # center of dash is at 50% of x_height above baseline
    "quote_height":    0.320,   # '  "
}

# ── Character → typographic category ─────────────────────────────────────────
CAT_SHORT_LOWER = frozenset("acemnorsuvwxz")
CAT_ASCENDER    = frozenset("bdfhijklt")
CAT_DESCENDER   = frozenset("gjpqy")
CAT_UPPER       = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CAT_DIGIT       = frozenset("0123456789")
CAT_PERIOD      = frozenset(".")
CAT_COMMA       = frozenset(",")
CAT_COLON       = frozenset(":;")
CAT_EXCL        = frozenset("!?")
CAT_PAREN       = frozenset("()")
CAT_DASH        = frozenset("-")
CAT_QUOTE       = frozenset("'\"")


def _get_category(label: str) -> str:
    """Return the typographic category name for a single character label."""
    if not label or label == "?":
        return "short_lower"
    c = label[0]
    if c in CAT_SHORT_LOWER: return "short_lower"
    if c in CAT_ASCENDER:    return "ascender"
    if c in CAT_DESCENDER:   return "descender"
    if c in CAT_UPPER:       return "upper"
    if c in CAT_DIGIT:       return "digit"
    if c in CAT_PERIOD:      return "period"
    if c in CAT_COMMA:       return "comma"
    if c in CAT_COLON:       return "colon"
    if c in CAT_EXCL:        return "excl"
    if c in CAT_PAREN:       return "paren"
    if c in CAT_DASH:        return "dash"
    if c in CAT_QUOTE:       return "quote"
    return "short_lower"


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return float(s[len(s) // 2])


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Measure the author's handwriting from the manifest
# ══════════════════════════════════════════════════════════════════════════════

def extract_handwriting_metrics(manifest: dict, char_dir: str = "") -> dict:
    """
    Measure the author's actual typographic heights from the segmented manifest.

    Uses `rh` (raw ink height, no padding) to avoid measuring padding artefacts.
    Missing categories are inferred from available categories via TNR ratios.

    Returns
    -------
    hw_metrics : dict
        x_height_px          — median ink-height of short-lowercase characters
        cap_height_px        — median ink-height of uppercase characters
        ascender_height_px   — median ink-height of ascender characters
        descender_total_px   — median total ink-height of descender characters
        n_short_lower / n_upper / n_ascender / n_descender  — sample counts
    """
    buckets: dict[str, list[float]] = {
        "short_lower": [], "upper": [], "ascender": [], "descender": [],
    }

    for fname, info in manifest.items():
        if fname.startswith("_"):
            continue
        label = info.get("label", "?")
        if not label or label == "?":
            continue
        cat = _get_category(label)
        if cat not in buckets:
            continue
        # rh = raw ink height; falls back to padded h if absent
        rh = float(info.get("rh", info.get("h", 0)))
        if rh >= 4:
            buckets[cat].append(rh)

    x_h   = _median(buckets["short_lower"])  or None
    cap_h = _median(buckets["upper"])         or None
    asc_h = _median(buckets["ascender"])      or None
    dsc_h = _median(buckets["descender"])     or None

    # ── Infer x_height from whatever category is available ───────────────────
    if x_h is None or x_h < 4:
        if cap_h and cap_h >= 4:
            x_h = cap_h  / TNR["cap_height"]
        elif asc_h and asc_h >= 4:
            x_h = asc_h  / TNR["ascender_height"]
        elif dsc_h and dsc_h >= 4:
            x_h = dsc_h  / (TNR["x_height"] + TNR["descender_below"])
        else:
            x_h = 30.0   # absolute fallback

    # ── Fill missing categories via TNR ratios from x_h ──────────────────────
    if cap_h is None or cap_h < 4:
        cap_h = x_h * TNR["cap_height"]
    if asc_h is None or asc_h < 4:
        asc_h = x_h * TNR["ascender_height"]
    if dsc_h is None or dsc_h < 4:
        dsc_h = x_h * (TNR["x_height"] + TNR["descender_below"])

    return {
        "x_height_px":         x_h,
        "cap_height_px":       cap_h,
        "ascender_height_px":  asc_h,
        "descender_total_px":  dsc_h,
        "n_short_lower":       len(buckets["short_lower"]),
        "n_upper":             len(buckets["upper"]),
        "n_ascender":          len(buckets["ascender"]),
        "n_descender":         len(buckets["descender"]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Compute per-category pixel target heights
# ══════════════════════════════════════════════════════════════════════════════

def compute_normalization_targets(hw_metrics: dict,
                                  target_x_height_px: float,
                                  blend: float = 0.35) -> dict:
    """
    Blend the author's measured height ratios with TNR ratios, then scale
    to `target_x_height_px`.

    Parameters
    ----------
    hw_metrics : dict
        Output of extract_handwriting_metrics().
    target_x_height_px : float
        Desired pixel height for short-lowercase characters in the output.
    blend : float, 0–1
        0.0 = pure author style; 1.0 = strict TNR; 0.35 = recommended.

    Returns
    -------
    targets : dict
        Per-category pixel heights, plus `baseline_y` (= target_x_height_px
        rounded to int) and `blend` for reference.
    """
    blend = float(np.clip(blend, 0.0, 1.0))
    tx    = float(target_x_height_px)
    x_h   = hw_metrics["x_height_px"]

    # Author's measured ratios (relative to author's x_height)
    cap_ratio  = hw_metrics["cap_height_px"]      / x_h
    asc_ratio  = hw_metrics["ascender_height_px"] / x_h
    dsc_ratio  = max(0.0, hw_metrics["descender_total_px"] / x_h - 1.0)

    def _blend(author_r: float, tnr_r: float) -> float:
        return (1.0 - blend) * author_r + blend * tnr_r

    # Blended ratios
    cap_r   = _blend(cap_ratio,  TNR["cap_height"])
    asc_r   = _blend(asc_ratio,  TNR["ascender_height"])
    dsc_r   = _blend(dsc_ratio,  TNR["descender_below"])

    # Pixel heights derived from blended ratios and target x_height
    cap_px   = tx * cap_r
    asc_px   = tx * asc_r
    dsc_below_px = tx * dsc_r

    # Punctuation: always pure TNR (no per-author measurement available)
    per_px   = tx * TNR["period_height"]
    com_px   = tx * TNR["comma_height"]
    col_px   = tx * TNR["colon_height"]
    exc_px   = tx * TNR["excl_height"]
    par_px   = tx * TNR["paren_height"]
    dsh_px   = tx * TNR["dash_height"]
    dsh_c    = tx * TNR["dash_center"]   # center of dash above baseline
    quo_px   = tx * TNR["quote_height"]

    # Enforce sensible minimums (a 1px character is not useful)
    def _at_least(v: float, lo: float = 4.0) -> float:
        return max(v, lo)

    return {
        # ── Core zones ───────────────────────────────────────────────────
        "x_height_px":         round(tx, 1),
        "cap_height_px":       round(_at_least(cap_px),    1),
        "ascender_height_px":  round(_at_least(asc_px),    1),
        "descender_below_px":  round(max(0.0, dsc_below_px), 1),
        "descender_total_px":  round(_at_least(tx + dsc_below_px), 1),
        # ── Punctuation ──────────────────────────────────────────────────
        "period_height_px":    round(_at_least(per_px, 3), 1),
        "comma_height_px":     round(_at_least(com_px, 4), 1),
        "colon_height_px":     round(_at_least(col_px, 8), 1),
        "excl_height_px":      round(_at_least(exc_px),    1),
        "paren_height_px":     round(_at_least(par_px),    1),
        "dash_height_px":      round(_at_least(dsh_px, 2), 1),
        "dash_center_px":      round(dsh_c, 1),
        "quote_height_px":     round(_at_least(quo_px, 4), 1),
        # ── Metadata ─────────────────────────────────────────────────────
        "blend":               round(blend, 3),
        # baseline_y = x_height from the top of descender canvas;
        # for all non-descender chars it equals the image height.
        "baseline_ref_px":     int(round(tx)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Normalise a single character image
# ══════════════════════════════════════════════════════════════════════════════

def normalize_character(img: np.ndarray,
                        label: str,
                        targets: dict,
                        h_pad: int = 2) -> tuple[np.ndarray, int]:
    """
    Resize `img` (uint8 grayscale) to the proportionally correct pixel height
    for its typographic category, then embed it in a canvas sized so that
    `baseline_y` (int, rows from the TOP of the returned image) marks the
    writing baseline.

    Characters are baseline-aligned during stitching:
        canvas[shared_baseline - char_baseline_y : ..., x : ...] = char_img

    Parameters
    ----------
    img : np.ndarray
        Grayscale uint8 crop of the character (tight ink bounds preferred).
    label : str
        Single character that the crop represents.
    targets : dict
        Output of compute_normalization_targets().
    h_pad : int
        Horizontal padding (pixels) added to left AND right of final image
        so adjacent characters have a small gap.  Does NOT affect baseline_y.

    Returns
    -------
    (norm_img, baseline_y) : (np.ndarray, int)
        norm_img  — uint8 grayscale, ink white on black background
        baseline_y — row index (from top) where the baseline falls
    """
    h_src, w_src = img.shape[:2]
    if h_src < 1 or w_src < 1:
        fh = max(4, int(targets["x_height_px"]))
        return np.zeros((fh, max(2, fh // 2)), dtype=np.uint8), fh

    cat = _get_category(label)
    tx  = targets["x_height_px"]     # reference: x_height in pixels (float)

    # ── Determine target resize height and canvas geometry ────────────────────
    #
    # Conventions:
    #   above_px — pixels from top of canvas to baseline  (= baseline_y)
    #   below_px — pixels from baseline to bottom of canvas
    #   resize_h — height to which the source ink image is scaled
    #   canvas_top — first row of the character image inside the canvas
    #     (used for dash / quote which float rather than sit on the baseline)
    #
    # For most categories: resize_h = above_px, canvas_top = 0.
    # ─────────────────────────────────────────────────────────────────────────

    if cat == "short_lower":
        resize_h   = max(4, int(round(targets["x_height_px"])))
        above_px   = resize_h
        below_px   = 0
        canvas_top = 0

    elif cat == "ascender":
        resize_h   = max(4, int(round(targets["ascender_height_px"])))
        above_px   = resize_h
        below_px   = 0
        canvas_top = 0

    elif cat == "upper":
        resize_h   = max(4, int(round(targets["cap_height_px"])))
        above_px   = resize_h
        below_px   = 0
        canvas_top = 0

    elif cat == "digit":
        resize_h   = max(4, int(round(targets["cap_height_px"])))
        above_px   = resize_h
        below_px   = 0
        canvas_top = 0

    elif cat == "descender":
        # Total image = x_height (above baseline) + descender tail (below)
        above_px   = max(4, int(round(tx)))
        below_px   = max(0, int(round(targets["descender_below_px"])))
        resize_h   = above_px + below_px
        canvas_top = 0

    elif cat == "period":
        resize_h   = max(3, int(round(targets["period_height_px"])))
        above_px   = resize_h      # period sits on baseline (baseline = bottom)
        below_px   = 0
        canvas_top = 0

    elif cat == "comma":
        resize_h   = max(4, int(round(targets["comma_height_px"])))
        above_px   = resize_h      # comma sits on baseline
        below_px   = 0
        canvas_top = 0

    elif cat == "colon":
        resize_h   = max(8, int(round(targets["colon_height_px"])))
        above_px   = resize_h
        below_px   = 0
        canvas_top = 0

    elif cat == "excl":
        resize_h   = max(4, int(round(targets["excl_height_px"])))
        above_px   = resize_h
        below_px   = 0
        canvas_top = 0

    elif cat == "paren":
        resize_h   = max(4, int(round(targets["paren_height_px"])))
        above_px   = resize_h
        below_px   = 0
        canvas_top = 0

    elif cat == "dash":
        # Dash is tiny; it floats at 50 % of x_height above baseline.
        # Canvas: height = above_px = x_height_px  (no below-baseline portion).
        # Dash pixels are positioned so their centre row = dash_center_px from baseline.
        dash_h     = max(2, int(round(targets["dash_height_px"])))
        center_abs = int(round(targets["dash_center_px"]))  # above baseline
        above_px   = max(4, int(round(tx)))
        below_px   = 0
        resize_h   = dash_h
        # first row of dash inside canvas (measured from canvas top):
        #   canvas_top = above_px - center_abs - dash_h // 2
        canvas_top = max(0, above_px - center_abs - dash_h // 2)

    elif cat == "quote":
        # Quotes float high — their bottom sits near cap_height above baseline.
        # Canvas extends from top of quote down to baseline.
        resize_h   = max(4, int(round(targets["quote_height_px"])))
        above_px   = max(resize_h, int(round(targets["cap_height_px"])))
        below_px   = 0
        canvas_top = 0   # quote sits at very top of canvas

    else:
        # Unknown — treat like short_lower
        resize_h   = max(4, int(round(tx)))
        above_px   = resize_h
        below_px   = 0
        canvas_top = 0

    # ── Scale source image to resize_h, preserving aspect ratio ──────────────
    scale_h     = resize_h / h_src
    target_w    = max(2, int(round(w_src * scale_h)))
    resized     = cv2.resize(img, (target_w, resize_h),
                             interpolation=cv2.INTER_AREA)

    # ── Build canvas ──────────────────────────────────────────────────────────
    canvas_h = above_px + below_px
    canvas_w = target_w

    if canvas_h == resize_h and canvas_top == 0:
        # No extra space needed — resized image IS the canvas
        canvas = resized.astype(np.uint8)
    else:
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        y0 = canvas_top
        y1 = min(canvas_h, y0 + resize_h)
        if y1 > y0:
            canvas[y0:y1, :] = resized[:y1 - y0, :]

    # ── Horizontal padding (does not affect baseline_y) ───────────────────────
    if h_pad > 0:
        canvas = cv2.copyMakeBorder(
            canvas, 0, 0, h_pad, h_pad,
            cv2.BORDER_CONSTANT, value=0
        )

    baseline_y = above_px   # row index (from top) where baseline falls
    return canvas.astype(np.uint8), baseline_y


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Build the normalised atlas for all segmented characters
# ══════════════════════════════════════════════════════════════════════════════

def build_normalized_atlas(
        char_dir: str,
        blend: float = 0.35,
        target_x_height_px: float | None = None,
) -> tuple[dict, dict, dict]:
    """
    Load all segmented characters from `char_dir`, measure author metrics,
    compute normalised target sizes, and normalise every character.

    Parameters
    ----------
    char_dir : str
        Directory containing manifest.json and char_XXXX.png files.
    blend : float
        Blend factor, 0–1 (see module docstring).
    target_x_height_px : float | None
        Pixel height for x-height characters in the output.
        None  → use the author's own measured x_height (proportions are
                 corrected but overall scale is preserved).

    Returns
    -------
    atlas : dict { label → (norm_img, baseline_y) }
        norm_img  : uint8 grayscale, ink on black
        baseline_y: int, row from top where baseline falls
    targets : dict
        Output of compute_normalization_targets().
    hw_metrics : dict
        Output of extract_handwriting_metrics().
    """
    manifest_path = os.path.join(char_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return {}, {}, {}

    with open(manifest_path) as f:
        manifest = __import__("json").load(f)

    # ── Step 1: measure author metrics ────────────────────────────────────────
    hw_metrics = extract_handwriting_metrics(manifest, char_dir)

    # ── Step 2: determine target x_height ────────────────────────────────────
    if target_x_height_px is None:
        target_x_height_px = hw_metrics["x_height_px"]
    target_x_height_px = float(np.clip(target_x_height_px, 12.0, 300.0))

    # ── Step 3: compute target sizes ─────────────────────────────────────────
    targets = compute_normalization_targets(hw_metrics, target_x_height_px, blend)

    # ── Step 4: normalise each character ─────────────────────────────────────
    atlas: dict[str, tuple[np.ndarray, int]] = {}

    entries = sorted(
        ((k, v) for k, v in manifest.items() if not k.startswith("_")),
        key=lambda kv: kv[1].get("index", 0)
    )

    for fname, info in entries:
        label = info.get("label", "?")
        if label == "?" or label in atlas:
            continue

        img_path = os.path.join(char_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img_h, img_w = img.shape

        # ── Crop to tight ink bounding box ────────────────────────────────
        # The saved crop is binary[pad_y:pad_y+pad_h, pad_x:pad_x+pad_w].
        # `rx, ry` are absolute coords in the original image; the image file's
        # top-left corresponds to `x, y` from the manifest.
        pad_x = info.get("x", 0)
        pad_y = info.get("y", 0)
        rx    = info.get("rx", pad_x) - pad_x   # offset within saved image
        ry    = info.get("ry", pad_y) - pad_y
        rw    = info.get("rw", img_w)
        rh    = info.get("rh", img_h)

        rx = int(np.clip(rx, 0, img_w - 1))
        ry = int(np.clip(ry, 0, img_h - 1))
        rw = int(np.clip(rw, 1, img_w - rx))
        rh = int(np.clip(rh, 1, img_h - ry))

        ink_crop = img[ry: ry + rh, rx: rx + rw]
        if ink_crop.size == 0:
            ink_crop = img   # fallback: use padded image

        norm_img, baseline_y = normalize_character(ink_crop, label, targets)
        atlas[label] = (norm_img, baseline_y)

    return atlas, targets, hw_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Normalisation preview / debug image
# ══════════════════════════════════════════════════════════════════════════════

def save_normalization_preview(
        atlas: dict,
        targets: dict,
        hw_metrics: dict,
        output_path: str,
        cols: int = 10,
) -> None:
    """
    Save a debug grid image (BGR PNG) showing every normalised character placed
    on a shared writing baseline (bright green horizontal rule).

    Characters are laid out in CHARSET order (a-z A-Z 0-9 punctuation).
    Characters missing from the atlas are shown as a blue placeholder box.
    """
    # Import here to avoid circular dependency at module load time
    from utils.segmentation import CHARSET

    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    CELL_PAD   = 8    # vertical padding above ascender line and below descender
    LABEL_H    = 16   # height reserved below each cell for the character label
    BL_COLOR   = (0, 220, 60)     # baseline rule — bright green
    MISS_COLOR = (80, 80, 200)    # placeholder — muted blue
    BG_COLOR   = 32               # dark background

    tx        = targets["x_height_px"]
    max_above = int(round(targets["ascender_height_px"]))   # tallest thing above baseline
    max_below = int(round(targets.get("descender_below_px", 0)))   # deepest below

    # Row geometry (in cell coordinates)
    row_above = CELL_PAD + max_above   # rows from cell top to baseline
    row_below = max_below + CELL_PAD   # rows from baseline to cell bottom
    cell_h    = row_above + row_below + LABEL_H

    # Column width — enough for a wide char (m, w) at typical aspect ratio
    cell_w    = max(28, int(round(tx * 1.1)) + 2 * 4)   # 4px h_pad per side

    n_chars = len(CHARSET)
    n_cols  = min(cols, n_chars)
    n_rows  = math.ceil(n_chars / n_cols)

    HEADER_H   = 60
    canvas_h   = HEADER_H + n_rows * cell_h
    canvas_w   = n_cols * cell_w
    canvas     = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)

    # ── Header ───────────────────────────────────────────────────────────────
    blend_str = f"{targets['blend']:.2f}"
    cv2.putText(canvas,
                f"Normalization Preview  "
                f"x_h={tx:.0f}px  blend={blend_str}",
                (6, 18), FONT, 0.42, (210, 210, 210), 1, cv2.LINE_AA)

    n_sl  = hw_metrics.get("n_short_lower", 0)
    n_up  = hw_metrics.get("n_upper", 0)
    n_asc = hw_metrics.get("n_ascender", 0)
    n_dsc = hw_metrics.get("n_descender", 0)
    cv2.putText(canvas,
                f"cap={targets['cap_height_px']:.0f}px  "
                f"asc={targets['ascender_height_px']:.0f}px  "
                f"desc_below={targets['descender_below_px']:.0f}px  "
                f"samples: sl={n_sl} up={n_up} asc={n_asc} dsc={n_dsc}",
                (6, 36), FONT, 0.32, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(canvas,
                f"author x_h={hw_metrics.get('x_height_px', 0):.1f}px  "
                f"cap={hw_metrics.get('cap_height_px', 0):.1f}px  "
                f"asc={hw_metrics.get('ascender_height_px', 0):.1f}px",
                (6, 52), FONT, 0.30, (120, 160, 120), 1, cv2.LINE_AA)

    # ── Character grid ────────────────────────────────────────────────────────
    for ci, ch in enumerate(CHARSET):
        row_i  = ci // n_cols
        col_i  = ci  % n_cols
        cell_x = col_i * cell_w
        cell_y = HEADER_H + row_i * cell_h

        # ── Baseline rule ─────────────────────────────────────────────────
        bl_y = cell_y + row_above
        cv2.line(canvas, (cell_x, bl_y), (cell_x + cell_w - 1, bl_y),
                 BL_COLOR, 1)

        # ── x_height rule (lighter green) ─────────────────────────────────
        xh_y = bl_y - int(round(tx))
        if xh_y >= cell_y:
            cv2.line(canvas,
                     (cell_x, xh_y), (cell_x + cell_w - 1, xh_y),
                     (0, 120, 40), 1)

        # ── Character ─────────────────────────────────────────────────────
        if ch in atlas:
            char_img, char_bl = atlas[ch]
            char_h, char_w    = char_img.shape[:2]

            # Place so char's baseline_y aligns with cell's baseline row
            y_top  = bl_y - char_bl      # canvas row of char image top
            x_left = cell_x + max(0, (cell_w - char_w) // 2)

            # Clip to canvas bounds
            src_y0 = max(0, -y_top)
            dst_y0 = max(0,  y_top)
            src_x0 = max(0,  cell_x - x_left)
            dst_x0 = max(0,  x_left)

            h_copy = min(char_h - src_y0, canvas_h - dst_y0)
            w_copy = min(char_w - src_x0, canvas_w - dst_x0)

            if h_copy > 0 and w_copy > 0:
                # Grayscale → BGR for the canvas
                patch = char_img[src_y0: src_y0 + h_copy,
                                 src_x0: src_x0 + w_copy]
                bgr   = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
                canvas[dst_y0: dst_y0 + h_copy,
                       dst_x0: dst_x0 + w_copy] = bgr

        else:
            # Placeholder box
            box_top = cell_y + CELL_PAD
            box_bot = bl_y
            if box_bot > box_top:
                cv2.rectangle(canvas,
                              (cell_x + 3, box_top),
                              (cell_x + cell_w - 4, box_bot),
                              MISS_COLOR, 1)

        # ── Cell divider (faint) ──────────────────────────────────────────
        cv2.line(canvas,
                 (cell_x + cell_w - 1, cell_y),
                 (cell_x + cell_w - 1, cell_y + cell_h - LABEL_H - 1),
                 (60, 60, 60), 1)

        # ── Label ─────────────────────────────────────────────────────────
        label_y = cell_y + cell_h - 4
        # Avoid special characters that cv2 can't render: use repr for those
        safe_ch = ch if ch.isascii() and ch.isprintable() else "?"
        cv2.putText(canvas, safe_ch,
                    (cell_x + 4, label_y),
                    FONT, 0.30, (180, 180, 100), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, canvas)
