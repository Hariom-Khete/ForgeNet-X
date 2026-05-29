"""ForgeNet-X — Handwriting Generation (with normalised atlas)"""

import os
import json
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Layout constants ──────────────────────────────────────────────────────────
# These are used in BOTH the legacy flat-atlas path and (as fallbacks) in the
# normalised path.  Normalised spacing is derived from targets['x_height_px'].
LINE_SPACING   = 20    # vertical gap between rendered text rows (pixels)
PAGE_MARGIN_X  = 40
PAGE_MARGIN_Y  = 36
JITTER_Y_MAX   = 2     # ± rows max drift amplitude for smooth baseline wander

CHARS_PER_LINE = 48    # word-wrap column width

# Legacy (flat-atlas) constants kept for the fallback path
CHAR_H         = 56
INTER_CHAR_GAP = 4
WORD_SPACE_W   = 26

assert INTER_CHAR_GAP < WORD_SPACE_W, "INTER_CHAR_GAP must be < WORD_SPACE_W"

# Proportional spacing coefficients (× x_height_px) used in normalised path
NORM_LETTER_SPACING = 0.08   # gap between letter ink edges (tighter)
NORM_WORD_SPACING   = 0.90   # width of a word space (1.20 × 0.75)

WINDOWS_FONTS = [
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\cour.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
]


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def generate_handwriting(
        processed: dict,
        text: str,
        output_path: str,
        char_dir: str = None,
        blend: float = 0.35,
        target_x_height_px: float = None,
) -> str:
    """
    Render `text` as a handwritten image using crops from `char_dir`.

    Parameters
    ----------
    processed : dict
        Output of preprocess_image() — used only when char_dir is absent.
    text : str
        The text to render.
    output_path : str
        Where to save the result PNG.
    char_dir : str | None
        Directory containing manifest.json and char_*.png.  When supplied the
        normalised-atlas path is used; otherwise falls back to re-segmenting
        `processed["resized"]` (legacy mode).
    blend : float  [0.0 – 1.0]
        0.0 = pure author style, 1.0 = strict TNR, 0.35 = recommended.
    target_x_height_px : float | None
        Desired pixel height for x-height characters.  None uses the author's
        own measured x_height (proportions corrected, overall scale preserved).

    Returns
    -------
    str
        Absolute path to the saved image.
    """
    from utils.normalization import build_normalized_atlas  # lazy import

    atlas   = {}
    targets = None

    # ── Try normalised atlas from manifest ────────────────────────────────────
    if char_dir and os.path.isdir(char_dir):
        atlas, targets, _ = build_normalized_atlas(
            char_dir,
            blend=blend,
            target_x_height_px=target_x_height_px,
        )

    # ── Fallback: re-segment the binary and use a flat atlas ──────────────────
    if not atlas:
        flat_atlas = _extract_atlas_legacy(
            processed.get("resized", np.zeros((100, 100), dtype=np.uint8))
        )
        # Convert flat atlas to normalised-style tuples using a simple estimate
        atlas, targets = _wrap_flat_atlas(flat_atlas, target_x_height_px)

    lines     = _wrap_text(text, CHARS_PER_LINE)
    line_imgs = [_render_line(ln, atlas, targets) for ln in lines]
    page      = _build_page(line_imgs)

    page   = cv2.GaussianBlur(page, (3, 3), 0)
    result = cv2.bitwise_not(page)
    result = _bake_watermark(result)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, result)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Atlas builders
# ══════════════════════════════════════════════════════════════════════════════

def _extract_atlas_legacy(binary: np.ndarray) -> dict:
    """Re-run segmentation on the binary image; return flat {label: img}."""
    from utils.segmentation import segment_characters
    import tempfile, shutil

    tmp_dir      = tempfile.mkdtemp(prefix="forgenet_hw_")
    fake_proc    = {
        "resized": binary, "binary": binary,
        "height": binary.shape[0], "width": binary.shape[1], "path": "synthetic",
    }
    try:
        segment_characters(fake_proc, tmp_dir, "atlas.png")
        manifest_path = os.path.join(tmp_dir, "chars_atlas", "manifest.json")
        if not os.path.exists(manifest_path):
            return {}
        with open(manifest_path) as f:
            manifest = json.load(f)
        atlas_dir = os.path.join(tmp_dir, "chars_atlas")
        flat = {}
        for fname, info in sorted(manifest.items(),
                                  key=lambda kv: kv[1].get("index", 0)):
            if fname.startswith("_"):
                continue
            label = info.get("label", "?")
            if label == "?" or label in flat:
                continue
            img = cv2.imread(os.path.join(atlas_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                flat[label] = img
        return flat
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _wrap_flat_atlas(
        flat: dict,
        target_x_height_px: float = None,
) -> tuple[dict, dict]:
    """
    Convert a flat {label: img} atlas to normalised-style {label: (img, bl)}.
    A simple uniform resize is applied (no proportional correction).
    Used only when no manifest is available.
    """
    # Estimate x_height from median image height
    if flat:
        heights = sorted(img.shape[0] for img in flat.values())
        med_h   = float(heights[len(heights) // 2])
    else:
        med_h = float(CHAR_H)

    tx = float(target_x_height_px) if target_x_height_px else med_h

    # Build a minimal targets dict with identity ratios
    targets = {
        "x_height_px":        tx,
        "cap_height_px":      tx,
        "ascender_height_px": tx,
        "descender_below_px": tx * 0.35,
        "descender_total_px": tx * 1.35,
        "period_height_px":   max(4, tx * 0.16),
        "comma_height_px":    max(4, tx * 0.28),
        "colon_height_px":    max(4, tx * 0.85),
        "excl_height_px":     tx,
        "paren_height_px":    tx,
        "dash_height_px":     max(2, tx * 0.09),
        "dash_center_px":     tx * 0.50,
        "quote_height_px":    max(4, tx * 0.32),
        "blend":              0.0,
        "baseline_ref_px":    int(round(tx)),
    }

    atlas = {}
    for label, img in flat.items():
        resized = _resize_to_height(img, int(round(tx)))
        # baseline at bottom of uniformly-scaled image
        atlas[label] = (resized, resized.shape[0])

    return atlas, targets


# ══════════════════════════════════════════════════════════════════════════════
# Core rendering — normalised path
# ══════════════════════════════════════════════════════════════════════════════

def _smooth_baseline_drift(n: int, max_drift: int) -> list:
    """
    Generate a smooth, slowly-evolving baseline drift sequence of length `n`.

    Uses an exponentially-weighted moving average so the baseline wanders
    gently across the line (like handwriting on ruled paper) rather than
    jumping independently for every character.

    alpha=0.18 gives roughly one complete drift cycle per 10–12 characters.
    """
    if n <= 0:
        return []
    alpha  = 0.18     # smoothing weight on new random target (lower = smoother)
    drift  = 0.0
    target = 0.0
    result = []
    for _ in range(n):
        # Occasionally pick a new random target direction
        if random.random() < 0.25:
            target = random.uniform(-max_drift, max_drift)
        drift = alpha * target + (1.0 - alpha) * drift
        # Gentle mean-reversion so drift never locks at extreme
        drift *= 0.96
        result.append(int(round(drift)))
    return result


def _render_line(
        line: str,
        atlas: dict,
        targets: dict,
) -> np.ndarray:
    """
    Stitch character tiles for one text line into a single horizontal strip.

    All characters are placed so their `baseline_y` aligns with a shared
    baseline row inside the strip.  A smoothly-drifting per-character offset
    (slow random walk) adds natural baseline undulation without the zigzag
    caused by independent per-character jitter.

    atlas  : { label → (norm_img, baseline_y) }
    targets: output of compute_normalization_targets()
    """
    tx = targets["x_height_px"]

    # ── Row geometry ──────────────────────────────────────────────────────────
    #   max_above — maximum extent above baseline  (ascender_height + drift room)
    #   max_below — maximum extent below baseline  (descender_below + drift room)
    max_above = int(round(targets["ascender_height_px"])) + JITTER_Y_MAX
    max_below = int(round(targets.get("descender_below_px", 0))) + JITTER_Y_MAX
    row_h     = max_above + max_below
    shared_bl = max_above   # row index in strip where the nominal baseline sits

    # Proportional spacing
    gap_w   = max(1, int(round(tx * NORM_LETTER_SPACING)))
    space_w = max(8, int(round(tx * NORM_WORD_SPACING)))

    # ── Pre-generate smooth baseline drift for every non-space character ──────
    n_chars   = sum(1 for c in line if c != " ")
    drifts    = _smooth_baseline_drift(n_chars, JITTER_Y_MAX)
    drift_idx = 0

    tiles: list[np.ndarray] = []

    for char in line:
        # ── Word space ────────────────────────────────────────────────────
        if char == " ":
            tiles.append(np.zeros((row_h, space_w), dtype=np.uint8))
            continue

        # ── Look up character in atlas; fall back to font rendering ───────
        if char in atlas:
            char_img, char_bl = atlas[char]
        else:
            char_img, char_bl = _fallback_char_norm(char, targets)

        char_h, char_w = char_img.shape[:2]

        # ── Smooth baseline drift (NOT independent per-char jitter) ──────
        drift = drifts[drift_idx] if drift_idx < len(drifts) else 0
        drift_idx += 1

        # y_top = row in strip where top of character image lands
        y_top = shared_bl - char_bl + drift

        # ── Paste character into tile with safe clipping ──────────────────
        tile  = np.zeros((row_h, char_w), dtype=np.uint8)
        src_y = max(0, -y_top)
        dst_y = max(0,  y_top)
        h_cpy = min(char_h - src_y, row_h - dst_y)
        if h_cpy > 0:
            tile[dst_y: dst_y + h_cpy, :] = char_img[src_y: src_y + h_cpy, :]

        tiles.append(tile)
        tiles.append(np.zeros((row_h, gap_w), dtype=np.uint8))

    if not tiles:
        return np.zeros((row_h, 10), dtype=np.uint8)

    return np.hstack(tiles)


def _build_page(line_imgs: list) -> np.ndarray:
    """Stack line strips vertically with page margins."""
    if not line_imgs:
        return np.zeros((200, 400), dtype=np.uint8)

    max_w = max(img.shape[1] for img in line_imgs)
    rows: list[np.ndarray] = []

    for img in line_imgs:
        h, w = img.shape[:2]
        if w < max_w:
            img = np.hstack([img, np.zeros((h, max_w - w), dtype=np.uint8)])
        rows.append(img)
        rows.append(np.zeros((LINE_SPACING, max_w), dtype=np.uint8))

    page  = np.vstack(rows)
    ph, pw = page.shape[:2]
    canvas = np.zeros((ph + PAGE_MARGIN_Y * 2, pw + PAGE_MARGIN_X * 2), dtype=np.uint8)
    canvas[PAGE_MARGIN_Y: PAGE_MARGIN_Y + ph,
           PAGE_MARGIN_X: PAGE_MARGIN_X + pw] = page
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Fallback character rendering (font-based, returns normalised tuple)
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_char_norm(char: str, targets: dict) -> tuple[np.ndarray, int]:
    """
    Render `char` using a Windows system font at the proportionally correct
    height and return `(img, baseline_y)` matching the normalised-atlas format.
    """
    from utils.normalization import _get_category   # lazy import

    cat = _get_category(char)

    # Choose render height based on category
    cat_key_map = {
        "short_lower": "x_height_px",
        "ascender":    "ascender_height_px",
        "upper":       "cap_height_px",
        "digit":       "cap_height_px",
        "descender":   "descender_total_px",
        "period":      "period_height_px",
        "comma":       "comma_height_px",
        "colon":       "colon_height_px",
        "excl":        "excl_height_px",
        "paren":       "paren_height_px",
        "dash":        "x_height_px",
        "quote":       "cap_height_px",
    }
    render_h = int(round(targets.get(cat_key_map.get(cat, "x_height_px"),
                                     targets["x_height_px"])))
    render_h = max(8, render_h)

    # Square canvas: width = render_h (generous; will be tight-cropped)
    img_pil = Image.new("L", (render_h * 2, render_h), color=0)
    draw    = ImageDraw.Draw(img_pil)

    font = None
    for fp in WINDOWS_FONTS:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, max(8, render_h - 4))
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), char, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    tx_  = max(0, (render_h * 2 - tw) // 2 - bbox[0])
    ty_  = max(0, (render_h - th) // 2 - bbox[1])
    draw.text((tx_, ty_), char, fill=210, font=font)

    arr = np.array(img_pil, dtype=np.uint8)

    # Tight-crop to ink bounding box
    rows = np.any(arr > 32, axis=1)
    cols = np.any(arr > 32, axis=0)
    if rows.any() and cols.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        arr = arr[r0: r1 + 1, c0: c1 + 1]

    # Resize tight crop to render_h
    if arr.shape[0] > 0 and arr.shape[1] > 0:
        arr = _resize_to_height(arr, render_h)
    else:
        arr = np.zeros((render_h, max(4, render_h // 2)), dtype=np.uint8)

    # baseline_y: for non-descenders, baseline is at the bottom of the image
    if cat == "descender":
        tx_ref = int(round(targets["x_height_px"]))
        baseline_y = tx_ref
    else:
        baseline_y = arr.shape[0]

    return arr, baseline_y


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Scale image to target_h, preserving aspect ratio."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, max(1, target_h // 2)), dtype=np.uint8)
    scale = target_h / h
    return cv2.resize(
        img, (max(1, int(round(w * scale))), target_h),
        interpolation=cv2.INTER_AREA
    )


def _bake_watermark(img: np.ndarray) -> np.ndarray:
    """Overlay a repeating diagonal '[SYNTHETIC - ForgeNet-X]' watermark."""
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w     = img_rgb.shape[:2]
    pil_base = Image.fromarray(img_rgb).convert("RGBA")
    overlay  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw     = ImageDraw.Draw(overlay)

    text      = "[SYNTHETIC - ForgeNet-X]"
    font_size = max(16, min(w, h) // 10)

    font = None
    for fp in WINDOWS_FONTS:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    bbox   = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    step_y = max(th + 30, 70)
    for y in range(-h, h + step_y, step_y):
        draw.text(((w - tw) // 2, y), text, fill=(100, 100, 100, 90), font=font)

    overlay = overlay.rotate(25, expand=False)
    result  = Image.alpha_composite(pil_base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


def _wrap_text(text: str, max_chars: int) -> list:
    """Word-wrap text so no line exceeds max_chars characters."""
    words, lines, cur = text.split(), [], ""
    for word in words:
        if len(cur) + len(word) + (1 if cur else 0) <= max_chars:
            cur = (cur + " " + word).strip() if cur else word
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines if lines else [text[:max_chars]]
