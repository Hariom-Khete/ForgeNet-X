"""
utils/handwriting.py
────────────────────
ForgeNet-X — Handwriting Generation Module  (Windows-compatible)

HOW THE ATLAS IS BUILT
───────────────────────
Previous bug: _extract_atlas() re-ran segmentation from scratch in a
temp directory, completely ignoring the corrected manifest that
segment_characters() already wrote.  Labels were re-assigned by
left-to-right contour index with no line detection — undoing all the
work the new segmentation module does.

Fix: the app layer now passes the char_dir path to generate_handwriting().
_load_atlas_from_manifest() reads the manifest.json that segment_characters()
produced and builds  { label → crop_image }  directly from it.  The labels
in that manifest are the correct ones from the new line-aware segmentation.

GENERATION PIPELINE
───────────────────
1. Load atlas from manifest  { "a": ndarray, "b": ndarray, … }
2. Word-wrap input text to CHARS_PER_LINE columns
3. For each output line:
   a. Look up each character's crop in atlas
   b. Resize crop to target height (proportional width preserved)
   c. Apply small random vertical jitter  (natural rhythm)
   d. Apply small random horizontal spacing variation
   e. Hstack all tiles into a line strip
4. Add LINE_SPACING gap between lines
5. Add page margins
6. Mild Gaussian blur  (ink-bleed simulation)
7. Invert to white background / dark ink
8. Save and return path
"""

import os
import json
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Layout ────────────────────────────────────────────────────────────────────
CHAR_H         = 56    # target height for each character tile (px)
SPACE_W_FACTOR = 0.45  # space width = SPACE_W_FACTOR × CHAR_H
LINE_SPACING   = 20    # extra vertical gap between lines (px)
PAGE_MARGIN_X  = 40    # left/right page margin (px)
PAGE_MARGIN_Y  = 36    # top/bottom page margin (px)
JITTER_Y_MAX   = 5     # max random vertical displacement per character (px)
KERN_VARY      = 3     # ±px random kerning variation between characters
CHARS_PER_LINE = 48    # characters before word-wrap

# Windows system font search order for fallback rendering
WINDOWS_FONTS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\cour.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_handwriting(processed: dict,
                         text: str,
                         output_path: str,
                         char_dir: str = None) -> str:
    """
    Render `text` as a handwritten image using crops from the uploaded sample.

    Parameters
    ----------
    processed    : dict  — from preprocessing.preprocess_image()
    text         : str   — input text to render
    output_path  : str   — where to save the output PNG
    char_dir     : str   — folder that contains manifest.json from
                           segment_characters().  If supplied, labels come
                           from the correct manifest.  If None, falls back
                           to re-running segmentation (legacy behaviour).

    Returns
    -------
    str  — absolute path of saved image
    """
    # ── Build atlas ───────────────────────────────────────────────────────────
    if char_dir and os.path.isdir(char_dir):
        atlas = _load_atlas_from_manifest(char_dir)
    else:
        # Legacy fallback: re-segment (loses corrected labels but still works)
        atlas = _extract_atlas_legacy(processed["resized"])

    if not atlas:
        # Nothing segmented — render entirely with fallback font
        atlas = {}

    # ── Render ────────────────────────────────────────────────────────────────
    lines     = _wrap_text(text, CHARS_PER_LINE)
    line_imgs = [_render_line(ln, atlas) for ln in lines]
    page      = _build_page(line_imgs)

    # Mild blur → ink bleed
    page   = cv2.GaussianBlur(page, (3, 3), 0)
    result = cv2.bitwise_not(page)   # invert: white BG, dark ink

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, result)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# ATLAS BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_atlas_from_manifest(char_dir: str) -> dict:
    """
    Read the manifest.json written by segment_characters() and build
    { charset_label → resized_crop_ndarray }.

    If multiple crops share the same label (shouldn't happen with a correct
    segmentation of an a-z A-Z 0-9 sheet, but can if the user wrote more
    than one set), the first one in reading order wins.

    Returns  { "a": ndarray, "b": ndarray, … }
    """
    manifest_path = os.path.join(char_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return {}

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Sort by global index so reading-order wins on duplicates
    entries = sorted(manifest.items(), key=lambda kv: kv[1].get("index", 0))

    atlas = {}
    for fname, info in entries:
        label = info.get("label", "?")
        if label == "?" or label in atlas:
            continue   # skip unknowns and duplicates

        crop_path = os.path.join(char_dir, fname)
        img = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        atlas[label] = _resize_to_height(img, CHAR_H)

    return atlas


def _extract_atlas_legacy(binary: np.ndarray) -> dict:
    """
    Fallback: re-run segmentation in a temp dir when no char_dir is given.
    Uses the new segment_characters() so line detection still applies.
    """
    from utils.segmentation import segment_characters, CHARSET
    import tempfile, shutil

    tmp_dir = tempfile.mkdtemp(prefix="forgenet_hw_")
    fake_processed = {
        "resized": binary,
        "binary" : binary,
        "height" : binary.shape[0],
        "width"  : binary.shape[1],
        "path"   : "synthetic",
    }
    try:
        paths = segment_characters(fake_processed, tmp_dir, "atlas.png")
        char_dir = os.path.join(tmp_dir, "chars_atlas")
        atlas = _load_atlas_from_manifest(char_dir)
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return atlas


# ══════════════════════════════════════════════════════════════════════════════
# RENDERING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_line(line: str, atlas: dict) -> np.ndarray:
    """
    Stitch character tiles for one line of text into a single strip.

    Each character:
      • gets its crop from atlas (or a fallback font render)
      • is resized to CHAR_H height (width proportional)
      • has a small random Y jitter applied
      • has a small random kerning gap added on the right

    Returns ndarray of shape  (CHAR_H + 2*JITTER_Y_MAX,  total_width)
    """
    row_h = CHAR_H + JITTER_Y_MAX * 2
    tiles = []
    space_w = max(4, int(CHAR_H * SPACE_W_FACTOR))

    for char in line:
        if char == " ":
            # Space tile — randomise width slightly for natural rhythm
            w = space_w + random.randint(-2, 4)
            tiles.append(np.zeros((row_h, max(w, 4)), dtype=np.uint8))
            continue

        # Get crop
        if char in atlas:
            raw = atlas[char].copy()
        else:
            raw = _fallback_char(char)

        # Resize to CHAR_H, preserve aspect ratio
        raw = _resize_to_height(raw, CHAR_H)

        # Random jitter
        jitter = random.randint(-JITTER_Y_MAX, JITTER_Y_MAX)
        tile   = np.zeros((row_h, raw.shape[1]), dtype=np.uint8)
        y_start = JITTER_Y_MAX + jitter
        y_end   = y_start + CHAR_H
        if 0 <= y_start and y_end <= row_h:
            tile[y_start:y_end, :] = raw
        else:
            tile[JITTER_Y_MAX:JITTER_Y_MAX + CHAR_H, :] = raw

        # Small random kerning gap after character
        kern = random.randint(-KERN_VARY, KERN_VARY)
        gap  = max(1, 2 + kern)
        gap_tile = np.zeros((row_h, gap), dtype=np.uint8)

        tiles.append(tile)
        tiles.append(gap_tile)

    if not tiles:
        return np.zeros((row_h, 10), dtype=np.uint8)

    return np.hstack(tiles)


def _build_page(line_imgs: list) -> np.ndarray:
    """
    Stack line strips vertically with LINE_SPACING between them,
    then add page margins.
    """
    if not line_imgs:
        return np.zeros((200, 400), dtype=np.uint8)

    max_w = max(img.shape[1] for img in line_imgs)
    rows  = []

    for img in line_imgs:
        h, w = img.shape
        # Pad right edge to page width
        if w < max_w:
            pad = np.zeros((h, max_w - w), dtype=np.uint8)
            img = np.hstack([img, pad])
        rows.append(img)
        # Inter-line gap
        rows.append(np.zeros((LINE_SPACING, max_w), dtype=np.uint8))

    page = np.vstack(rows)
    ph, pw = page.shape

    # Add margins
    canvas = np.zeros(
        (ph + PAGE_MARGIN_Y * 2,
         pw + PAGE_MARGIN_X * 2),
        dtype=np.uint8
    )
    canvas[PAGE_MARGIN_Y:PAGE_MARGIN_Y + ph,
           PAGE_MARGIN_X:PAGE_MARGIN_X + pw] = page
    return canvas


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Scale image so height == target_h, preserving aspect ratio."""
    h, w  = img.shape
    if h == 0 or w == 0:
        return np.zeros((target_h, max(1, target_h // 2)), dtype=np.uint8)
    scale = target_h / h
    new_w = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _fallback_char(char: str) -> np.ndarray:
    """
    Render a single character using a Windows system font.
    Used when the character is absent from the atlas.
    """
    size    = CHAR_H
    img_pil = Image.new("L", (size, size), color=0)
    draw    = ImageDraw.Draw(img_pil)

    font = None
    for fp in WINDOWS_FONTS:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size - 8)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    bbox   = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx     = max(0, (size - tw) // 2)
    ty     = max(0, (size - th) // 2)
    draw.text((tx, ty), char, fill=210, font=font)

    arr = np.array(img_pil, dtype=np.uint8)
    return arr


def _wrap_text(text: str, max_chars: int) -> list:
    """Word-wrap text so no line exceeds max_chars characters."""
    words = text.split()
    lines = []
    cur   = ""
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
