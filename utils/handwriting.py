"""ForgeNet-X — Handwriting Generation"""

import os
import json
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Layout ────────────────────────────────────────────────────────────────────
CHAR_H         = 56
LINE_SPACING   = 20
PAGE_MARGIN_X  = 40
PAGE_MARGIN_Y  = 36
JITTER_Y_MAX   = 4
CHARS_PER_LINE = 48

INTER_CHAR_GAP = 6   # gap between chars within a word
WORD_SPACE_W   = 22  # gap between words — must stay > INTER_CHAR_GAP

assert INTER_CHAR_GAP < WORD_SPACE_W, "INTER_CHAR_GAP must be < WORD_SPACE_W"

WINDOWS_FONTS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\cour.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_handwriting(processed: dict, text: str, output_path: str,
                         char_dir: str = None) -> str:
    """Render text as a handwritten image using crops from the uploaded sample."""
    if char_dir and os.path.isdir(char_dir):
        atlas = _load_atlas_from_manifest(char_dir)
    else:
        atlas = _extract_atlas_legacy(processed["resized"])

    if not atlas:
        atlas = {}

    lines     = _wrap_text(text, CHARS_PER_LINE)
    line_imgs = [_render_line(ln, atlas) for ln in lines]
    page      = _build_page(line_imgs)

    page   = cv2.GaussianBlur(page, (3, 3), 0)
    result = cv2.bitwise_not(page)
    result = _bake_watermark(result)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, result)
    return output_path


# ── Atlas builders ────────────────────────────────────────────────────────────

def _load_atlas_from_manifest(char_dir: str) -> dict:
    """Build { label → crop_ndarray } from manifest.json."""
    manifest_path = os.path.join(char_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return {}

    with open(manifest_path) as f:
        manifest = json.load(f)

    entries = sorted(
        ((k, v) for k, v in manifest.items() if not k.startswith("_")),
        key=lambda kv: kv[1].get("index", 0)
    )

    atlas = {}
    for fname, info in entries:
        label = info.get("label", "?")
        if label == "?" or label in atlas:
            continue
        img = cv2.imread(os.path.join(char_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            atlas[label] = _resize_to_height(img, CHAR_H)

    return atlas


def _extract_atlas_legacy(binary: np.ndarray) -> dict:
    """Fallback: re-run segmentation when no char_dir is supplied."""
    from utils.segmentation import segment_characters
    import tempfile, shutil

    tmp_dir = tempfile.mkdtemp(prefix="forgenet_hw_")
    fake_processed = {
        "resized": binary, "binary": binary,
        "height": binary.shape[0], "width": binary.shape[1], "path": "synthetic",
    }
    try:
        segment_characters(fake_processed, tmp_dir, "atlas.png")
        atlas = _load_atlas_from_manifest(os.path.join(tmp_dir, "chars_atlas"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return atlas


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _render_line(line: str, atlas: dict) -> np.ndarray:
    """Stitch character tiles for one line into a single strip."""
    row_h = CHAR_H + JITTER_Y_MAX * 2
    tiles = []

    for char in line:
        if char == " ":
            tiles.append(np.zeros((row_h, WORD_SPACE_W), dtype=np.uint8))
            continue

        raw     = atlas[char].copy() if char in atlas else _fallback_char(char)
        raw     = _resize_to_height(raw, CHAR_H)
        jitter  = random.randint(-JITTER_Y_MAX, JITTER_Y_MAX)
        tile    = np.zeros((row_h, raw.shape[1]), dtype=np.uint8)
        y_start = JITTER_Y_MAX + jitter
        y_end   = y_start + CHAR_H
        if 0 <= y_start and y_end <= row_h:
            tile[y_start:y_end, :] = raw
        else:
            tile[JITTER_Y_MAX:JITTER_Y_MAX + CHAR_H, :] = raw

        tiles.append(tile)
        tiles.append(np.zeros((row_h, INTER_CHAR_GAP), dtype=np.uint8))

    if not tiles:
        return np.zeros((row_h, 10), dtype=np.uint8)

    return np.hstack(tiles)


def _build_page(line_imgs: list) -> np.ndarray:
    """Stack line strips vertically with margins."""
    if not line_imgs:
        return np.zeros((200, 400), dtype=np.uint8)

    max_w = max(img.shape[1] for img in line_imgs)
    rows  = []
    for img in line_imgs:
        h, w = img.shape
        if w < max_w:
            img = np.hstack([img, np.zeros((h, max_w - w), dtype=np.uint8)])
        rows.append(img)
        rows.append(np.zeros((LINE_SPACING, max_w), dtype=np.uint8))

    page = np.vstack(rows)
    ph, pw = page.shape
    canvas = np.zeros((ph + PAGE_MARGIN_Y*2, pw + PAGE_MARGIN_X*2), dtype=np.uint8)
    canvas[PAGE_MARGIN_Y:PAGE_MARGIN_Y+ph, PAGE_MARGIN_X:PAGE_MARGIN_X+pw] = page
    return canvas


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Scale image to target_h, preserving aspect ratio."""
    h, w = img.shape
    if h == 0 or w == 0:
        return np.zeros((target_h, max(1, target_h // 2)), dtype=np.uint8)
    scale = target_h / h
    return cv2.resize(img, (max(1, int(w * scale)), target_h), interpolation=cv2.INTER_AREA)


def _fallback_char(char: str) -> np.ndarray:
    """Render a character using a Windows system font when missing from atlas."""
    size    = CHAR_H
    img_pil = Image.new("L", (size, size), color=0)
    draw    = ImageDraw.Draw(img_pil)

    font = None
    for fp in WINDOWS_FONTS:
        if os.path.exists(fp):
            try:    font = ImageFont.truetype(fp, size - 8); break
            except: continue
    if font is None:
        font = ImageFont.load_default()

    bbox   = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((max(0, (size-tw)//2), max(0, (size-th)//2)), char, fill=210, font=font)

    return np.array(img_pil, dtype=np.uint8)


def _bake_watermark(img: np.ndarray) -> np.ndarray:
    """Overlay a repeating diagonal '[SYNTHETIC – ForgeNet-X]' watermark."""
    if img.ndim == 2: img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:             img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w     = img_rgb.shape[:2]
    pil_base = Image.fromarray(img_rgb).convert("RGBA")
    overlay  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw     = ImageDraw.Draw(overlay)

    text      = "[SYNTHETIC – ForgeNet-X]"
    font_size = max(16, min(w, h) // 10)

    font = None
    for fp in WINDOWS_FONTS:
        if os.path.exists(fp):
            try:    font = ImageFont.truetype(fp, font_size); break
            except: continue
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
    """Word-wrap text so no line exceeds max_chars."""
    words, lines, cur = text.split(), [], ""
    for word in words:
        if len(cur) + len(word) + (1 if cur else 0) <= max_chars:
            cur = (cur + " " + word).strip() if cur else word
        else:
            if cur: lines.append(cur)
            cur = word
    if cur: lines.append(cur)
    return lines if lines else [text[:max_chars]]
