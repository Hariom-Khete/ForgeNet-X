"""
utils/handwriting.py
────────────────────
ForgeNet-X — Handwriting Generation Module

Strategy (MVP — corpus stitching):
  1. From the uploaded handwriting sheet, extract one representative crop
     per unique character using segmentation.
  2. For each character in the user's input text, look up the crop.
  3. Stitch crops left-to-right with natural spacing and slight random
     vertical jitter to mimic real handwriting rhythm.
  4. Apply a mild perspective / elastic warp for added realism.
  5. Save the result to disk and return the path.

If a character is not found in the extracted atlas (e.g. numbers when
only letters were sampled), a synthetic fallback tile is rendered using
a handwriting-style PIL font as a placeholder.

Advanced path (post-MVP):
  Replace this module with a trained conditional GAN or Transformer-based
  model (see ADVANCED_NOTES at the bottom of this file).
"""

import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Layout constants ──────────────────────────────────────────────────────────
CHAR_H          = 48    # px — target height for each character tile
SPACE_W         = 14    # px — width of a space character
LINE_SPACING    = 18    # px — gap between lines
PAGE_MARGIN_X   = 30    # px — left/right page margin
PAGE_MARGIN_Y   = 30    # px — top/bottom page margin
JITTER_Y_MAX    = 4     # px — max random vertical displacement per character
CHARS_PER_LINE  = 50    # characters before automatic line-wrap


def generate_handwriting(processed: dict,
                         text: str,
                         output_path: str) -> str:
    """
    Generate a handwritten-style image of `text` using character crops
    extracted from the supplied handwriting sample.

    Parameters
    ----------
    processed   : dict  — from preprocessing.preprocess_image()
    text        : str   — input text to render
    output_path : str   — where to save the output PNG

    Returns
    -------
    str  — absolute path of the saved image
    """
    binary = processed["resized"]

    # ── 1. Extract per-character atlas from the sample ────────────────────────
    atlas = _extract_atlas(binary)

    # ── 2. Render each character ──────────────────────────────────────────────
    lines      = _wrap_text(text, CHARS_PER_LINE)
    line_imgs  = [_render_line(line, atlas) for line in lines]

    # ── 3. Stack lines into a page ────────────────────────────────────────────
    page = _stack_lines(line_imgs)
    page = _elastic_distortion(page)

    # ── 4. Post-processing: mild blur for ink bleed effect ────────────────────
    page = cv2.GaussianBlur(page, (3, 3), 0)

    # ── 5. Invert back to white background / dark ink ─────────────────────────
    result = cv2.bitwise_not(page)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_atlas(binary: np.ndarray) -> dict:
    from utils.segmentation import segment_characters
    import tempfile, shutil

    tmp_dir = tempfile.mkdtemp(prefix="forgenet_atlas_")

    fake_processed = {
        "resized": binary,
        "binary": binary,
        "height": binary.shape[0],
        "width": binary.shape[1],
        "path": "synthetic",
    }

    paths = segment_characters(fake_processed, tmp_dir, "atlas.png")

    # 🔥 Simple sequential mapping (safe fallback)
    charset = list(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    atlas = {}

    for idx, fpath in enumerate(paths):
        if idx >= len(charset):
            break

        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = _resize_to_height(img, CHAR_H)
        atlas[charset[idx]] = img

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return atlas


def _resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Scale image so its height == target_h (preserving aspect ratio)."""
    h, w   = img.shape
    scale  = target_h / h
    new_w  = max(1, int(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _fallback_char(char: str) -> np.ndarray:
    """
    Render `char` using a system font as a fallback when the character
    is absent from the atlas.
    Tries common Windows font paths first, then falls back to PIL default.
    """
    size    = CHAR_H
    img_pil = Image.new("L", (size, size), color=0)
    draw    = ImageDraw.Draw(img_pil)

    # Windows font search order
    WINDOWS_FONTS = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\cour.ttf",       # Courier New
        r"C:\Windows\Fonts\verdana.ttf",
    ]
    font = None
    for fpath in WINDOWS_FONTS:
        if os.path.exists(fpath):
            try:
                font = ImageFont.truetype(fpath, size - 10)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    bbox   = draw.textbbox((0, 0), char, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx     = (size - tw) // 2
    ty     = (size - th) // 2
    ink_intensity = random.randint(180, 255)
    draw.text((tx, ty), char, fill=ink_intensity, font=font)

    arr = np.array(img_pil)
    return arr


def _render_line(line: str, atlas: dict) -> np.ndarray:
    row_h = CHAR_H + JITTER_Y_MAX * 2
    tiles = []

    for char in line:
        # 🔹 Handle space
        if char == " ":
            variable_space = random.randint(8, 20)
            tile = np.zeros((row_h, variable_space), dtype=np.uint8)
            tiles.append(tile)
            continue

        # 🔹 Get character image
        if char in atlas:
            raw = atlas[char]
        else:
            raw = _fallback_char(char)

        # 🔹 Apply jitter
        jitter = random.randint(-JITTER_Y_MAX, JITTER_Y_MAX)
        tile = _place_with_jitter(raw, row_h, jitter)

        # 🔹 Add character
        tiles.append(tile)

        # 🔹 Add spacing AFTER character (important)
        extra_space = random.randint(1, 4)
        space_tile = np.zeros((row_h, extra_space), dtype=np.uint8)
        tiles.append(space_tile)

    # 🔹 Remove last extra space sometimes
    if tiles and np.random.rand() > 0.5:
        tiles.pop()

    if not tiles:
        return np.zeros((row_h, 10), dtype=np.uint8)

    return np.hstack(tiles)


def _place_with_jitter(char_img, row_h, jitter):
    h, w = char_img.shape
    canvas = np.zeros((row_h, w + 4), dtype=np.uint8)

    y_start = JITTER_Y_MAX + jitter
    x_jitter = random.randint(0, 3)

    if y_start + h <= row_h:
        canvas[y_start:y_start+h, x_jitter:x_jitter+w] = char_img

    return canvas


def _stack_lines(line_imgs: list) -> np.ndarray:
    """
    Vertically concatenate line images into a single page, adding
    a white margin around the page.
    """
    if not line_imgs:
        return np.zeros((200, 400), dtype=np.uint8)

    max_w  = max(img.shape[1] for img in line_imgs)
    rows   = []

    for img in line_imgs:
        h, w = img.shape
        if w < max_w:
            pad   = np.zeros((h, max_w - w), dtype=np.uint8)
            img   = np.hstack([img, pad])
        rows.append(img)

    page     = np.vstack(rows)

    # Add margins
    page_h, page_w = page.shape
    canvas = np.zeros(
        (page_h + PAGE_MARGIN_Y * 2, page_w + PAGE_MARGIN_X * 2),
        dtype=np.uint8
    )
    canvas[PAGE_MARGIN_Y:PAGE_MARGIN_Y + page_h,
           PAGE_MARGIN_X:PAGE_MARGIN_X + page_w] = page

    return canvas


def _wrap_text(text: str, max_chars: int) -> list:
    """
    Word-wrap text so no line exceeds max_chars characters.
    """
    words  = text.split()
    lines  = []
    cur    = ""

    for word in words:
        if len(cur) + len(word) + 1 <= max_chars:
            cur = (cur + " " + word).strip()
        else:
            if cur:
                lines.append(cur)
            cur = word

    if cur:
        lines.append(cur)

    return lines if lines else [text[:max_chars]]

def _elastic_distortion(image):
    alpha = 5
    sigma = 3

    shape = image.shape
    dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (17, 17), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)


# ──────────────────────────────────────────────────────────────────────────────
# ADVANCED_NOTES (Post-MVP upgrade paths)
# ──────────────────────────────────────────────────────────────────────────────
"""
OPTION A — CNN + BiLSTM (sequence generation)
  - Encode input text as one-hot vectors
  - Use IAM Handwriting DB for training
  - Architecture: CNN feature extractor → BiLSTM → attention → stroke params
  - Library: PyTorch or TensorFlow/Keras

OPTION B — Conditional GAN (ScrabbleGAN / HandwritingGAN)
  - Generator conditioned on character-level embeddings
  - Discriminator assesses realism + legibility
  - Training data: IAM / RIMES / Bentham
  - Higher realism but requires GPU training (~4–8 hrs on RTX 3060)

OPTION C — Transformer (DALLE-style)
  - Treat handwriting as image patches (ViT backbone)
  - Fine-tune on a small personal dataset
  - Best realism, highest compute cost
"""
