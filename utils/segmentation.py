"""
utils/segmentation.py
─────────────────────
ForgeNet-X — Character Segmentation Module  (Windows-compatible)

Pipeline
────────
  1. Find all external contours in the binarised image
  2. Filter by minimum / maximum bounding-box size
  3. Sort into reading order (top → bottom, left → right)
  4. Crop + pad each character  →  save as individual PNGs
  5. Write JSON manifest  {filename: {x,y,w,h, label, charset_index}}

Visual confirmation (new)
─────────────────────────
  save_segmentation_overview()
      Draws coloured bounding boxes on a copy of the source image.
      Each box is numbered and colour-coded by its assigned charset label.
      A legend strip at the bottom lists index → character label.

  save_character_atlas_sheet()
      Tiles all cropped characters onto a single sheet, 10 per row.
      Under each tile: the crop index  +  the assigned charset label
      (or "?" if unlabelled), shown on a dark background strip.
"""

import os
import json
import math
import cv2
import numpy as np


# ── Tunable constants ─────────────────────────────────────────────────────────
MIN_CHAR_W  = 5
MIN_CHAR_H  = 8
MAX_CHAR_W  = 200
CHAR_PAD    = 4

# Charset used for heuristic label assignment (same order as handwriting.py)
CHARSET = (
    list("abcdefghijklmnopqrstuvwxyz") +
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    list("0123456789") +
    list(".,!?;:'\"()-")
)

# Colours per column  (BGR)  — cycles through 8 distinct hues
BOX_COLORS = [
    (52,  152, 219),   # blue
    (39,  174,  96),   # green
    (231,  76,  60),   # red
    (241, 196,  15),   # yellow
    (155,  89, 182),   # purple
    (230, 126,  34),   # orange
    ( 26, 188, 156),   # teal
    (236,  95, 128),   # pink
]

FONT       = cv2.FONT_HERSHEY_SIMPLEX
BANNER_H   = 38
BANNER_BG  = (26, 26, 46)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def segment_characters(processed: dict,
                       output_dir: str,
                       base_name: str) -> list:
    """
    Extract individual character crops from a preprocessed image.

    Returns
    -------
    list[str]  —  absolute paths to saved PNG crops, in reading order
    """
    binary   = processed["resized"].copy()
    stem     = os.path.splitext(base_name)[0]
    char_dir = os.path.join(output_dir, f"chars_{stem}")
    os.makedirs(char_dir, exist_ok=True)

    # 1. Contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # 2. Filter
    valid_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if MIN_CHAR_W <= w <= MAX_CHAR_W and MIN_CHAR_H <= h:
            valid_boxes.append((x, y, w, h))

    # 3. Reading-order sort
    valid_boxes.sort(key=lambda b: (b[1] // 20, b[0]))

    # 4. Crop & save
    paths    = []
    manifest = {}
    img_h, img_w = binary.shape

    for idx, (x, y, w, h) in enumerate(valid_boxes):
        x1 = max(0, x - CHAR_PAD)
        y1 = max(0, y - CHAR_PAD)
        x2 = min(img_w, x + w + CHAR_PAD)
        y2 = min(img_h, y + h + CHAR_PAD)

        crop  = binary[y1:y2, x1:x2]
        label = CHARSET[idx] if idx < len(CHARSET) else "?"
        fname = f"char_{idx:04d}.png"
        fpath = os.path.join(char_dir, fname)
        cv2.imwrite(fpath, crop)

        paths.append(fpath)
        manifest[fname] = {
            "x": x1, "y": y1,
            "w": int(x2 - x1), "h": int(y2 - y1),
            "label": label,
            "index": idx,
        }

    # 5. Manifest
    with open(os.path.join(char_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return paths


def save_segmentation_overview(processed: dict,
                                char_paths: list,
                                output_dir: str,
                                base_name: str) -> str:
    """
    Draw every detected bounding box onto the binary image.

    Each box is:
      • Coloured by its column index (cycles through BOX_COLORS)
      • Labelled with  #index / charset-label  above the box
      • Numbered in reading order

    A top banner shows total character count + image dimensions.
    A bottom legend strip maps colour → charset range.

    Returns
    -------
    str  —  path to saved overview PNG
    """
    binary = processed["resized"].copy()
    vis    = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    vis    = _scale_to_max_width(vis, 1200)
    h, w   = vis.shape[:2]

    # Re-derive boxes from the saved manifest (avoids re-running detection)
    stem     = os.path.splitext(base_name)[0]
    char_dir = os.path.join(output_dir, f"chars_{stem}")
    manifest_path = os.path.join(char_dir, "manifest.json")

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        boxes_info = sorted(manifest.values(), key=lambda b: b["index"])
    else:
        boxes_info = []

    # Scale factor (image was scaled for display)
    orig_w = processed["resized"].shape[1]
    sf     = w / orig_w   # scale factor applied by _scale_to_max_width

    for info in boxes_info:
        idx   = info["index"]
        label = info.get("label", "?")
        color = BOX_COLORS[idx % len(BOX_COLORS)]

        # Scale coords
        x1 = int(info["x"] * sf)
        y1 = int(info["y"] * sf)
        x2 = int((info["x"] + info["w"]) * sf)
        y2 = int((info["y"] + info["h"]) * sf)

        # Draw box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

        # Label tag above box  (#idx / char)
        tag    = f"{idx}/{label}"
        fs     = 0.32
        (tw, th), bl = cv2.getTextSize(tag, FONT, fs, 1)
        tag_x  = max(0, x1)
        tag_y  = max(th + 2, y1 - 2)

        # Tag background pill
        cv2.rectangle(vis,
                      (tag_x - 1,  tag_y - th - 1),
                      (tag_x + tw + 1, tag_y + bl),
                      color, -1)
        cv2.putText(vis, tag, (tag_x, tag_y), FONT, fs,
                    (0, 0, 0), 1, cv2.LINE_AA)

    # ── Top banner ────────────────────────────────────────────────────────────
    banner = np.zeros((BANNER_H, w, 3), dtype=np.uint8)
    banner[:] = BANNER_BG
    banner[:, :4] = (231, 76, 60)   # red accent bar
    cv2.putText(banner,
                f"Step 6 — Character Segmentation Overview",
                (12, 24), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner,
                f"Contour detection  •  {len(boxes_info)} characters found"
                f"  •  Boxes colour-coded by index  •  label = charset position",
                (12, 36), FONT, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

    annotated = np.vstack([banner, vis])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"seg_overview_{stem}.png")
    cv2.imwrite(out_path, annotated)
    return out_path


def save_character_atlas_sheet(char_paths: list,
                                output_dir: str,
                                base_name: str,
                                tile_size: int = 64,
                                cols: int = 10) -> str:
    """
    Tile all character crops into a single labelled sheet.

    Each tile shows:
      • The character crop centred on a dark background
      • A bottom strip with  #index  and  assigned charset label
      • Colour-matched border (same palette as overview)

    Returns
    -------
    str  —  path to saved atlas sheet PNG
    """
    if not char_paths:
        return ""

    stem     = os.path.splitext(base_name)[0]
    char_dir = os.path.join(output_dir, f"chars_{stem}")
    manifest_path = os.path.join(char_dir, "manifest.json")

    # Load label info
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    # fname → label map
    label_map = {fname: info.get("label", "?")
                 for fname, info in manifest.items()}

    LABEL_H  = 18   # px  height of the label strip under each tile
    cell_h   = tile_size + LABEL_H
    n        = len(char_paths)
    rows     = math.ceil(n / cols)

    sheet_w  = cols * tile_size
    sheet_h  = rows * cell_h

    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)
    sheet[:] = (20, 20, 30)   # very dark background

    for idx, path in enumerate(char_paths):
        row  = idx // cols
        col  = idx  % cols
        x0   = col * tile_size
        y0   = row * cell_h

        color = BOX_COLORS[idx % len(BOX_COLORS)]
        fname = os.path.basename(path)
        label = label_map.get(fname, "?")

        # ── Crop tile ─────────────────────────────────────────────────────────
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            tile_bgr = _fit_crop_to_tile(img, tile_size, color)
        else:
            tile_bgr = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)

        sheet[y0: y0 + tile_size, x0: x0 + tile_size] = tile_bgr

        # ── Label strip ───────────────────────────────────────────────────────
        ly0 = y0 + tile_size
        ly1 = y0 + cell_h
        cv2.rectangle(sheet, (x0, ly0), (x0 + tile_size, ly1), color, -1)

        # Index in small font
        idx_tag = f"#{idx}"
        cv2.putText(sheet, idx_tag,
                    (x0 + 2, ly0 + 9),
                    FONT, 0.28, (0, 0, 0), 1, cv2.LINE_AA)

        # Character label centred
        lbl_tag  = f'"{label}"'
        (lw, _), _ = cv2.getTextSize(lbl_tag, FONT, 0.32, 1)
        lx = x0 + (tile_size - lw) // 2
        cv2.putText(sheet, lbl_tag,
                    (lx, ly0 + 15),
                    FONT, 0.32, (0, 0, 0), 1, cv2.LINE_AA)

        # Thin border around tile
        cv2.rectangle(sheet, (x0, y0), (x0 + tile_size, y0 + cell_h),
                      color, 1)

    # ── Top banner ────────────────────────────────────────────────────────────
    banner = np.zeros((BANNER_H, sheet_w, 3), dtype=np.uint8)
    banner[:] = BANNER_BG
    banner[:, :4] = (155, 89, 182)   # purple accent
    cv2.putText(banner,
                f"Step 7 — Segmented Character Atlas  ({n} characters)",
                (12, 24), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner,
                f"Each tile: crop image  +  index (#N)  +  assigned charset label"
                f"  •  Heuristic mapping: left-to-right → a-z A-Z 0-9",
                (12, 36), FONT, 0.36, (200, 200, 200), 1, cv2.LINE_AA)

    final = np.vstack([banner, sheet])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"seg_atlas_{stem}.png")
    cv2.imwrite(out_path, final)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Supporting utilities (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────

def segment_lines(binary: np.ndarray) -> list:
    h_proj    = np.sum(binary, axis=1).astype(float)
    threshold = np.max(h_proj) * 0.05
    in_text   = h_proj > threshold
    lines, start = [], None
    for i, active in enumerate(in_text):
        if active and start is None:
            start = i
        elif not active and start is not None:
            lines.append(binary[start:i, :])
            start = None
    if start is not None:
        lines.append(binary[start:, :])
    return lines


def build_character_atlas(char_paths: list, char_size: int = 32) -> dict:
    atlas = {}
    for path in char_paths:
        key = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        atlas[key] = _fit_to_square(img, char_size)
    return atlas


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fit_to_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w   = img.shape
    scale  = size / max(h, w)
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas  = np.zeros((size, size), dtype=np.uint8)
    y_off   = (size - new_h) // 2
    x_off   = (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _fit_crop_to_tile(img: np.ndarray, tile: int, border_color) -> np.ndarray:
    """
    Centre a grayscale crop in a tile×tile BGR canvas.
    White pixels (ink) are shown in the tile's accent colour; background is dark.
    """
    h, w   = img.shape
    scale  = (tile - 6) / max(h, w)   # leave 3px padding all around
    new_w  = max(1, int(w * scale))
    new_h  = max(1, int(h * scale))
    small  = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Dark canvas
    canvas = np.zeros((tile, tile, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 45)

    y0 = (tile - new_h) // 2
    x0 = (tile - new_w) // 2

    # Coloured ink (where pixel > 0)
    ink_mask = small > 0
    canvas[y0:y0+new_h, x0:x0+new_w][ink_mask] = border_color

    return canvas


def _scale_to_max_width(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
