"""
utils/segmentation.py
─────────────────────
ForgeNet-X — Character Segmentation Module  (Windows-compatible)

WHY CONTOUR-FIRST FAILS FOR i, j, k
──────────────────────────────────────
The previous approach detected contours first, then tried to merge dots
into their parent letters using area/distance heuristics.  This is
fundamentally fragile because:

  • i-body is a thin stroke — small area → classified as "satellite"
  • j-body is adjacent to i — small horizontal gap → false merge
  • k has a detached diagonal stroke — two contours → either merge or skip

No set of fixed thresholds can reliably distinguish "dot above its letter"
from "adjacent thin letter" across all handwriting sizes and styles.

OPTIMAL APPROACH: VERTICAL PROJECTION VALLEYS (no CNN required)
────────────────────────────────────────────────────────────────
Reference: Cursive Handwriting Segmentation — hybrid vertical projection
+ skeleton analysis (JATIT 2024); Two-stage segmentation using background
skeleton and vertical projection (ScienceDirect 2006).

Key insight: instead of asking "which contours belong together?", ask
"where are the vertical gaps between characters?"

  1. LINE DETECTION  (horizontal projection profile)
     Sum each row → find gap rows → extract horizontal line strips.

  2. CHARACTER DETECTION  (vertical projection profile per line strip)
     Sum each column within the strip → find valley columns where ink
     sum drops below a threshold → those valleys ARE the character
     boundaries.  No merging required.  The dot of 'i' and the body of
     'i' are in the same column band → they are one character naturally.

  3. BOUNDARY REFINEMENT
     Valleys that are too narrow (< MIN_VALLEY_W px) are noise.
     Valleys that are too wide might be a real space between words —
     keep them as inter-character gaps, don't split the word.

  4. BOUNDING BOX EXTRACTION
     For each column band between two valleys, find the y-extent of the
     ink pixels → tight bounding box.  Pad by CHAR_PAD.

  5. READING ORDER  (automatic — left→right within each line, top→bottom)
     Because bands are already ordered left→right by column position.

VISUAL OUTPUTS (unchanged API)
────────────────────────────────
  save_segmentation_overview()   — bounding boxes on binary image
  save_character_atlas_sheet()   — tiled crop grid
  save_line_debug_image()        — projection profile graph
"""

import os
import json
import math
import cv2
import numpy as np


# ── Charset ───────────────────────────────────────────────────────────────────
CHARSET = (
    list("abcdefghijklmnopqrstuvwxyz") +
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    list("0123456789") +
    list(".,!?;:'\"()-")
)

# ── Line detection ────────────────────────────────────────────────────────────
LINE_PROJ_THRESH  = 0.04   # row ink-sum / max must exceed this → "text row"
LINE_GAP_FACTOR   = 0.6    # fallback: new line if y-gap > this × median_h

# ── Vertical projection (character separation) ────────────────────────────────
# A column is a "valley" (gap between characters) when its ink sum is below
# VALLEY_THRESH × (mean ink sum of non-zero columns in this line strip).
# A column is a gap when its ink sum <= VALLEY_ABS_MIN absolute pixels.
# Using a tiny absolute floor (not % of mean) because:
#   • True inter-character gaps have ZERO ink
#   • Low-ink columns (crossbars, thin strokes) still have substantial ink
#   • % of mean incorrectly flags crossbar columns as gaps
# We allow up to 2 stray ink pixels per column from preprocessing noise.
VALLEY_ABS_MIN    = 2      # columns with ink_sum <= this are 'gap' columns
MIN_VALLEY_W      = 1      # px — valleys narrower than this are noise
MIN_VALLEY_W      = 2      # px — gaps narrower than this between bands
                           #      are noise (e.g. k stroke-to-diagonal),
                           #      not true inter-character spaces
MIN_CHAR_W        = 3      # px — character bands narrower than this → skip
                           #      must be 3 to keep i, j, l, 1 (naturally thin)
MIN_CHAR_H        = 5      # px — character bands shorter than this → skip
MAX_CHAR_W        = 350    # px — wider than this → likely merged word, split later
CHAR_PAD          = 4      # px — padding around each crop

# ── Visuals ───────────────────────────────────────────────────────────────────
LINE_COLORS = [
    ( 52, 152, 219),  # blue
    ( 39, 174,  96),  # green
    (231,  76,  60),  # red
    (241, 196,  15),  # yellow
    (155,  89, 182),  # purple
    (230, 126,  34),  # orange
    ( 26, 188, 156),  # teal
    (236,  95, 128),  # pink
    (  0, 180, 240),  # cyan
    (180, 230,  70),  # lime
]
FONT      = cv2.FONT_HERSHEY_SIMPLEX
BANNER_H  = 44
BANNER_BG = (26, 26, 46)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def segment_characters(processed: dict,
                       output_dir: str,
                       base_name: str) -> list:
    """
    Segment characters using vertical projection valleys within each line.

    Returns list of saved crop paths in reading order
    (top-to-bottom lines, left-to-right within each line).

    Also writes:
        <char_dir>/manifest.json
        <char_dir>/line_map.json
    """
    binary       = processed["resized"].copy()
    img_h, img_w = binary.shape
    stem         = os.path.splitext(base_name)[0]
    char_dir     = os.path.join(output_dir, f"chars_{stem}")
    os.makedirs(char_dir, exist_ok=True)

    # ── Step 1: Detect text lines ─────────────────────────────────────────────
    line_bands = _find_line_bands(binary)
    if not line_bands:
        _write_empty_manifest(char_dir)
        return []

    # ── Step 2: Per-line vertical projection → character bounding boxes ───────
    all_boxes   = []   # list of (x, y, w, h, line_idx)
    for line_idx, (y0, y1) in enumerate(line_bands):
        strip = binary[y0:y1 + 1, :]
        if strip.shape[0] == 0:
            continue
        char_cols = _find_character_columns(strip)  # list of (col_x, col_w)
        for (cx, cw) in char_cols:
            # Find tight vertical extent within this column band
            col_strip = strip[:, cx:cx + cw]
            rows_with_ink = np.where(np.any(col_strip > 0, axis=1))[0]
            if len(rows_with_ink) == 0:
                continue
            local_y0 = int(rows_with_ink[0])
            local_y1 = int(rows_with_ink[-1])
            abs_y    = y0 + local_y0
            char_h   = local_y1 - local_y0 + 1
            if char_h < MIN_CHAR_H or cw < MIN_CHAR_W:
                continue
            all_boxes.append((cx, abs_y, cw, char_h, line_idx))

    # ── Step 3: Assign global index, label, save crops ───────────────────────
    paths    = []
    manifest = {}
    line_map = {}

    for global_idx, (x, y, w, h, line_idx) in enumerate(all_boxes):
        x1 = max(0, x - CHAR_PAD)
        y1 = max(0, y - CHAR_PAD)
        x2 = min(img_w, x + w + CHAR_PAD)
        y2 = min(img_h, y + h + CHAR_PAD)

        crop  = binary[y1:y2, x1:x2]
        label = CHARSET[global_idx] if global_idx < len(CHARSET) else "?"
        fname = f"char_{global_idx:04d}.png"
        fpath = os.path.join(char_dir, fname)
        cv2.imwrite(fpath, crop)
        paths.append(fpath)

        manifest[fname] = {
            "index" : global_idx,
            "label" : label,
            "line"  : line_idx,
            "x"     : int(x1), "y": int(y1),
            "w"     : int(x2 - x1), "h": int(y2 - y1),
            "rx"    : int(x),  "ry": int(y),
            "rw"    : int(w),  "rh": int(h),
        }
        line_map.setdefault(line_idx, []).append(global_idx)

    # ── Step 4: Write manifests ───────────────────────────────────────────────
    with open(os.path.join(char_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(char_dir, "line_map.json"), "w") as f:
        json.dump({str(k): v for k, v in line_map.items()}, f, indent=2)

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# VERTICAL PROJECTION CORE
# ══════════════════════════════════════════════════════════════════════════════

def _find_character_columns(strip: np.ndarray) -> list:
    """
    Given a binary line strip (height × width), use a vertical projection
    profile to find column bands that contain characters.

    Returns list of (x_start, width) tuples in left→right order.
    Strips wider than MAX_CHAR_W are sub-divided recursively.
    """
    h, w = strip.shape
    # Column ink sums
    v_proj = np.sum(strip, axis=0).astype(float)

    if np.max(v_proj) == 0:
        return []

    # A column is part of a character if its ink sum exceeds VALLEY_ABS_MIN.
    # Using an absolute floor (not % of mean) so that low-ink columns like
    # crossbars (which still have real ink) are NOT treated as gaps.
    # True inter-character gaps have ink sum ≈ 0 from well-preprocessed images.
    is_ink = v_proj > VALLEY_ABS_MIN

    # Find contiguous ink bands
    bands = []
    in_band = False
    start   = 0
    for col in range(w):
        if is_ink[col] and not in_band:
            start   = col
            in_band = True
        elif not is_ink[col] and in_band:
            band_w = col - start
            if band_w >= MIN_CHAR_W:
                bands.append((start, band_w))
            in_band = False
    if in_band:
        band_w = w - start
        if band_w >= MIN_CHAR_W:
            bands.append((start, band_w))

    # Sub-divide any band that is suspiciously wide
    # ── Merge bands separated by tiny gaps (≤ MIN_VALLEY_W) ─────────────────
    # Handles characters with detached strokes like k, x, z whose diagonal
    # may have a 1-2 px column gap from the main vertical stroke.
    merged = []
    for bx, bw in bands:
        if merged:
            prev_end = merged[-1][0] + merged[-1][1]
            gap = bx - prev_end
            if gap <= MIN_VALLEY_W:
                # Extend the previous band to absorb this one
                merged[-1] = (merged[-1][0],
                               bx + bw - merged[-1][0])
                continue
        merged.append((bx, bw))

    result = []
    for (bx, bw) in merged:
        if bw > MAX_CHAR_W:
            sub = _split_wide_band(strip[:, bx:bx + bw])
            result.extend([(bx + sx, sw) for (sx, sw) in sub])
        else:
            result.append((bx, bw))

    return result


def _split_wide_band(strip: np.ndarray) -> list:
    """
    Recursively split a wide column band by finding its deepest valley.
    Used when two characters have no true zero-column gap between them
    (e.g. 'm' and 'n' written close together).
    """
    h, w = strip.shape
    v_proj = np.sum(strip, axis=0).astype(float)

    # Find the column with the minimum ink sum in the middle 80% of the band
    margin  = max(MIN_CHAR_W, w // 10)
    search  = v_proj[margin: w - margin]
    if len(search) == 0:
        return [(0, w)]
    split_col = int(np.argmin(search)) + margin

    left_w  = split_col
    right_w = w - split_col
    result  = []
    if left_w  >= MIN_CHAR_W: result.append((0,         left_w))
    if right_w >= MIN_CHAR_W: result.append((split_col, right_w))
    return result if result else [(0, w)]


# ══════════════════════════════════════════════════════════════════════════════
# LINE DETECTION (unchanged from previous version)
# ══════════════════════════════════════════════════════════════════════════════

def _find_line_bands(binary: np.ndarray) -> list:
    """
    Return list of (y_start, y_end) pixel bands containing text rows,
    derived from the horizontal projection profile.
    """
    h_proj    = np.sum(binary, axis=1).astype(float)
    max_v     = float(np.max(h_proj)) if np.max(h_proj) > 0 else 1.0
    threshold = max_v * LINE_PROJ_THRESH

    in_text = h_proj > threshold
    bands   = []
    start   = None
    for row_idx, active in enumerate(in_text):
        if active and start is None:
            start = row_idx
        elif not active and start is not None:
            bands.append((start, row_idx - 1))
            start = None
    if start is not None:
        bands.append((start, len(h_proj) - 1))

    # If projection gives nothing useful, fall back to full image
    if not bands:
        return [(0, binary.shape[0] - 1)]

    return bands


def _find_line_boundaries_from_projection(h_proj: np.ndarray) -> list:
    """Return y-positions of line break boundaries (for debug image)."""
    max_v     = float(np.max(h_proj)) if np.max(h_proj) > 0 else 1.0
    threshold = max_v * LINE_PROJ_THRESH
    in_text   = h_proj > threshold
    bounds    = []
    for i in range(1, len(in_text)):
        if in_text[i - 1] and not in_text[i]:
            bounds.append(i)
    return bounds


# ══════════════════════════════════════════════════════════════════════════════
# VISUAL OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_segmentation_overview(processed: dict,
                                char_paths: list,
                                output_dir: str,
                                base_name: str,
                                char_dir: str = None) -> str:
    """Annotated image: bounding boxes colour-coded by line, labelled."""
    binary = processed["resized"].copy()
    vis    = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    vis    = _scale_to_max_width(vis, 1280)
    h, w   = vis.shape[:2]
    orig_w = processed["resized"].shape[1]
    sf     = w / orig_w

    stem = os.path.splitext(base_name)[0]
    if char_dir is None:
        char_dir = os.path.join(output_dir, f"chars_{stem}")
    manifest = _load_json(os.path.join(char_dir, "manifest.json"))
    if not manifest:
        return ""

    boxes_info = sorted(manifest.values(), key=lambda b: b["index"])
    n_lines    = max((b["line"] for b in boxes_info), default=0) + 1

    # Draw each bounding box
    for info in boxes_info:
        idx   = info["index"]
        label = info.get("label", "?")
        line  = info.get("line", 0)
        color = LINE_COLORS[line % len(LINE_COLORS)]

        rx = int(info.get("rx", info["x"]) * sf)
        ry = int(info.get("ry", info["y"]) * sf)
        rw = int(info.get("rw", info["w"]) * sf)
        rh = int(info.get("rh", info["h"]) * sf)

        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), color, 1)

        tag = f'#{idx}"{label}"'
        fs  = 0.28
        (tw, th), bl = cv2.getTextSize(tag, FONT, fs, 1)
        tx = max(0, rx)
        ty = max(th + 2, ry - 2)
        cv2.rectangle(vis, (tx-1, ty-th-1), (tx+tw+2, ty+bl+1), color, -1)
        cv2.putText(vis, tag, (tx, ty), FONT, fs, (0,0,0), 1, cv2.LINE_AA)

    # Line sidebar stripes
    line_yranges = {}
    for info in boxes_info:
        ln = info["line"]
        ry = int(info.get("ry", info["y"]) * sf)
        rh = int(info.get("rh", info["h"]) * sf)
        if ln not in line_yranges:
            line_yranges[ln] = [ry, ry + rh]
        else:
            line_yranges[ln][0] = min(line_yranges[ln][0], ry)
            line_yranges[ln][1] = max(line_yranges[ln][1], ry + rh)

    for ln, (y0, y1) in line_yranges.items():
        color = LINE_COLORS[ln % len(LINE_COLORS)]
        cv2.rectangle(vis, (0, y0), (6, y1), color, -1)
        cv2.putText(vis, f"L{ln}", (8, y0+12), FONT, 0.35, color, 1, cv2.LINE_AA)

    # Banner
    banner = np.zeros((BANNER_H, w, 3), dtype=np.uint8)
    banner[:] = BANNER_BG
    banner[:, :5] = (231, 76, 60)
    cv2.putText(banner,
                f"Segmentation  —  {len(boxes_info)} chars  |  {n_lines} line(s)"
                f"  |  method: vertical projection valleys",
                (12, 20), FONT, 0.48, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(banner,
                "Colour = line  •  Tag: #index\"label\"  •  Box = column-band extent",
                (12, 38), FONT, 0.34, (180,180,180), 1, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"seg_overview_{stem}.png")
    cv2.imwrite(out_path, np.vstack([banner, vis]))
    return out_path


def save_character_atlas_sheet(char_paths: list,
                                output_dir: str,
                                base_name: str,
                                tile_size: int = 72,
                                cols: int = 10,
                                char_dir: str = None) -> str:
    """Tiled grid of all character crops, labelled by index + charset label."""
    if not char_paths:
        return ""

    stem = os.path.splitext(base_name)[0]
    if char_dir is None:
        char_dir = os.path.join(output_dir, f"chars_{stem}")
    manifest = _load_json(os.path.join(char_dir, "manifest.json"))
    if not manifest:
        return ""

    label_map = {fname: info for fname, info in manifest.items()}

    LABEL_H = 20
    cell_h  = tile_size + LABEL_H
    n       = len(char_paths)
    rows    = math.ceil(n / cols)

    sheet = np.zeros((rows * cell_h, cols * tile_size, 3), dtype=np.uint8)
    sheet[:] = (18, 18, 28)

    for idx, path in enumerate(char_paths):
        row = idx // cols
        col = idx  % cols
        x0  = col * tile_size
        y0  = row * cell_h

        fname = os.path.basename(path)
        info  = label_map.get(fname, {})
        label = info.get("label", "?")
        line  = info.get("line",   0)
        color = LINE_COLORS[line % len(LINE_COLORS)]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        tile_bgr = (_fit_crop_to_tile(img, tile_size, color)
                    if img is not None
                    else np.zeros((tile_size, tile_size, 3), dtype=np.uint8))
        sheet[y0:y0+tile_size, x0:x0+tile_size] = tile_bgr

        # Label strip
        ly0 = y0 + tile_size
        cv2.rectangle(sheet, (x0, ly0), (x0+tile_size, y0+cell_h), color, -1)
        cv2.putText(sheet, f"#{idx}",
                    (x0+2, ly0+9), FONT, 0.28, (0,0,0), 1, cv2.LINE_AA)
        lbl_txt = f'"{label}"'
        (lw, _), _ = cv2.getTextSize(lbl_txt, FONT, 0.34, 1)
        lx = x0 + (tile_size - lw) // 2
        cv2.putText(sheet, lbl_txt, (lx, ly0+16), FONT, 0.34, (0,0,0), 1, cv2.LINE_AA)
        cv2.rectangle(sheet, (x0, y0), (x0+tile_size, y0+cell_h), color, 1)

    bw = sheet.shape[1]
    banner = np.zeros((BANNER_H, bw, 3), dtype=np.uint8)
    banner[:] = BANNER_BG
    banner[:, :5] = (155, 89, 182)
    cv2.putText(banner,
                f"Character Atlas  —  {n} crops  |  colour = writing line"
                f"  |  label = CHARSET[index]",
                (12, 20), FONT, 0.50, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(banner,
                "Order: top-to-bottom lines, left-to-right columns  →  matches a-z A-Z 0-9",
                (12, 38), FONT, 0.34, (180,180,180), 1, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"seg_atlas_{stem}.png")
    cv2.imwrite(out_path, np.vstack([banner, sheet]))
    return out_path


def save_line_debug_image(processed: dict,
                           output_dir: str,
                           base_name: str) -> str:
    """Horizontal projection profile graph with detected line boundaries."""
    binary = processed["resized"]
    img_h, img_w = binary.shape
    stem = os.path.splitext(base_name)[0]

    h_proj = np.sum(binary, axis=1).astype(float)
    max_v  = float(np.max(h_proj)) if np.max(h_proj) > 0 else 1.0

    GRAPH_W  = 400
    canvas_h = img_h
    canvas   = np.zeros((canvas_h, GRAPH_W, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 28)

    thresh_v = max_v * LINE_PROJ_THRESH
    for row in range(min(img_h, canvas_h)):
        bar_len = int(h_proj[row] / max_v * (GRAPH_W - 60))
        col_bg  = (39,174,96) if h_proj[row] > thresh_v else (70,70,90)
        cv2.line(canvas, (0, row), (bar_len, row), col_bg, 1)

    thresh_x = int(thresh_v / max_v * (GRAPH_W - 60))
    cv2.line(canvas, (thresh_x,0), (thresh_x, canvas_h-1), (231,76,60), 1)

    boundaries = _find_line_boundaries_from_projection(h_proj)
    for by in boundaries:
        if by < canvas_h:
            cv2.line(canvas, (0,by), (GRAPH_W-1,by), (241,196,15), 1)
            cv2.putText(canvas, "line break", (2, by-2),
                        FONT, 0.28, (241,196,15), 1, cv2.LINE_AA)

    banner = np.zeros((BANNER_H, GRAPH_W, 3), dtype=np.uint8)
    banner[:] = BANNER_BG
    banner[:, :5] = (241, 196, 15)
    cv2.putText(banner, "Horizontal Projection Profile",
                (10, 20), FONT, 0.46, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(banner, "Green=ink rows  Red=threshold  Yellow=line breaks",
                (10, 38), FONT, 0.32, (180,180,180), 1, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"seg_linedebug_{stem}.png")
    cv2.imwrite(out_path, np.vstack([banner, canvas]))
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# TILE / DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fit_crop_to_tile(img: np.ndarray, tile: int, color) -> np.ndarray:
    h, w   = img.shape
    pad    = 4
    scale  = (tile - pad * 2) / max(h, w, 1)
    new_w  = max(1, int(w * scale))
    new_h  = max(1, int(h * scale))
    small  = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((tile, tile, 3), (30,30,45), dtype=np.uint8)
    y0 = (tile - new_h) // 2
    x0 = (tile - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w][small > 0] = color
    return canvas


def _fit_to_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w   = img.shape
    scale  = size / max(h, w, 1)
    new_w  = max(1, int(w * scale))
    new_h  = max(1, int(h * scale))
    r      = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    yo = (size - new_h) // 2
    xo = (size - new_w) // 2
    canvas[yo:yo+new_h, xo:xo+new_w] = r
    return canvas


def _scale_to_max_width(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _write_empty_manifest(char_dir: str) -> None:
    with open(os.path.join(char_dir, "manifest.json"), "w") as f:
        json.dump({}, f)
    with open(os.path.join(char_dir, "line_map.json"), "w") as f:
        json.dump({}, f)


# ── Backward compatibility ────────────────────────────────────────────────────
def segment_lines(binary: np.ndarray) -> list:
    return _find_line_bands(binary)

def build_character_atlas(char_paths: list, char_size: int = 32) -> dict:
    atlas = {}
    for path in char_paths:
        key = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        atlas[key] = _fit_to_square(img, char_size)
    return atlas
