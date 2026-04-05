"""
utils/segmentation.py
─────────────────────
ForgeNet-X — Character Segmentation Module  (Windows-compatible)

THE CORE PROBLEM WITH THE OLD APPROACH
───────────────────────────────────────
Old code sorted contours with  key=lambda b: (b[1] // 20, b[0])
This divides every y-coordinate by 20 to decide which "row" a character
belongs to.  That is wrong in three ways:

  1. The divisor 20 has no relation to actual character height.  A 60-px
     tall letter at y=0 and one at y=19 land in different row buckets even
     though they are on the same line.

  2. On a blank (unruled) page, lines have no fixed pixel position.
     The only way to find them is to CLUSTER the bounding-box tops, not
     divide by an arbitrary constant.

  3. Multi-stroke characters (i, j, !, :, ;) have their dot as a separate
     contour.  Each dot consumed a charset slot, shifting every label after
     it by +1 forever.

NEW PIPELINE
────────────
Step 1  Extract all external contours, filter by size.

Step 2  Merge nearby vertically-overlapping contours.
        If two boxes overlap vertically by > VERTICAL_OVERLAP_THRESH % of
        the shorter one's height AND their horizontal gap is less than
        MAX_DOT_MERGE_GAP px, they are merged into one bounding box.
        This handles i-dot, j-dot, !, :, ;, and lightly touching strokes.

Step 3  Line detection via horizontal projection profile on the BINARY
        image (not on arbitrary box y-coordinates).
        We sum pixel rows → find peaks of ink density → between each pair
        of peaks there is a gap row → use those gaps as line boundaries.
        Fallback: if projection gives too few lines, cluster box centres
        by y-position using a gap-threshold equal to median_char_height.

Step 4  Within each detected line, sort boxes left → right by x.
        Concatenate lines top → bottom.  This is the true reading order.

Step 5  Crop, pad, save each character.  Label = CHARSET[global_index].

Step 6  Write manifest JSON and line-map JSON (which charset indices
        belong to which line, for debugging / PDF annotation).

VISUAL OUTPUTS
──────────────
  save_segmentation_overview()   — bounding boxes on binary image,
                                   colour-coded by line, labelled with
                                   #index "char"

  save_character_atlas_sheet()   — tiled crops grid, 10 per row,
                                   with index + label under each tile

  save_line_debug_image()        — horizontal projection profile graph
                                   with detected line boundaries overlaid
"""

import os
import json
import math
import cv2
import numpy as np


# ── Charset (must match handwriting.py) ──────────────────────────────────────
CHARSET = (
    list("abcdefghijklmnopqrstuvwxyz") +
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    list("0123456789") +
    list(".,!?;:'\"()-")
)

# ── Size filter ───────────────────────────────────────────────────────────────
MIN_CHAR_W = 4     # px  — narrower boxes are noise
MIN_CHAR_H = 6     # px  — shorter boxes are noise
MAX_CHAR_W = 300   # px  — wider boxes are likely merged words (reject)
MAX_CHAR_H = 400   # px  — taller than this is likely a page artefact

# ── Dot/stroke merge parameters ───────────────────────────────────────────────
# A satellite (dot/accent) is merged into a parent ONLY when ALL of these hold:
#
#   1. satellite area  <  DOT_AREA_RATIO × median character area
#      Keeps full-sized letters like 'i' body from being treated as dots.
#
#   2. vertical gap between the two boxes  ≤  VERT_GAP_MERGE px
#      The dot sits just above the letter; the gap is tiny.
#
#   3. horizontal offset between centres  ≤  DOT_MAX_HORIZ_OFFSET × parent_width
#      A real i-dot sits almost directly above the body (offset < ~0.6×width).
#      Two side-by-side letters like 'h' and 'i' have offset >> parent_width,
#      so this rule blocks the false merge.
#
#   4. The satellite must be ABOVE the parent (top of satellite < top of parent)
#      or overlap vertically. Side-by-side letters are at the same Y → blocked.
#      Exception: descender dots (j, !) can be below.

DOT_AREA_RATIO        = 0.20   # satellite area must be < 20% of median area
VERT_GAP_MERGE        = 6     # px  max vertical gap between dot bottom and body top
DOT_MAX_HORIZ_OFFSET  = 0.7   # dot centre must be within 70% of parent width
                               # from parent centre — blocks side-by-side merges

# ── Line detection ────────────────────────────────────────────────────────────
# Projection profile: a pixel row is "text" if its ink sum > this fraction
# of the maximum ink-row sum in the whole image.
LINE_PROJ_THRESH    = 0.04   # 4 % of max row sum

# Fallback clustering: if projection gives < 2 lines, cluster by y-gap.
# Two boxes are on different lines if their y-centre gap > this × median_h.
LINE_GAP_FACTOR     = 0.6    # 60 % of median character height

# ── Crop padding ──────────────────────────────────────────────────────────────
CHAR_PAD = 4   # px added on all sides of each crop

# ── Visual constants ──────────────────────────────────────────────────────────
# One colour per line (cycles if > 10 lines)
LINE_COLORS = [
    (52,  152, 219),  # blue
    (39,  174,  96),  # green
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
    Full segmentation pipeline.  Returns list of saved crop paths in
    correct reading order (line by line, left to right within each line).
    Also writes:
        <char_dir>/manifest.json     — per-crop metadata + label
        <char_dir>/line_map.json     — which global indices are on which line
    """
    binary   = processed["resized"].copy()
    img_h, img_w = binary.shape
    stem     = os.path.splitext(base_name)[0]
    char_dir = os.path.join(output_dir, f"chars_{stem}")
    os.makedirs(char_dir, exist_ok=True)

    # ── Step 1: Raw contours → filtered bounding boxes ────────────────────────
    raw_boxes = _extract_raw_boxes(binary)
    if not raw_boxes:
        _write_empty_manifest(char_dir)
        return []

    # ── Step 2: Merge dot strokes into their parent characters ────────────────
    merged_boxes = _merge_dot_strokes(raw_boxes)

    # ── Step 3: Detect text lines ─────────────────────────────────────────────
    lines_of_boxes = _assign_lines(binary, merged_boxes)

    # ── Step 4: Sort within lines → build global reading-order list ───────────
    ordered_boxes = []
    line_map      = {}   # line_idx → [global_idx, ...]
    global_idx    = 0
    for line_idx, line_boxes in enumerate(lines_of_boxes):
        # Sort each line left → right by box x
        line_boxes_sorted = sorted(line_boxes, key=lambda b: b[0])
        line_map[line_idx] = []
        for box in line_boxes_sorted:
            ordered_boxes.append((box, line_idx))
            line_map[line_idx].append(global_idx)
            global_idx += 1

    # ── Step 5: Crop, save, build manifest ────────────────────────────────────
    paths    = []
    manifest = {}

    for idx, (box, line_idx) in enumerate(ordered_boxes):
        x, y, w, h = box
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
            "index"    : idx,
            "label"    : label,
            "line"     : line_idx,
            "x"        : int(x1),
            "y"        : int(y1),
            "w"        : int(x2 - x1),
            "h"        : int(y2 - y1),
            # raw box (before padding) for overlap drawing
            "rx"       : int(x),
            "ry"       : int(y),
            "rw"       : int(w),
            "rh"       : int(h),
        }

    # ── Step 6: Write manifests ───────────────────────────────────────────────
    with open(os.path.join(char_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(char_dir, "line_map.json"), "w") as f:
        json.dump({str(k): v for k, v in line_map.items()}, f, indent=2)

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# VISUAL OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_segmentation_overview(processed: dict,
                                char_paths: list,
                                output_dir: str,
                                base_name: str,
                                char_dir: str = None) -> str:
    """
    Annotated image showing:
      • Bounding boxes colour-coded by line number
      • Label tag above each box: #index "char"
      • Line separator markers on the left edge
    """
    binary = processed["resized"].copy()
    vis    = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    vis    = _scale_to_max_width(vis, 1280)
    h, w   = vis.shape[:2]
    orig_w = processed["resized"].shape[1]
    sf     = w / orig_w

    stem = os.path.splitext(base_name)[0]
    # char_dir is passed explicitly; fall back to output_dir/chars_{stem}
    if char_dir is None:
        char_dir = os.path.join(output_dir, f"chars_{stem}")
    manifest = _load_json(os.path.join(char_dir, "manifest.json"))
    line_map = _load_json(os.path.join(char_dir, "line_map.json"))

    if not manifest:
        return ""

    boxes_info = sorted(manifest.values(), key=lambda b: b["index"])
    n_lines    = max((b["line"] for b in boxes_info), default=0) + 1

    for info in boxes_info:
        idx   = info["index"]
        label = info.get("label", "?")
        line  = info.get("line", 0)
        color = LINE_COLORS[line % len(LINE_COLORS)]

        # Use raw (un-padded) box coords for drawing
        rx = int(info.get("rx", info["x"]) * sf)
        ry = int(info.get("ry", info["y"]) * sf)
        rw = int(info.get("rw", info["w"]) * sf)
        rh = int(info.get("rh", info["h"]) * sf)

        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), color, 1)

        # Label tag: "#idx «char»"
        tag = f'#{idx} "{label}"'
        fs  = 0.30
        (tw, th), bl = cv2.getTextSize(tag, FONT, fs, 1)
        tx = max(0, rx)
        ty = max(th + 2, ry - 2)
        cv2.rectangle(vis,
                      (tx - 1, ty - th - 1),
                      (tx + tw + 2, ty + bl + 1),
                      color, -1)
        cv2.putText(vis, tag, (tx, ty), FONT, fs, (0, 0, 0), 1, cv2.LINE_AA)

    # ── Line number sidebar ───────────────────────────────────────────────────
    # Find y-range of each line and draw a coloured stripe on the left
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
        label_txt = f"L{ln}"
        cv2.putText(vis, label_txt, (8, y0 + 12),
                    FONT, 0.35, color, 1, cv2.LINE_AA)

    # ── Banner ────────────────────────────────────────────────────────────────
    banner = np.zeros((BANNER_H, w, 3), dtype=np.uint8)
    banner[:] = BANNER_BG
    banner[:, :5] = (231, 76, 60)
    cv2.putText(banner,
                f"Segmentation Overview  —  {len(boxes_info)} characters  "
                f"across {n_lines} line(s)  •  colour = line number",
                (12, 20), FONT, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner,
                "Boxes: raw contour bounding rect  "
                "•  Tag: #global_index \"charset_label\"  "
                "•  Left stripe: line band",
                (12, 38), FONT, 0.34, (180, 180, 180), 1, cv2.LINE_AA)

    out_path = os.path.join(output_dir, f"seg_overview_{stem}.png")
    cv2.imwrite(out_path, np.vstack([banner, vis]))
    return out_path


def save_character_atlas_sheet(char_paths: list,
                                output_dir: str,
                                base_name: str,
                                tile_size: int = 72,
                                cols: int = 10,
                                char_dir: str = None) -> str:
    """
    Grid of all character crops, 10 per row (one row = one line of writing).
    Colour of each tile matches its line colour from the overview.
    """
    if not char_paths:
        return ""

    stem = os.path.splitext(base_name)[0]
    # char_dir is passed explicitly; fall back to output_dir/chars_{stem}
    if char_dir is None:
        char_dir = os.path.join(output_dir, f"chars_{stem}")
    manifest = _load_json(os.path.join(char_dir, "manifest.json"))
    if not manifest:
        return ""

    # fname → metadata
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

        # Tile image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        tile_bgr = (_fit_crop_to_tile(img, tile_size, color)
                    if img is not None
                    else np.zeros((tile_size, tile_size, 3), dtype=np.uint8))
        sheet[y0:y0 + tile_size, x0:x0 + tile_size] = tile_bgr

        # Label strip
        ly0 = y0 + tile_size
        cv2.rectangle(sheet, (x0, ly0), (x0 + tile_size, y0 + cell_h), color, -1)

        # #index top-left of strip
        cv2.putText(sheet, f"#{idx}",
                    (x0 + 2, ly0 + 9),
                    FONT, 0.28, (0, 0, 0), 1, cv2.LINE_AA)

        # label centred
        lbl_txt = f'"{label}"'
        (lw, _), _ = cv2.getTextSize(lbl_txt, FONT, 0.34, 1)
        lx = x0 + (tile_size - lw) // 2
        cv2.putText(sheet, lbl_txt, (lx, ly0 + 16),
                    FONT, 0.34, (0, 0, 0), 1, cv2.LINE_AA)

        # tile border
        cv2.rectangle(sheet, (x0, y0), (x0 + tile_size, y0 + cell_h), color, 1)

    # Banner
    bw = sheet.shape[1]
    banner = np.zeros((BANNER_H, bw, 3), dtype=np.uint8)
    banner[:] = BANNER_BG
    banner[:, :5] = (155, 89, 182)
    cv2.putText(banner,
                f"Character Atlas  —  {n} crops  •  colour = writing line  "
                f"•  label = CHARSET[index]",
                (12, 20), FONT, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner,
                "Order: top-to-bottom lines, left-to-right within each line  "
                "→  matches charset mapping a-z A-Z 0-9 ...",
                (12, 38), FONT, 0.34, (180, 180, 180), 1, cv2.LINE_AA)

    out_path = os.path.join(output_dir, f"seg_atlas_{stem}.png")
    cv2.imwrite(out_path, np.vstack([banner, sheet]))
    return out_path


def save_line_debug_image(processed: dict,
                           output_dir: str,
                           base_name: str) -> str:
    """
    Horizontal projection profile graph with detected line boundaries overlaid.
    Useful for verifying that line detection found the right gaps.
    """
    binary = processed["resized"]
    img_h, img_w = binary.shape
    stem = os.path.splitext(base_name)[0]

    h_proj = np.sum(binary, axis=1).astype(float)
    max_v  = float(np.max(h_proj)) if np.max(h_proj) > 0 else 1.0

    GRAPH_W = 400
    canvas_h = img_h
    canvas   = np.zeros((canvas_h, GRAPH_W, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 28)

    # Draw projection bars
    thresh_v = max_v * LINE_PROJ_THRESH
    for row in range(img_h):
        if row >= canvas_h:
            break
        bar_len = int(h_proj[row] / max_v * (GRAPH_W - 60))
        col_bg  = (39, 174, 96) if h_proj[row] > thresh_v else (70, 70, 90)
        cv2.line(canvas, (0, row), (bar_len, row), col_bg, 1)

    # Threshold line
    thresh_x = int(thresh_v / max_v * (GRAPH_W - 60))
    cv2.line(canvas, (thresh_x, 0), (thresh_x, canvas_h - 1),
             (231, 76, 60), 1)

    # Detect and draw line boundaries
    boundaries = _find_line_boundaries_from_projection(h_proj)
    for by in boundaries:
        if by < canvas_h:
            cv2.line(canvas, (0, by), (GRAPH_W - 1, by), (241, 196, 15), 1)
            cv2.putText(canvas, "line break", (2, by - 2),
                        FONT, 0.28, (241, 196, 15), 1, cv2.LINE_AA)

    # Banner
    banner = np.zeros((BANNER_H, GRAPH_W, 3), dtype=np.uint8)
    banner[:] = BANNER_BG
    banner[:, :5] = (241, 196, 15)
    cv2.putText(banner, "Horizontal Projection Profile",
                (10, 20), FONT, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner,
                f"Green=ink rows  Red=threshold  Yellow=line breaks",
                (10, 38), FONT, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

    combined = np.vstack([banner, canvas])
    out_path = os.path.join(output_dir, f"seg_linedebug_{stem}.png")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(out_path, combined)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# CORE SEGMENTATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _extract_raw_boxes(binary: np.ndarray) -> list:
    """
    Run findContours and return filtered (x,y,w,h) tuples.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (MIN_CHAR_W <= w <= MAX_CHAR_W and
                MIN_CHAR_H <= h <= MAX_CHAR_H):
            boxes.append((x, y, w, h))
    return boxes


def _merge_dot_strokes(boxes: list) -> list:
    """
    Merge i-dots, j-dots, and accent marks into their parent character box.

    WHY THE OLD APPROACH MERGED 'h' AND 'i'
    ─────────────────────────────────────────
    The old code checked only (a) area < threshold and (b) horizontal centre
    distance < fixed px limit.  The 'i' body is a thin vertical stroke whose
    area is often < 25% of the median — so it was classified as a satellite.
    'h' and 'i' are side-by-side on the same baseline, so their horizontal
    centre distance is small.  Both conditions passed → false merge.

    NEW RULES — all four must hold for a merge to happen
    ──────────────────────────────────────────────────────
    1. satellite area  <  DOT_AREA_RATIO × median_area
       (i-dot is tiny relative to most characters; 'i' body is not)

    2. vertical gap  ≤  VERT_GAP_MERGE px
       (dot is just above or just touching the body)

    3. horizontal offset between centres  ≤  DOT_MAX_HORIZ_OFFSET × parent_width
       A real i-dot sits nearly directly above the body: offset ≈ 0.
       Two adjacent letters like h/i have offset ≈ 1–3× the parent width.
       This single rule eliminates all false side-by-side merges.

    4. satellite must be ABOVE the parent top, OR overlap it vertically.
       Side-by-side letters share almost the same y-band → their vertical
       centres are equal → the satellite is not "above" the parent.
       Real dots sit above the midpoint of the parent body.
    """
    if not boxes:
        return []

    areas        = [w * h for (x, y, w, h) in boxes]
    median_area  = float(np.median(areas))
    small_thresh = median_area * DOT_AREA_RATIO

    rects  = [[x, y, w, h] for (x, y, w, h) in boxes]
    # Largest-first so parents are processed before their potential satellites
    order  = sorted(range(len(rects)), key=lambda i: areas[i], reverse=True)
    merged_into = {}

    for i in order:
        if i in merged_into:
            continue
        ai = rects[i][2] * rects[i][3]
        if ai >= small_thresh:
            continue   # this is a main character box, not a satellite

        xi, yi, wi, hi = rects[i]
        cx_i = xi + wi / 2.0
        cy_i = yi + hi / 2.0

        best_parent = None
        best_score  = float("inf")   # lower = better (vertical gap)

        for j in order:
            if j == i or j in merged_into:
                continue
            aj = rects[j][2] * rects[j][3]
            if aj < small_thresh:
                continue   # never merge small into small

            xj, yj, wj, hj = rects[j]
            cx_j = xj + wj / 2.0
            cy_j = yj + hj / 2.0

            # ── Rule 3: horizontal proximity relative to parent width ─────────
            horiz_offset = abs(cx_i - cx_j)
            if wj > 0 and horiz_offset > DOT_MAX_HORIZ_OFFSET * wj:
                continue   # too far sideways — side-by-side letter, not a dot

            # ── Rule 2: vertical gap ──────────────────────────────────────────
            top_i, bot_i = yi, yi + hi
            top_j, bot_j = yj, yj + hj
            vert_gap = max(top_i - bot_j,   # satellite above parent
                           top_j - bot_i,   # satellite below parent
                           0)
            if vert_gap > VERT_GAP_MERGE:
                continue   # too far vertically

            # ── Rule 4: satellite must be above parent midpoint ───────────────
            # (allows j-descender dots which are below, but blocks side-by-side)
            # We allow merges where the satellite centre is above the parent
            # centre, OR where they genuinely overlap vertically.
            overlaps_vertically = not (bot_i <= top_j or bot_j <= top_i)
            satellite_above     = cy_i < cy_j   # satellite centre above parent centre

            if not overlaps_vertically and not satellite_above:
                # satellite is below the parent and they don't overlap —
                # could be a j-dot; allow only if it's very close
                if vert_gap > 3:
                    continue

            if vert_gap < best_score:
                best_score  = vert_gap
                best_parent = j

        if best_parent is not None:
            px, py, pw, ph = rects[best_parent]
            new_x  = min(px, xi)
            new_y  = min(py, yi)
            new_x2 = max(px + pw, xi + wi)
            new_y2 = max(py + ph, yi + hi)
            rects[best_parent] = [new_x, new_y, new_x2 - new_x, new_y2 - new_y]
            merged_into[i] = best_parent

    return [tuple(r) for i, r in enumerate(rects) if i not in merged_into]


def _assign_lines(binary: np.ndarray, boxes: list) -> list:
    """
    Cluster bounding boxes into text lines.

    Method 1 — Horizontal projection profile (preferred)
    ─────────────────────────────────────────────────────
    Sum pixel values across each row of the binary image.
    Rows where the sum falls below LINE_PROJ_THRESH × max_sum are "gaps".
    A text line is a contiguous band of non-gap rows.
    Each box is assigned to the line whose y-band its vertical centre falls in.

    Method 2 — Y-centre clustering (fallback)
    ─────────────────────────────────────────
    If the projection gives fewer than 2 bands (e.g. very sparse image),
    fall back to sorting boxes by y-centre and splitting whenever the gap
    between consecutive sorted y-centres exceeds LINE_GAP_FACTOR × median_h.

    Returns
    ───────
    list of lists:  [ [boxes_in_line_0], [boxes_in_line_1], … ]
    Each inner list is in the original (un-sorted) order — caller sorts L→R.
    """
    if not boxes:
        return []

    # ── Method 1: projection profile ─────────────────────────────────────────
    line_bands = _find_line_bands(binary)

    if len(line_bands) >= 2 or (len(line_bands) == 1 and len(boxes) > 0):
        assignment = _assign_boxes_to_bands(boxes, line_bands)
        if _assignment_looks_valid(assignment):
            return assignment

    # ── Method 2: y-centre gap clustering ────────────────────────────────────
    return _cluster_by_y_gap(boxes)


def _find_line_bands(binary: np.ndarray) -> list:
    """
    Return list of (y_start, y_end) pixel bands that contain text rows,
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

    return bands


def _find_line_boundaries_from_projection(h_proj: np.ndarray) -> list:
    """Return list of y-pixel positions where line breaks occur (for debug image)."""
    max_v     = float(np.max(h_proj)) if np.max(h_proj) > 0 else 1.0
    threshold = max_v * LINE_PROJ_THRESH
    in_text   = h_proj > threshold
    boundaries = []
    for i in range(1, len(in_text)):
        if in_text[i - 1] and not in_text[i]:
            boundaries.append(i)
    return boundaries


def _assign_boxes_to_bands(boxes: list, bands: list) -> list:
    """
    Place each box into the band whose y-range contains the box's y-centre.
    Boxes that fall outside all bands go into the nearest band.
    Returns list of lists (one per band, preserving band order).
    """
    lines = [[] for _ in bands]
    for box in boxes:
        x, y, w, h = box
        cy = y + h // 2   # vertical centre of this box

        best_band = None
        best_dist = float("inf")
        for band_idx, (y0, y1) in enumerate(bands):
            band_cy = (y0 + y1) / 2
            if y0 <= cy <= y1:
                best_band = band_idx
                break
            dist = min(abs(cy - y0), abs(cy - y1))
            if dist < best_dist:
                best_dist = dist
                best_band = band_idx

        if best_band is not None:
            lines[best_band].append(box)

    # Drop empty lines (can happen if a band had no boxes)
    return [ln for ln in lines if ln]


def _assignment_looks_valid(assignment: list) -> bool:
    """
    Sanity check: the assignment is valid if every line has at least
    one box, and no single line contains more than 80 % of all boxes
    (which would indicate a mis-detected single-line scenario).
    """
    if not assignment:
        return False
    total = sum(len(ln) for ln in assignment)
    if total == 0:
        return False
    max_in_line = max(len(ln) for ln in assignment)
    # If one line has everything, projection likely failed
    if len(assignment) == 1:
        return True   # genuinely one line — that's fine
    return max_in_line / total < 0.95


def _cluster_by_y_gap(boxes: list) -> list:
    """
    Fallback line detection: sort boxes by y-centre, then split into a
    new line whenever the jump between consecutive y-centres exceeds
    LINE_GAP_FACTOR × median_character_height.
    """
    if not boxes:
        return []

    heights     = [h for (x, y, w, h) in boxes]
    median_h    = float(np.median(heights))
    gap_thresh  = LINE_GAP_FACTOR * median_h

    # Sort by y-centre
    sorted_boxes = sorted(boxes, key=lambda b: b[1] + b[3] // 2)

    lines   = [[sorted_boxes[0]]]
    for box in sorted_boxes[1:]:
        prev_cy = lines[-1][-1][1] + lines[-1][-1][3] // 2
        curr_cy = box[1] + box[3] // 2
        if curr_cy - prev_cy > gap_thresh:
            lines.append([])   # start new line
        lines[-1].append(box)

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# TILE / DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fit_crop_to_tile(img: np.ndarray, tile: int, color) -> np.ndarray:
    """Centre a grayscale crop in a tile×tile BGR canvas; ink shown in color."""
    h, w   = img.shape
    pad    = 4
    scale  = (tile - pad * 2) / max(h, w, 1)
    new_w  = max(1, int(w * scale))
    new_h  = max(1, int(h * scale))
    small  = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((tile, tile, 3), (30, 30, 45), dtype=np.uint8)
    y0 = (tile - new_h) // 2
    x0 = (tile - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w][small > 0] = color
    return canvas


def _fit_to_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w   = img.shape
    scale  = size / max(h, w, 1)
    new_w  = max(1, int(w * scale))
    new_h  = max(1, int(h * scale))
    r      = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    yo     = (size - new_h) // 2
    xo     = (size - new_w) // 2
    canvas[yo:yo + new_h, xo:xo + new_w] = r
    return canvas


def _scale_to_max_width(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
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


# ── Kept for backward compatibility with handwriting.py ──────────────────────
def segment_lines(binary: np.ndarray) -> list:
    return [band for band in _find_line_bands(binary)]


def build_character_atlas(char_paths: list, char_size: int = 32) -> dict:
    atlas = {}
    for path in char_paths:
        key = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        atlas[key] = _fit_to_square(img, char_size)
    return atlas
