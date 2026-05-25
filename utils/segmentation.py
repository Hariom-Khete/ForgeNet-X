"""ForgeNet-X — Character Segmentation (CCA + satellite merge)"""

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
)  # 73 total

# ── Line detection ────────────────────────────────────────────────────────────
LINE_PROJ_THRESH = 0.06   # rows below 6% of peak ink treated as gaps
MIN_LINE_BAND_H  = 12
MIN_LINE_SEP     = 3      # bands separated by ≥3 blank rows kept distinct

# ── Blob filtering ────────────────────────────────────────────────────────────
MIN_BLOB_AREA    = 20
MAX_BLOB_W_RATIO = 0.40
MAX_BLOB_H_RATIO = 0.80

# ── Character size limits ─────────────────────────────────────────────────────
MIN_CHAR_W = 3
MIN_CHAR_H = 5
MAX_CHAR_W = 200
CHAR_PAD   = 4

# ── Satellite merge ───────────────────────────────────────────────────────────
SAT_AREA_RATIO    = 0.20
SAT_H_MAX_RATIO   = 0.35
SAT_V_GAP_MAX     = 18
SAT_H_OVERLAP_MIN = 0.50
PAIR_V_GAP_MAX    = 32

# ── Split ─────────────────────────────────────────────────────────────────────
SPLIT_DEPTH_RATIO    = 0.15  # valley below this % of peak → genuine gap
ADAPTIVE_SPLIT_RATIO = 2.5   # blob wider than this × p25 blob width → try split

# ── Visuals ───────────────────────────────────────────────────────────────────
LINE_COLORS = [
    ( 52, 152, 219), ( 39, 174,  96), (231,  76,  60), (241, 196,  15),
    (155,  89, 182), (230, 126,  34), ( 26, 188, 156), (236,  95, 128),
    (  0, 180, 240), (180, 230,  70),
]
FONT      = cv2.FONT_HERSHEY_SIMPLEX
BANNER_H  = 44
BANNER_BG = (26, 26, 46)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def segment_characters(processed: dict, output_dir: str, base_name: str) -> list:
    """Segment characters using CCA + line detection + satellite merge. Returns crop paths."""
    binary       = processed["resized"].copy()
    img_h, img_w = binary.shape
    stem         = os.path.splitext(base_name)[0]
    char_dir     = os.path.join(output_dir, f"chars_{stem}")
    os.makedirs(char_dir, exist_ok=True)

    # Step 1: CCA
    blobs = _extract_blobs(binary, img_w, img_h)
    if not blobs:
        _write_empty_manifest(char_dir)
        return []

    # Step 2: Line detection — horizontal projection primary, blob centroid fallback
    line_bands = _find_line_bands(binary)
    if len(line_bands) <= 1:
        MIN_PRIMARY_H = max(15, int(np.median([b['h'] for b in blobs]) * 0.3))
        primary_blobs = [b for b in blobs if b['h'] >= MIN_PRIMARY_H]
        if len(primary_blobs) < 4:
            primary_blobs = blobs
        line_bands = _find_line_bands_from_blobs(primary_blobs, img_h)
    if not line_bands:
        _write_empty_manifest(char_dir)
        return []

    # Step 3: Assign all blobs to line bands
    line_blobs = {i: [] for i in range(len(line_bands))}
    for blob in blobs:
        line_blobs[_assign_blob_to_line(blob['cy'], line_bands)].append(blob)

    # Step 4: Satellite merge + edge snap + adaptive split
    all_boxes = []
    for line_idx in range(len(line_bands)):
        lblobs = line_blobs[line_idx]

        if lblobs:
            py_top      = min(b['y']          for b in lblobs)
            py_bot      = max(b['y'] + b['h'] for b in lblobs)
            line_band_h = max(1, py_bot - py_top + 1)
            blob_widths = sorted(b['w'] for b in lblobs)
            p25_idx     = max(0, len(blob_widths) // 4 - 1)
            p25_w       = float(blob_widths[p25_idx])
            adaptive_max_w = max(int(p25_w * ADAPTIVE_SPLIT_RATIO), 35)
        else:
            y0, y1      = line_bands[line_idx]
            line_band_h = max(1, y1 - y0 + 1)
            adaptive_max_w = MAX_CHAR_W

        chars = _merge_satellites(lblobs, line_band_h)
        for (cx, cy, cw, ch) in chars:
            (cx, cy, cw, ch) = _snap_to_ink_edges(binary, cx, cy, cw, ch)
            if cw < MIN_CHAR_W or ch < MIN_CHAR_H:
                continue
            if cw > adaptive_max_w:
                sub = _split_blob_if_wide(binary, cx, cy, cw, ch, max_w=adaptive_max_w)
                for (sx, sy, sw, sh) in sub:
                    if sw >= MIN_CHAR_W and sh >= MIN_CHAR_H:
                        all_boxes.append((sx, sy, sw, sh, line_idx))
            else:
                all_boxes.append((cx, cy, cw, ch, line_idx))

    all_boxes.sort(key=lambda b: (b[4], b[0]))

    # Step 5: Structural context validation
    detected = len(all_boxes)
    expected = len(CHARSET)
    delta    = detected - expected
    quality  = ("exact" if delta == 0
                else f"near ({delta:+d})" if abs(delta) <= 3
                else f"poor ({delta:+d})")

    # Step 6: Save crops + manifests
    paths    = []
    manifest = {}
    line_map = {}

    for global_idx, (x, y, w, h, line_idx) in enumerate(all_boxes):
        x1 = max(0, x - CHAR_PAD);  y1 = max(0, y - CHAR_PAD)
        x2 = min(img_w, x + w + CHAR_PAD);  y2 = min(img_h, y + h + CHAR_PAD)

        crop  = binary[y1:y2, x1:x2]
        label = CHARSET[global_idx] if global_idx < len(CHARSET) else "?"
        fname = f"char_{global_idx:04d}.png"
        fpath = os.path.join(char_dir, fname)
        cv2.imwrite(fpath, crop)
        paths.append(fpath)

        manifest[fname] = {
            "index": global_idx, "label": label, "line": line_idx,
            "x": int(x1), "y": int(y1), "w": int(x2-x1), "h": int(y2-y1),
            "rx": int(x), "ry": int(y), "rw": int(w), "rh": int(h),
        }
        line_map.setdefault(line_idx, []).append(global_idx)

    manifest_with_meta = {
        "_meta": {
            "detected": detected, "expected": expected,
            "delta": delta, "quality": quality,
            "n_lines": len(line_bands), "charset": "a-z A-Z 0-9 punctuation",
        },
        **manifest
    }

    with open(os.path.join(char_dir, "manifest.json"), "w") as f:
        json.dump(manifest_with_meta, f, indent=2)
    with open(os.path.join(char_dir, "line_map.json"), "w") as f:
        json.dump({str(k): v for k, v in line_map.items()}, f, indent=2)

    return paths


# ══════════════════════════════════════════════════════════════════════════════
# CCA core
# ══════════════════════════════════════════════════════════════════════════════

def _extract_blobs(binary: np.ndarray, img_w: int, img_h: int) -> list:
    """Run CCA, return list of blob dicts {x, y, w, h, area, cx, cy}."""
    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8, ltype=cv2.CV_32S)

    blobs = []
    for lbl in range(1, n_labels):
        x    = int(stats[lbl, cv2.CC_STAT_LEFT])
        y    = int(stats[lbl, cv2.CC_STAT_TOP])
        w    = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h    = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        cx   = float(centroids[lbl, 0])
        cy   = float(centroids[lbl, 1])

        if area < MIN_BLOB_AREA:              continue
        if w > img_w * MAX_BLOB_W_RATIO:      continue
        if h > img_h * MAX_BLOB_H_RATIO:      continue

        blobs.append({'x': x, 'y': y, 'w': w, 'h': h, 'area': area, 'cx': cx, 'cy': cy})

    return blobs


def _assign_blob_to_line(blob_cy: float, line_bands: list) -> int:
    """Return index of the line band whose range contains blob_cy."""
    for i, (y0, y1) in enumerate(line_bands):
        if y0 <= blob_cy <= y1:
            return i
    best_i, best_d = 0, float('inf')
    for i, (y0, y1) in enumerate(line_bands):
        d = abs(blob_cy - (y0 + y1) / 2.0)
        if d < best_d:
            best_d = d; best_i = i
    return best_i


def _merge_satellites(blobs: list, line_band_h: int) -> list:
    """Merge satellite blobs (i-dots, j-dots, punctuation components) into parents."""
    if not blobs:       return []
    if len(blobs) == 1: return [(blobs[0]['x'], blobs[0]['y'], blobs[0]['w'], blobs[0]['h'])]

    areas       = sorted(b['area'] for b in blobs)
    median_area = areas[len(areas) // 2]
    sat_thresh  = median_area * SAT_AREA_RATIO
    sat_max_h   = line_band_h * SAT_H_MAX_RATIO

    def _is_sat(b):
        return b['area'] < sat_thresh and b['h'] <= sat_max_h

    primaries  = [b for b in blobs if not _is_sat(b)]
    satellites = [b for b in blobs if     _is_sat(b)]

    if not primaries:
        primaries  = list(blobs)
        satellites = []

    mboxes = [[b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']] for b in primaries]

    for sat in satellites:
        sx, sy   = sat['x'], sat['y']
        sx2, sy2 = sx + sat['w'], sy + sat['h']
        best_i, best_d = -1, float('inf')

        for i, m in enumerate(mboxes):
            mx, my, mx2, my2 = m
            v_gap     = max(0, sy - my2, my - sy2)
            if v_gap > SAT_V_GAP_MAX: continue
            h_ov      = max(0, min(sx2, mx2) - max(sx, mx))
            h_ov_frac = h_ov / max(sat['w'], mx2 - mx, 1)
            if h_ov_frac < SAT_H_OVERLAP_MIN: continue
            d = abs((sx + sx2) / 2.0 - (mx + mx2) / 2.0)
            if d < best_d:
                best_d = d; best_i = i

        if best_i >= 0:
            m = mboxes[best_i]
            mboxes[best_i] = [min(m[0], sx), min(m[1], sy), max(m[2], sx2), max(m[3], sy2)]
        else:
            mboxes.append([sx, sy, sx2, sy2])

    result = [(m[0], m[1], m[2]-m[0], m[3]-m[1]) for m in mboxes]
    result.sort(key=lambda r: r[0])

    # Two-component pair merge (for ; : " etc.)
    short_h = line_band_h * SAT_H_MAX_RATIO
    merged  = []
    skip    = set()
    for i in range(len(result)):
        if i in skip: continue
        xi, yi, wi, hi = result[i]
        if hi > short_h:
            merged.append(result[i]); continue
        best_j, best_vgap = -1, float('inf')
        for j in range(i + 1, len(result)):
            if j in skip: continue
            xj, yj, wj, hj = result[j]
            if hj > short_h: continue
            h_ov      = max(0, min(xi+wi, xj+wj) - max(xi, xj))
            h_ov_frac = h_ov / max(wi, wj, 1)
            if h_ov_frac < SAT_H_OVERLAP_MIN: continue
            v_gap = max(0, yj-(yi+hi), yi-(yj+hj))
            if v_gap > PAIR_V_GAP_MAX: continue
            if v_gap < best_vgap:
                best_vgap = v_gap; best_j = j
        if best_j >= 0:
            xj, yj, wj, hj = result[best_j]
            merged.append((min(xi,xj), min(yi,yj),
                           max(xi+wi,xj+wj)-min(xi,xj),
                           max(yi+hi,yj+hj)-min(yi,yj)))
            skip.add(best_j)
        else:
            merged.append(result[i])

    merged.sort(key=lambda r: r[0])
    return merged


def _snap_to_ink_edges(binary: np.ndarray, x: int, y: int, w: int, h: int) -> tuple:
    """Shrink bounding box to the tightest ink boundary."""
    img_h, img_w = binary.shape
    x1 = max(0, x);  y1 = max(0, y)
    x2 = min(img_w, x+w);  y2 = min(img_h, y+h)
    if x2 <= x1 or y2 <= y1: return (x, y, w, h)

    crop = binary[y1:y2, x1:x2]
    if not np.any(crop > 0): return (x, y, w, h)

    rows_ink = np.where(np.any(crop > 0, axis=1))[0]
    cols_ink = np.where(np.any(crop > 0, axis=0))[0]
    if len(rows_ink) == 0 or len(cols_ink) == 0: return (x, y, w, h)

    return (x1 + int(cols_ink[0]),  y1 + int(rows_ink[0]),
            int(cols_ink[-1]) - int(cols_ink[0]) + 1,
            int(rows_ink[-1]) - int(rows_ink[0]) + 1)


def _split_blob_if_wide(binary: np.ndarray,
                        bx: int, by: int, bw: int, bh: int,
                        max_w: int = None) -> list:
    """Recursively split an over-wide blob at its deepest vertical-projection valley."""
    if max_w is None: max_w = MAX_CHAR_W
    if bw <= max_w:   return [(bx, by, bw, bh)]

    img_h, img_w = binary.shape
    x1 = max(0, bx);  y1 = max(0, by)
    x2 = min(img_w, bx+bw);  y2 = min(img_h, by+bh)

    strip = binary[y1:y2, x1:x2]
    h, w  = strip.shape
    if w == 0 or h == 0: return [(bx, by, bw, bh)]

    v_proj   = np.sum(strip, axis=0).astype(float)
    peak_val = float(np.max(v_proj))
    if peak_val == 0: return [(bx, by, bw, bh)]

    margin = max(3, w // 8)
    search = v_proj[margin: w - margin]
    if len(search) == 0: return [(bx, by, bw, bh)]

    min_val   = float(np.min(search))
    split_rel = int(np.argmin(search)) + margin

    if min_val > peak_val * SPLIT_DEPTH_RATIO:
        return [(bx, by, bw, bh)]

    results = []
    for (sub_x, sub_w) in [(0, split_rel), (split_rel, w - split_rel)]:
        if sub_w < MIN_CHAR_W: continue
        sub = strip[:, sub_x:sub_x + sub_w]
        rows_with_ink = np.where(np.any(sub > 0, axis=1))[0]
        if len(rows_with_ink) == 0: continue
        sub_y0 = int(rows_with_ink[0]);  sub_y1 = int(rows_with_ink[-1])
        sh = sub_y1 - sub_y0 + 1
        if sub_w < MIN_CHAR_W or sh < MIN_CHAR_H: continue
        abs_x, abs_y = x1 + sub_x, y1 + sub_y0
        (abs_x, abs_y, sub_w, sh) = _snap_to_ink_edges(binary, abs_x, abs_y, sub_w, sh)
        if sub_w < MIN_CHAR_W or sh < MIN_CHAR_H: continue
        results.extend(_split_blob_if_wide(binary, abs_x, abs_y, sub_w, sh, max_w))

    return results if results else [(bx, by, bw, bh)]


# ══════════════════════════════════════════════════════════════════════════════
# Line detection
# ══════════════════════════════════════════════════════════════════════════════

def _find_line_bands_from_blobs(blobs: list, img_h: int,
                                gap_factor: float = 2.0,
                                min_gap: float = 10.0) -> list:
    """Gap detection on y-centroids + Voronoi partition → line bands."""
    if not blobs: return [(0, img_h - 1)]

    sorted_blobs = sorted(blobs, key=lambda b: b['cy'])
    ys = [b['cy'] for b in sorted_blobs]
    if len(ys) == 1: return [(0, img_h - 1)]

    gaps      = [ys[i+1] - ys[i] for i in range(len(ys)-1)]
    threshold = max(min_gap, float(np.median(gaps)) * gap_factor)

    groups = [[sorted_blobs[0]]]
    for i, g in enumerate(gaps):
        if g >= threshold: groups.append([sorted_blobs[i+1]])
        else:              groups[-1].append(sorted_blobs[i+1])

    means      = [float(np.mean([b['cy'] for b in g])) for g in groups]
    boundaries = [0]
    for i in range(len(means)-1):
        boundaries.append(int((means[i] + means[i+1]) / 2))
    boundaries.append(img_h - 1)

    return [(boundaries[k], boundaries[k+1]) for k in range(len(groups))]


def _find_line_bands(binary: np.ndarray) -> list:
    """Horizontal-projection line detection — primary line detector."""
    h_proj    = np.sum(binary, axis=1).astype(float)
    max_v     = float(np.max(h_proj)) if np.max(h_proj) > 0 else 1.0
    threshold = max_v * LINE_PROJ_THRESH
    in_text   = h_proj > threshold

    raw   = []
    start = None
    for row_idx, active in enumerate(in_text):
        if active and start is None:              start = row_idx
        elif not active and start is not None:    raw.append([start, row_idx - 1]); start = None
    if start is not None: raw.append([start, len(h_proj)-1])

    raw = [b for b in raw if b[1] - b[0] + 1 >= MIN_LINE_BAND_H]
    if raw:
        merged = [raw[0]]
        for b in raw[1:]:
            if b[0] - merged[-1][1] - 1 < MIN_LINE_SEP: merged[-1][1] = b[1]
            else:                                         merged.append(b)
        raw = merged

    return [(b[0], b[1]) for b in raw] if raw else [(0, binary.shape[0]-1)]


def _find_line_boundaries_from_projection(h_proj: np.ndarray) -> list:
    """Return y-positions of line break boundaries (for debug image)."""
    max_v     = float(np.max(h_proj)) if np.max(h_proj) > 0 else 1.0
    threshold = max_v * LINE_PROJ_THRESH
    in_text   = h_proj > threshold
    return [i for i in range(1, len(in_text)) if in_text[i-1] and not in_text[i]]


# ══════════════════════════════════════════════════════════════════════════════
# Visual outputs
# ══════════════════════════════════════════════════════════════════════════════

def save_segmentation_overview(processed: dict, char_paths: list,
                                output_dir: str, base_name: str,
                                char_dir: str = None) -> str:
    """Annotated image: bounding boxes colour-coded by line."""
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
    if not manifest: return ""

    boxes_info = sorted(
        (v for k, v in manifest.items() if not k.startswith("_")),
        key=lambda b: b["index"])
    n_lines = max((b["line"] for b in boxes_info), default=0) + 1

    for info in boxes_info:
        color = LINE_COLORS[info.get("line", 0) % len(LINE_COLORS)]
        rx = int(info.get("rx", info["x"]) * sf);  ry = int(info.get("ry", info["y"]) * sf)
        rw = int(info.get("rw", info["w"]) * sf);  rh = int(info.get("rh", info["h"]) * sf)
        cv2.rectangle(vis, (rx, ry), (rx+rw, ry+rh), color, 1)
        tag = f'#{info["index"]}"{info.get("label","?")}"'
        fs  = 0.28
        (tw, th), bl = cv2.getTextSize(tag, FONT, fs, 1)
        tx = max(0, rx);  ty = max(th+2, ry-2)
        cv2.rectangle(vis, (tx-1, ty-th-1), (tx+tw+2, ty+bl+1), color, -1)
        cv2.putText(vis, tag, (tx, ty), FONT, fs, (0,0,0), 1, cv2.LINE_AA)

    line_yranges = {}
    for info in boxes_info:
        ln = info["line"]
        ry = int(info.get("ry", info["y"]) * sf);  rh = int(info.get("rh", info["h"]) * sf)
        if ln not in line_yranges: line_yranges[ln] = [ry, ry+rh]
        else:
            line_yranges[ln][0] = min(line_yranges[ln][0], ry)
            line_yranges[ln][1] = max(line_yranges[ln][1], ry+rh)
    for ln, (y0, y1) in line_yranges.items():
        color = LINE_COLORS[ln % len(LINE_COLORS)]
        cv2.rectangle(vis, (0, y0), (6, y1), color, -1)
        cv2.putText(vis, f"L{ln}", (8, y0+12), FONT, 0.35, color, 1, cv2.LINE_AA)

    banner = np.zeros((BANNER_H, w, 3), dtype=np.uint8)
    banner[:] = BANNER_BG;  banner[:, :5] = (231, 76, 60)
    cv2.putText(banner,
        f"Segmentation  —  {len(boxes_info)} chars  |  {n_lines} line(s)  |  CCA + satellite merge",
        (12, 20), FONT, 0.48, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(banner, "Colour = line  •  Tag: #index\"label\"  •  Box = merged CCA bbox",
        (12, 38), FONT, 0.34, (180,180,180), 1, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"seg_overview_{stem}.png")
    cv2.imwrite(out_path, np.vstack([banner, vis]))
    return out_path


def save_character_atlas_sheet(char_paths: list, output_dir: str, base_name: str,
                                tile_size: int = 72, cols: int = 10,
                                char_dir: str = None) -> str:
    """Tiled grid of all character crops, labelled by index + charset label."""
    if not char_paths: return ""

    stem = os.path.splitext(base_name)[0]
    if char_dir is None:
        char_dir = os.path.join(output_dir, f"chars_{stem}")
    manifest = _load_json(os.path.join(char_dir, "manifest.json"))
    if not manifest: return ""

    label_map = {fname: info for fname, info in manifest.items() if not fname.startswith("_")}

    LABEL_H = 20
    cell_h  = tile_size + LABEL_H
    n       = len(char_paths)
    rows    = math.ceil(n / cols)
    sheet   = np.zeros((rows * cell_h, cols * tile_size, 3), dtype=np.uint8)
    sheet[:] = (18, 18, 28)

    for idx, path in enumerate(char_paths):
        row = idx // cols;  col = idx % cols
        x0  = col * tile_size;  y0 = row * cell_h
        fname = os.path.basename(path)
        info  = label_map.get(fname, {})
        label = info.get("label", "?")
        color = LINE_COLORS[info.get("line", 0) % len(LINE_COLORS)]

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        tile_bgr = (_fit_crop_to_tile(img, tile_size, color)
                    if img is not None
                    else np.zeros((tile_size, tile_size, 3), dtype=np.uint8))
        sheet[y0:y0+tile_size, x0:x0+tile_size] = tile_bgr

        ly0 = y0 + tile_size
        cv2.rectangle(sheet, (x0, ly0), (x0+tile_size, y0+cell_h), color, -1)
        cv2.putText(sheet, f"#{idx}", (x0+2, ly0+9), FONT, 0.28, (0,0,0), 1, cv2.LINE_AA)
        lbl_txt = f'"{label}"'
        (lw, _), _ = cv2.getTextSize(lbl_txt, FONT, 0.34, 1)
        cv2.putText(sheet, lbl_txt, (x0+(tile_size-lw)//2, ly0+16), FONT, 0.34, (0,0,0), 1, cv2.LINE_AA)
        cv2.rectangle(sheet, (x0, y0), (x0+tile_size, y0+cell_h), color, 1)

    bw = sheet.shape[1]
    banner = np.zeros((BANNER_H, bw, 3), dtype=np.uint8)
    banner[:] = BANNER_BG;  banner[:, :5] = (155, 89, 182)
    cv2.putText(banner,
        f"Character Atlas  —  {n} crops  |  colour = writing line  |  label = CHARSET[index]",
        (12, 20), FONT, 0.50, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(banner, "Order: top→bottom lines, left→right columns  →  matches a-z A-Z 0-9",
        (12, 38), FONT, 0.34, (180,180,180), 1, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"seg_atlas_{stem}.png")
    cv2.imwrite(out_path, np.vstack([banner, sheet]))
    return out_path


def save_line_debug_image(processed: dict, output_dir: str, base_name: str) -> str:
    """Horizontal projection profile graph with detected line boundaries."""
    binary = processed["resized"]
    img_h, img_w = binary.shape
    stem   = os.path.splitext(base_name)[0]

    h_proj = np.sum(binary, axis=1).astype(float)
    max_v  = float(np.max(h_proj)) if np.max(h_proj) > 0 else 1.0

    GRAPH_W  = 400
    canvas   = np.zeros((img_h, GRAPH_W, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 28)

    thresh_v = max_v * LINE_PROJ_THRESH
    for row in range(min(img_h, img_h)):
        bar_len = int(h_proj[row] / max_v * (GRAPH_W - 60))
        col_bg  = (39, 174, 96) if h_proj[row] > thresh_v else (70, 70, 90)
        cv2.line(canvas, (0, row), (bar_len, row), col_bg, 1)

    thresh_x = int(thresh_v / max_v * (GRAPH_W - 60))
    cv2.line(canvas, (thresh_x, 0), (thresh_x, img_h-1), (231, 76, 60), 1)

    for by in _find_line_boundaries_from_projection(h_proj):
        if by < img_h:
            cv2.line(canvas, (0, by), (GRAPH_W-1, by), (241, 196, 15), 1)
            cv2.putText(canvas, "line break", (2, by-2), FONT, 0.28, (241,196,15), 1, cv2.LINE_AA)

    banner = np.zeros((BANNER_H, GRAPH_W, 3), dtype=np.uint8)
    banner[:] = BANNER_BG;  banner[:, :5] = (241, 196, 15)
    cv2.putText(banner, "Horizontal Projection Profile",
        (10, 20), FONT, 0.46, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(banner, "Green=ink rows  Red=threshold  Yellow=line breaks",
        (10, 38), FONT, 0.32, (180,180,180), 1, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"seg_linedebug_{stem}.png")
    cv2.imwrite(out_path, np.vstack([banner, canvas]))
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Tile / display helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fit_crop_to_tile(img: np.ndarray, tile: int, color) -> np.ndarray:
    h, w  = img.shape
    pad   = 4
    scale = (tile - pad*2) / max(h, w, 1)
    new_w = max(1, int(w*scale));  new_h = max(1, int(h*scale))
    small  = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((tile, tile, 3), (30, 30, 45), dtype=np.uint8)
    y0 = (tile - new_h) // 2;  x0 = (tile - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w][small > 0] = color
    return canvas


def _fit_to_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w  = img.shape
    scale = size / max(h, w, 1)
    new_w = max(1, int(w*scale));  new_h = max(1, int(h*scale))
    r     = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[(size-new_h)//2:(size-new_h)//2+new_h,
           (size-new_w)//2:(size-new_w)//2+new_w] = r
    return canvas


def _scale_to_max_width(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w: return img
    scale = max_w / w
    return cv2.resize(img, (max_w, int(h*scale)), interpolation=cv2.INTER_AREA)


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {}


def _write_empty_manifest(char_dir: str) -> None:
    with open(os.path.join(char_dir, "manifest.json"), "w") as f: json.dump({}, f)
    with open(os.path.join(char_dir, "line_map.json"),  "w") as f: json.dump({}, f)


# ── Backward compatibility ────────────────────────────────────────────────────
def segment_lines(binary: np.ndarray) -> list:
    return _find_line_bands(binary)

def build_character_atlas(char_paths: list, char_size: int = 32) -> dict:
    atlas = {}
    for path in char_paths:
        key = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            atlas[key] = _fit_to_square(img, char_size)
    return atlas
