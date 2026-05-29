"""ForgeNet-X — Character Segmentation (CCA + baseline clustering)"""

import os
import json
import math
import warnings
import cv2
import numpy as np


# ── Charset ───────────────────────────────────────────────────────────────────
CHARSET = (
    list("abcdefghijklmnopqrstuvwxyz") +
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
    list("0123456789") +
    list(".,!?;:'\"()-")
)  # 73 total

# ── Line detection (debug projection image only) ───────────────────────────────
LINE_PROJ_THRESH = 0.10   # rows below 10% of peak treated as gaps
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

# ── Seam-boundary conflict resolution ─────────────────────────────────────────
NEAR_SEAM_PX      = 10    # blob centroid within this many rows of a seam → ambiguous
SEAM_H_OVERLAP_FR = 0.30  # fraction of min-width overlap → upper slot "taken"

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
    """Segment characters using CCA + baseline clustering + satellite merge. Returns crop paths."""
    binary       = processed["resized"].copy()
    img_h, img_w = binary.shape
    stem         = os.path.splitext(base_name)[0]
    char_dir     = os.path.join(output_dir, f"chars_{stem}")
    os.makedirs(char_dir, exist_ok=True)

    # Step 1: CCA — extract every connected ink island
    blobs = _extract_blobs(binary, img_w, img_h)
    if not blobs:
        _write_empty_manifest(char_dir)
        return []

    # Step 2: Baseline estimation — use only taller blobs to anchor curves
    med_h = float(sorted(b['h'] for b in blobs)[len(blobs) // 2])
    MIN_PRIMARY_H = max(15, int(med_h * 0.3))
    primary_blobs = [b for b in blobs if b['h'] >= MIN_PRIMARY_H]
    if len(primary_blobs) < 4:
        primary_blobs = blobs

    baselines = _estimate_baselines(primary_blobs, img_h)
    if not baselines:
        _write_empty_manifest(char_dir)
        return []
    n_lines = len(baselines)

    # Step 3: Seam-based line separation — DP minimum-energy paths between lines
    seams = _find_seams_between_lines(binary, baselines, img_h, img_w)

    # Persist seam arrays so the overlay visualisation can load them later
    with open(os.path.join(char_dir, "seams.json"), "w") as _sf:
        json.dump([s.tolist() for s in seams], _sf)

    # Persist histogram valley boundaries for overlay visualisation (3c)
    _pblob_med_h = float(sorted(b['h'] for b in primary_blobs)[len(primary_blobs) // 2])
    valley_boundaries, _ = _compute_histogram_valleys(primary_blobs, img_h, _pblob_med_h)
    with open(os.path.join(char_dir, "valleys.json"), "w") as _vf:
        json.dump(valley_boundaries, _vf)

    # Step 4: Assign blobs to lines — two-pass conflict-aware seam assignment
    #
    # Pass A — definite: blob centroid is clearly inside one seam region.
    # Pass B — ambiguous: blob centroid is within NEAR_SEAM_PX rows of a seam;
    #   these are processed after all clear blobs are placed so we can query
    #   which upper-line slots are already occupied.
    #
    # Conflict rule (per user spec):
    #   1. Compute vertical pixel gap from the ambiguous blob to the nearest
    #      already-placed blob in the upper candidate line and the lower one.
    #   2. Give priority to the upper line UNLESS a blob already placed there
    #      overlaps horizontally with this blob (slot taken).
    #   3. If the upper slot is taken → assign to the lower line.
    #      If the upper slot is free  → assign to the upper line (priority).
    #   Least-ambiguous blobs (farthest from the seam) are resolved first so
    #   that their placements anchor subsequent decisions.

    line_blobs = {i: [] for i in range(n_lines)}
    ambiguous  = []   # list of (blob, nearest_seam_index)

    for blob in blobs:
        if seams:
            bx = int(np.clip(blob['cx'], 0, img_w - 1))
            by = blob['cy']
            # Find the nearest seam and its distance
            best_si, best_d = 0, float('inf')
            for si, s in enumerate(seams):
                d = abs(by - int(s[bx]))
                if d < best_d:
                    best_d, best_si = d, si
            if best_d < NEAR_SEAM_PX:
                ambiguous.append((blob, best_si))
                continue
        li = (_assign_blob_to_seam_region(blob, seams, n_lines, img_h, img_w)
              if seams else _assign_blob_to_baseline(blob, baselines))
        line_blobs[li].append(blob)

    # Pass B — resolve ambiguous blobs using center-dot method, least-ambiguous first
    #
    # For each ambiguous blob:
    #   1. Find the nearest already-placed blob in the upper candidate line
    #      and the nearest in the lower candidate line — the "confusion pair".
    #   2. Place a center dot at the vertical midpoint between the two confusion
    #      characters' centroids.  This boundary is built from actual character
    #      positions so it's independent of (and more accurate than) the DP seam.
    #   3. The ambiguous blob is closer to whichever character lies on the same
    #      side of the center dot  →  that is the preferred line.
    #   4. Priority rule: if preferred line is upper AND the upper slot is free
    #      → assign upper.  If upper slot is already taken → lower.
    #      If preferred line is lower → lower.
    if seams and ambiguous:
        ambiguous.sort(
            key=lambda t: abs(t[0]['cy'] -
                              int(seams[t[1]][int(np.clip(t[0]['cx'], 0, img_w - 1))])),
            reverse=True,   # farthest from seam first → most confident placements anchor later ones
        )
        for blob, seam_idx in ambiguous:
            upper_line = seam_idx          # line above seams[seam_idx]
            lower_line = seam_idx + 1      # line below seams[seam_idx]

            # Step 1 — find the confusion pair (nearest blob in each candidate line)
            upper_nb = _nearest_blob_in_line(blob, line_blobs[upper_line])
            lower_nb = _nearest_blob_in_line(blob, line_blobs[lower_line])

            # Step 2 — compute center dot  (vertical midpoint between the pair)
            if upper_nb is None and lower_nb is None:
                # No reference blobs anywhere yet — use the seam itself as fallback
                bx_      = int(np.clip(blob['cx'], 0, img_w - 1))
                center_y = float(seams[seam_idx][bx_])
            elif upper_nb is None:
                # Upper line still empty → this blob should seed it (prefer upper)
                center_y = float('inf')   # blob.cy always < inf → prefer_upper = True
            elif lower_nb is None:
                # Lower line still empty → fall back to seam boundary
                bx_      = int(np.clip(blob['cx'], 0, img_w - 1))
                center_y = float(seams[seam_idx][bx_])
            else:
                center_y = (upper_nb['cy'] + lower_nb['cy']) / 2.0

            # Step 3 — which side of the center dot is the blob on?
            prefer_upper = blob['cy'] <= center_y

            # Step 4 — priority + slot-taken check
            upper_taken = _has_h_overlap(blob, line_blobs[upper_line])

            if upper_taken:
                line_blobs[lower_line].append(blob)
            elif prefer_upper:
                line_blobs[upper_line].append(blob)
            else:
                line_blobs[lower_line].append(blob)

    elif ambiguous:
        # No seams (single-line document) — fall back to baseline assignment
        for blob, _ in ambiguous:
            li = _assign_blob_to_baseline(blob, baselines)
            line_blobs[li].append(blob)

    # Step 4b: Per-line outlier eviction
    # The last seam region is a catch-all for everything below it, including
    # isolated ink specks far below the text body.  For each line, compute the
    # median centroid-Y of its assigned blobs and discard any blob whose cy is
    # more than 2 × med_h away from that median.  Real characters (punctuation
    # included) cluster tightly around the line median; stray specks do not.
    for li in range(n_lines):
        lb = line_blobs[li]
        if len(lb) <= 2:               # too few blobs to compute a reliable median
            continue
        cys    = sorted(b['cy'] for b in lb)
        med_cy = cys[len(cys) // 2]
        line_blobs[li] = [b for b in lb if abs(b['cy'] - med_cy) <= med_h * 2.0]

    # Step 5: Per-line — satellite merge + edge snap + adaptive split
    all_boxes = []
    for line_idx in range(n_lines):
        lblobs = line_blobs[line_idx]

        if lblobs:
            py_top         = min(b['y']          for b in lblobs)
            py_bot         = max(b['y'] + b['h'] for b in lblobs)
            line_band_h    = max(1, py_bot - py_top + 1)
            blob_widths    = sorted(b['w'] for b in lblobs)
            p25_idx        = max(0, len(blob_widths) // 4 - 1)
            p25_w          = float(blob_widths[p25_idx])
            adaptive_max_w = max(int(p25_w * ADAPTIVE_SPLIT_RATIO), 35)
        else:
            line_band_h    = max(1, int(med_h * 2))
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

    # Step 5b: detect and split any box that still has ≥2 merged primary blobs
    split_boxes = []
    for box in all_boxes:
        split_boxes.extend(_split_merged_chars(binary, *box, med_h=med_h))
    all_boxes = split_boxes

    all_boxes.sort(key=lambda b: (b[4], b[0]))
    all_boxes = all_boxes[:len(CHARSET)]   # prune anything beyond 73

    # Step 6: Structural context validation
    detected = len(all_boxes)
    expected = len(CHARSET)
    delta    = detected - expected
    quality  = ("exact" if delta == 0
                else f"near ({delta:+d})" if abs(delta) <= 3
                else f"poor ({delta:+d})")

    # Step 7: Save crops + manifests
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
            "n_lines": n_lines, "charset": "a-z A-Z 0-9 punctuation",
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


def _split_merged_chars(binary: np.ndarray, x: int, y: int, w: int, h: int,
                         line_idx: int, med_h: float) -> list:
    """Detect and split a box that contains two merged characters.

    Runs CCA on the cropped region to find all non-satellite primary blobs.
    Tries both a HORIZONTAL split (side-by-side characters joined by the
    satellite merge) and a VERTICAL split (cross-line characters stacked when
    the DP seam cut too deep and pulled a character from the next line into
    the current one).

    Split selection:
      - Measure the largest horizontal gap (between right edge of one primary
        and left edge of the next, sorted by X).
      - Measure the largest vertical gap   (between bottom edge of one primary
        and top edge of the next, sorted by Y).
      - Both gaps must exceed MIN_SPLIT_GAP = max(5, med_h * 0.15) to be
        considered real (rules out ink breaks inside a single glyph).
      - Prefer the axis whose gap is larger; fall back to the other if valid.

    Returns a list of (x, y, w, h, line_idx) tuples.
    """
    img_h, img_w = binary.shape
    x1 = max(0, x);           y1 = max(0, y)
    x2 = min(img_w, x + w);   y2 = min(img_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return [(x, y, w, h, line_idx)]

    crop = binary[y1:y2, x1:x2]
    if not np.any(crop > 0):
        return [(x, y, w, h, line_idx)]

    n_labels, _labels, stats, _cents = cv2.connectedComponentsWithStats(
        crop, connectivity=8, ltype=cv2.CV_32S)

    crop_blobs = []
    for lbl in range(1, n_labels):
        bx_ = int(stats[lbl, cv2.CC_STAT_LEFT])
        by_ = int(stats[lbl, cv2.CC_STAT_TOP])
        bw_ = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh_ = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        ba  = int(stats[lbl, cv2.CC_STAT_AREA])
        if ba < MIN_BLOB_AREA:
            continue
        crop_blobs.append({'x': bx_, 'y': by_, 'w': bw_, 'h': bh_, 'area': ba})

    if not crop_blobs:
        return [(x, y, w, h, line_idx)]

    max_area = max(b['area'] for b in crop_blobs)
    sat_area = max_area * SAT_AREA_RATIO
    sat_h    = med_h * SAT_H_MAX_RATIO

    # Primaries: large enough AND tall enough to not be an accent dot
    primaries = [b for b in crop_blobs
                 if b['area'] >= sat_area and b['h'] >= sat_h]

    if len(primaries) <= 1:
        return [(x, y, w, h, line_idx)]

    # Minimum gap in pixels below which we treat it as ink noise, not a real split
    MIN_SPLIT_GAP = max(5, int(med_h * 0.15))

    # ── Horizontal gap (side-by-side characters) ──────────────────────────────
    by_x        = sorted(primaries, key=lambda b: b['x'])
    best_h_gap  = -1
    h_split_col = -1
    for i in range(len(by_x) - 1):
        gap = by_x[i + 1]['x'] - (by_x[i]['x'] + by_x[i]['w'])
        if gap > best_h_gap:
            best_h_gap  = gap
            lr           = by_x[i]['x'] + by_x[i]['w']
            rl           = by_x[i + 1]['x']
            h_split_col  = (lr + rl) // 2        # local col inside crop

    # ── Vertical gap (cross-line stacked characters) ──────────────────────────
    by_y        = sorted(primaries, key=lambda b: b['y'])
    best_v_gap  = -1
    v_split_row = -1
    for i in range(len(by_y) - 1):
        gap = by_y[i + 1]['y'] - (by_y[i]['y'] + by_y[i]['h'])
        if gap > best_v_gap:
            best_v_gap  = gap
            ub           = by_y[i]['y'] + by_y[i]['h']
            lt           = by_y[i + 1]['y']
            v_split_row  = (ub + lt) // 2        # local row inside crop

    # ── Choose split axis ─────────────────────────────────────────────────────
    h_valid = (best_h_gap >= MIN_SPLIT_GAP and
               0 < h_split_col < (x2 - x1))
    v_valid = (best_v_gap >= MIN_SPLIT_GAP and
               0 < v_split_row < (y2 - y1))

    def _make_result(rx, ry, rw, rh):
        if rw < MIN_CHAR_W or rh < MIN_CHAR_H:
            return None
        rx2, ry2, rw2, rh2 = _snap_to_ink_edges(binary, rx, ry, rw, rh)
        if rw2 >= MIN_CHAR_W and rh2 >= MIN_CHAR_H:
            return (rx2, ry2, rw2, rh2, line_idx)
        return None

    if v_valid and (not h_valid or best_v_gap >= best_h_gap):
        # Vertical split: horizontal cut at row v_split_row
        split_abs_y = y1 + v_split_row
        parts = [
            _make_result(x1, y1,         x2 - x1, v_split_row),
            _make_result(x1, split_abs_y, x2 - x1, y2 - split_abs_y),
        ]
        results = [p for p in parts if p is not None]
        return results if results else [(x, y, w, h, line_idx)]

    if h_valid:
        # Horizontal split: vertical cut at column h_split_col
        split_abs_x = x1 + h_split_col
        parts = [
            _make_result(x1,         y1, h_split_col,      y2 - y1),
            _make_result(split_abs_x, y1, x2 - split_abs_x, y2 - y1),
        ]
        results = [p for p in parts if p is not None]
        return results if results else [(x, y, w, h, line_idx)]

    return [(x, y, w, h, line_idx)]


# ══════════════════════════════════════════════════════════════════════════════
# Baseline estimation
# ══════════════════════════════════════════════════════════════════════════════

def _compute_histogram_valleys(blobs: list, img_h: int, med_h: float) -> tuple:
    """Build a centroid-Y density histogram and return valley-based line boundaries.

    Uses LOCAL-PROMINENCE valley detection instead of a global threshold:
      1. Build a 1-D density histogram of centroid-Y values.
      2. Smooth with a short box-filter to reduce per-bin noise.
      3. Find local maxima (peaks) above 8 % of the global peak.
      4. Merge peaks that are closer than med_h * 0.7 px (sub-peaks of one line).
      5. Between each pair of adjacent peaks, locate the deepest valley.
         Accept it as a line boundary only when:
             valley_value <= VALLEY_RATIO * min(left_peak, right_peak)
         VALLEY_RATIO = 0.60 means the valley must be at least 40 % below
         the shorter of its two neighbouring peaks.  This adapts to local
         density, so close lines whose inter-row valley never dips below the
         global 15 % threshold are still correctly split.
      6. Restrict accepted boundaries to the text region (between first and
         last significant-density bin) to exclude empty image margins.

    Args:
        blobs  -- list of blob dicts with 'cy' key
        img_h  -- image height in pixels
        med_h  -- median blob height (sets bin width and smoothing)

    Returns:
        boundaries -- sorted list of Y pixel positions (float) of accepted valleys
        groups     -- list of blob lists, one per detected line (empty groups pruned)
    """
    BIN_PX  = max(2, int(med_h / 8))
    n_bins  = int(img_h / BIN_PX) + 2
    hist    = np.zeros(n_bins, dtype=float)
    for b in blobs:
        bi = int(b['cy'] / BIN_PX)
        if 0 <= bi < n_bins:
            hist[bi] += 1.0

    # Smooth enough to merge sub-peaks within one physical line (spread ≈ 15-25 px)
    # while keeping inter-line valleys (30-80 px wide) clearly visible.
    smooth_w     = max(5, int(med_h / 2 / BIN_PX))
    hist_s       = _smooth1d(hist, smooth_w)
    global_peak  = float(np.max(hist_s)) if hist_s.max() > 0 else 0.0
    if global_peak == 0:
        return [], [list(blobs)]

    MIN_PEAK_H   = global_peak * 0.08          # ignore tiny noise bumps
    # Two peaks must be ≥ 1.2 × med_h apart to count as distinct lines.
    # Within-line centroid spread is ~0.3-0.5 × med_h, so this safely separates
    # character-height variation (false peaks) from genuine line-to-line gaps.
    MIN_PEAK_SEP = max(3, int(med_h * 1.2 / BIN_PX))
    # Valley must drop to ≤ 35 % of the lower adjacent peak.
    # Genuine inter-line gaps → valley ≈ 0-10 % (passes).
    # Within-line undulations → valley ≈ 70-90 % (rejected).
    VALLEY_RATIO = 0.35

    # ── Step 1: local maxima ──────────────────────────────────────────────────
    raw_peaks = [
        i for i in range(1, len(hist_s) - 1)
        if hist_s[i] > hist_s[i - 1]
        and hist_s[i] >= hist_s[i + 1]
        and hist_s[i] >= MIN_PEAK_H
    ]

    # ── Step 2: merge peaks that are too close (keep the taller one) ──────────
    merged: list = []
    for p in raw_peaks:
        if merged and p - merged[-1] < MIN_PEAK_SEP:
            if hist_s[p] > hist_s[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)
    peaks = merged

    # ── Step 3: valley between each adjacent peak pair ────────────────────────
    boundaries: list = []
    for i in range(len(peaks) - 1):
        p1, p2  = peaks[i], peaks[i + 1]
        sub     = hist_s[p1: p2 + 1]
        v_idx   = p1 + int(np.argmin(sub))
        v_val   = float(hist_s[v_idx])
        lower_p = min(float(hist_s[p1]), float(hist_s[p2]))
        if lower_p > 0 and v_val <= lower_p * VALLEY_RATIO:
            boundaries.append(float(v_idx) * BIN_PX)

    # ── Step 4: restrict to text region (exclude empty image margins) ─────────
    text_bins = np.where(hist_s >= MIN_PEAK_H)[0]
    if len(text_bins) == 0:
        return [], [list(blobs)]
    first_t = int(text_bins[0])
    last_t  = int(text_bins[-1])
    boundaries = [b for b in boundaries
                  if first_t * BIN_PX < b < last_t * BIN_PX]
    boundaries.sort()

    # ── Step 5: assign blobs to segments ─────────────────────────────────────
    groups = [[] for _ in range(len(boundaries) + 1)]
    for b in blobs:
        seg = sum(1 for bdy in boundaries if b['cy'] > bdy)
        groups[seg].append(b)
    groups = [g for g in groups if g]

    return boundaries, groups


def _estimate_baselines(blobs: list, img_h: int) -> list:
    """Cluster blobs into text lines and fit a polynomial baseline per line.

    Step 1 — histogram-valley grouping.
              Build a 1-D centroid-Y density histogram (bin ≈ med_h/8 px),
              smooth it, find valleys (< 15 % of peak) as line boundaries,
              and assign blobs to segments.  More robust than gap-based
              clustering when blobs from different lines interleave in Y.
    Step 2 — merge groups whose mean-Y is within med_h × 0.18 of each other
              (repairs over-splits caused by histogram quantisation).
    Step 3 — for each group fit np.polyfit(cx, cy, deg):
              deg = 1 (linear) for < 8 blobs, deg = 2 (parabola) for ≥ 8.

    Returns baselines sorted top → bottom, each a dict with keys:
        blobs   — list of member blob dicts
        coeffs  — np.polyfit coefficients (use np.polyval to evaluate)
        deg     — polynomial degree used
    """
    if not blobs:
        return []

    med_h = float(sorted(b['h'] for b in blobs)[len(blobs) // 2])

    # Step 1: histogram-valley line detection
    boundaries, groups = _compute_histogram_valleys(blobs, img_h, med_h)

    # Step 2: merge groups with almost identical mean-Y (same line, two segments)
    # Kept below gap_thresh so we don't immediately re-merge what we just split.
    merge_thresh = med_h * 0.18  # was 0.5
    changed = True
    while changed and len(groups) > 1:
        changed = False
        means = [float(np.mean([b['cy'] for b in g])) for g in groups]
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if abs(means[i] - means[j]) < merge_thresh:
                    groups[i].extend(groups[j])
                    groups.pop(j)
                    changed = True
                    break
            if changed:
                break

    # Step 3: fit polynomial baseline per group
    baselines = []
    for grp in groups:
        xs  = np.array([b['cx'] for b in grp], dtype=float)
        cys = np.array([b['cy'] for b in grp], dtype=float)
        deg = 2 if len(grp) >= 8 else 1

        if len(grp) == 1:
            # Single blob — constant horizontal baseline at its centroid-Y
            coeffs = np.array([0.0, cys[0]])
            deg    = 1
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    coeffs = np.polyfit(xs, cys, deg)
            except Exception:
                coeffs = np.array([0.0, float(np.mean(cys))])
                deg    = 1

        baselines.append({'blobs': grp, 'coeffs': coeffs, 'deg': deg})

    # Sort top → bottom by mean centroid-Y
    baselines.sort(key=lambda bl: float(np.mean([b['cy'] for b in bl['blobs']])))
    return baselines


def _assign_blob_to_baseline(blob: dict, baselines: list) -> int:
    """Return index of the baseline whose curve is nearest to blob's centroid-Y.

    Evaluates each polynomial at blob['cx'] and returns the index with the
    smallest absolute vertical distance to blob['cy'].  Works for both straight
    (deg-1) and gently curved (deg-2) baselines.
    """
    bx = blob['cx']
    by = blob['cy']
    best_i, best_d = 0, float('inf')
    for i, bl in enumerate(baselines):
        expected_y = float(np.polyval(bl['coeffs'], bx))
        d = abs(by - expected_y)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _smooth1d(arr: np.ndarray, w: int) -> np.ndarray:
    """Box-filter smoothing for a 1D array (pure numpy, no scipy)."""
    if w <= 1 or len(arr) == 0:
        return arr.astype(float)
    pad     = w // 2
    padded  = np.pad(arr.astype(float), pad, mode='edge')
    cum     = np.zeros(len(padded) + 1, dtype=float)
    cum[1:] = np.cumsum(padded)
    result  = (cum[w:] - cum[:-w]) / float(w)
    return result[:len(arr)]


def _bottom_contour_of_line(blobs: list, img_w: int, img_h: int,
                             smooth_w: int = 25) -> np.ndarray:
    """Bottom bounding-box edge per column for a set of blobs, interpolated then smoothed.

    For each column x, records the lowest y-extent among blobs that span that
    column.  Gaps (columns covered by no blob) are filled by linear interpolation;
    the result is then box-filtered to remove the staircase artefact left by
    discrete bbox edges — giving a smooth reference curve for the seam band.
    """
    contour = np.full(img_w, np.nan)

    for b in blobs:
        x1  = max(0,     b['x'])
        x2  = min(img_w, b['x'] + b['w'])
        bot = float(b['y'] + b['h'])
        seg = contour[x1:x2]
        contour[x1:x2] = np.where(np.isnan(seg) | (bot > seg), bot, seg)

    known = np.where(~np.isnan(contour))[0]
    if len(known) == 0:
        return np.full(img_w, float(img_h) / 2.0)
    contour = np.interp(np.arange(img_w, dtype=float), known, contour[known])
    return _smooth1d(contour, smooth_w)


def _find_seams_between_lines(binary: np.ndarray, baselines: list,
                              img_h: int, img_w: int,
                              energy_radius: int = 3,
                              half_band_up: int = 15,
                              half_band_down: int = 15) -> list:
    """Find minimum-energy seams between adjacent text lines (midpoint-anchored).

    Steps:
      1. Detect components         -- done by _estimate_baselines.
      2. Compute seam anchor       -- midpoint between adjacent baseline curves
                                     evaluated at each column x.  Using the
                                     centroid midpoint rather than the upper
                                     line's bottom contour prevents descenders
                                     (y, g, p, q) from dragging the seam into
                                     the next line and mis-assigning characters.
      3. Create search band        -- symmetric strip (half_band px above and
                                     below the midpoint anchor).  The DP finds
                                     the whitespace gap inside this band.
      4. Find lowest energy path   -- left-to-right DP within the sliding strip,
                                     each step may shift +/-1 in band coordinates.
      5. Get final seam            -- backtrack, convert band indices to absolute rows.

    Energy model: local vertical ink density (cumsum box-filter, no scipy).
    Returns a list of (img_w,) int32 arrays -- one per inter-line gap.
    seams[k][x] = absolute row y of seam k at column x.
    """
    if len(baselines) <= 1:
        return []

    # Precompute vertically-smoothed energy for the whole image (cumsum trick)
    r      = energy_radius
    padded = np.vstack([np.zeros((r, img_w), dtype=np.float32),
                        binary.astype(np.float32),
                        np.zeros((r, img_w), dtype=np.float32)])
    cum         = np.vstack([np.zeros((1, img_w), dtype=np.float32),
                             np.cumsum(padded, axis=0)])
    energy_full = (cum[2*r+1 : img_h+2*r+1] - cum[:img_h]) / (2*r + 1)

    band_h     = half_band_up + half_band_down + 1
    rows_local = np.arange(band_h, dtype=np.int16)
    col_idx    = np.tile(np.arange(img_w), (band_h, 1))   # (band_h, img_w)

    seams = []
    for k in range(len(baselines) - 1):

        # Step 2: midpoint between adjacent baseline curves -- sliding band centre.
        # Evaluate both baseline polynomials at every column and take their mean.
        # This anchors the seam equidistant from both text lines at each x, so
        # descenders on line k and ascenders on line k+1 cannot pull it off-centre.
        x_range    = np.arange(img_w, dtype=float)
        bl_upper_y = np.polyval(baselines[k]['coeffs'],     x_range)
        bl_lower_y = np.polyval(baselines[k + 1]['coeffs'], x_range)
        midpoint   = (bl_upper_y + bl_lower_y) / 2.0
        center     = np.clip(midpoint, half_band_up,
                             img_h - 1 - half_band_down).astype(int)        # (img_w,)

        # Step 3: build energy strip in band-relative coordinates (band_h, img_w)
        offsets      = np.arange(-half_band_up, half_band_down + 1, dtype=int)
        abs_y        = np.clip(offsets[:, np.newaxis] + center[np.newaxis, :],
                               0, img_h - 1)                               # (band_h, img_w)
        energy_strip = energy_full[abs_y, col_idx]                         # (band_h, img_w)

        # Step 4: DP — accumulate minimum-cost path left → right
        cumcost = energy_strip.copy()
        parent  = np.zeros((band_h, img_w), dtype=np.int16)

        for x in range(1, img_w):
            prev = cumcost[:, x - 1]

            prev_up        = np.empty(band_h, dtype=np.float32)
            prev_up[0]     = np.inf
            prev_up[1:]    = prev[:-1]

            prev_down      = np.empty(band_h, dtype=np.float32)
            prev_down[-1]  = np.inf
            prev_down[:-1] = prev[1:]

            stacked   = np.stack([prev_up, prev, prev_down])
            min_idx   = np.argmin(stacked, axis=0).astype(np.int16)

            cumcost[:, x] = energy_strip[:, x] + np.min(stacked, axis=0)
            parent[:, x]  = np.clip(rows_local + min_idx - 1, 0, band_h - 1)

        # Step 5: backtrack in band-relative coords → absolute rows
        seam_local = np.zeros(img_w, dtype=np.int32)
        r_idx      = int(np.argmin(cumcost[:, img_w - 1]))
        for x in range(img_w - 1, -1, -1):
            seam_local[x] = r_idx
            if x > 0:
                r_idx = int(parent[r_idx, x])

        # seam_local in [0, band_h-1]; offset from centre = seam_local - half_band_up
        seam = np.clip(center + (seam_local - half_band_up), 0, img_h - 1).astype(np.int32)
        seams.append(seam)

    return seams


def _assign_blob_to_seam_region(blob: dict, seams: list,
                                 n_lines: int, img_h: int, img_w: int) -> int:
    """Return the line index whose seam-bounded region contains the blob.

    Line 0  : blob cy  < seams[0][cx]
    Line k  : seams[k-1][cx] <= blob cy < seams[k][cx]
    Line n-1: blob cy >= seams[-1][cx]

    A blob that sits exactly on a seam boundary falls into the lower line.
    """
    bx = int(np.clip(blob['cx'], 0, img_w - 1))
    by = blob['cy']

    for i in range(n_lines):
        top = 0         if i == 0           else int(seams[i - 1][bx])
        bot = img_h     if i == n_lines - 1 else int(seams[i][bx])
        if top <= by < bot:
            return i

    # Fallback: scan seams top-to-bottom, assign to region just above first seam above blob
    for i, s in enumerate(seams):
        if by < int(s[bx]):
            return i
    return n_lines - 1


def _nearest_blob_in_line(blob: dict, candidates: list):
    """Return the blob in candidates whose centroid is closest to blob (Euclidean).

    Returns None when candidates is empty — caller must handle that case.
    Used to find the 'confusion pair': the nearest character in the upper line
    and the nearest character in the lower line that compete for an ambiguous blob.
    """
    if not candidates:
        return None
    bx, by = blob['cx'], blob['cy']
    return min(candidates,
               key=lambda c: (c['cx'] - bx) ** 2 + (c['cy'] - by) ** 2)


def _has_h_overlap(blob: dict, candidates: list) -> bool:
    """Return True if any candidate's bbox overlaps blob horizontally.

    Overlap threshold = SEAM_H_OVERLAP_FR × min(blob width, candidate width),
    with a hard floor of 2 px so tiny fragments never falsely block a slot.
    This mirrors the SAT_H_OVERLAP_MIN logic used in satellite merging but at
    a lower fraction because we only need to detect 'same horizontal slot', not
    prove the blobs are part of the same glyph.
    """
    b_x1 = blob['x']
    b_x2 = blob['x'] + blob['w']
    for c in candidates:
        c_x1 = c['x']
        c_x2 = c['x'] + c['w']
        overlap   = min(b_x2, c_x2) - max(b_x1, c_x1)
        threshold = max(2, int(min(blob['w'], c['w']) * SEAM_H_OVERLAP_FR))
        if overlap >= threshold:
            return True
    return False


def _find_line_bands(binary: np.ndarray) -> list:
    """Horizontal-projection line detection (debug image only)."""
    h_proj    = np.sum(binary, axis=1).astype(float)
    img_h     = binary.shape[0]
    max_v     = float(np.max(h_proj)) if np.max(h_proj) > 0 else 1.0
    threshold = max_v * LINE_PROJ_THRESH
    in_text   = h_proj > threshold

    raw   = []
    start = None
    for row_idx, active in enumerate(in_text):
        if active and start is None:           start = row_idx
        elif not active and start is not None: raw.append([start, row_idx - 1]); start = None
    if start is not None: raw.append([start, img_h - 1])

    raw = [b for b in raw if b[1] - b[0] + 1 >= MIN_LINE_BAND_H]
    if raw:
        merged = [raw[0]]
        for b in raw[1:]:
            if b[0] - merged[-1][1] - 1 < MIN_LINE_SEP: merged[-1][1] = b[1]
            else:                                         merged.append(b)
        raw = merged

    return [(b[0], b[1]) for b in raw] if raw else [(0, img_h - 1)]


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
        f"Segmentation -- {len(boxes_info)} chars | {n_lines} line(s) | CCA + seam separation",
        (12, 20), FONT, 0.48, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(banner, "Colour = line | Tag: #index\"label\" | Box = merged CCA bbox",
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
        f"Character Atlas -- {n} crops | colour = writing line | label = CHARSET[index]",
        (12, 20), FONT, 0.50, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(banner, "Order: top-to-bottom lines, left-to-right columns -> matches a-z A-Z 0-9",
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


def save_seam_overlay_image(processed: dict, char_paths: list,
                            output_dir: str, base_name: str,
                            char_dir: str = None) -> str:
    """Merged view: character boxes + seam paths + projection valleys on one canvas.

    Combines what 3a (bounding boxes) and 3c (line breaks) show separately into
    a single annotated image so the seam boundaries and the valleys that informed
    them can be compared against the actual character regions at a glance.

    Layers (bottom → top):
      1. Binary handwriting image (greyscale → BGR)
      2. Character bounding boxes, colour-coded by line
      3. Left-edge line-range bars + L0/L1/… labels
      4. Horizontal-projection valley markers  (yellow horizontal lines)
      5. Seam paths drawn as polylines  (bright contrasting colours)
    """
    binary   = processed["resized"].copy()
    vis      = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    vis      = _scale_to_max_width(vis, 1280)
    h, w     = vis.shape[:2]
    orig_w   = processed["resized"].shape[1]
    sf       = w / orig_w

    stem = os.path.splitext(base_name)[0]
    if char_dir is None:
        char_dir = os.path.join(output_dir, f"chars_{stem}")

    manifest = _load_json(os.path.join(char_dir, "manifest.json"))
    if not manifest:
        return ""

    boxes_info = sorted(
        (v for k, v in manifest.items() if not k.startswith("_")),
        key=lambda b: b["index"])

    # Layer 2: character bounding boxes
    for info in boxes_info:
        color = LINE_COLORS[info.get("line", 0) % len(LINE_COLORS)]
        rx = int(info.get("rx", info["x"]) * sf)
        ry = int(info.get("ry", info["y"]) * sf)
        rw = int(info.get("rw", info["w"]) * sf)
        rh = int(info.get("rh", info["h"]) * sf)
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), color, 1)

    # Layer 3: left-edge line bars
    line_yranges = {}
    for info in boxes_info:
        ln  = info["line"]
        ry  = int(info.get("ry", info["y"]) * sf)
        rh  = int(info.get("rh", info["h"]) * sf)
        if ln not in line_yranges:
            line_yranges[ln] = [ry, ry + rh]
        else:
            line_yranges[ln][0] = min(line_yranges[ln][0], ry)
            line_yranges[ln][1] = max(line_yranges[ln][1], ry + rh)
    for ln, (y0, y1) in line_yranges.items():
        color = LINE_COLORS[ln % len(LINE_COLORS)]
        cv2.rectangle(vis, (0, y0), (6, y1), color, -1)
        cv2.putText(vis, f"L{ln}", (8, y0 + 12), FONT, 0.35, color, 1, cv2.LINE_AA)

    # Layer 4: horizontal-projection valley markers (thin yellow, background reference)
    h_proj = np.sum(binary, axis=1).astype(float)
    for valley_y in _find_line_boundaries_from_projection(h_proj):
        y_s = int(valley_y * sf)
        if 0 <= y_s < h:
            cv2.line(vis, (0, y_s), (w - 1, y_s), (0, 215, 255), 1)
            cv2.putText(vis, "proj-valley", (w - 82, y_s - 3),
                        FONT, 0.30, (0, 215, 255), 1, cv2.LINE_AA)

    # Layer 4b: histogram valley boundaries — these are the actual line-split
    # boundaries computed by _compute_histogram_valleys and used by the pipeline.
    # Drawn as bold bright-green dashed horizontal lines so they stand out clearly.
    HIST_V_COLOR = (0, 255, 128)   # bright spring-green
    valley_json_path = os.path.join(char_dir, "valleys.json")
    if os.path.exists(valley_json_path):
        with open(valley_json_path) as _vf:
            hist_valleys = json.load(_vf)
        for vy in hist_valleys:
            y_s = int(vy * sf)
            if 0 <= y_s < h:
                # Dashed line: alternate 12-px drawn / 6-px skipped segments
                x = 0
                draw = True
                while x < w:
                    seg_end = min(x + (12 if draw else 6), w)
                    if draw:
                        cv2.line(vis, (x, y_s), (seg_end, y_s), HIST_V_COLOR, 2)
                    x = seg_end
                    draw = not draw
                cv2.putText(vis, "hist-valley", (w - 85, y_s - 5),
                            FONT, 0.32, HIST_V_COLOR, 1, cv2.LINE_AA)

    # Layer 5: intra-box vertical gap annotation
    # For every character box, crop the original binary and run CCA.
    # If 2+ blobs exist inside the box, the box was formed by merging separate
    # ink islands (e.g. 'i' + dot, 'j' + dot, or two characters merged by mistake).
    # Draw a vertical bracket between the bottom of the upper blob and the top of
    # the lower blob, labelled with the pixel gap, so the merge decision is visible.
    orig_binary = processed["resized"]   # full-res binary for accurate CCA
    orig_h, orig_w = orig_binary.shape
    GAP_COLOR = (0, 220, 255)            # bright cyan

    for info in boxes_info:
        # Use raw (rx, ry, rw, rh) coordinates — these match orig_binary
        rx = int(info.get("rx", info["x"]))
        ry = int(info.get("ry", info["y"]))
        rw = int(info.get("rw", info["w"]))
        rh = int(info.get("rh", info["h"]))

        x1 = max(0, rx);           y1 = max(0, ry)
        x2 = min(orig_w, rx + rw); y2 = min(orig_h, ry + rh)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = orig_binary[y1:y2, x1:x2]
        if not np.any(crop > 0):
            continue

        n_lbl, _lbl, stats, _cents = cv2.connectedComponentsWithStats(
            crop, connectivity=8, ltype=cv2.CV_32S)

        # Collect blobs inside this crop (skip background label 0)
        inner = []
        for lbl in range(1, n_lbl):
            ba = int(stats[lbl, cv2.CC_STAT_AREA])
            if ba < MIN_BLOB_AREA:
                continue
            by_ = int(stats[lbl, cv2.CC_STAT_TOP])
            bh_ = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            inner.append((by_, bh_, ba))   # (top, height, area) — local coords

        if len(inner) < 2:
            continue   # single blob — nothing to measure

        # Sort by vertical position (top of blob)
        inner.sort(key=lambda t: t[0])

        # Measure gap between bottom of upper blob and top of lower blob
        upper_bot_local = inner[0][0] + inner[0][1]   # y + h of topmost blob
        lower_top_local = inner[1][0]                  # y of second blob
        gap_px = lower_top_local - upper_bot_local     # may be negative if overlapping

        # Convert local crop coords → scaled canvas coords
        upper_bot_s = int((y1 + upper_bot_local) * sf)
        lower_top_s = int((y1 + lower_top_local) * sf)
        box_cx_s    = int((x1 + (x2 - x1) / 2) * sf)  # horizontal centre of box

        # Clamp to canvas
        upper_bot_s = max(0, min(h - 1, upper_bot_s))
        lower_top_s = max(0, min(h - 1, lower_top_s))
        mid_y_s     = (upper_bot_s + lower_top_s) // 2

        # Draw bracket centred on the box
        cv2.line(vis, (box_cx_s, upper_bot_s), (box_cx_s, lower_top_s),
                 GAP_COLOR, 1, cv2.LINE_AA)
        cv2.line(vis, (box_cx_s - 4, upper_bot_s),
                 (box_cx_s + 4, upper_bot_s), GAP_COLOR, 1)
        cv2.line(vis, (box_cx_s - 4, lower_top_s),
                 (box_cx_s + 4, lower_top_s), GAP_COLOR, 1)

        # Label: gap value in original pixels
        label = f"{gap_px}px"
        cv2.putText(vis, label, (box_cx_s + 5, mid_y_s + 4),
                    FONT, 0.30, GAP_COLOR, 1, cv2.LINE_AA)

    # Layer 6: seam paths
    SEAM_PALETTE = [(0, 255, 255), (255, 100, 0), (220, 0, 220), (80, 255, 80)]
    seam_path = os.path.join(char_dir, "seams.json")
    if os.path.exists(seam_path):
        with open(seam_path) as _sf:
            seam_data = json.load(_sf)
        for k, seam_list in enumerate(seam_data):
            sc       = SEAM_PALETTE[k % len(SEAM_PALETTE)]
            seam_arr = np.array(seam_list, dtype=float)
            step     = max(1, len(seam_arr) // 300)
            pts      = [(int(x * sf), int(seam_arr[x] * sf))
                        for x in range(0, len(seam_arr), step)]
            for i in range(len(pts) - 1):
                cv2.line(vis, pts[i], pts[i + 1], sc, 2, cv2.LINE_AA)
            if pts:
                cv2.putText(vis, f"Seam {k + 1}",
                            (pts[0][0] + 4, pts[0][1] - 5),
                            FONT, 0.38, sc, 1, cv2.LINE_AA)

    n_lines = max((b["line"] for b in boxes_info), default=0) + 1

    banner     = np.zeros((BANNER_H, w, 3), dtype=np.uint8)
    banner[:]  = BANNER_BG
    banner[:, :5] = (0, 200, 255)
    cv2.putText(banner,
        f"Line Separation -- {n_lines} line(s) | seam paths | valleys | char boxes",
        (12, 20), FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(banner,
        "Curves=seams | Green=hist-valleys (line splits) | Yellow=proj-valleys | Cyan=intra-box gap",
        (12, 38), FONT, 0.34, (180, 180, 180), 1, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"seg_seam_overlay_{stem}.png")
    cv2.imwrite(out_path, np.vstack([banner, vis]))
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
