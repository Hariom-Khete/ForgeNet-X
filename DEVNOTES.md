# ForgeNet-X — Developer Notes

All technical rationale, algorithm explanations, and tuning history lives here.

---

## preprocessing.py

### Why not Otsu alone?
Otsu picks one global threshold. Handwriting photos have uneven lighting, paper texture, and a small ink fraction (~5–15 %) — all of which bias Otsu toward the background. Solution: adaptive Gaussian threshold (per-pixel local mean) AND-ed with Otsu on CLAHE-normalised image. Both must agree a pixel is ink; false positives from either alone are eliminated.

### Pipeline stages
1. Bilateral filter — preserves ink edges, smooths paper grain
2. CLAHE — normalises local contrast tile-by-tile
3. Adaptive Gaussian threshold — per-pixel local threshold (blockSize=31, C=10)
4. Otsu on CLAHE — secondary vote AND-ed with adaptive
5. Morphological open (2×2) — removes isolated noise specks
6. Morphological close (1×1, effectively no-op) — intentionally kept small; a 2×2 close was bridging inter-character gaps and fusing adjacent letters into one CCA blob
7. Blob area filter (< 20 px²) — removes surviving paper grain
8. Resize to 1024 px wide

### Why MORPH_CLOSE_K = 1×1 (not 2×2)
A 2×2 closing kernel bridges any ≤ 1 px gap. On a handwriting atlas sheet, adjacent characters like 'u' and 'q' can be just 1–2 px apart. The bridge makes them one CCA blob; downstream splitting can't recover it because the bridging column has ~5 % of peak ink (above any useful SPLIT_DEPTH_RATIO). A 1×1 kernel is effectively a no-op — it does not bridge anything.

---

## segmentation.py

### Why CCA instead of vertical projection
Vertical projection treats every low-ink column as a boundary. Wide glyphs (m, w, M, W) have internal low-ink valleys → false splits. Detached components (i-dot, j-dot) have two projection bands → false merges. CCA reasons about ink blobs, not ink columns — each connected blob is one unit regardless of internal valleys.

### Line detection: gap detection + Voronoi partition
1. Sort all primary-blob centroids by y
2. Consecutive centroid gaps: within a row ≈ 2–8 px, between rows ≈ 30+ px
3. Split at gaps ≥ max(15, median_gap × 3.0) → row groups
4. Band boundaries = midpoints between adjacent row means (Voronoi) — non-overlapping, tile [0, img_h-1]

Why Voronoi over pixel-extent bands: pixel-extent bands can overlap when ascenders/descenders from adjacent rows reach into each other; Voronoi cannot.

Why centroid-based over horizontal projection: projection merges rows packed 3–5 px apart (descenders fill the gap); centroids are always ~30 px apart even for tightly packed rows.

### Satellite merge
A satellite is a blob with area < 20 % of line median AND height < 35 % of line pixel height. The height cap prevents thin-but-tall letters (i-body, l, 1) being swallowed when wide glyphs (m, w) dominate the median area.

### Two-component pair merge
Characters like ; : " consist of two satellite-sized blobs. Both fail the primary test so they'd emerge as two separate characters. The pair merge at the end of _merge_satellites finds pairs of short boxes with ≥ 50 % horizontal overlap and gap ≤ 32 px and merges them.

### Edge snapping
After satellite merge the bounding box can include whitespace between a primary and satellite. _snap_to_ink_edges shrinks each box to the tightest ink boundary before the width test and before saving crops.

### Adaptive split
ADAPTIVE_SPLIT_RATIO = 2.5: blob wider than 2.5 × p25-blob-width triggers split attempt.
Uses p25 (not median) because merged blobs inflate the median and can hide above the threshold.
SPLIT_DEPTH_RATIO = 0.15: valley must drop below 15 % of peak to be accepted as a genuine gap. Touching character pairs: ~0–5 %. Wide single-glyph valleys (m, M, W): ~20–60 %. 15 % sits between them.
Split is recursive — handles triple-touching chains.

### Tuning history (SPLIT_DEPTH_RATIO)
- 0.30 → check was never reached (MAX_CHAR_W guard too high)
- 0.15 → first real attempt; still missed morphologically-bridged pairs
- 0.003125 → extreme halving experiment; revealed real root cause was preprocessing not splitting
- 0.15 → restored after fixing MORPH_CLOSE_K

### Structural context validation
CHARSET has 73 entries (a-z, A-Z, 0-9, .,!?;:'"()-). Detected count vs 73 gives quality: "exact", "near (±N)", or "poor (±N)". Stored in manifest _meta block; surfaced in UI as colour-coded badge.

---

## handwriting.py

### Atlas loading
_load_atlas_from_manifest reads manifest.json written by segment_characters() and builds { label → crop_ndarray }. Previously _extract_atlas() re-ran segmentation from scratch, losing the corrected manifest labels.

### Spacing contract
INTER_CHAR_GAP (6 px) < WORD_SPACE_W (22 px) — enforced by assert. Both are fixed (not random) so word boundaries are always visually wider than character boundaries. Only vertical jitter (±4 px) is random, preserving the handwritten feel without affecting legibility.

### Watermark
_bake_watermark overlays a repeating diagonal "[SYNTHETIC – ForgeNet-X]" using PIL alpha compositing at 90/255 opacity. Prevents outputs being submitted as genuine samples.

---

## signature_analysis.py

### Metric weights
- SSIM 40 % — captures luminance, contrast, and local structure jointly
- Hu Moments 25 % — 7 rotation/scale/translation-invariant shape descriptors; log-transformed before distance
- Histogram correlation 20 % — ink density distribution across 256 bins
- Pixel Euclidean 15 % — raw pixel difference after both images are resized to 256×256

### Risk thresholds
≥ 80 % → Low Risk (green), 50–79 % → Moderate (orange), < 50 % → High Risk (red)

---

## provenance.py

### Scoring
- No EXIF: +15
- No camera make/model: +10
- Known synthesis software in EXIF Software tag: +40 (photoshop, gimp, opencv, python, forgenet, etc.)
- Background purity > 70 % pure-255 pixels: +20
- Laplacian variance < 8.0 in background region: +15

Score < 30 → Likely Human, 30–59 → Possibly Synthetic, ≥ 60 → Likely Synthetic.

---

## pdf_generator.py

### Report structure
- Cover: title, timestamp, system info table
- §1: original + generated handwriting images, provenance block
- §2: all 8 preprocessing stage images
- §3: segmentation overview, character atlas, line debug image, charset label map
- §4: signature images, metric table, risk badge, feature comparison table
- §5: conclusion + disclaimer

---

## app.py

### _visual_store
In-memory dict: { hw_filename → { pipeline, seg_overview, atlas_sheet, line_debug, char_dir, provenance } }. Fine for single-user offline tool; replace with Redis/DB for multi-user.

### Generation audit log
Every /generate_text call appends a JSONL record to outputs/generation_log.jsonl with timestamp, hw_filename, text_length, and purpose_declared=True.

---

## Known Issues / To-Do

- Segmentation still not perfect for all 73 characters in one pass
- Morphological close reduced to 1×1 — may need a smarter stroke-repair step that doesn't bridge inter-character gaps
- SAT_H_MAX_RATIO and SAT_AREA_RATIO may need per-line tuning for mixed uppercase/lowercase lines
- Provenance scorer gives false "Likely Synthetic" for phone photos saved through messaging apps (they strip EXIF)
