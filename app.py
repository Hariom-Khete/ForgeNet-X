"""ForgeNet-X — Flask Application"""

import os
import uuid
import json
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename

from utils.preprocessing      import preprocess_image, save_pipeline_visuals
from utils.segmentation       import (segment_characters,
                                      save_segmentation_overview,
                                      save_character_atlas_sheet,
                                      save_line_debug_image,
                                      save_seam_overlay_image)
from utils.handwriting        import generate_handwriting
from utils.normalization      import build_normalized_atlas, save_normalization_preview
from utils.signature_analysis import analyze_signatures
from utils.pdf_generator      import generate_report
from utils.provenance         import analyze_provenance


# ── App config ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "forgenet-x-secret-2024"

BASE_DIR       = Path(__file__).resolve().parent
UPLOAD_HW      = BASE_DIR / "uploads"  / "handwriting"
UPLOAD_SIG     = BASE_DIR / "uploads"  / "signatures"
OUTPUT_GEN     = BASE_DIR / "outputs"  / "generated"
OUTPUT_REPORTS = BASE_DIR / "outputs"  / "reports"
OUTPUT_VISUALS = BASE_DIR / "outputs"  / "visuals"
GEN_LOG_PATH   = BASE_DIR / "outputs"  / "generation_log.jsonl"

for _d in [UPLOAD_HW, UPLOAD_SIG, OUTPUT_GEN, OUTPUT_REPORTS, OUTPUT_VISUALS]:
    _d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

_visual_store: dict = {}  # hw_filename → { pipeline, seg_overview, atlas_sheet, ... }


# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def uid_name(fn: str) -> str:
    return f"{uuid.uuid4().hex}.{fn.rsplit('.',1)[1].lower()}"

def _log_generation(hw_filename: str, text_length: int) -> None:
    record = {
        "timestamp"       : datetime.now().isoformat(timespec="seconds"),
        "hw_filename"     : hw_filename,
        "text_length"     : text_length,
        "purpose_declared": True,
    }
    with open(GEN_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_handwriting", methods=["POST"])
def upload_handwriting():
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400

    f = request.files["file"]
    if not f.filename or not allowed_file(f.filename):
        return jsonify({"error": "Invalid file"}), 400

    fname = uid_name(secure_filename(f.filename))
    fpath = UPLOAD_HW / fname
    f.save(str(fpath))

    processed   = preprocess_image(str(fpath))
    pp_paths    = save_pipeline_visuals(processed, output_dir=str(OUTPUT_VISUALS), base_name=fname)
    char_paths  = segment_characters(processed, output_dir=str(OUTPUT_GEN), base_name=fname)

    stem     = os.path.splitext(fname)[0]
    char_dir = str(OUTPUT_GEN / f"chars_{stem}")

    seg_overview = save_segmentation_overview(processed, char_paths,
                                              output_dir=str(OUTPUT_VISUALS),
                                              base_name=fname, char_dir=char_dir)
    atlas_sheet  = save_character_atlas_sheet(char_paths,
                                              output_dir=str(OUTPUT_VISUALS),
                                              base_name=fname, char_dir=char_dir)
    line_debug   = save_seam_overlay_image(processed, char_paths,
                                          output_dir=str(OUTPUT_VISUALS),
                                          base_name=fname, char_dir=char_dir)
    provenance   = analyze_provenance(str(fpath))

    _visual_store[fname] = {
        "pipeline"    : pp_paths,
        "seg_overview": seg_overview,
        "atlas_sheet" : atlas_sheet,
        "line_debug"  : line_debug,
        "char_dir"    : char_dir,
        "provenance"  : provenance,
    }

    # ── Normalisation preview (debug grid on shared baseline) ─────────────────
    norm_preview = None
    try:
        atlas_n, targets_n, hw_n = build_normalized_atlas(
            char_dir, blend=0.35, target_x_height_px=None
        )
        if atlas_n:
            norm_preview_name = f"norm_{stem}.png"
            norm_preview_path = str(OUTPUT_VISUALS / norm_preview_name)
            save_normalization_preview(atlas_n, targets_n, hw_n, norm_preview_path)
            norm_preview = norm_preview_name
    except Exception:
        pass   # non-fatal; don't break the upload if normalisation preview fails

    _visual_store[fname]["norm_preview"] = norm_preview

    seg_quality   = {}
    manifest_path = os.path.join(char_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as _mf:
            seg_quality = json.load(_mf).get("_meta", {})

    return jsonify({
        "message"    : "Handwriting uploaded & processed",
        "filename"   : fname,
        "char_count" : len(char_paths),
        "stage_count": len(pp_paths),
        "provenance" : provenance,
        "seg_quality": seg_quality,
    })


@app.route("/generate_text", methods=["POST"])
def generate_text():
    data        = request.get_json(force=True)
    hw_filename = data.get("hw_filename", "")
    input_text  = data.get("text", "").strip()

    if not hw_filename or not input_text:
        return jsonify({"error": "hw_filename and text are required"}), 400

    hw_path = UPLOAD_HW / hw_filename
    if not hw_path.exists():
        return jsonify({"error": "Handwriting source not found"}), 404

    out_name  = f"gen_{hw_filename}"
    out_path  = OUTPUT_GEN / out_name
    processed = preprocess_image(str(hw_path))
    char_dir  = _visual_store.get(hw_filename, {}).get("char_dir", None)
    generate_handwriting(processed, input_text, str(out_path), char_dir=char_dir)
    _log_generation(hw_filename, len(input_text))

    return jsonify({
        "message"     : "Synthetic sample generated and watermarked",
        "output_file" : out_name,
        "download_url": url_for("download_file", folder="generated", filename=out_name),
    })


@app.route("/upload_signature", methods=["POST"])
def upload_signature():
    results = {}
    for field in ("original", "test"):
        if field not in request.files:
            return jsonify({"error": f"Missing: {field}"}), 400
        f = request.files[field]
        if not f.filename or not allowed_file(f.filename):
            return jsonify({"error": f"Invalid file: {field}"}), 400
        fname = f"{field}_{uid_name(secure_filename(f.filename))}"
        f.save(str(UPLOAD_SIG / fname))
        results[field] = fname
    return jsonify({"message": "Signatures uploaded", **results})


@app.route("/analyze_signature", methods=["POST"])
def analyze_signature():
    data = request.get_json(force=True)
    orig = data.get("original")
    test = data.get("test")
    if not orig or not test:
        return jsonify({"error": "Both filenames required"}), 400
    for name, label in [(orig, "original"), (test, "test")]:
        if not (UPLOAD_SIG / name).exists():
            return jsonify({"error": f"File not found: {label}"}), 404
    return jsonify(analyze_signatures(str(UPLOAD_SIG / orig), str(UPLOAD_SIG / test)))


@app.route("/download_report", methods=["POST"])
def download_report():
    data     = request.get_json(force=True)
    hw_file  = data.get("hw_filename",  "")
    gen_file = data.get("gen_filename", "")
    orig_sig = data.get("original_sig", "")
    test_sig = data.get("test_sig",     "")
    analysis = data.get("analysis",     {})

    report_name = f"report_{uuid.uuid4().hex[:8]}.pdf"
    report_path = OUTPUT_REPORTS / report_name

    def _p(folder, name):
        if not name: return None
        p = folder / name
        return str(p) if p.exists() else None

    visuals = _visual_store.get(hw_file, {})

    generate_report(
        hw_path        = _p(UPLOAD_HW,  hw_file),
        gen_path       = _p(OUTPUT_GEN, gen_file),
        orig_sig       = _p(UPLOAD_SIG, orig_sig),
        test_sig       = _p(UPLOAD_SIG, test_sig),
        analysis       = analysis,
        output_path    = str(report_path),
        pipeline_paths = visuals.get("pipeline",     {}),
        seg_overview   = visuals.get("seg_overview", None),
        atlas_sheet    = visuals.get("atlas_sheet",  None),
        line_debug     = visuals.get("line_debug",   None),
        provenance     = visuals.get("provenance",   None),
    )

    return send_file(str(report_path), as_attachment=True,
                     download_name="ForgeNet-X_Report.pdf")


@app.route("/download/<folder>/<filename>")
def download_file(folder, filename):
    folder_map = {
        "generated": OUTPUT_GEN,
        "reports"  : OUTPUT_REPORTS,
        "visuals"  : OUTPUT_VISUALS,
    }
    d = folder_map.get(folder)
    if not d: return jsonify({"error": "Invalid folder"}), 400
    p = d / filename
    if not p.exists(): return jsonify({"error": "File not found"}), 404
    return send_file(str(p), as_attachment=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
