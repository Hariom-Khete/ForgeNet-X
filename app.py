"""
ForgeNet-X — Flask Application  (Windows-compatible)

Visual confirmation is now baked into every processing step:
  • upload_handwriting  →  saves 5 preprocessing stage PNGs
                           saves segmentation overview PNG
                           saves character atlas sheet PNG
  • generate_text       →  saves generated output PNG
  • download_report     →  bundles all the above into the PDF
"""

import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename

from utils.preprocessing      import preprocess_image, save_pipeline_visuals
from utils.segmentation       import (segment_characters,
                                      save_segmentation_overview,
                                      save_character_atlas_sheet,
                                      save_line_debug_image)
from utils.handwriting        import generate_handwriting
from utils.signature_analysis import analyze_signatures
from utils.pdf_generator      import generate_report

# ── App config ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "forgenet-x-secret-2024"

BASE_DIR       = Path(__file__).resolve().parent
UPLOAD_HW      = BASE_DIR / "uploads"  / "handwriting"
UPLOAD_SIG     = BASE_DIR / "uploads"  / "signatures"
OUTPUT_GEN     = BASE_DIR / "outputs"  / "generated"
OUTPUT_REPORTS = BASE_DIR / "outputs"  / "reports"
OUTPUT_VISUALS = BASE_DIR / "outputs"  / "visuals"   # NEW: all annotation PNGs

for _d in [UPLOAD_HW, UPLOAD_SIG, OUTPUT_GEN, OUTPUT_REPORTS, OUTPUT_VISUALS]:
    _d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ── In-memory session store ───────────────────────────────────────────────────
# Maps upload filename → visual paths dict so /download_report can retrieve them.
# Fine for a single-user offline tool; replace with Redis/DB for multi-user.
_visual_store: dict = {}   # { hw_filename: { "pipeline": {...}, "seg_overview": "...", ... } }

# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(fn: str) -> bool:
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def uid_name(fn: str) -> str:
    return f"{uuid.uuid4().hex}.{fn.rsplit('.',1)[1].lower()}"

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload_handwriting", methods=["POST"])
def upload_handwriting():
    """
    1. Save upload
    2. Preprocess  →  save 5 stage visuals
    3. Segment     →  save overview PNG  +  atlas sheet PNG
    4. Return counts + filenames for the frontend
    """
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400

    f = request.files["file"]
    if not f.filename or not allowed_file(f.filename):
        return jsonify({"error": "Invalid file"}), 400

    fname = uid_name(secure_filename(f.filename))
    fpath = UPLOAD_HW / fname
    f.save(str(fpath))

    # ── Preprocess ────────────────────────────────────────────────────────────
    processed = preprocess_image(str(fpath))

    # ── Save pipeline visuals (5 annotated stage images) ─────────────────────
    pp_paths = save_pipeline_visuals(processed,
                                     output_dir=str(OUTPUT_VISUALS),
                                     base_name=fname)

    # ── Segment characters ────────────────────────────────────────────────────
    char_paths = segment_characters(processed,
                                    output_dir=str(OUTPUT_GEN),
                                    base_name=fname)

    # ── Segmentation overview (bounding-box annotation image) ─────────────────
    # Derive char_dir here so all visual functions share the same path
    stem     = os.path.splitext(fname)[0]
    char_dir = str(OUTPUT_GEN / f"chars_{stem}")

    seg_overview = save_segmentation_overview(processed,
                                              char_paths,
                                              output_dir=str(OUTPUT_VISUALS),
                                              base_name=fname,
                                              char_dir=char_dir)

    # ── Character atlas sheet (tiled crop grid) ───────────────────────────────
    atlas_sheet  = save_character_atlas_sheet(char_paths,
                                              output_dir=str(OUTPUT_VISUALS),
                                              base_name=fname,
                                              char_dir=char_dir)

    # ── Line detection debug image ────────────────────────────────────────────
    line_debug   = save_line_debug_image(processed,
                                         output_dir=str(OUTPUT_VISUALS),
                                         base_name=fname)

    # Store all visual paths + char_dir keyed by upload filename
    _visual_store[fname] = {
        "pipeline"    : pp_paths,
        "seg_overview": seg_overview,
        "atlas_sheet" : atlas_sheet,
        "line_debug"  : line_debug,
        "char_dir"    : char_dir,
    }

    return jsonify({
        "message"    : "Handwriting uploaded & processed",
        "filename"   : fname,
        "char_count" : len(char_paths),
        "stage_count": len(pp_paths),
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

    out_name = f"gen_{hw_filename}"
    out_path = OUTPUT_GEN / out_name

    processed = preprocess_image(str(hw_path))
    # Retrieve the char_dir with the correct manifest from the upload step
    char_dir  = _visual_store.get(hw_filename, {}).get("char_dir", None)
    generate_handwriting(processed, input_text, str(out_path),
                         char_dir=char_dir)

    return jsonify({
        "message"      : "Handwriting generated",
        "output_file"  : out_name,
        "download_url" : url_for("download_file",
                                 folder="generated", filename=out_name),
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

    return jsonify(analyze_signatures(str(UPLOAD_SIG / orig),
                                      str(UPLOAD_SIG / test)))


@app.route("/download_report", methods=["POST"])
def download_report():
    """
    Passes all visual confirmation paths to generate_report so the PDF
    contains the full pipeline, segmentation overview, and atlas sheet.
    """
    data     = request.get_json(force=True)
    hw_file  = data.get("hw_filename",  "")
    gen_file = data.get("gen_filename", "")
    orig_sig = data.get("original_sig", "")
    test_sig = data.get("test_sig",     "")
    analysis = data.get("analysis",     {})

    report_name = f"report_{uuid.uuid4().hex[:8]}.pdf"
    report_path = OUTPUT_REPORTS / report_name

    def _p(folder, name):
        if not name:
            return None
        p = folder / name
        return str(p) if p.exists() else None

    # Retrieve stored visual paths for this handwriting upload
    visuals = _visual_store.get(hw_file, {})

    generate_report(
        hw_path      = _p(UPLOAD_HW,  hw_file),
        gen_path     = _p(OUTPUT_GEN, gen_file),
        orig_sig     = _p(UPLOAD_SIG, orig_sig),
        test_sig     = _p(UPLOAD_SIG, test_sig),
        analysis     = analysis,
        output_path  = str(report_path),
        # ── NEW visual confirmation args ──────────────────────────────────────
        pipeline_paths = visuals.get("pipeline",     {}),
        seg_overview   = visuals.get("seg_overview", None),
        atlas_sheet    = visuals.get("atlas_sheet",  None),
        line_debug     = visuals.get("line_debug",   None),
    )

    return send_file(str(report_path),
                     as_attachment=True,
                     download_name="ForgeNet-X_Report.pdf")


@app.route("/download/<folder>/<filename>")
def download_file(folder, filename):
    folder_map = {
        "generated": OUTPUT_GEN,
        "reports"  : OUTPUT_REPORTS,
        "visuals"  : OUTPUT_VISUALS,
    }
    d = folder_map.get(folder)
    if not d:
        return jsonify({"error": "Invalid folder"}), 400
    p = d / filename
    if not p.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(p), as_attachment=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
