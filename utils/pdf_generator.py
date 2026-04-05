"""
utils/pdf_generator.py
───────────────────────
ForgeNet-X — PDF Report Generator  (Windows-compatible)

Report structure
────────────────
  Page 1   Cover  (title, timestamp, system info table)

  §1  Handwriting Analysis
        1a  Original vs Generated images  (side-by-side)

  §2  Preprocessing Pipeline  ← NEW
        2a  5-stage annotated images  (one per row, full-width)
            Each image already carries its own step banner from preprocessing.py

  §3  Character Segmentation  ← NEW
        3a  Segmentation overview  (bounding boxes + colour-coded labels)
        3b  Character atlas sheet  (tiled crops with index + label)
        3c  Label assignment explanation table

  §4  Signature Forgery Analysis
        4a  Original vs test signature  (side-by-side)
        4b  Per-metric score table
        4c  Risk badge
        4d  Feature comparison table

  §5  Conclusion & Disclaimer
"""

import os
from datetime import datetime

from reportlab.lib           import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units     import cm
from reportlab.platypus      import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, HRFlowable, PageBreak,
)
from reportlab.lib.enums     import TA_CENTER, TA_LEFT


# ── Brand colours ─────────────────────────────────────────────────────────────
C_PRIMARY   = colors.HexColor("#1A1A2E")
C_ACCENT    = colors.HexColor("#0F3460")
C_RED       = colors.HexColor("#E94560")
C_LIGHT     = colors.HexColor("#F5F5F5")
C_SUCCESS   = colors.HexColor("#27AE60")
C_WARNING   = colors.HexColor("#F39C12")
C_DANGER    = colors.HexColor("#C0392B")
C_PURPLE    = colors.HexColor("#9B59B6")
C_TEAL      = colors.HexColor("#1ABC9C")

RISK_COLORS = {"green": C_SUCCESS, "orange": C_WARNING, "red": C_DANGER}

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm
INNER_W = PAGE_W - 2 * MARGIN   # usable content width


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(
    hw_path        : "str | None",
    gen_path       : "str | None",
    orig_sig       : "str | None",
    test_sig       : "str | None",
    analysis       : dict,
    output_path    : str,
    # Visual confirmation
    pipeline_paths : dict  = None,   # {stage_key: png_path}
    seg_overview   : "str | None" = None,
    atlas_sheet    : "str | None" = None,
    line_debug     : "str | None" = None,
) -> str:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.8*cm, bottomMargin=1.4*cm,
    )

    S = _styles()
    story = []

    # § Cover
    story += _cover(S)

    # § 1  Handwriting
    if hw_path or gen_path:
        story += _section_handwriting(hw_path, gen_path, S)

    # § 2  Preprocessing pipeline
    if pipeline_paths:
        story += _section_preprocessing(pipeline_paths, S)

    # § 3  Segmentation
    if seg_overview or atlas_sheet or line_debug:
        story += _section_segmentation(seg_overview, atlas_sheet, line_debug, S)

    # § 4  Signature analysis
    if orig_sig or test_sig or analysis:
        story += _section_signature(orig_sig, test_sig, analysis, S)

    # § 5  Conclusion
    story += _section_conclusion(analysis, S)

    doc.build(story,
              onFirstPage=_header_footer,
              onLaterPages=_header_footer)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Section builders
# ──────────────────────────────────────────────────────────────────────────────

def _cover(S) -> list:
    e = []
    e.append(Spacer(1, 2.5*cm))
    e.append(Paragraph("ForgeNet-X", S["title"]))
    e.append(Paragraph(
        "AI Handwriting Imitation &amp; Signature Forgery Analysis System",
        S["subtitle"]))
    e.append(HRFlowable(width="100%", thickness=2, color=C_RED, spaceAfter=10))
    e.append(Spacer(1, 0.8*cm))
    e.append(Paragraph("Visual Confirmation Report", S["section_h"]))

    ts = datetime.now().strftime("%d %B %Y  |  %H:%M:%S")
    e.append(Paragraph(f"Generated on: {ts}", S["meta"]))
    e.append(Paragraph("Confidential — For authorised use only", S["meta_red"]))
    e.append(Spacer(1, 1.5*cm))

    info = [
        ["System",    "ForgeNet-X v1.0"],
        ["Purpose",   "Handwriting imitation & signature forgery detection"],
        ["Method",    "Image processing + Computer vision (OpenCV · scikit-image)"],
        ["Libraries", "Flask · OpenCV · NumPy · scikit-image · ReportLab"],
        ["Report",    "Full pipeline visuals, segmentation atlas, forgery metrics"],
    ]
    t = Table(info, colWidths=[4.5*cm, 11.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), C_LIGHT),
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",      (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [colors.white, C_LIGHT]),
        ("BOX",           (0,0), (-1,-1), 0.5, C_ACCENT),
        ("INNERGRID",     (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    e.append(t)
    e.append(PageBreak())
    return e


# ── § 1 ───────────────────────────────────────────────────────────────────────
def _section_handwriting(hw_path, gen_path, S) -> list:
    e = []
    e += _section_title("1", "Handwriting Analysis", C_ACCENT, S)
    e.append(Paragraph(
        "The original handwriting sample uploaded by the user (left) and the "
        "synthetically generated text rendered in the same writing style (right).",
        S["body"]))
    e.append(Spacer(1, 0.3*cm))

    half = INNER_W / 2 - 0.3*cm
    row  = []
    for path, label in [(hw_path, "Original Handwriting Sample"),
                        (gen_path, "Generated Handwriting Output")]:
        col = [Paragraph(label, S["cap"])]
        if path and os.path.exists(path):
            col.append(Image(path, width=half, height=5*cm, kind="proportional"))
        else:
            col.append(Paragraph("[Not available]", S["body"]))
        row.append(col)

    t = Table([row], colWidths=[half + 0.3*cm, half + 0.3*cm])
    t.setStyle(TableStyle([
        ("VALIGN",    (0,0), (-1,-1), "TOP"),
        ("ALIGN",     (0,0), (-1,-1), "CENTER"),
        ("BOX",       (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("INNERGRID", (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("TOPPADDING",(0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    e.append(t)
    e.append(Spacer(1, 0.5*cm))
    return e


# ── § 2 ───────────────────────────────────────────────────────────────────────
PIPELINE_DESCS = {
    "original"  : ("BGR colour image loaded directly from disk.",
                   "No processing applied — serves as the ground-truth reference."),
    "gray"      : ("Converted to single-channel luminance using cv2.COLOR_BGR2GRAY.",
                   "Eliminates colour variation so binarisation operates on intensity only."),
    "bilateral" : ("Bilateral filter (d=9, σ_colour=75, σ_space=75) — edge-preserving denoise.",
                   "Smooths paper grain and texture while keeping ink-stroke edges sharp."),
    "clahe"     : ("CLAHE (Contrast-Limited Adaptive Histogram Equalisation) on bilateral image.",
                   "Normalises local contrast tile-by-tile — neutralises shadows and highlights."),
    "adaptive"  : ("Adaptive Gaussian Threshold — primary binariser (blockSize=31, C=10).",
                   "Each pixel gets its own local threshold — immune to global lighting variation."),
    "otsu_mask" : ("Otsu threshold applied to the CLAHE image — used as a secondary vote.",
                   "AND-blended with adaptive: only pixels both methods call ink are kept."),
    "binary"    : ("AND(adaptive, otsu) → morphOpen → morphClose → blob area filter (≥20px²).",
                   "Final cleaned binary: false-positive paper noise and tiny specks removed."),
    "resized"   : ("Aspect-preserving resize to 1024px wide + re-binarise after interpolation.",
                   "Standardised working image used for all character segmentation."),
}

STAGE_COLORS = {
    "original"  : C_TEAL,
    "gray"      : C_SUCCESS,
    "bilateral" : C_WARNING,
    "clahe"     : colors.HexColor("#E67E22"),
    "adaptive"  : C_DANGER,
    "otsu_mask" : C_PURPLE,
    "binary"    : colors.HexColor("#1ABC9C"),
    "resized"   : colors.HexColor("#EC407A"),
}

def _section_preprocessing(pipeline_paths: dict, S) -> list:
    e = []
    e.append(PageBreak())
    e += _section_title("2", "Image Preprocessing Pipeline", C_ACCENT, S)
    e.append(Paragraph(
        "Every uploaded image passes through five sequential processing stages "
        "before characters are extracted.  Each stage is shown below at full "
        "report width.  The coloured banner on each image was added by ForgeNet-X "
        "to identify the stage name, description, and key pixel statistics.",
        S["body"]))
    e.append(Spacer(1, 0.4*cm))

    # Ordered display
    stage_order = ["original", "gray", "bilateral", "clahe", "adaptive", "otsu_mask", "binary", "resized"]

    for i, key in enumerate(stage_order):
        path = pipeline_paths.get(key)
        if not path or not os.path.exists(path):
            continue

        title_txt, detail_txt = PIPELINE_DESCS.get(key, ("", ""))
        acc = STAGE_COLORS.get(key, C_ACCENT)

        # Stage number badge + title
        badge_data = [[
            Paragraph(f"Stage {i+1}", S["badge_num"]),
            Paragraph(f"<b>{title_txt}</b><br/>"
                      f"<font size='7' color='grey'>{detail_txt}</font>",
                      S["body"]),
        ]]
        badge_t = Table(badge_data, colWidths=[2*cm, INNER_W - 2*cm])
        badge_t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,0), acc),
            ("TEXTCOLOR",  (0,0), (0,0), colors.white),
            ("ALIGN",      (0,0), (0,0), "CENTER"),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("BOX",        (0,0), (-1,-1), 0.5, acc),
        ]))
        e.append(badge_t)

        # Full-width image
        e.append(Image(path, width=INNER_W, height=5.5*cm, kind="proportional"))
        e.append(Spacer(1, 0.5*cm))

    return e


# ── § 3 ───────────────────────────────────────────────────────────────────────
def _section_segmentation(seg_overview, atlas_sheet, line_debug, S) -> list:
    e = []
    e.append(PageBreak())
    e += _section_title("3", "Character Segmentation", C_ACCENT, S)

    # 3a — Overview
    e.append(_sub_header("3a — Segmentation Overview  (Bounding Boxes)", C_PURPLE, S))
    e.append(Paragraph(
        "The preprocessed binary image is scanned for external contours.  "
        "Each contour that passes the size filter (width 5–200 px, height ≥ 8 px) "
        "is wrapped in a coloured bounding box.  The tag above each box shows "
        "<b>#index / charset-label</b>, where the label is the character that "
        "would be assigned to that crop if the sheet follows a standard a–z practice layout.",
        S["body"]))
    e.append(Spacer(1, 0.3*cm))

    if seg_overview and os.path.exists(seg_overview):
        e.append(Image(seg_overview, width=INNER_W, height=7*cm, kind="proportional"))
    else:
        e.append(Paragraph("[Segmentation overview not available]", S["body"]))

    e.append(Spacer(1, 0.5*cm))

    # 3b — Atlas sheet
    e.append(_sub_header("3b — Character Atlas Sheet  (Individual Crops)", C_PURPLE, S))
    e.append(Paragraph(
        "Every extracted character crop is shown in the grid below.  "
        "Each tile displays the ink pixels in the colour assigned to that "
        "crop's index.  The dark strip beneath each tile shows the crop's "
        "<b>#index</b> (top-left) and its <b>assigned label</b> (centre).  "
        "Labels are heuristic — they follow the left-to-right reading order "
        "of the source sheet mapped onto the charset: a–z → A–Z → 0–9 → punctuation.",
        S["body"]))
    e.append(Spacer(1, 0.3*cm))

    if atlas_sheet and os.path.exists(atlas_sheet):
        e.append(Image(atlas_sheet, width=INNER_W, height=9*cm, kind="proportional"))
    else:
        e.append(Paragraph("[Atlas sheet not available]", S["body"]))

    e.append(Spacer(1, 0.4*cm))

    # 3c — Line detection debug image
    e.append(_sub_header("3c — Line Detection  (Horizontal Projection Profile)", C_PURPLE, S))
    e.append(Paragraph(
        "The graph shows the horizontal projection profile used to detect text-line "
        "boundaries on the blank page.  Green bars = ink rows above threshold.  "
        "Yellow lines = detected line breaks used to group contours into lines "
        "before left-to-right sorting is applied within each line.  "
        "This replaces the old arbitrary y//20 row-bucket hack.",
        S["body"]))
    e.append(Spacer(1, 0.3*cm))
    if line_debug and os.path.exists(line_debug):
        e.append(Image(line_debug, width=5*cm, height=9*cm, kind="proportional"))
    else:
        e.append(Paragraph("[Line debug image not available]", S["body"]))

    e.append(Spacer(1, 0.5*cm))

    # 3d — Label assignment explanation table
    e.append(_sub_header("3d — Charset Label Assignment Map", C_PURPLE, S))
    e.append(Paragraph(
        "The table below shows exactly which label is assigned to each crop index. "
        "Index 0 → 'a', index 1 → 'b', … index 25 → 'z', index 26 → 'A', etc.  "
        "If more crops are found than charset entries, they are marked '?'.",
        S["body"]))
    e.append(Spacer(1, 0.2*cm))
    e.append(_charset_table(S))
    e.append(Spacer(1, 0.3*cm))

    return e


def _charset_table(S) -> Table:
    """Build a compact index→label reference table (10 columns)."""
    from utils.segmentation import CHARSET
    cols_per_row = 13
    header = ["#", "Label"] * cols_per_row
    rows   = [header]
    chunk  = []
    for i, ch in enumerate(CHARSET):
        chunk.append(str(i))
        chunk.append(f'"{ch}"')
        if len(chunk) == cols_per_row * 2:
            rows.append(chunk)
            chunk = []
    if chunk:
        # Pad to full width
        while len(chunk) < cols_per_row * 2:
            chunk.append("")
        rows.append(chunk)

    col_w = INNER_W / (cols_per_row * 2)
    t = Table(rows, colWidths=[col_w] * (cols_per_row * 2))
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), C_PRIMARY),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 6),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, C_LIGHT]),
        ("BOX",           (0,0), (-1,-1), 0.3, C_ACCENT),
        ("INNERGRID",     (0,0), (-1,-1), 0.2, colors.lightgrey),
        ("TOPPADDING",    (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))
    return t


# ── § 4 ───────────────────────────────────────────────────────────────────────
def _section_signature(orig_sig, test_sig, analysis, S) -> list:
    e = []
    e.append(PageBreak())
    e += _section_title("4", "Signature Forgery Analysis", C_ACCENT, S)

    half = INNER_W / 2 - 0.3*cm

    # 4a — Images
    e.append(_sub_header("4a — Signature Images", C_TEAL, S))
    row = []
    for path, label in [(orig_sig, "Original Signature  (Reference)"),
                        (test_sig, "Test Signature  (Under Scrutiny)")]:
        col = [Paragraph(label, S["cap"])]
        if path and os.path.exists(path):
            col.append(Image(path, width=half, height=4*cm, kind="proportional"))
        else:
            col.append(Paragraph("[Not available]", S["body"]))
        row.append(col)

    t = Table([row], colWidths=[half + 0.3*cm, half + 0.3*cm])
    t.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("BOX",          (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("INNERGRID",    (0,0), (-1,-1), 0.5, colors.lightgrey),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ]))
    e.append(t)
    e.append(Spacer(1, 0.4*cm))

    if analysis:
        # 4b — Score breakdown
        e.append(_sub_header("4b — Similarity Metric Breakdown", C_TEAL, S))
        score_data = [
            ["Metric", "Score (%)", "Weight", "What it measures"],
            ["SSIM — Structural Similarity",
             str(analysis.get("ssim_score","N/A")), "40%",
             "Luminance, contrast, and local structure jointly"],
            ["Hu Moments — Shape Invariants",
             str(analysis.get("hu_score","N/A")), "25%",
             "Global shape (rotation / scale / translation invariant)"],
            ["Histogram Correlation",
             str(analysis.get("hist_score","N/A")), "20%",
             "Ink-density distribution across intensity bins"],
            ["Pixel Euclidean Distance",
             str(analysis.get("pixel_score","N/A")), "15%",
             "Direct pixel-by-pixel difference after alignment"],
        ]
        st = TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), C_ACCENT),
            ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, C_LIGHT]),
            ("ALIGN",         (1,1), (2,-1), "CENTER"),
            ("BOX",           (0,0), (-1,-1), 0.5, C_ACCENT),
            ("INNERGRID",     (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ])
        st_tbl = Table(score_data,
                       colWidths=[6*cm, 2.3*cm, 1.7*cm, 6*cm])
        st_tbl.setStyle(st)
        e.append(st_tbl)
        e.append(Spacer(1, 0.4*cm))

        # 4c — Risk badge
        e.append(_sub_header("4c — Risk Classification", C_TEAL, S))
        score    = analysis.get("similarity_score", 0)
        risk     = analysis.get("risk_level",  "Unknown")
        risk_col = RISK_COLORS.get(analysis.get("risk_color","red"), C_DANGER)

        badge = [[
            Paragraph(f"Composite Score: <b>{score:.1f}%</b>", S["badge_txt"]),
            Paragraph(f"Risk Level: <b>{risk}</b>",            S["badge_txt"]),
        ]]
        bt = Table(badge, colWidths=[INNER_W/2, INNER_W/2])
        bt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (0,0), C_ACCENT),
            ("BACKGROUND",    (1,0), (1,0), risk_col),
            ("TEXTCOLOR",     (0,0), (-1,-1), colors.white),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ]))
        e.append(bt)
        e.append(Spacer(1, 0.4*cm))

        # 4d — Feature comparison
        fo = analysis.get("features_original", {})
        ft = analysis.get("features_test",     {})
        if fo or ft:
            e.append(_sub_header("4d — Feature Comparison", C_TEAL, S))
            feat_labels = {
                "ink_density_pct"   : "Ink Density (%)",
                "aspect_ratio"      : "Aspect Ratio (W/H)",
                "stroke_width_px"   : "Avg Stroke Width (px)",
                "n_stroke_segments" : "Stroke Segments (contours)",
                "centroid_x"        : "Signature Centroid X",
                "centroid_y"        : "Signature Centroid Y",
                "sig_width_px"      : "Bounding Width (px)",
                "sig_height_px"     : "Bounding Height (px)",
            }
            rows = [["Feature", "Original", "Test", "Δ Delta", "Interpretation"]]
            for key, lbl in feat_labels.items():
                ov = fo.get(key, "—")
                tv = ft.get(key, "—")
                try:
                    d = round(float(tv) - float(ov), 3)
                    ds = f"{d:+.3f}"
                    interp = ("Similar" if abs(d) < 0.05 * max(abs(float(ov)), 1)
                              else "Differs")
                except Exception:
                    ds, interp = "—", "—"
                rows.append([lbl, str(ov), str(tv), ds, interp])

            ft_tbl = Table(rows,
                           colWidths=[5.5*cm, 2.2*cm, 2.2*cm, 2.2*cm, 3.9*cm])
            ft_tbl.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0), C_PRIMARY),
                ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
                ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",      (0,0), (-1,-1), 7.5),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, C_LIGHT]),
                ("ALIGN",         (1,1), (-1,-1), "CENTER"),
                ("BOX",           (0,0), (-1,-1), 0.5, C_PRIMARY),
                ("INNERGRID",     (0,0), (-1,-1), 0.25, colors.lightgrey),
                ("TOPPADDING",    (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ]))
            e.append(ft_tbl)

    e.append(Spacer(1, 0.4*cm))
    return e


# ── § 5 ───────────────────────────────────────────────────────────────────────
def _section_conclusion(analysis, S) -> list:
    e = []
    e.append(PageBreak())
    e += _section_title("5", "Conclusion", C_ACCENT, S)
    verdict = analysis.get("verdict", "No analysis data available.")
    e.append(Paragraph(verdict, S["body"]))
    e.append(Spacer(1, 0.6*cm))
    e.append(Paragraph(
        "<i>This report is generated automatically by ForgeNet-X and is intended "
        "for academic and research purposes only.  All results must be reviewed "
        "by a qualified forensic document examiner before any legal, financial, "
        "or investigative action is taken.</i>",
        S["disclaimer"]))
    return e


# ──────────────────────────────────────────────────────────────────────────────
# Shared layout helpers
# ──────────────────────────────────────────────────────────────────────────────

def _section_title(num, text, color, S) -> list:
    return [
        Paragraph(f"{num}.  {text}", S["section_h"]),
        HRFlowable(width="100%", thickness=1, color=color, spaceAfter=8),
    ]

def _sub_header(text, color, S) -> Paragraph:
    return Paragraph(
        f'<font color="{color.hexval()}">{text}</font>',
        S["sub_h"])


def _header_footer(canvas, doc):
    canvas.saveState()
    # Top bar
    canvas.setFillColor(C_PRIMARY)
    canvas.rect(MARGIN, PAGE_H - 1.4*cm,
                PAGE_W - 2*MARGIN, 0.55*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.drawString(MARGIN + 4, PAGE_H - 0.99*cm,
                      "ForgeNet-X  |  Visual Confirmation Report  |  Confidential")
    # Bottom bar
    canvas.setFillColor(C_PRIMARY)
    canvas.rect(MARGIN, 0.55*cm,
                PAGE_W - 2*MARGIN, 0.38*cm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica", 6.5)
    ts = datetime.now().strftime("%d %b %Y %H:%M")
    canvas.drawString(MARGIN + 4, 0.65*cm, f"Generated: {ts}")
    canvas.drawRightString(PAGE_W - MARGIN - 4, 0.65*cm, f"Page {doc.page}")
    canvas.restoreState()


# ──────────────────────────────────────────────────────────────────────────────
# Paragraph styles
# ──────────────────────────────────────────────────────────────────────────────

def _styles() -> dict:
    B = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("fnx_title", parent=B["Title"],
            fontSize=24, textColor=C_PRIMARY, spaceAfter=6,
            alignment=TA_CENTER, fontName="Helvetica-Bold"),
        "subtitle": ParagraphStyle("fnx_sub", parent=B["Normal"],
            fontSize=10.5, textColor=C_ACCENT, spaceAfter=10,
            alignment=TA_CENTER),
        "section_h": ParagraphStyle("fnx_sh", parent=B["Heading1"],
            fontSize=12, textColor=C_PRIMARY, spaceBefore=12, spaceAfter=3,
            fontName="Helvetica-Bold"),
        "sub_h": ParagraphStyle("fnx_subh", parent=B["Heading2"],
            fontSize=9.5, textColor=C_ACCENT, spaceBefore=8, spaceAfter=3,
            fontName="Helvetica-Bold"),
        "body": ParagraphStyle("fnx_body", parent=B["Normal"],
            fontSize=8.5, spaceAfter=5, leading=13),
        "cap": ParagraphStyle("fnx_cap", parent=B["Normal"],
            fontSize=7.5, textColor=C_ACCENT, alignment=TA_CENTER,
            spaceAfter=3, fontName="Helvetica-Bold"),
        "meta": ParagraphStyle("fnx_meta", parent=B["Normal"],
            fontSize=8.5, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=3),
        "meta_red": ParagraphStyle("fnx_metared", parent=B["Normal"],
            fontSize=8.5, textColor=C_RED, alignment=TA_CENTER, spaceAfter=3,
            fontName="Helvetica-Bold"),
        "badge_num": ParagraphStyle("fnx_bnum", parent=B["Normal"],
            fontSize=9, textColor=colors.white, alignment=TA_CENTER,
            fontName="Helvetica-Bold"),
        "badge_txt": ParagraphStyle("fnx_btxt", parent=B["Normal"],
            fontSize=9.5, textColor=colors.white, alignment=TA_CENTER),
        "disclaimer": ParagraphStyle("fnx_disc", parent=B["Normal"],
            fontSize=7.5, textColor=colors.grey, leading=11),
    }
