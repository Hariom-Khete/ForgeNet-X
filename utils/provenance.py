"""ForgeNet-X — Image Provenance & Synthetic Origin Classifier"""

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


# ── Synthetic software keywords ───────────────────────────────────────────────
_SYNTHETIC_SOFTWARE_KW = [
    "photoshop", "gimp", "illustrator", "inkscape", "paint.net",
    "affinity", "corel", "canva", "midjourney", "dall-e", "stable diffusion",
    "firefly", "opencv", "python", "pillow", "forgenet",
]

# ── Scoring weights ───────────────────────────────────────────────────────────
_W_NO_EXIF       = 15
_W_NO_CAMERA     = 10
_W_SOFTWARE_FLAG = 40
_W_BG_PURITY     = 20
_W_LOW_NOISE     = 15


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_provenance(image_path: str) -> dict:
    """Analyse image provenance. Returns classification dict with score 0–100."""
    flags = []
    score = 0.0

    exif_data    = _read_exif(image_path)
    has_exif     = bool(exif_data)
    camera_make  = exif_data.get("Make")
    camera_model = exif_data.get("Model")
    software     = exif_data.get("Software")
    dt_original  = exif_data.get("DateTimeOriginal") or exif_data.get("DateTime")

    if not has_exif:
        score += _W_NO_EXIF
        flags.append("No EXIF metadata — image may not originate from a camera")

    if software:
        if any(kw in software.lower() for kw in _SYNTHETIC_SOFTWARE_KW):
            score += _W_SOFTWARE_FLAG
            flags.append(f'EXIF Software tag indicates digital tool: "{software}"')

    if not camera_make and not camera_model:
        score += _W_NO_CAMERA
        flags.append("No camera make/model in metadata")

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is not None:
        bg_pts, bg_flag = _check_background_purity(img_gray)
        score += bg_pts
        if bg_flag: flags.append(bg_flag)

        noise_pts, noise_flag = _check_noise_level(img_gray)
        score += noise_pts
        if noise_flag: flags.append(noise_flag)

    score = float(min(100.0, max(0.0, score)))

    if score < 30:
        origin_label = "Likely Human"
        origin_color = "green"
        summary = f"Image shows characteristics consistent with a genuine scanned handwriting sample (origin score: {score:.0f}/100)."
    elif score < 60:
        origin_label = "Possibly Synthetic"
        origin_color = "orange"
        summary = f"Image has ambiguous provenance indicators (origin score: {score:.0f}/100). Manual review recommended."
    else:
        origin_label = "Likely Synthetic"
        origin_color = "red"
        summary = f"Image shows strong indicators of digital/synthetic origin (origin score: {score:.0f}/100). Treat with caution."

    return {
        "has_exif"         : has_exif,
        "camera_make"      : camera_make,
        "camera_model"     : camera_model,
        "software"         : software,
        "datetime_original": dt_original,
        "synthetic_score"  : round(score, 1),
        "origin_label"     : origin_label,
        "origin_color"     : origin_color,
        "flags"            : flags,
        "summary"          : summary,
    }


# ── Heuristic checks ──────────────────────────────────────────────────────────

def _check_background_purity(gray: np.ndarray) -> tuple:
    purity = float(np.sum(gray == 255)) / gray.size
    if purity > 0.70:
        return (_W_BG_PURITY,
                f"Background is {purity*100:.1f}% pure white (255) — typical of digitally generated images")
    return (0, None)


def _check_noise_level(gray: np.ndarray) -> tuple:
    bg_mask = gray > 200
    if np.sum(bg_mask) < 100:
        return (0, None)
    lap      = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    variance = float(np.var(lap[bg_mask]))
    if variance < 8.0:
        return (_W_LOW_NOISE,
                f"Very low background noise (Laplacian σ²={variance:.2f}) — consistent with digital generation")
    return (0, None)


def _read_exif(image_path: str) -> dict:
    result = {}
    try:
        img = Image.open(image_path)
        raw = img._getexif()
        if raw is None: return {}
        for tag_id, value in raw.items():
            tag = TAGS.get(tag_id, str(tag_id))
            if isinstance(value, (str, int, float)):
                result[tag] = str(value).strip()
    except Exception:
        pass
    return result
