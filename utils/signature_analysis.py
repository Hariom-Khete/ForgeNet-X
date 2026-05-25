"""ForgeNet-X — Signature Forgery Analysis"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ── Weights ───────────────────────────────────────────────────────────────────
WEIGHT_SSIM  = 0.40
WEIGHT_HU    = 0.25
WEIGHT_HIST  = 0.20
WEIGHT_PIXEL = 0.15

# ── Risk thresholds ───────────────────────────────────────────────────────────
THRESHOLD_LOW      = 80.0
THRESHOLD_MODERATE = 50.0

CANVAS_SIZE = (256, 256)


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_signatures(original_path: str, test_path: str) -> dict:
    """Compare an original signature against a test signature. Returns result dict."""
    orig_bin = _load_binary(original_path)
    test_bin = _load_binary(test_path)

    if orig_bin is None: raise FileNotFoundError(f"Cannot read: {original_path}")
    if test_bin is None: raise FileNotFoundError(f"Cannot read: {test_path}")

    orig_c = cv2.resize(orig_bin, CANVAS_SIZE, interpolation=cv2.INTER_AREA)
    test_c = cv2.resize(test_bin, CANVAS_SIZE, interpolation=cv2.INTER_AREA)

    ssim_score  = _ssim_score(orig_c, test_c)
    hu_score    = _hu_moment_score(orig_c, test_c)
    hist_score  = _histogram_score(orig_c, test_c)
    pixel_score = _pixel_score(orig_c, test_c)

    composite  = (WEIGHT_SSIM  * ssim_score +
                  WEIGHT_HU    * hu_score   +
                  WEIGHT_HIST  * hist_score +
                  WEIGHT_PIXEL * pixel_score)
    similarity = round(float(np.clip(composite, 0.0, 100.0)), 2)

    risk_level, risk_color, verdict = _classify_risk(similarity)

    return {
        "similarity_score" : similarity,
        "risk_level"       : risk_level,
        "risk_color"       : risk_color,
        "ssim_score"       : round(ssim_score,  2),
        "hu_score"         : round(hu_score,    2),
        "hist_score"       : round(hist_score,  2),
        "pixel_score"      : round(pixel_score, 2),
        "features_original": _extract_features(orig_c),
        "features_test"    : _extract_features(test_c),
        "verdict"          : verdict,
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def _ssim_score(img1, img2) -> float:
    score, _ = ssim(img1, img2, full=True)
    return float(np.clip(score * 100.0, 0, 100))


def _hu_moment_score(img1, img2) -> float:
    def log_t(m):
        return np.array([-np.sign(v) * np.log10(abs(v) + 1e-10) for v in m])
    lm1 = log_t(cv2.HuMoments(cv2.moments(img1)).flatten())
    lm2 = log_t(cv2.HuMoments(cv2.moments(img2)).flatten())
    return float(np.clip(100.0 * np.exp(-0.1 * np.linalg.norm(lm1 - lm2)), 0, 100))


def _histogram_score(img1, img2) -> float:
    h1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    cv2.normalize(h1, h1); cv2.normalize(h2, h2)
    corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return float(np.clip((corr + 1.0) / 2.0 * 100.0, 0, 100))


def _pixel_score(img1, img2) -> float:
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    return float(np.clip(100.0 * (1.0 - mse / 65025.0), 0, 100))


# ── Feature extraction ────────────────────────────────────────────────────────

def _extract_features(binary: np.ndarray) -> dict:
    """Extract interpretable style features from a binarised signature."""
    ink_px  = int(np.sum(binary > 0))
    density = round(ink_px / binary.size * 100, 2)

    coords = np.argwhere(binary > 0)
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        sig_h  = int(y_max - y_min + 1)
        sig_w  = int(x_max - x_min + 1)
        aspect = round(sig_w / sig_h, 3) if sig_h else 0.0
    else:
        sig_h = sig_w = 0; aspect = 0.0

    dist     = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    nonzero  = dist[dist > 0]
    stroke_w = round(float(np.mean(nonzero)) * 2, 2) if len(nonzero) else 0.0

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(binary)
    cx = round(M["m10"] / M["m00"], 1) if M["m00"] != 0 else 0.0
    cy = round(M["m01"] / M["m00"], 1) if M["m00"] != 0 else 0.0

    return {
        "ink_density_pct"  : density,
        "aspect_ratio"     : aspect,
        "stroke_width_px"  : stroke_w,
        "n_stroke_segments": len(contours),
        "centroid_x"       : cx,
        "centroid_y"       : cy,
        "sig_width_px"     : sig_w,
        "sig_height_px"    : sig_h,
    }


# ── Risk classification ───────────────────────────────────────────────────────

def _classify_risk(score: float) -> tuple:
    if score >= THRESHOLD_LOW:
        return ("Low Risk", "green",
                f"The test signature closely matches the original (similarity: {score:.1f}%). "
                "No significant signs of forgery detected.")
    elif score >= THRESHOLD_MODERATE:
        return ("Moderate Risk", "orange",
                f"Partial similarity detected (similarity: {score:.1f}%). "
                "Some structural differences observed. Manual review recommended.")
    else:
        return ("High Risk", "red",
                f"Low similarity with the original signature (similarity: {score:.1f}%). "
                "Significant differences detected — likely forgery.")


def _load_binary(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    _, binary = cv2.threshold(cv2.GaussianBlur(img, (3,3), 0), 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary
