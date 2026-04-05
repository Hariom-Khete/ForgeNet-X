"""
utils/signature_analysis.py
────────────────────────────
ForgeNet-X — Signature Forgery Analysis Module

Pipeline:
  1. Preprocess both signatures (grayscale, binary, resize to same canvas)
  2. Extract feature vectors
       • Structural (contour shape, Hu moments)
       • Statistical (histogram of oriented gradients proxy)
       • Stroke (width, density, aspect ratio)
  3. Compute similarity metrics
       • SSIM  — Structural Similarity Index
       • Hu Moment distance
       • Histogram correlation
       • Pixel Euclidean distance (after alignment)
  4. Weighted fusion → final score (0–100 %)
  5. Risk classification
       ≥ 80 % → Low Risk
       50–79 % → Moderate Risk
       < 50 % → High Risk

Returns a rich dict that the Flask route serialises to JSON.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ── Weights for the composite similarity score ────────────────────────────────
WEIGHT_SSIM      = 0.40
WEIGHT_HU        = 0.25
WEIGHT_HIST      = 0.20
WEIGHT_PIXEL     = 0.15

# ── Risk thresholds ───────────────────────────────────────────────────────────
THRESHOLD_LOW      = 80.0   # ≥ 80 % → Low Risk
THRESHOLD_MODERATE = 50.0   # 50–79 % → Moderate Risk
                             # < 50 % → High Risk

# ── Standard canvas for comparison ───────────────────────────────────────────
CANVAS_SIZE = (256, 256)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def analyze_signatures(original_path: str, test_path: str) -> dict:
    """
    Compare an original signature against a test signature.

    Parameters
    ----------
    original_path : str  — path to the genuine signature image
    test_path     : str  — path to the signature under scrutiny

    Returns
    -------
    dict with keys:
        similarity_score   float   — 0–100 %
        risk_level         str     — "Low Risk" | "Moderate Risk" | "High Risk"
        risk_color         str     — "green" | "orange" | "red"
        ssim_score         float
        hu_score           float
        hist_score         float
        pixel_score        float
        features_original  dict    — stroke / shape features of original
        features_test      dict    — stroke / shape features of test
        verdict            str     — human-readable conclusion
    """
    # ── Load & preprocess ─────────────────────────────────────────────────────
    orig_bin = _load_binary(original_path)
    test_bin = _load_binary(test_path)

    if orig_bin is None:
        raise FileNotFoundError(f"Cannot read: {original_path}")
    if test_bin is None:
        raise FileNotFoundError(f"Cannot read: {test_path}")

    # Resize both to the same canvas
    orig_c = cv2.resize(orig_bin, CANVAS_SIZE, interpolation=cv2.INTER_AREA)
    test_c = cv2.resize(test_bin, CANVAS_SIZE, interpolation=cv2.INTER_AREA)

    # ── Compute individual similarity metrics ─────────────────────────────────
    ssim_score  = _ssim_score(orig_c, test_c)
    hu_score    = _hu_moment_score(orig_c, test_c)
    hist_score  = _histogram_score(orig_c, test_c)
    pixel_score = _pixel_score(orig_c, test_c)

    # ── Weighted composite ────────────────────────────────────────────────────
    composite = (
        WEIGHT_SSIM  * ssim_score  +
        WEIGHT_HU    * hu_score    +
        WEIGHT_HIST  * hist_score  +
        WEIGHT_PIXEL * pixel_score
    )
    similarity = round(float(np.clip(composite, 0.0, 100.0)), 2)

    # ── Risk classification ───────────────────────────────────────────────────
    risk_level, risk_color, verdict = _classify_risk(similarity)

    # ── Feature extraction for display ───────────────────────────────────────
    feat_orig = _extract_features(orig_c)
    feat_test = _extract_features(test_c)

    return {
        "similarity_score"   : similarity,
        "risk_level"         : risk_level,
        "risk_color"         : risk_color,
        "ssim_score"         : round(ssim_score, 2),
        "hu_score"           : round(hu_score, 2),
        "hist_score"         : round(hist_score, 2),
        "pixel_score"        : round(pixel_score, 2),
        "features_original"  : feat_orig,
        "features_test"      : feat_test,
        "verdict"            : verdict,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Similarity Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _ssim_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index (0–1) → scaled to 0–100.
    SSIM captures luminance, contrast, and structure simultaneously.
    """
    score, _ = ssim(img1, img2, full=True)
    return float(np.clip(score * 100.0, 0, 100))


def _hu_moment_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compare Hu Moments (7 rotation/scale/translation-invariant moments).
    Hu moments capture the global shape of the signature silhouette.
    Distance → similarity via exponential decay.
    """
    m1 = cv2.HuMoments(cv2.moments(img1)).flatten()
    m2 = cv2.HuMoments(cv2.moments(img2)).flatten()

    # Log-transform (standard practice) to bring values to similar range
    def log_transform(m):
        return np.array([-np.sign(v) * np.log10(abs(v) + 1e-10) for v in m])

    lm1 = log_transform(m1)
    lm2 = log_transform(m2)

    distance = np.linalg.norm(lm1 - lm2)
    # Convert distance → similarity [0, 100]
    similarity = 100.0 * np.exp(-0.1 * distance)
    return float(np.clip(similarity, 0, 100))


def _histogram_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compare normalised pixel-intensity histograms via correlation.
    Histogram correlation ∈ [-1, 1] → mapped to [0, 100].
    """
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    # Map [-1, 1] → [0, 100]
    similarity = (correlation + 1.0) / 2.0 * 100.0
    return float(np.clip(similarity, 0, 100))


def _pixel_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Normalised pixel-level Euclidean distance → similarity.
    Both images must already be the same size.
    """
    diff       = img1.astype(float) - img2.astype(float)
    mse        = np.mean(diff ** 2)
    # Max possible MSE for uint8 images = 255^2 = 65025
    similarity = 100.0 * (1.0 - mse / 65025.0)
    return float(np.clip(similarity, 0, 100))


# ──────────────────────────────────────────────────────────────────────────────
# Feature Extraction (for display / report)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_features(binary: np.ndarray) -> dict:
    """
    Extract interpretable style features from a binarised signature image.

    Returns
    -------
    dict with numeric feature values
    """
    # Ink pixel density
    total_px   = binary.size
    ink_px     = int(np.sum(binary > 0))
    density    = round(ink_px / total_px * 100, 2)

    # Bounding-box aspect ratio of the signature
    coords = np.argwhere(binary > 0)
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        sig_h  = int(y_max - y_min + 1)
        sig_w  = int(x_max - x_min + 1)
        aspect = round(sig_w / sig_h, 3) if sig_h else 0.0
    else:
        sig_h, sig_w, aspect = 0, 0, 0.0

    # Average stroke width via distance transform
    dist       = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    nonzero    = dist[dist > 0]
    stroke_w   = round(float(np.mean(nonzero)) * 2, 2) if len(nonzero) else 0.0

    # Number of contours (≈ stroke segments)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    n_contours  = len(contours)

    # Centroid
    M = cv2.moments(binary)
    if M["m00"] != 0:
        cx = round(M["m10"] / M["m00"], 1)
        cy = round(M["m01"] / M["m00"], 1)
    else:
        cx, cy = 0.0, 0.0

    return {
        "ink_density_pct"   : density,
        "aspect_ratio"      : aspect,
        "stroke_width_px"   : stroke_w,
        "n_stroke_segments" : n_contours,
        "centroid_x"        : cx,
        "centroid_y"        : cy,
        "sig_width_px"      : sig_w,
        "sig_height_px"     : sig_h,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Risk Classification
# ──────────────────────────────────────────────────────────────────────────────

def _classify_risk(score: float) -> tuple:
    """
    Map similarity score to (risk_level, color, verdict) triple.
    """
    if score >= THRESHOLD_LOW:
        return (
            "Low Risk",
            "green",
            f"The test signature closely matches the original "
            f"(similarity: {score:.1f}%). "
            "No significant signs of forgery detected."
        )
    elif score >= THRESHOLD_MODERATE:
        return (
            "Moderate Risk",
            "orange",
            f"Partial similarity detected (similarity: {score:.1f}%). "
            "Some structural differences observed. Manual review recommended."
        )
    else:
        return (
            "High Risk",
            "red",
            f"Low similarity with the original signature "
            f"(similarity: {score:.1f}%). "
            "Significant differences detected — likely forgery."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_binary(path: str) -> np.ndarray | None:
    """Load an image and return a binarised (ink=255) copy."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary
