"""
Engineered image-quality feature extraction.

All features are computed from classical image-processing signals and are
used both as (a) direct interpretable quality indicators and (b) inputs to
the learned classifiers in train.py / predict.py. Keeping every feature
interpretable is what powers the explainability layer in predict.py.
"""
import cv2
import numpy as np


FEATURE_NAMES = [
    "laplacian_var",        # sharpness (higher = sharper)
    "tenengrad",             # alternate sharpness measure (gradient energy)
    "mean_brightness",       # 0-255
    "brightness_std",        # contrast proxy
    "underexposed_frac",     # fraction of pixels below dark threshold
    "overexposed_frac",      # fraction of pixels above bright threshold
    "noise_estimate",        # residual std after denoising (sensor/compression noise)
    "high_freq_energy",      # proportion of energy in high frequency band (FFT)
    "edge_density",          # fraction of pixels flagged as edges (Canny)
    "colorfulness",          # Hasler-Susstrunk colorfulness metric
    "saturation_mean",       # mean HSV saturation
    "block_artifact_score",  # blockiness score (JPEG-style 8x8 block edges)
    "entropy",                # Shannon entropy of grayscale histogram
]


def _load_bgr(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def compute_features(image_bytes: bytes) -> dict:
    """Compute the full engineered feature vector for one image.

    Raises ValueError if the bytes cannot be decoded as an image.
    """
    img = _load_bgr(image_bytes)
    if img is None or img.size == 0:
        raise ValueError("Could not decode image (corrupted or unsupported format)")

    h, w = img.shape[:2]
    # Normalize scale so features are comparable across resolutions
    max_dim = 800
    scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
    if scale != 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- Sharpness ---
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx ** 2 + gy ** 2))

    # --- Exposure / brightness ---
    mean_brightness = float(gray.mean())
    brightness_std = float(gray.std())
    underexposed_frac = float(np.mean(gray < 30))
    overexposed_frac = float(np.mean(gray > 235))

    # --- Noise estimate: residual energy after median denoising ---
    denoised = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float64)
    noise_estimate = float(np.std(gray - denoised))

    # --- Frequency-domain energy ratio (high freq vs total) ---
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    ch, cw = magnitude.shape[0] // 2, magnitude.shape[1] // 2
    r = min(ch, cw) // 4
    low_mask = np.zeros_like(magnitude, dtype=bool)
    low_mask[ch - r:ch + r, cw - r:cw + r] = True
    total_energy = magnitude.sum() + 1e-8
    high_freq_energy = float(1.0 - (magnitude[low_mask].sum() / total_energy))

    # --- Edges ---
    edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
    edge_density = float(np.mean(edges > 0))

    # --- Colorfulness (Hasler & Susstrunk) ---
    b, g, r_ = img[:, :, 0].astype(np.float64), img[:, :, 1].astype(np.float64), img[:, :, 2].astype(np.float64)
    rg = r_ - g
    yb = 0.5 * (r_ + g) - b
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

    saturation_mean = float(hsv[:, :, 1].mean())

    # --- Blockiness / compression-artifact score ---
    block_artifact_score = _block_artifact_score(gray)

    # --- Entropy ---
    hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))

    return {
        "laplacian_var": laplacian_var,
        "tenengrad": tenengrad,
        "mean_brightness": mean_brightness,
        "brightness_std": brightness_std,
        "underexposed_frac": underexposed_frac,
        "overexposed_frac": overexposed_frac,
        "noise_estimate": noise_estimate,
        "high_freq_energy": high_freq_energy,
        "edge_density": edge_density,
        "colorfulness": colorfulness,
        "saturation_mean": saturation_mean,
        "block_artifact_score": block_artifact_score,
        "entropy": entropy,
    }


def _block_artifact_score(gray: np.ndarray, block=8) -> float:
    """Estimate JPEG-style blockiness: mean gradient discontinuity aligned to
    the 8x8 block grid relative to gradient magnitude elsewhere."""
    gy = np.abs(np.diff(gray, axis=0))
    gx = np.abs(np.diff(gray, axis=1))

    row_idx = np.arange(gy.shape[0])
    col_idx = np.arange(gx.shape[1])
    on_grid_rows = (row_idx % block) == (block - 1)
    on_grid_cols = (col_idx % block) == (block - 1)

    if gy[on_grid_rows, :].size == 0 or gx[:, on_grid_cols].size == 0:
        return 0.0

    grid_energy = gy[on_grid_rows, :].mean() + gx[:, on_grid_cols].mean()
    total_energy = gy.mean() + gx.mean() + 1e-8
    return float(grid_energy / total_energy)


def feature_vector(features: dict) -> np.ndarray:
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)