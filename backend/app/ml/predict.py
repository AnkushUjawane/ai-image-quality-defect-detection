"""
Inference + explainability layer.

Loads the trained per-issue RandomForest classifiers, runs them on the
engineered feature vector of an uploaded image, and turns the raw
probabilities into the structured JSON contract described in the
assessment (section 7), including severity, confidence, an overall
quality score/label, and a human-readable explanation for each finding.
"""
import os
import joblib
import numpy as np

from .features import compute_features, feature_vector, FEATURE_NAMES

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "model.joblib")

_BUNDLE = None

# Explanation templates: which raw feature(s) explain a given issue, and how
# to phrase the interpretable justification.
EXPLAIN_RULES = {
    "blur": lambda f: (
        f"Laplacian variance (sharpness) is {f['laplacian_var']:.1f}; low values indicate "
        f"few sharp edges, consistent with blur."
    ),
    "underexposure": lambda f: (
        f"Mean brightness is {f['mean_brightness']:.1f}/255 with "
        f"{f['underexposed_frac']*100:.1f}% of pixels below the dark threshold (30/255)."
    ),
    "overexposure": lambda f: (
        f"{f['overexposed_frac']*100:.1f}% of pixels exceed the bright threshold (235/255), "
        f"mean brightness {f['mean_brightness']:.1f}/255."
    ),
    "noise": lambda f: (
        f"Residual noise estimate after denoising is {f['noise_estimate']:.2f} "
        f"(std of high-frequency residual); elevated values indicate sensor/compression noise."
    ),
    "corruption": lambda f: (
        f"Block-artifact score {f['block_artifact_score']:.2f} and high-frequency energy "
        f"ratio {f['high_freq_energy']:.2f} indicate compression artifacts or corrupted regions."
    ),
}

SEVERITY_WEIGHT = {"low": 8, "medium": 18, "high": 32}


def _load_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        try:
            _BUNDLE = joblib.load(MODEL_PATH)
        except Exception:
            _BUNDLE = {"fallback": True}
    return _BUNDLE


def _severity_from_proba(p: float) -> str:
    if p < 0.4:
        return "low"
    if p < 0.7:
        return "medium"
    return "high"


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _fallback_issue_probabilities(feats: dict) -> dict:
    return {
        "blur": _clamp01((120.0 - feats["laplacian_var"]) / 120.0),
        "underexposure": _clamp01(max(feats["underexposed_frac"], (70.0 - feats["mean_brightness"]) / 70.0)),
        "overexposure": _clamp01(max(feats["overexposed_frac"], (feats["mean_brightness"] - 185.0) / 70.0)),
        "noise": _clamp01((feats["noise_estimate"] - 10.0) / 20.0),
        "corruption": _clamp01(max((feats["block_artifact_score"] - 1.6) / 1.2, (feats["high_freq_energy"] - 0.9) / 0.1)),
    }


def analyze_image(image_bytes: bytes) -> dict:
    """Full pipeline: validate -> extract features -> predict -> score -> explain.

    Raises ValueError on invalid/corrupted/unreadable images (caller maps
    this to an HTTP 400).
    """
    if not image_bytes or len(image_bytes) < 16:
        raise ValueError("Empty or unreadably small file")

    feats = compute_features(image_bytes)  # raises ValueError if undecodable
    bundle = _load_bundle()
    if bundle.get("fallback"):
        issue_probabilities = _fallback_issue_probabilities(feats)
    else:
        vec = feature_vector(feats).reshape(1, -1)
        scaler = bundle["scaler"]
        models = bundle["models"]
        vec_s = scaler.transform(vec)
        issue_probabilities = {
            issue: (
                float(clf.predict_proba(vec_s)[0][1])
                if len(clf.classes_) > 1
                else float(clf.predict(vec_s)[0])
            )
            for issue, clf in models.items()
        }

    issues = []
    penalty = 0.0
    for issue, proba in issue_probabilities.items():
        present = proba >= 0.5
        if present:
            severity = _severity_from_proba(proba)
            issues.append({
                "type": issue,
                "severity": severity,
                "confidence": round(proba, 3),
                "explanation": EXPLAIN_RULES[issue](feats),
            })
            penalty += SEVERITY_WEIGHT[severity] * proba

    quality_score = int(round(max(0, min(100, 100 - penalty))))

    if quality_score >= 80 and not any(i["severity"] == "high" for i in issues):
        quality_label = "ACCEPTABLE"
    elif quality_score >= 50:
        quality_label = "DEGRADED"
    else:
        quality_label = "DEFECTIVE"

    issues.sort(key=lambda i: -i["confidence"])

    return {
        "quality_score": quality_score,
        "quality_label": quality_label,
        "issues": issues,
        "image_stats": {k: round(v, 3) for k, v in feats.items()},
    }