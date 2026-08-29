"""
Automated backend tests (bonus item, section 13).

Run with:  cd backend && DATA_DIR=/tmp/iq_test_data pytest -v
"""
import io
import os
import sys

import numpy as np
from PIL import Image

os.environ.setdefault("DATA_DIR", "/tmp/iq_test_data")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.ml import predict
from app.main import app

client = TestClient(app)


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_analyze_dark_image_flags_underexposure():
    arr = (np.random.randint(0, 20, (200, 200, 3)))
    r = client.post("/api/analyze", files={"file": ("dark.png", _png_bytes(arr), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["quality_score"] <= 100
    types = [i["type"] for i in body["issues"]]
    assert "underexposure" in types


def test_analyze_bright_image_flags_overexposure():
    arr = np.full((200, 200, 3), 250, dtype=np.uint8)
    r = client.post("/api/analyze", files={"file": ("bright.png", _png_bytes(arr), "image/png")})
    assert r.status_code == 200
    types = [i["type"] for i in r.json()["issues"]]
    assert "overexposure" in types


def test_analyze_rejects_bad_content_type():
    r = client.post("/api/analyze", files={"file": ("note.txt", b"hello world", "text/plain")})
    assert r.status_code == 400


def test_analyze_rejects_corrupted_image():
    r = client.post("/api/analyze", files={"file": ("bad.png", b"not a real png file", "image/png")})
    assert r.status_code == 422


def test_analyze_rejects_empty_file():
    r = client.post("/api/analyze", files={"file": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_history_and_detail_roundtrip():
    arr = np.random.randint(100, 150, (150, 150, 3))
    r = client.post("/api/analyze", files={"file": ("mid.png", _png_bytes(arr), "image/png")})
    assert r.status_code == 200
    analysis_id = r.json()["id"]

    r2 = client.get("/api/analyses?limit=5")
    assert r2.status_code == 200
    assert any(item["id"] == analysis_id for item in r2.json())

    r3 = client.get(f"/api/analyses/{analysis_id}")
    assert r3.status_code == 200
    assert r3.json()["id"] == analysis_id


def test_get_nonexistent_analysis_returns_404():
    r = client.get("/api/analyses/999999")
    assert r.status_code == 404


def test_fallback_model_is_used_when_bundle_load_fails(monkeypatch):
    monkeypatch.setattr(predict, "_BUNDLE", None)

    def _raise_load_error(_):
        raise EOFError("corrupted model bundle")

    monkeypatch.setattr(predict.joblib, "load", _raise_load_error)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    arr = np.full((200, 200, 3), 250, dtype=np.uint8)
    r = client.post("/api/analyze", files={"file": ("bright-fallback.png", _png_bytes(arr), "image/png")})
    assert r.status_code == 200
    assert "overexposure" in [i["type"] for i in r.json()["issues"]]