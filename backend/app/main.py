import json
import os
import uuid
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import desc

from . import models, schemas
from .database import engine, get_db, DATA_DIR
from .ml.predict import analyze_image

APP_VERSION = "1.0.0"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB

IMAGES_DIR = os.path.join(DATA_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI-Powered Image Quality & Defect Detection API",
    version=APP_VERSION,
    description="Upload an image to receive an automated quality assessment "
                 "(blur, exposure, noise, corruption, defects).",
)

origins = os.environ.get("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if origins == "*" else origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


def _model_is_loaded() -> bool:
    try:
        from .ml.predict import _load_bundle
        _load_bundle()
        return True
    except Exception:
        return False


@app.get("/api/health", response_model=schemas.HealthResponse, tags=["system"])
def health_check():
    return schemas.HealthResponse(
        status="ok", model_loaded=_model_is_loaded(), version=APP_VERSION
    )


@app.post("/api/analyze", response_model=schemas.AnalysisResult, tags=["analysis"])
async def analyze(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. "
                   f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}",
        )

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(raw) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 15 MB limit")

    try:
        result = analyze_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Could not analyze image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal analysis error: {e}")

    # persist a copy of the image for later viewing in history
    ext = os.path.splitext(file.filename or "")[1] or ".png"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = os.path.join(IMAGES_DIR, stored_name)
    with open(stored_path, "wb") as f:
        f.write(raw)

    record = models.Analysis(
        filename=file.filename or stored_name,
        content_type=file.content_type,
        file_size=len(raw),
        image_path=stored_name,
        quality_score=result["quality_score"],
        quality_label=result["quality_label"],
        issues_json=json.dumps(result["issues"]),
        image_stats_json=json.dumps(result["image_stats"]),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return schemas.AnalysisResult(
        id=record.id,
        filename=record.filename,
        quality_score=record.quality_score,
        quality_label=record.quality_label,
        issues=result["issues"],
        image_stats=result["image_stats"],
        image_url=f"/images/{stored_name}",
        created_at=record.created_at.isoformat() if record.created_at else datetime.utcnow().isoformat(),
    )


@app.get("/api/analyses", response_model=list[schemas.AnalysisListItem], tags=["analysis"])
def list_analyses(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(models.Analysis)
        .order_by(desc(models.Analysis.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )
    out = []
    for r in rows:
        issues = json.loads(r.issues_json)
        out.append(schemas.AnalysisListItem(
            id=r.id,
            filename=r.filename,
            quality_score=r.quality_score,
            quality_label=r.quality_label,
            issue_count=len(issues),
            image_url=f"/images/{r.image_path}" if r.image_path else None,
            created_at=r.created_at.isoformat() if r.created_at else "",
        ))
    return out


@app.get("/api/analyses/{analysis_id}", response_model=schemas.AnalysisResult, tags=["analysis"])
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    r = db.query(models.Analysis).filter(models.Analysis.id == analysis_id).first()
    if not r:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return schemas.AnalysisResult(
        id=r.id,
        filename=r.filename,
        quality_score=r.quality_score,
        quality_label=r.quality_label,
        issues=json.loads(r.issues_json),
        image_stats=json.loads(r.image_stats_json),
        image_url=f"/images/{r.image_path}" if r.image_path else None,
        created_at=r.created_at.isoformat() if r.created_at else "",
    )


@app.get("/", tags=["system"])
def root():
    return {"service": "image-quality-detection-api", "docs": "/docs", "health": "/api/health"}