# APERTURE — AI-Powered Image Quality & Defect Detection

A full-stack application that accepts an image and automatically evaluates its
visual quality, detecting blur, under/overexposure, noise, corruption, and
flagging an overall ACCEPTABLE / DEGRADED / DEFECTIVE verdict — using a
**hybrid classical-CV + machine-learning** pipeline (no external AI/vision
APIs, no API keys required).

```
┌────────────┐      multipart/form-data       ┌──────────────┐      feature vector      ┌──────────────────┐
│  Frontend   │ ─────────────────────────────▶ │  FastAPI      │ ────────────────────────▶│  ML pipeline       │
│ (React +    │ ◀───────────────────────────── │  backend      │ ◀────────────────────────│  (OpenCV features  │
│  Vite)      │      JSON quality report        │  + SQLite     │   per-issue predictions   │  + RandomForests)  │
└────────────┘                                 └──────────────┘                           └──────────────────┘
```

## 1. Project layout

```
aperture-image-quality-detector/
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app, REST endpoints
│   │   ├── models.py          SQLAlchemy ORM model (Analysis)
│   │   ├── schemas.py         Pydantic response schemas
│   │   ├── database.py        SQLite engine/session setup
│   │   └── ml/
│   │       ├── features.py            13 engineered image-quality features (OpenCV/NumPy)
│   │       ├── generate_dataset.py    Synthetic clean-image + controlled-degradation generator
│   │       ├── train.py               Trains 5 RandomForest classifiers + evaluation report
│   │       ├── predict.py             Inference + scoring + explainability
│   │       ├── model.joblib           Trained model bundle (committed, ready to run)
│   │       ├── evaluation_report.md   Precision/recall/F1/ROC-AUC + failure analysis
│   │       └── evaluation_report.json
│   ├── tests/test_api.py      Automated backend tests (pytest)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── main.jsx, App.jsx / App.css   React entry point + top-level layout/state
│   │   │                                  (App.css: layout grid, panels, footer)
│   │   ├── api/client.js                  fetch wrapper (analyze, history, health)
│   │   ├── hooks/useHealthCheck.js
│   │   ├── styles/variables.css           global design tokens, resets
│   │   ├── config.js
│   │   └── components/                    each in its own folder with a co-located .css
│   │       ├── TopBar/, Dropzone/, ResultsPanel/,
│   │       └── IssueCard/, StatsGrid/, HistoryList/
│   ├── index.html, vite.config.js, package.json, eslint.config.js
│   ├── nginx.conf, Dockerfile             multi-stage: npm build → nginx serve
│   └── .env.example                       VITE_API_BASE_URL
├── sample_images/             Example images spanning every quality condition
├── docker-compose.yml
└── README.md
```

## 2. How the AI/ML works

### 2.1 Features (`app/ml/features.py`)
Every image is reduced to **13 interpretable features** computed with OpenCV/NumPy:
sharpness (Laplacian variance, Tenengrad gradient energy), brightness mean/std,
under/overexposed pixel fractions, a denoise-residual noise estimate, FFT
high-frequency energy ratio, Canny edge density, colorfulness, mean HSV
saturation, an 8×8 JPEG block-artifact score, and grayscale histogram entropy.

### 2.2 Model (`train.py` / `predict.py`)
This is a **hybrid approach**: the engineered features above feed **5 independent
RandomForest binary classifiers** (one per issue — blur, underexposure,
overexposure, noise, corruption), each outputting a probability that becomes
the issue's `confidence`, thresholded into `low` / `medium` / `high` `severity`.
An overall `quality_score` (0–100) is derived by penalizing the base score of
100 proportionally to each detected issue's severity × confidence; the
`quality_label` (ACCEPTABLE / DEGRADED / DEFECTIVE) is derived from that score.

Why RandomForest-on-engineered-features rather than an end-to-end CNN: it stays
fully interpretable (see §2.4), trains in seconds without a GPU, and — per the
assessment's "no external AI services" constraint — needs no pretrained
backbone weights downloaded from the internet.

### 2.3 Data (`generate_dataset.py`)
No external dataset or network access is used. Instead, ~1,400 synthetic
"clean" base images (gradients, shapes, checkerboards, procedural textures,
text) are generated, and controlled degradations (Gaussian blur, brightness
scaling, Gaussian noise, block corruption + heavy JPEG re-encoding) are applied
at randomized severities to build labeled training data — exactly the
"generate controlled image-quality degradations from clean images" option
described in the assessment (§8). ~22% of samples are left clean as the
ACCEPTABLE class. The dataset is regenerated deterministically (`random.seed(42)`).

Regenerate + retrain from scratch:
```bash
cd backend/app/ml
python3 generate_dataset.py   # writes data/synthetic/ (~1400 PNGs + manifest.json)
python3 train.py              # trains model.joblib + evaluation_report.{md,json}
```

### 2.4 Explainability (`predict.py`)
Every detected issue ships with a plain-language `explanation` string built
from the actual feature values that drove the decision (e.g. *"Mean brightness
is 5.9/255 with 100.0% of pixels below the dark threshold (30/255)"*), plus
the raw `image_stats` are returned alongside the verdict so the reasoning is
independently checkable. `train.py`'s evaluation report additionally records
each classifier's top contributing features (via RandomForest feature
importances).

### 2.5 Evaluation
See [`backend/app/ml/evaluation_report.md`](backend/app/ml/evaluation_report.md)
for full precision/recall/F1/ROC-AUC and confusion matrices on a held-out 20%
test split, plus a discussion of failure cases and limitations (e.g. reduced
recall on images with 2–3 simultaneous degradations, and the synthetic→real
domain-gap caveat).

Headline test-set numbers (RandomForest, 1120 train / 280 test):

| Issue | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| blur | 0.68 | 0.76 | 0.72 | 0.93 |
| underexposure | 0.76 | 0.59 | 0.66 | 0.84 |
| overexposure | 0.85 | 0.73 | 0.78 | 0.92 |
| noise | 0.98 | 0.85 | 0.91 | 0.98 |
| corruption | 0.93 | 0.90 | 0.92 | 0.99 |

## 3. API reference

Base URL: `http://localhost:8000` (see `docker-compose.yml` to change ports).
Interactive Swagger docs are auto-served at **`/docs`**.

| Method | Path | Description |
|---|---|---|
| `GET`  | `/api/health` | Health/status check (`{status, model_loaded, version}`) |
| `POST` | `/api/analyze` | Upload an image (`multipart/form-data`, field `file`); returns the full analysis and persists it |
| `GET`  | `/api/analyses?limit=20&offset=0` | Paginated analysis history (summaries) |
| `GET`  | `/api/analyses/{id}` | Full stored result for one past analysis |
| `GET`  | `/images/{filename}` | Static serving of the uploaded image copy |

### Example request
```bash
curl -F "file=@sample_images/03_underexposed.png;type=image/png" \
     http://localhost:8000/api/analyze
```

### Example response
```json
{
  "id": 3,
  "filename": "03_underexposed.png",
  "quality_score": 72,
  "quality_label": "DEGRADED",
  "issues": [
    {
      "type": "underexposure",
      "severity": "high",
      "confidence": 0.868,
      "explanation": "Mean brightness is 31.4/255 with 68.2% of pixels below the dark threshold (30/255)."
    }
  ],
  "image_stats": { "laplacian_var": 210.4, "mean_brightness": 31.4, "...": "..." },
  "image_url": "/images/9f2c....png",
  "created_at": "2026-08-29T05:44:21"
}
```

### Error handling
- `400` — missing/empty file or unsupported content type
- `413` — file exceeds the 15 MB limit
- `422` — file is readable as an upload but not decodable as an image (corrupted/invalid)
- `404` — analysis id not found (history lookup)
- `500` — unexpected internal error (caught and reported without crashing the request)

## 4. Running locally (without Docker)

Requires Python 3.11+ and Node 18+.

```bash
cd backend
pip install -r requirements.txt
DATA_DIR=./data uvicorn app.main:app --reload --port 8000
```

In a second terminal:
```bash
cd frontend
npm install
cp .env.example .env   # VITE_API_BASE_URL=http://localhost:8000
npm run dev
```
Open `http://localhost:5173` (Vite's dev server). See `frontend/README.md`
for the component breakdown.

Run tests:
```bash
cd backend
DATA_DIR=/tmp/iq_test_data pytest -v
```

## 5. Running with Docker (preferred)

```bash
docker compose up --build
```
- Backend: `http://localhost:8000` (docs at `/docs`, health at `/api/health`)
- Frontend: `http://localhost:3000`
- SQLite database and uploaded-image copies persist in the `backend_data`
  named Docker volume across restarts (`docker compose down` without `-v`
  preserves history; add `-v` to wipe it).

Configurable environment variables:

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `DATABASE_URL` | backend | `sqlite:////app/data/analyses.db` | swap in Postgres etc. by pointing to a different SQLAlchemy URL |
| `DATA_DIR` | backend | `/app/data` | where the SQLite file + uploaded image copies live |
| `CORS_ORIGINS` | backend | `*` | comma-separated allowed origins |
| `VITE_API_BASE_URL` | frontend (build arg) | `http://localhost:8000` | the URL the **browser** uses to reach the backend; baked into the React bundle at build time — see `docker-compose.yml`'s `build.args` |

To point the backend at PostgreSQL instead of SQLite, set e.g.
`DATABASE_URL=postgresql://user:pass@host:5432/dbname` and add a `psycopg2-binary`
line to `backend/requirements.txt`; no application code changes are required
since all access goes through SQLAlchemy.

### Model loading & inference after deployment
`model.joblib` (the trained RandomForest bundle) is committed inside
`backend/app/ml/` and copied into the Docker image at build time — no
training or network access happens at container startup. `app/ml/predict.py`
lazy-loads it once per process (`joblib.load`, cached in a module-level
global) and every `/api/analyze` request reuses the in-memory model for
inference; there is no per-request model reload.

## 6. Frontend

A React (Vite) single-page app styled as a machine-vision inspection console:
drag-and-drop intake with a live "scan" animation, a quality-score readout
with per-issue explanations and confidence, the raw image-statistics panel,
and a persistent history log of past inspections (click any row to reload
that result). Loading, success, and error states are all handled explicitly
via component state — no external state library. See `frontend/README.md`
for the full component/hook breakdown.


## 7. Design notes / judgment calls

- **Why per-issue binary classifiers instead of one multi-class model**:
  issues are not mutually exclusive (an image can be blurry *and*
  underexposed), so independent binary classifiers with their own
  probability/severity are more faithful than a single softmax.
- **Why SQLite by default**: zero external dependency, trivially reproducible
  by graders; `DATABASE_URL` is fully swappable to Postgres (see above).
- **Severity thresholds** (`low <0.4`, `medium 0.4–0.7`, `high ≥0.7` predicted
  probability) and the score-penalty weights are documented, fixed constants
  in `predict.py` — not hidden magic numbers — so they're easy to audit or
  retune.
- **Corruption vs. hard invalid files**: a file that isn't decodable at all
  (empty, truncated, wrong format) is rejected at the API layer with `422`
  before ever reaching the model; the `corruption` ML class instead targets
  images that decode fine but contain compression artifacts / corrupted
  *regions*.

## 8. Bonus work implemented

- ✅ Automated backend tests (`backend/tests/test_api.py`, 8 tests, pytest)
- ✅ Health/status endpoint (`/api/health`) with model-load check
- ✅ Full explainability layer (plain-language, feature-grounded per-issue explanations)
- ✅ Dockerized, reproducible deployment with a named persistent volume

Not implemented (documented as future work): quality heatmaps/localization,
confidence calibration, model versioning, CI/CD, batch analysis endpoint.