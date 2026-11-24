# Shape & Shade — Skin Tone Extraction & Undertone Analysis API

A modular FastAPI microservice for automated facial skin patch extraction and machine learning–based undertone classification (**Warm**, **Cool**, **Neutral**). Built as the skin-tone analysis engine for the **Shape & Shade** styling recommendation system.

---

## What This Project Does

The pipeline processes face images (file uploads or webcam frames) through two main stages:

1. **Stage 1 — Facial Landmark Detection & Skin Patch Extraction (`app/core/skin_extractor.py`, `app/core/feature_extraction.py`)**
   - Applies image downscaling (`max_dim=800`) and **Gray World white balance** normalization (`app/core/image_processing.py`) to reduce ambient lighting bias.
   - Detects 478 facial landmarks via **MediaPipe Face Landmarker** (`models/face_landmarker.task`), checks head pose orientation, and crops four standardized `15x15` skin regions (**forehead**, **left cheek**, **right cheek**, **chin**).
   - Filters non-skin pixels (hair/shadows) using **YCrCb skin thresholding** and exports flattened **225-pixel RGB and CIELAB matrices** per region.

2. **Stage 2 — Statistical Color Feature Aggregation & Undertone Classification (`app/core/undertone_classifier.py`)**
   - Aggregates **12D color features** (`mean` and `std` across `L*`, `a*`, `b*` and `R`, `G`, `B`) across valid facial skin patches.
   - Runs inference using a trained **Random Forest Classifier** (`models/undertone_model.pkl`) to return the predicted undertone (`warm`, `cool`, or `neutral`), confidence score, and per-class probability distribution.

3. **Web Dashboard & Analysis History (`app/static/index.html`, `app/main.py`)**
   - Includes an interactive web dashboard at `/` and OpenAPI/Swagger documentation at `/docs`, plus in-memory history tracking for uploaded and webcam captures.

---

## Project Structure

```text
skin_tone_extraction/
├── .github/workflows/
│   └── ci-cd.yml                  # Automated GitHub Actions pytest CI pipeline
├── app/
│   ├── core/
│   │   ├── feature_extraction.py  # 15x15 RGB/LAB matrix & 12D feature aggregation
│   │   ├── image_processing.py    # Resizing, Gray World WB, YCrCb skin mask, LAB stats
│   │   ├── skin_extractor.py      # MediaPipe FaceLandmarker wrapper & pose check
│   │   └── undertone_classifier.py# Production sklearn model & label encoder wrapper
│   ├── static/
│   │   └── index.html             # Frontend web UI dashboard
│   ├── main.py                    # FastAPI application & REST endpoints
│   └── schemas.py                 # Pydantic request/response schemas
├── dataset/                       # Training images grouped by class (cool/warm/neutral)
├── models/
│   ├── face_landmarker.task       # MediaPipe face landmark model (downloaded via script)
│   ├── undertone_model.pkl        # Trained RandomForest undertone classifier
│   ├── label_encoder.pkl          # Target label encoder
│   └── training_metrics.json      # Cross-validation & training evaluation report
├── scripts/
│   ├── download_model.py          # Downloads MediaPipe face_landmarker.task
│   ├── fetch_matrices.py          # Helper script to inspect extracted JSON matrices
│   └── train_undertone_model.py   # Feature extraction & model training CLI
├── tests/
│   ├── assets/lena.jpg            # Reference face image for end-to-end testing
│   └── test_pipeline.py           # Unit & end-to-end FastAPI integration tests
├── Dockerfile                     # Container image definition for cloud deployment
├── requirements.txt               # Python dependencies
└── README.md
```

---

## Project Setup Guidelines

### Prerequisites
- **Python 3.11 – 3.14**
- **Git**

### 1. Clone the Repository
```bash
git clone [https://github.com/AlgorithmKoPhyo/shape-and-shade-api.git](https://github.com/AlgorithmKoPhyo/shape-and-shade-api.git)
cd shape-and-shade-api
```

### 2. Create & Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the MediaPipe Face Landmarker Model
*(Required if `models/face_landmarker.task` is not already present locally)*
```bash
python scripts/download_model.py
```

---

## Configuration & Running Commands

### Environment Configuration (Windows OpenBLAS Fix)
On Windows systems with high CPU core counts, restrict OpenBLAS thread allocation before running `scikit-learn` or `pytest` to prevent memory allocation errors:

```powershell
$env:OPENBLAS_NUM_THREADS=1
```

### 1. Start the FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- **Web Dashboard:** `http://127.0.0.1:8000/`
- **Interactive Swagger Docs:** `http://127.0.0.1:8000/docs`
- **Health Check:** `http://127.0.0.1:8000/health`

### 2. Train / Retrain the Undertone Model
Place labeled face images inside `dataset/cool/`, `dataset/warm/`, and `dataset/neutral/`, then run:
```bash
python scripts/train_undertone_model.py
```
This regenerates `dataset/dataset_features.csv` and updates `models/undertone_model.pkl`, `models/label_encoder.pkl`, and `models/training_metrics.json`.

### 3. Run Automated Tests (`pytest`)
```bash
pytest -v
```

### 4. Build & Run with Docker
```bash
docker build -t shape-and-shade-api .
docker run -p 8000:8000 shape-and-shade-api
```

---

## API Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive skin tone analysis web UI |
| `GET` | `/health` | Verifies API status, MediaPipe model, and undertone classifier readiness |
| `POST` | `/api/v1/extract-skin-regions` | Extracts 4-region `15x15` RGB/LAB matrices and predicts skin undertone |
| `GET` | `/api/v1/history` | Returns all past analysis records in the current session |
| `GET` | `/api/v1/history/latest` | Returns the most recent webcam capture analysis |