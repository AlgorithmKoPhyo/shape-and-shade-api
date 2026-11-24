"""FastAPI server for skin region pixel extraction and undertone classification."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.feature_extraction import (
    build_pixel_patch_response,
    build_regions_metadata,
    extract_face_feature_vector,
)
from app.core.image_processing import (
    STANDARD_PATCH_SIZE,
    apply_gray_world_wb,
    resize_image_if_large,
)
from app.core.skin_extractor import (
    FaceNotDetectedError,
    ModelNotFoundError,
    SkinRegionExtractor,
)
from app.core.undertone_classifier import (
    UndertoneClassifier,
    UndertoneModelNotFoundError,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"
STATIC_DIR = BASE_DIR / "app" / "static"
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "application/octet-stream",
}

_extractor: SkinRegionExtractor | None = None
_classifier: UndertoneClassifier | None = None
prediction_history: list[dict[str, Any]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _extractor, _classifier

    logger.info("Starting skin tone API; loading models...")
    try:
        _extractor = SkinRegionExtractor(model_path=MODEL_PATH)
    except ModelNotFoundError as exc:
        logger.error("%s", exc)
        _extractor = None

    try:
        classifier = UndertoneClassifier()
        classifier.load()
        _classifier = classifier
    except UndertoneModelNotFoundError as exc:
        logger.warning("%s", exc)
        _classifier = None

    yield

    if _extractor is not None:
        _extractor.close()
        _extractor = None
    _classifier = None
    logger.info("Skin tone API shutdown complete.")


app = FastAPI(
    title="Skin Tone Analysis API",
    description=(
        "Skin region extraction and undertone classification API: face landmark detection, "
        "15x15 RGB/LAB pixel matrix export, and ML-based Warm/Cool/Neutral prediction."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _decode_upload(contents: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR OpenCV image."""
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    buffer = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode image. Upload a valid JPEG, PNG, WEBP, or BMP file.",
        )
    return image


def _get_extractor() -> SkinRegionExtractor:
    if _extractor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Face landmarker model is not loaded. "
                "Run 'python scripts/download_model.py' and restart the server."
            ),
        )
    return _extractor


def _infer_input_source(filename: str | None) -> str:
    if not filename:
        return "upload"
    lowered = filename.lower()
    if "webcam" in lowered or "capture" in lowered:
        return "webcam"
    return "upload"


def _build_history_entry(
    filename: str | None,
    content_type: str | None,
    pixel_payload: dict[str, Any],
    quality: dict[str, Any],
    undertone_analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "content_type": content_type,
        "source": _infer_input_source(filename),
        "patch_dimensions": pixel_payload["patch_dimensions"],
        "total_pixels_per_patch": pixel_payload["total_pixels_per_patch"],
        "region_patches": pixel_payload["region_patches"],
        "undertone_analysis": undertone_analysis,
        "quality": quality,
    }


def _append_prediction_history(entry: dict[str, Any]) -> None:
    prediction_history.append(entry)
    logger.info(
        "Recorded analysis history entry #%s (source=%s, regions=%s)",
        len(prediction_history),
        entry.get("source"),
        list(entry.get("region_patches", {}).keys()),
    )


@app.get("/api/v1/history", tags=["History"])
async def get_prediction_history() -> dict[str, Any]:
    """Return all past skin tone analysis results."""
    return {
        "status": "success",
        "count": len(prediction_history),
        "history": prediction_history,
    }


@app.get("/api/v1/history/latest", tags=["History"])
async def get_latest_webcam_history() -> JSONResponse:
    """Return the most recent webcam capture analysis."""
    webcam_entries = [
        entry for entry in prediction_history if entry.get("source") == "webcam"
    ]

    if not webcam_entries:
        return JSONResponse(
            status_code=404,
            content={
                "status": "empty",
                "message": "No webcam capture analyses found in history.",
            },
        )

    latest = webcam_entries[-1]
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "patch_dimensions": latest["patch_dimensions"],
            "total_pixels_per_patch": latest["total_pixels_per_patch"],
            "region_patches": latest["region_patches"],
            "undertone_analysis": latest.get("undertone_analysis"),
            "timestamp": latest["timestamp"],
            "filename": latest.get("filename"),
            "source": latest.get("source"),
            "quality": latest.get("quality"),
        },
    )


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    """Serve the skin tone analysis web dashboard."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    return FileResponse(index_path)


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "face_model_loaded": _extractor is not None,
        "face_model_path": str(MODEL_PATH),
        "undertone_model_loaded": _classifier is not None and _classifier.is_loaded,
    }


@app.post("/api/v1/extract-skin-regions", tags=["Skin Extraction"])
async def extract_skin_regions_endpoint(
    file: UploadFile = File(..., description="Face image (upload or webcam frame)"),
) -> JSONResponse:
    """
    Process an uploaded image, extract 15x15 RGB/LAB pixel matrices,
    and classify skin undertone (Warm / Cool / Neutral) if the model is loaded.
    """
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type: {content_type}",
        )

    try:
        contents = await file.read()
        original_image = _decode_upload(contents)

        processed_image, was_resized, original_size = resize_image_if_large(
            original_image,
            max_dim=800,
        )
        balanced_image = apply_gray_world_wb(processed_image)

        extractor = _get_extractor()
        extraction = extractor.extract(balanced_image)
        pixel_payload = build_pixel_patch_response(
            extraction,
            patch_size=STANDARD_PATCH_SIZE,
        )

        # Stage 2: 12D Feature Aggregation & Undertone Classification
        aggregated = extract_face_feature_vector(extraction)
        undertone_analysis: dict[str, Any] | None = None
        if _classifier is not None and _classifier.is_loaded:
            prediction = _classifier.predict(aggregated["feature_vector"])
            undertone_analysis = {
                "undertone": prediction["undertone"],
                "confidence": prediction["confidence"],
                "probabilities": prediction["probabilities"],
                "features": aggregated["features"],
                "regions_used": aggregated["regions_used"],
            }
        else:
            undertone_analysis = {
                "status": "model_not_loaded",
                "features": aggregated["features"],
                "regions_used": aggregated["regions_used"],
            }

        processed_height, processed_width = balanced_image.shape[:2]
        pose_status = extraction.get("pose_status", "unknown")

        message = "Skin region pixel matrices and undertone analyzed successfully."
        if pose_status == "face_turned_sideways":
            message = (
                "Analysis complete, but the face appears turned sideways. "
                "For best accuracy, face the camera directly."
            )

        quality = {
            "pose_status": pose_status,
            "pose_metadata": extraction.get("pose_metadata", {}),
            "face_width": extraction.get("face_width"),
            "dynamic_patch_size": extraction.get("patch_size"),
            "output_patch_size": STANDARD_PATCH_SIZE,
        }

        response_payload = {
            "status": "success",
            "message": message,
            "undertone_analysis": undertone_analysis,
            "patch_dimensions": pixel_payload["patch_dimensions"],
            "total_pixels_per_patch": pixel_payload["total_pixels_per_patch"],
            "region_patches": pixel_payload["region_patches"],
            "input": {
                "filename": file.filename,
                "content_type": file.content_type,
            },
            "image": {
                "original_width": original_size["width"],
                "original_height": original_size["height"],
                "processed_width": processed_width,
                "processed_height": processed_height,
                "was_resized": was_resized,
                "white_balance_applied": True,
            },
            "quality": quality,
            "regions_metadata": build_regions_metadata(extraction),
        }

        history_entry = _build_history_entry(
            filename=file.filename,
            content_type=file.content_type,
            pixel_payload=pixel_payload,
            quality=quality,
            undertone_analysis=undertone_analysis,
        )
        _append_prediction_history(history_entry)
        response_payload["history_id"] = len(prediction_history)

        return JSONResponse(status_code=200, content=response_payload)

    except FaceNotDetectedError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "message": str(exc),
                "error_code": "FACE_NOT_DETECTED",
            },
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=422,
            content={
                "status": "error",
                "message": str(exc),
                "error_code": "PATCH_EXTRACTION_FAILED",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during skin tone analysis")
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)