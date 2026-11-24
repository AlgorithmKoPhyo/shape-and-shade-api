from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.core.image_processing import (
    STANDARD_PATCH_SIZE,
    TOTAL_PATCH_PIXELS,
    apply_gray_world_wb,
    compute_patch_color_features,
    extract_patch_pixel_matrices,
    resize_image_if_large,
)
from app.core.skin_extractor import SkinRegionExtractor
from app.main import app

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def test_resize_and_white_balance() -> None:
    large_img = np.full((1200, 1600, 3), [120, 150, 200], dtype=np.uint8)
    resized, was_resized, orig = resize_image_if_large(large_img, max_dim=800)
    assert was_resized is True
    assert max(resized.shape[:2]) == 800
    assert orig == {"width": 1600, "height": 1200}

    balanced = apply_gray_world_wb(resized)
    assert balanced.shape == resized.shape
    assert balanced.dtype == np.uint8


def test_patch_rgb_and_lab_matrices() -> None:
    dummy_skin_patch = np.full((20, 20, 3), [130, 160, 210], dtype=np.uint8)
    matrices = extract_patch_pixel_matrices(dummy_skin_patch)

    assert len(matrices["rgb_pixels"]) == TOTAL_PATCH_PIXELS
    assert len(matrices["lab_pixels"]) == TOTAL_PATCH_PIXELS
    assert len(matrices["rgb_pixels"][0]) == 3
    assert len(matrices["lab_pixels"][0]) == 3

    features_out = compute_patch_color_features(dummy_skin_patch)
    assert len(features_out["feature_vector"]) == 12


def test_mediapipe_model_loads() -> None:
    with SkinRegionExtractor() as extractor:
        assert extractor.model_path.is_file()


def test_api_health_and_no_face_handling() -> None:
    with TestClient(app) as client:
        health_res = client.get("/health")
        assert health_res.status_code == 200
        health_data = health_res.json()
        assert health_data["face_model_loaded"] is True
        assert health_data["undertone_model_loaded"] is True

        # Send a blank image (no face) and verify clean 422 error response
        import cv2

        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", blank)
        res = client.post(
            "/api/v1/extract-skin-regions",
            files={"file": ("blank.jpg", encoded.tobytes(), "image/jpeg")},
        )
        assert res.status_code == 422
        assert res.json()["error_code"] == "FACE_NOT_DETECTED"


def test_end_to_end_lena_extraction_and_undertone() -> None:
    lena_path = ASSETS_DIR / "lena.jpg"
    assert lena_path.is_file(), f"Missing test asset: {lena_path}"

    with TestClient(app) as client:
        with lena_path.open("rb") as image_file:
            res = client.post(
                "/api/v1/extract-skin-regions",
                files={"file": ("lena.jpg", image_file, "image/jpeg")},
            )

        assert res.status_code == 200
        payload = res.json()
        assert payload["status"] == "success"
        assert payload["patch_dimensions"] == f"{STANDARD_PATCH_SIZE}x{STANDARD_PATCH_SIZE}"
        assert payload["total_pixels_per_patch"] == TOTAL_PATCH_PIXELS

        # Stage 1 check: 4 facial skin region pixel matrices (225 pixels each)
        region_patches = payload["region_patches"]
        for region in ("forehead", "left_cheek", "right_cheek", "chin"):
            assert region in region_patches
            assert len(region_patches[region]["rgb_pixels"]) == TOTAL_PATCH_PIXELS
            assert len(region_patches[region]["lab_pixels"]) == TOTAL_PATCH_PIXELS

        # Stage 2 check: Undertone ML prediction + 12D features
        undertone_data = payload["undertone_analysis"]
        assert undertone_data["undertone"] in {"warm", "cool", "neutral"}
        assert 0.0 <= undertone_data["confidence"] <= 1.0
        assert set(undertone_data["probabilities"].keys()) == {"warm", "cool", "neutral"}
        assert len(undertone_data["features"]) == 12