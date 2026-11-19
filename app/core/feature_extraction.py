"""Build raw per-pixel RGB/LAB patch payloads from skin extraction output."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from app.core.image_processing import (
    STANDARD_PATCH_SIZE,
    extract_patch_pixel_matrices,
)

REGION_NAMES: list[str] = ["forehead", "left_cheek", "right_cheek", "chin"]


def decode_patch_from_base64(patch_base64: str) -> np.ndarray:
    """Decode a base64-encoded PNG patch into a BGR numpy array."""
    buffer = np.frombuffer(base64.b64decode(patch_base64), dtype=np.uint8)
    patch = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if patch is None:
        raise ValueError("Failed to decode patch image from base64.")
    return patch


def build_region_patches_payload(
    extraction: dict[str, Any],
    patch_size: int = STANDARD_PATCH_SIZE,
) -> dict[str, dict[str, Any]]:
    """
    Convert extracted skin regions into 15x15 (225-pixel) RGB/LAB matrices.

    Skips regions that failed extraction or have no patch image data.
    """
    regions = extraction.get("regions", {})
    region_patches: dict[str, dict[str, Any]] = {}

    for region_name in REGION_NAMES:
        region = regions.get(region_name)
        if not region or region.get("status") == "error":
            continue

        patch_base64 = region.get("patch_base64")
        if not patch_base64:
            continue

        patch = decode_patch_from_base64(patch_base64)
        region_patches[region_name] = extract_patch_pixel_matrices(
            patch,
            patch_size=patch_size,
        )

    return region_patches


def build_regions_metadata(extraction: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return lightweight region metadata (bbox/center/status) without patch images."""
    metadata: dict[str, dict[str, Any]] = {}

    for region_name in REGION_NAMES:
        region = extraction.get("regions", {}).get(region_name)
        if not region:
            continue

        metadata[region_name] = {
            "status": region.get("status"),
            "center_x": region.get("center_x"),
            "center_y": region.get("center_y"),
            "bbox": region.get("bbox"),
            "landmark_indices": region.get("landmark_indices"),
        }

    return metadata


def build_pixel_patch_response(
    extraction: dict[str, Any],
    patch_size: int = STANDARD_PATCH_SIZE,
) -> dict[str, Any]:
    """Assemble the standard raw pixel-matrix API payload."""
    region_patches = build_region_patches_payload(extraction, patch_size=patch_size)

    if not region_patches:
        raise ValueError("No valid skin region patches available for pixel extraction.")

    return {
        "patch_dimensions": f"{patch_size}x{patch_size}",
        "total_pixels_per_patch": patch_size * patch_size,
        "region_patches": region_patches,
    }
