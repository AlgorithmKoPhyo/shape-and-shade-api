"""Pydantic schemas for Skin Tone Analysis API requests and responses."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class RegionPatchMatrix(BaseModel):
    rgb_pixels: list[list[int]] = Field(
        ..., description="225 RGB pixel triplets [R, G, B] for a 15x15 skin patch."
    )
    lab_pixels: list[list[float]] = Field(
        ..., description="225 standardized CIELAB triplets [L*, a*, b*] for a 15x15 skin patch."
    )


class UndertoneAnalysisResult(BaseModel):
    undertone: str | None = Field(None, description="Predicted undertone: warm, cool, or neutral.")
    confidence: float | None = Field(None, description="Prediction confidence score (0.0 to 1.0).")
    probabilities: dict[str, float] | None = Field(
        None, description="Class probability distribution across warm, cool, and neutral."
    )
    features: dict[str, float] = Field(
        ..., description="Aggregated 12D LAB/RGB statistical color features."
    )
    regions_used: int = Field(..., description="Number of valid facial regions aggregated.")
    status: str | None = Field(None, description="Optional status when model artifact is unloaded.")


class QualityMetadata(BaseModel):
    pose_status: str
    pose_metadata: dict[str, Any]
    face_width: int | None
    dynamic_patch_size: int | None
    output_patch_size: int


class SkinToneAnalysisResponse(BaseModel):
    status: str
    message: str
    undertone_analysis: UndertoneAnalysisResult
    patch_dimensions: str
    total_pixels_per_patch: int
    region_patches: dict[str, RegionPatchMatrix]
    input: dict[str, str | None]
    image: dict[str, Any]
    quality: QualityMetadata
    regions_metadata: dict[str, dict[str, Any]]
    history_id: int


class HealthResponse(BaseModel):
    status: str
    face_model_loaded: bool
    face_model_path: str
    undertone_model_loaded: bool