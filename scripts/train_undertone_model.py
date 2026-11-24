#!/usr/bin/env python3
"""Train undertone classifier from labeled face images and persist production artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.feature_extraction import REGION_NAMES, aggregate_patch_features, decode_patch_from_base64
from app.core.image_processing import (
    FEATURE_KEYS,
    apply_gray_world_wb,
    compute_patch_color_features,
    resize_image_if_large,
)
from app.core.skin_extractor import FaceNotDetectedError, SkinRegionExtractor

logger = logging.getLogger(__name__)

DATASET_DIR = BASE_DIR / "dataset"
FEATURES_CSV = DATASET_DIR / "dataset_features.csv"
MODEL_PATH = BASE_DIR / "models" / "undertone_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"
METRICS_PATH = BASE_DIR / "models" / "training_metrics.json"
FACE_MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

CLASS_DIRS = ("cool", "warm", "neutral")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def discover_dataset_images(dataset_dir: Path) -> list[tuple[str, str, Path]]:
    samples: list[tuple[str, str, Path]] = []
    for label in CLASS_DIRS:
        class_dir = dataset_dir / label
        if not class_dir.is_dir():
            logger.warning("Missing class directory: %s", class_dir)
            continue

        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            samples.append((f"{label}/{image_path.name}", label, image_path))
    return samples


def extract_image_features(
    image_path: Path,
    extractor: SkinRegionExtractor,
    max_dim: int = 800,
) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    resized, _, _ = resize_image_if_large(image, max_dim=max_dim)
    balanced = apply_gray_world_wb(resized)
    extraction = extractor.extract(balanced)

    patch_features: list[dict[str, Any]] = []
    for region_name in REGION_NAMES:
        region = extraction["regions"].get(region_name, {})
        if region.get("status") == "error" or not region.get("patch_base64"):
            continue
        patch = decode_patch_from_base64(region["patch_base64"])
        patch_features.append(compute_patch_color_features(patch))

    if not patch_features:
        raise ValueError(f"No valid skin patches extracted from {image_path}")

    aggregated = aggregate_patch_features(patch_features)
    aggregated["pose_status"] = extraction.get("pose_status")
    return aggregated


def build_feature_dataset(
    samples: list[tuple[str, str, Path]],
    extractor: SkinRegionExtractor,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for index, (filename, label, image_path) in enumerate(samples, start=1):
        logger.info("[%s/%s] Processing %s", index, len(samples), filename)
        try:
            aggregated = extract_image_features(image_path, extractor)
            row = {"filename": filename, "label": label}
            row.update(aggregated["features"])
            rows.append(row)
        except FaceNotDetectedError as exc:
            logger.warning("Face not detected in %s: %s", filename, exc)
            failures.append(filename)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", filename, exc)
            failures.append(filename)

    return rows, failures


def save_features_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filename", "label", *FEATURE_KEYS]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved feature dataset to %s (%s rows)", csv_path, len(rows))


def train_and_evaluate(
    rows: list[dict[str, Any]],
    n_splits: int = 5,
) -> tuple[Any, LabelEncoder, dict[str, Any]]:
    if len(rows) < 2:
        raise ValueError("Need at least 2 labeled samples to train a model.")

    labels = [row["label"] for row in rows]
    x_matrix = np.array([[row[key] for key in FEATURE_KEYS] for row in rows], dtype=np.float64)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels)

    class_counts = {
        str(class_name): int(np.sum(y_encoded == index))
        for index, class_name in enumerate(encoder.classes_)
    }

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=1,
    )

    _, class_sample_counts = np.unique(y_encoded, return_counts=True)
    min_class_count = int(class_sample_counts.min())

    if len(rows) < 3 or min_class_count < 2:
        cv_strategy = LeaveOneOut()
        effective_splits = len(rows)
        logger.warning(
            "Small dataset detected (%s samples); using Leave-One-Out cross-validation.",
            len(rows),
        )
    else:
        effective_splits = max(2, min(n_splits, min_class_count, len(rows)))
        cv_strategy = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)

    cv_scores = cross_val_score(model, x_matrix, y_encoded, cv=cv_strategy, scoring="accuracy")

    model.fit(x_matrix, y_encoded)
    predictions = model.predict(x_matrix)

    conf_matrix = confusion_matrix(y_encoded, predictions, labels=np.arange(len(encoder.classes_)))
    report = classification_report(
        y_encoded,
        predictions,
        target_names=[str(name) for name in encoder.classes_],
        output_dict=True,
        zero_division=0,
    )

    feature_importances = {
        feature_name: round(float(importance), 6)
        for feature_name, importance in zip(FEATURE_KEYS, model.feature_importances_)
    }

    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(rows),
        "class_counts": class_counts,
        "feature_columns": FEATURE_KEYS,
        "cross_validation": {
            "folds": effective_splits,
            "accuracy_mean": round(float(cv_scores.mean()), 4),
            "accuracy_std": round(float(cv_scores.std()), 4),
            "accuracy_scores": [round(float(score), 4) for score in cv_scores],
        },
        "training_accuracy": round(float(np.mean(predictions == y_encoded)), 4),
        "confusion_matrix": {
            "labels": [str(name) for name in encoder.classes_],
            "matrix": conf_matrix.tolist(),
        },
        "classification_report": report,
        "feature_importances": feature_importances,
        "model_type": "RandomForestClassifier",
    }
    return model, encoder, metrics


def save_artifacts(model: Any, encoder: LabelEncoder, metrics: dict[str, Any]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved model artifacts to %s", MODEL_PATH.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train undertone classifier and save artifacts.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--features-csv", type=Path, default=FEATURES_CSV)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    samples = discover_dataset_images(args.dataset_dir)
    if not samples:
        raise SystemExit(
            "No training images found. Add images under dataset/cool, dataset/warm, and dataset/neutral."
        )

    logger.info("Discovered %s labeled images", len(samples))

    with SkinRegionExtractor(model_path=FACE_MODEL_PATH) as extractor:
        rows, failures = build_feature_dataset(samples, extractor)

    if not rows:
        raise SystemExit("Feature extraction failed for all images. Cannot train model.")

    save_features_csv(rows, args.features_csv)
    model, encoder, metrics = train_and_evaluate(rows, n_splits=args.cv_folds)
    metrics["failed_files"] = failures
    save_artifacts(model, encoder, metrics)

    logger.info(
        "Training complete | CV accuracy=%.3f +/- %.3f | classes=%s",
        metrics["cross_validation"]["accuracy_mean"],
        metrics["cross_validation"]["accuracy_std"],
        metrics["class_counts"],
    )


if __name__ == "__main__":
    main()