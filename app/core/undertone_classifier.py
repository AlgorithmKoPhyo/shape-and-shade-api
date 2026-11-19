"""Load and run the trained undertone classification model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from app.core.feature_extraction import FEATURE_KEYS

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "undertone_model.pkl"
DEFAULT_ENCODER_PATH = Path(__file__).resolve().parent.parent / "models" / "label_encoder.pkl"
DEFAULT_METRICS_PATH = Path(__file__).resolve().parent.parent / "models" / "training_metrics.json"


class UndertoneModelNotFoundError(FileNotFoundError):
    """Raised when undertone model artifacts are missing."""


class UndertoneClassifier:
    """Production wrapper for undertone prediction using saved sklearn artifacts."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        encoder_path: Path | str = DEFAULT_ENCODER_PATH,
        metrics_path: Path | str = DEFAULT_METRICS_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)
        self.metrics_path = Path(metrics_path)
        self.model: Any | None = None
        self.encoder: LabelEncoder | None = None
        self.training_metrics: dict[str, Any] | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.encoder is not None

    def load(self) -> None:
        """Load model, label encoder, and optional training metrics from disk."""
        if not self.model_path.is_file():
            raise UndertoneModelNotFoundError(
                f"Undertone model not found at '{self.model_path}'. "
                "Run 'python scripts/train_undertone_model.py' first."
            )
        if not self.encoder_path.is_file():
            raise UndertoneModelNotFoundError(
                f"Label encoder not found at '{self.encoder_path}'. "
                "Run 'python scripts/train_undertone_model.py' first."
            )

        self.model = joblib.load(self.model_path)
        self.encoder = joblib.load(self.encoder_path)

        if self.metrics_path.is_file():
            self.training_metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        else:
            self.training_metrics = None

        logger.info(
            "Loaded undertone classifier from %s (classes=%s)",
            self.model_path,
            list(self.encoder.classes_),
        )

    def predict(self, feature_vector: list[float] | np.ndarray) -> dict[str, Any]:
        """Predict undertone class with confidence scores."""
        if not self.is_loaded:
            raise RuntimeError("Undertone classifier is not loaded.")

        vector = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
        if vector.shape[1] != len(FEATURE_KEYS):
            raise ValueError(
                f"Expected {len(FEATURE_KEYS)} features, received {vector.shape[1]}."
            )

        probabilities = self.model.predict_proba(vector)[0]
        predicted_index = int(np.argmax(probabilities))
        undertone = str(self.encoder.inverse_transform([predicted_index])[0])

        class_probabilities = {
            str(class_name): round(float(probability), 4)
            for class_name, probability in zip(self.encoder.classes_, probabilities)
        }

        return {
            "undertone": undertone,
            "confidence": round(float(probabilities[predicted_index]), 4),
            "probabilities": class_probabilities,
            "feature_vector": [round(float(value), 4) for value in vector.flatten()],
        }
