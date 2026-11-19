#!/usr/bin/env python3
"""Download the MediaPipe Face Landmarker task model if not already present."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"


def download_model(force: bool = False) -> Path:
    """Download the face landmarker model into ``models/``."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and not force:
        print(f"Model already exists: {MODEL_PATH}")
        return MODEL_PATH

    print(f"Downloading model from:\n  {MODEL_URL}")
    print(f"Saving to:\n  {MODEL_PATH}")

    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "skin-tone-project/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to download model: {exc}") from exc

    MODEL_PATH.write_bytes(data)
    print(f"Download complete ({len(data):,} bytes).")
    return MODEL_PATH


if __name__ == "__main__":
    force_download = "--force" in sys.argv
    download_model(force=force_download)
