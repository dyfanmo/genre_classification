# scripts/train_classifier.py
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training_pipeline import run_training  # noqa: E402


if __name__ == "__main__":
    run_training()
