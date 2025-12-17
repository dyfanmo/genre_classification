import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from google.cloud import storage

from src.training_pipeline import (
    balance_by_downsampling,
    train_and_evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vertex AI training entrypoint"
    )
    parser.add_argument(
        "--data-uri",
        type=str,
        required=True,
        help="Path to embeddings CSV. Supports local path or gs://bucket/path.csv",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="If set, downsample each genre to the minimum class count.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for splitting / balancing.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="Override model output directory. If empty, uses AIP_MODEL_DIR.",
    )
    return parser.parse_args()


def download_from_gcs(gs_uri: str, local_path: Path) -> Path:
    assert gs_uri.startswith("gs://")
    _, _, bucket_name, *blob_parts = gs_uri.split("/")
    blob_name = "/".join(blob_parts)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    return local_path


def load_embeddings_csv(data_uri: str) -> pd.DataFrame:
    if data_uri.startswith("gs://"):
        tmp_path = Path("/tmp") / "panns_embeddings.csv"
        download_from_gcs(data_uri, tmp_path)
        return pd.read_csv(tmp_path)
    return pd.read_csv(data_uri)


def resolve_vertex_dirs(model_dir_override: str) -> Tuple[Path, Path]:
    model_dir = Path(
        model_dir_override
        if model_dir_override
        else os.environ.get("AIP_MODEL_DIR", "/model")
    )
    metric_dir = Path(os.environ.get("AIP_METRIC_DIR", "/metrics"))
    model_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    return model_dir, metric_dir


def write_metrics(metrics: Dict[str, Any], metric_dir: Path) -> None:
    out = {
        "accuracy": float(metrics["accuracy"]),
    }
    (metric_dir / "metrics.json").write_text(json.dumps(out, indent=2))


def main() -> None:
    args = parse_args()

    df = load_embeddings_csv(args.data_uri)

    if args.balance:
        df = balance_by_downsampling(
            df, label_column="genre", random_state=args.random_state
        )

    model, metrics = train_and_evaluate(
        df, test_size=0.2, random_state=args.random_state
    )

    model_dir, metric_dir = resolve_vertex_dirs(args.model_dir)

    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)

    write_metrics(metrics, metric_dir)

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metric_dir / 'metrics.json'}")
    print("Accuracy:", metrics["accuracy"])


if __name__ == "__main__":
    main()
