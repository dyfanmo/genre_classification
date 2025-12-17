from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
)
import joblib

from src.config import EMBEDDINGS_CSV


def load_embeddings(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def balance_by_downsampling(
    df: pd.DataFrame,
    label_column: str = "genre",
    random_state: int = 42,
) -> pd.DataFrame:

    class_counts = df[label_column].value_counts()
    min_count = class_counts.min()
    print("Class counts BEFORE balancing:")
    print(class_counts)
    print(f"\nBalancing all genres to {min_count} samples each.")

    balanced = df.groupby(label_column, group_keys=False).sample(
        n=min_count, random_state=random_state
    )

    print("\nClass counts AFTER balancing:")
    print(balanced[label_column].value_counts())
    return balanced


def split_features_labels(
    df: pd.DataFrame,
    label_column: str = "genre",
) -> Tuple[pd.DataFrame, pd.Series]:

    y = df[label_column]
    X = df.drop(columns=[label_column, "file"], errors="ignore")
    return X, y


def build_classifier_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, Dict[str, Any]]:

    X, y = split_features_labels(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = build_classifier_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    report_str = classification_report(y_val, y_pred, output_dict=False)

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "classification_report_str": report_str,
    }

    return pipeline, metrics


def save_trained_model(
    model: Pipeline,
    models_dir: Path,
    filename: str = "panns_logreg_genre.pkl",
) -> Path:

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / filename
    joblib.dump(model, model_path)
    return model_path


def run_training() -> None:

    df = load_embeddings(EMBEDDINGS_CSV)
    balanced_df = balance_by_downsampling(df)

    model, metrics = train_and_evaluate(balanced_df)

    print("\nValidation results:")
    print(metrics["classification_report_str"])
    print("Accuracy:", metrics["accuracy"])

    model_path = save_trained_model(model, Path("models"))
    print(f"\nSaved classifier to {model_path}")
