"""Train the K-Means customer segmentation model.

Usage:
    python src/train.py            # train with 3 clusters (default)
    N_CLUSTERS=4 python src/train.py

Artifacts produced:
    models/kmeans_model.pkl
    models/scaler.pkl
    models/model_metadata.json
    data/segmented_customers.csv
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.segmentation import FEATURES, build_cluster_profiles  # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", BASE_DIR / "models"))

DATASET_PATH = DATA_DIR / "customers.csv"
SEGMENTED_PATH = DATA_DIR / "segmented_customers.csv"
MODEL_PATH = MODELS_DIR / "kmeans_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

N_CLUSTERS = int(os.environ.get("N_CLUSTERS", "3"))
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, skip_blank_lines=True)

    missing = [column for column in FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURES).reset_index(drop=True)
    df = df.drop_duplicates(subset=FEATURES).reset_index(drop=True)

    if "CustomerID" not in df.columns:
        df.insert(0, "CustomerID", range(1, len(df) + 1))

    if len(df) < N_CLUSTERS:
        raise ValueError(
            f"Need at least {N_CLUSTERS} rows to fit {N_CLUSTERS} clusters, "
            f"found {len(df)}."
        )

    return df


def log_to_mlflow(params: dict, metrics: dict, artifacts: list[Path]) -> str | None:
    """Log the run to MLflow. Training still succeeds if MLflow is unavailable."""

    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed - skipping experiment tracking.")
        return None

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Customer-Segmentation")

    try:
        with mlflow.start_run(run_name=f"kmeans-k{N_CLUSTERS}"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "dataset": DATASET_PATH.name,
                    "training_rows": str(params.get("rows", "")),
                }
            )
            for artifact in artifacts:
                if artifact.is_file():
                    mlflow.log_artifact(str(artifact))
            return mlflow.active_run().info.run_id
    except Exception as exc:  # pragma: no cover - tracking must never block training
        print(f"MLflow logging skipped: {exc}")
        return None


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    print(f"Dataset loaded from {DATASET_PATH} ({len(df)} customers)")
    print(df.head())

    X = df[FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    clusters = model.fit_predict(X_scaled)

    df["Segment"] = clusters

    profiles = build_cluster_profiles(df, model, scaler)
    names = {profile["segment"]: profile["name"] for profile in profiles}
    df["SegmentName"] = df["Segment"].map(names)

    silhouette = None
    if 1 < N_CLUSTERS < len(df):
        silhouette = float(silhouette_score(X_scaled, clusters))

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata = {
        "algorithm": "KMeans",
        "n_clusters": N_CLUSTERS,
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "rows": int(len(df)),
        "inertia": round(float(model.inertia_), 4),
        "silhouette_score": round(silhouette, 4) if silhouette is not None else None,
        "trained_at": trained_at,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scaler": {
            "mean": [round(float(value), 4) for value in scaler.mean_],
            "scale": [round(float(value), 4) for value in scaler.scale_],
        },
        "cluster_profiles": profiles,
    }

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    df.to_csv(SEGMENTED_PATH, index=False)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    run_id = log_to_mlflow(
        params={
            "algorithm": "KMeans",
            "n_clusters": N_CLUSTERS,
            "random_state": RANDOM_STATE,
            "features": ",".join(FEATURES),
            "rows": len(df),
        },
        metrics={
            "inertia": float(model.inertia_),
            **({"silhouette_score": silhouette} if silhouette is not None else {}),
        },
        artifacts=[MODEL_PATH, SCALER_PATH, METADATA_PATH, SEGMENTED_PATH],
    )

    print("\nModel trained successfully!")
    print(f"Cluster profiles:")
    for profile in profiles:
        print(
            f"  Segment {profile['segment']} - {profile['name']}: "
            f"{profile['size']} customers, "
            f"avg income {profile['avg_income']:.0f}, "
            f"avg spending {profile['avg_spending_score']:.1f}"
        )
    print(f"Inertia: {model.inertia_:.2f}")
    if silhouette is not None:
        print(f"Silhouette score: {silhouette:.4f}")
    if run_id:
        print(f"MLflow run id: {run_id}")
    print(f"\nArtifacts saved to {MODELS_DIR} and {SEGMENTED_PATH}")


if __name__ == "__main__":
    main()
