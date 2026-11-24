"""Model loading, fallback training and prediction."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.segmentation import (
    FEATURES,
    build_cluster_profiles,
    describe_segment,
)

logger = logging.getLogger(__name__)


class SegmentationService:
    """Owns the trained K-Means model, scaler and cluster profiles."""

    def __init__(self, models_dir: Path, data_dir: Path, n_clusters: int = 3) -> None:
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.features = list(FEATURES)
        self.n_clusters = n_clusters
        self.model: KMeans | None = None
        self.scaler: StandardScaler | None = None
        self.metadata: dict = {}
        self.reference: pd.DataFrame | None = None
        self._segment_info: dict[int, dict] = {}
        self._lock = threading.Lock()
        self.loaded = False
        self.fallback_trained = False
        self.load_error: str | None = None

    # ------------------------------------------------------------------ load
    @property
    def model_path(self) -> Path:
        return self.models_dir / "kmeans_model.pkl"

    @property
    def scaler_path(self) -> Path:
        return self.models_dir / "scaler.pkl"

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / "model_metadata.json"

    def load(self) -> None:
        """Load persisted artifacts, retraining from the CSV if they are unusable."""

        self._load_reference_data()

        try:
            model = joblib.load(self.model_path)
            scaler = joblib.load(self.scaler_path)
            self._validate_model(model, scaler)
            self.model, self.scaler = model, scaler
            logger.info("Loaded model and scaler from %s", self.models_dir)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Could not load saved model (%s) - retraining", exc)
            self._train_fallback()

        if self.fallback_trained:
            self.metadata = {}
        else:
            self._load_metadata()
        self._refresh_segment_info()
        self.loaded = True

    def _load_reference_data(self) -> None:
        dataset_path = self.data_dir / "customers.csv"
        if not dataset_path.is_file():
            logger.warning("Reference dataset not found at %s", dataset_path)
            return
        try:
            df = pd.read_csv(dataset_path, skip_blank_lines=True)
            for column in self.features:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            self.reference = df.dropna(subset=self.features).reset_index(drop=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not read reference dataset: %s", exc)

    def _validate_model(self, model, scaler) -> None:
        if not hasattr(model, "predict") or not hasattr(model, "transform"):
            raise TypeError("Loaded object is not a fitted KMeans model")
        if getattr(scaler, "n_features_in_", None) != len(self.features):
            raise ValueError("Scaler feature count does not match the model features")
        if getattr(model, "n_features_in_", None) != len(self.features):
            raise ValueError("Model feature count does not match the model features")
        model.predict(np.zeros((1, len(self.features))))

    def _train_fallback(self) -> None:
        if self.reference is None or len(self.reference) < self.n_clusters:
            raise RuntimeError(
                "Model artifacts are missing and there is no dataset to retrain from"
            )

        X = self.reference[self.features]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)
        model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        model.fit(scaled)

        self.model, self.scaler = model, scaler
        self.fallback_trained = True

        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, self.model_path)
            joblib.dump(scaler, self.scaler_path)
            logger.info("Fallback model trained on startup and saved")
        except Exception as exc:  # pragma: no cover - read-only filesystem
            logger.warning("Could not persist fallback model: %s", exc)

    def _load_metadata(self) -> None:
        if not self.metadata_path.is_file():
            return
        try:
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read model metadata: %s", exc)

    def _refresh_segment_info(self) -> None:
        profiles: list[dict] = self.metadata.get("cluster_profiles") or []
        if not profiles and self.reference is not None:
            labelled = self.reference.copy()
            scaled = self.scaler.transform(labelled[self.features])
            labelled["Segment"] = self.model.predict(scaled)
            profiles = build_cluster_profiles(labelled, self.model, self.scaler)

        self._segment_info = {
            int(profile["segment"]): {
                "segment": int(profile["segment"]),
                "name": profile["name"],
                "description": profile.get("description")
                or describe_segment(profile["name"]),
                "avg_age": float(profile["avg_age"]),
                "avg_income": float(profile["avg_income"]),
                "avg_spending_score": float(profile["avg_spending_score"]),
                "avg_purchase_frequency": float(profile["avg_purchase_frequency"]),
            }
            for profile in profiles
        }

    # -------------------------------------------------------------- inference
    def predict(
        self,
        age: float,
        annual_income: float,
        spending_score: float,
        purchase_frequency: float,
    ) -> dict:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model is not loaded")

        values = pd.DataFrame(
            [[age, annual_income, spending_score, purchase_frequency]],
            columns=self.features,
        )

        with self._lock:
            scaled = self.scaler.transform(values)
            segment = int(self.model.predict(scaled)[0])
            distances = np.asarray(self.model.transform(scaled)[0], dtype=float)

        info = self._segment_info.get(segment, {})
        distance = float(distances[segment])
        centroid_distances = [
            {
                "segment": index,
                "name": self._segment_info.get(index, {}).get(
                    "name", f"Segment {index}"
                ),
                "distance": round(float(value), 4),
            }
            for index, value in enumerate(distances)
        ]

        return {
            "segment": segment,
            "segment_name": info.get("name", f"Segment {segment}"),
            "description": info.get(
                "description", describe_segment(info.get("name", ""))
            ),
            "distance_to_centroid": round(distance, 4),
            "similarity": round(1.0 / (1.0 + distance), 4),
            "centroid_distances": centroid_distances,
            "profile": {
                key: info.get(key)
                for key in (
                    "avg_age",
                    "avg_income",
                    "avg_spending_score",
                    "avg_purchase_frequency",
                )
            },
        }

    # ---------------------------------------------------------------- reports
    def segment_info(self) -> dict[int, dict]:
        return {key: dict(value) for key, value in self._segment_info.items()}

    def status(self) -> dict:
        metadata = self.metadata
        silhouette = metadata.get("silhouette_score")
        if silhouette is None and self.reference is not None and self.model is not None:
            scaled = self.scaler.transform(self.reference[self.features])
            labels = self.model.predict(scaled)
            if 1 < len(set(labels)) < len(labels):
                silhouette = float(silhouette_score(scaled, labels))
                silhouette = round(silhouette, 4)

        trained_at = metadata.get("trained_at")
        if not trained_at and self.model_path.is_file():
            trained_at = datetime.fromtimestamp(
                self.model_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds")

        return {
            "loaded": self.loaded and self.model is not None,
            "fallback_trained": self.fallback_trained,
            "load_error": self.load_error,
            "algorithm": "KMeans",
            "n_clusters": int(getattr(self.model, "n_clusters", self.n_clusters)),
            "features": self.features,
            "rows": metadata.get("rows") or (
                int(len(self.reference)) if self.reference is not None else None
            ),
            "inertia": round(float(self.model.inertia_), 4) if self.model is not None else None,
            "silhouette_score": silhouette,
            "trained_at": trained_at,
            "sklearn_version": metadata.get("sklearn_version", sklearn.__version__),
            "numpy_version": metadata.get("numpy_version", np.__version__),
            "pandas_version": metadata.get("pandas_version", pd.__version__),
        }
