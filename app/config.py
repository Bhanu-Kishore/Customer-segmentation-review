"""Application configuration and filesystem paths.

All paths can be overridden with environment variables so the same code runs
locally, inside Docker and on Render.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else (BASE_DIR / path)


DATA_DIR = _env_path("DATA_DIR", BASE_DIR / "data")
MODELS_DIR = _env_path("MODELS_DIR", BASE_DIR / "models")
STATIC_DIR = BASE_DIR / "app" / "static"
DB_PATH = _env_path("CUSTOMER_DB_PATH", DATA_DIR / "app.db")
DRIFT_REPORT_PATH = _env_path(
    "DRIFT_REPORT_PATH", BASE_DIR / "monitoring" / "data_drift_report.html"
)
CUSTOMERS_CSV_PATH = DATA_DIR / "customers.csv"

APP_ENV = os.environ.get("APP_ENV", "development")
APP_VERSION = "2.0.0"
API_TITLE = "Customer Segmentation API"
API_DESCRIPTION = (
    "K-Means customer segmentation service with customer management, "
    "analytics and data-drift monitoring."
)
