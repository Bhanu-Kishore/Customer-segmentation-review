"""Generate an Evidently data-drift report comparing reference and current data.

Usage:
    python -m monitoring.monitor
    python monitoring.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
REPORT_PATH = Path(
    os.environ.get("DRIFT_REPORT_PATH", BASE_DIR / "monitoring" / "data_drift_report.html")
)

REFERENCE_PATH = DATA_DIR / "customers.csv"
CURRENT_PATH = DATA_DIR / "current_customers.csv"


def main() -> None:
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(f"Reference dataset not found: {REFERENCE_PATH}")
    if not CURRENT_PATH.is_file():
        raise FileNotFoundError(f"Current dataset not found: {CURRENT_PATH}")

    reference_data = pd.read_csv(REFERENCE_PATH, skip_blank_lines=True)
    current_data = pd.read_csv(CURRENT_PATH, skip_blank_lines=True)

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference_data, current_data=current_data)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(REPORT_PATH))

    print("===================================")
    print("DATA DRIFT ANALYSIS COMPLETED")
    print("===================================")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
