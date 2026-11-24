"""Backwards-compatible entry point for the data-drift report.

Prefer: python -m monitoring.monitor
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "monitoring" / "monitor.py"),
        run_name="__main__",
    )
