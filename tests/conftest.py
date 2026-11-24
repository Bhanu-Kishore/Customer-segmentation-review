"""Pytest fixtures.

The environment is configured before importing the application so tests use a
throw-away SQLite database instead of the local development database.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

TEST_DB = Path(tempfile.mkdtemp(prefix="segmentation-tests-")) / "test.db"
os.environ["CUSTOMER_DB_PATH"] = str(TEST_DB)
os.environ["APP_ENV"] = "test"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client
