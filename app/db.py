"""SQLite-backed customer repository."""

from __future__ import annotations

import csv
import logging
import sqlite3
from datetime import datetime, timezone
from math import ceil
from pathlib import Path

from app.ml import SegmentationService

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    name               TEXT NOT NULL,
    age                INTEGER NOT NULL,
    annual_income      REAL NOT NULL,
    spending_score     REAL NOT NULL,
    purchase_frequency INTEGER NOT NULL,
    segment            INTEGER NOT NULL,
    segment_name       TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_customers_segment ON customers(segment);
CREATE INDEX IF NOT EXISTS idx_customers_name ON customers(name);
"""

COLUMNS = (
    "id",
    "name",
    "age",
    "annual_income",
    "spending_score",
    "purchase_frequency",
    "segment",
    "segment_name",
    "created_at",
    "updated_at",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {key: row[key] for key in COLUMNS}


class CustomerRepository:
    def __init__(self, db_path: Path, service: SegmentationService) -> None:
        self.db_path = Path(db_path)
        self.service = service

    # ------------------------------------------------------------- connection
    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.db_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM customers").fetchone()[0])

    # ------------------------------------------------------------------ seed
    def seed_from_csv(self, csv_path: Path) -> int:
        """Populate an empty database from the reference dataset."""

        if self.count() > 0 or not Path(csv_path).is_file():
            return 0

        required = {
            "Age",
            "AnnualIncome",
            "SpendingScore",
            "PurchaseFrequency",
        }
        rows = []
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not required.issubset(set(reader.fieldnames or [])):
                logger.warning("Reference dataset is missing required columns")
                return 0
            for index, raw in enumerate(reader, start=1):
                if not raw:
                    continue
                try:
                    age = float(raw["Age"])
                    annual_income = float(raw["AnnualIncome"])
                    spending_score = float(raw["SpendingScore"])
                    purchase_frequency = float(raw["PurchaseFrequency"])
                except (KeyError, TypeError, ValueError):
                    continue

                try:
                    customer_id = int(raw.get("CustomerID") or index)
                except (TypeError, ValueError):
                    customer_id = index

                prediction = self.service.predict(
                    age, annual_income, spending_score, purchase_frequency
                )
                now = _utc_now()
                rows.append(
                    (
                        customer_id,
                        f"Customer {customer_id:03d}",
                        int(age),
                        annual_income,
                        spending_score,
                        int(purchase_frequency),
                        prediction["segment"],
                        prediction["segment_name"],
                        now,
                        now,
                    )
                )

        if not rows:
            return 0

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO customers (
                    id, name, age, annual_income, spending_score,
                    purchase_frequency, segment, segment_name,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        logger.info("Seeded %d customers from %s", len(rows), csv_path)
        return len(rows)

    # ------------------------------------------------------------------- CRUD
    def list_customers(
        self,
        segment: int | None = None,
        search: str | None = None,
        page: int = 1,
        page_size: int = 10,
    ) -> dict:
        where: list[str] = []
        params: list = []

        if segment is not None:
            where.append("segment = ?")
            params.append(segment)
        if search:
            where.append("(name LIKE ? COLLATE NOCASE OR CAST(id AS TEXT) = ?)")
            params.extend([f"%{search.strip()}%", search.strip()])

        clause = f"WHERE {' AND '.join(where)}" if where else ""
        offset = (page - 1) * page_size

        with self._connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM customers {clause}", params
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM customers {clause}
                ORDER BY id ASC
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, offset],
            ).fetchall()

        return {
            "items": [_row_to_dict(row) for row in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": max(ceil(total / page_size), 1) if total else 0,
        }

    def get(self, customer_id: int) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM customers WHERE id = ?", (customer_id,)
            ).fetchone()
        return _row_to_dict(row) if row else None

    def create(
        self,
        name: str | None,
        age: int,
        annual_income: float,
        spending_score: float,
        purchase_frequency: int,
    ) -> dict:
        prediction = self.service.predict(
            age, annual_income, spending_score, purchase_frequency
        )
        now = _utc_now()
        clean_name = (name or "").strip() or "Unnamed customer"

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO customers (
                    name, age, annual_income, spending_score,
                    purchase_frequency, segment, segment_name,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_name,
                    age,
                    annual_income,
                    spending_score,
                    purchase_frequency,
                    prediction["segment"],
                    prediction["segment_name"],
                    now,
                    now,
                ),
            )
            customer_id = int(cursor.lastrowid)

        return self.get(customer_id)  # type: ignore[return-value]

    def update(self, customer_id: int, changes: dict) -> dict | None:
        current = self.get(customer_id)
        if current is None:
            return None

        merged = {
            "name": changes.get("name") or current["name"],
            "age": changes.get("age", current["age"]),
            "annual_income": changes.get("annual_income", current["annual_income"]),
            "spending_score": changes.get("spending_score", current["spending_score"]),
            "purchase_frequency": changes.get(
                "purchase_frequency", current["purchase_frequency"]
            ),
        }

        prediction = self.service.predict(
            merged["age"],
            merged["annual_income"],
            merged["spending_score"],
            merged["purchase_frequency"],
        )

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE customers SET
                    name = ?, age = ?, annual_income = ?, spending_score = ?,
                    purchase_frequency = ?, segment = ?, segment_name = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    merged["name"],
                    merged["age"],
                    merged["annual_income"],
                    merged["spending_score"],
                    merged["purchase_frequency"],
                    prediction["segment"],
                    prediction["segment_name"],
                    _utc_now(),
                    customer_id,
                ),
            )

        return self.get(customer_id)

    def delete(self, customer_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM customers WHERE id = ?", (customer_id,)
            )
            return cursor.rowcount > 0

    # -------------------------------------------------------------- analytics
    def all_customers(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM customers ORDER BY id ASC"
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def recent(self, limit: int = 5) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM customers ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def segment_overview(self) -> dict:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    segment,
                    segment_name,
                    COUNT(*) AS size,
                    ROUND(AVG(age), 2) AS avg_age,
                    ROUND(AVG(annual_income), 2) AS avg_income,
                    ROUND(AVG(spending_score), 2) AS avg_spending_score,
                    ROUND(AVG(purchase_frequency), 2) AS avg_purchase_frequency
                FROM customers
                GROUP BY segment, segment_name
                """
            ).fetchall()
            overall = connection.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    ROUND(AVG(age), 2) AS avg_age,
                    ROUND(AVG(annual_income), 2) AS avg_income,
                    ROUND(AVG(spending_score), 2) AS avg_spending_score,
                    ROUND(AVG(purchase_frequency), 2) AS avg_purchase_frequency
                FROM customers
                """
            ).fetchone()

        live = {int(row["segment"]): dict(row) for row in rows}
        total = int(overall["total"]) if overall else 0

        segments = []
        for segment_id, info in sorted(self.service.segment_info().items()):
            stats = live.get(segment_id)
            size = int(stats["size"]) if stats else 0
            segments.append(
                {
                    "segment": segment_id,
                    "name": info.get("name", stats["segment_name"] if stats else f"Segment {segment_id}"),
                    "description": info.get("description", ""),
                    "size": size,
                    "share": round(size / total, 4) if total else 0.0,
                    "avg_age": float(stats["avg_age"]) if stats and stats["avg_age"] is not None else 0.0,
                    "avg_income": float(stats["avg_income"]) if stats and stats["avg_income"] is not None else 0.0,
                    "avg_spending_score": float(stats["avg_spending_score"]) if stats and stats["avg_spending_score"] is not None else 0.0,
                    "avg_purchase_frequency": float(stats["avg_purchase_frequency"]) if stats and stats["avg_purchase_frequency"] is not None else 0.0,
                }
            )

        return {
            "total_customers": total,
            "segment_count": len(segments),
            "avg_age": float(overall["avg_age"]) if overall and overall["avg_age"] is not None else 0.0,
            "avg_income": float(overall["avg_income"]) if overall and overall["avg_income"] is not None else 0.0,
            "avg_spending_score": float(overall["avg_spending_score"]) if overall and overall["avg_spending_score"] is not None else 0.0,
            "avg_purchase_frequency": float(overall["avg_purchase_frequency"]) if overall and overall["avg_purchase_frequency"] is not None else 0.0,
            "segments": segments,
        }
