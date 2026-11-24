"""FastAPI application for customer segmentation."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.db import CustomerRepository
from app.ml import SegmentationService
from app.schemas import (
    CustomerCreate,
    CustomerListResponse,
    CustomerResponse,
    CustomerUpdate,
    PredictRequest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

service = SegmentationService(config.MODELS_DIR, config.DATA_DIR)
repository = CustomerRepository(config.DB_PATH, service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()
    repository.init_schema()
    seeded = repository.seed_from_csv(config.CUSTOMERS_CSV_PATH)
    if seeded:
        logger.info("Seeded %d customers into %s", seeded, config.DB_PATH)
    logger.info(
        "Startup complete | env=%s | db=%s | model_ready=%s",
        config.APP_ENV,
        config.DB_PATH,
        service.loaded,
    )
    yield


app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.APP_VERSION,
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


# --------------------------------------------------------------------- pages
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    index_path = config.STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


# ------------------------------------------------------------------- health
@app.get("/health", tags=["system"])
def health(response: Response) -> dict:
    database_ok = True
    customer_count = 0
    try:
        customer_count = repository.count()
    except Exception as exc:  # pragma: no cover - defensive
        database_ok = False
        logger.warning("Database health check failed: %s", exc)

    healthy = service.loaded and database_ok
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ok" if healthy else "degraded",
        "model_loaded": service.loaded,
        "database": "ok" if database_ok else "unavailable",
        "customers": customer_count,
        "environment": config.APP_ENV,
        "version": config.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ------------------------------------------------------------------ summary
@app.get("/api/summary", tags=["analytics"])
def summary() -> dict:
    overview = repository.segment_overview()
    model_status = service.status()
    return {
        **overview,
        "model": {
            "algorithm": model_status["algorithm"],
            "n_clusters": model_status["n_clusters"],
            "inertia": model_status["inertia"],
            "silhouette_score": model_status["silhouette_score"],
            "trained_at": model_status["trained_at"],
        },
        "recent_customers": repository.recent(limit=5),
    }


@app.get("/api/segments", tags=["analytics"])
def segments() -> dict:
    overview = repository.segment_overview()
    return {
        "total_customers": overview["total_customers"],
        "segment_count": overview["segment_count"],
        "segments": overview["segments"],
    }


@app.get("/api/analytics", tags=["analytics"])
def analytics() -> dict:
    customers = repository.all_customers()
    overview = repository.segment_overview()
    segment_names = {
        segment["segment"]: segment["name"] for segment in overview["segments"]
    }

    age_buckets = _bucketize(
        [row["age"] for row in customers],
        [24, 34, 44, 54, 100],
        ["18-24", "25-34", "35-44", "45-54", "55+"],
    )
    frequency_buckets = _bucketize(
        [row["purchase_frequency"] for row in customers],
        [4, 9, 14, 19, 365],
        ["0-4", "5-9", "10-14", "15-19", "20+"],
    )
    income_buckets = _bucketize(
        [row["annual_income"] for row in customers],
        [30_000, 60_000, 90_000, 120_000, 1_000_000],
        ["<30k", "30k-60k", "60k-90k", "90k-120k", "120k+"],
    )

    return {
        "total_customers": overview["total_customers"],
        "segments": overview["segments"],
        "scatter": [
            {
                "id": row["id"],
                "name": row["name"],
                "age": row["age"],
                "income": row["annual_income"],
                "spending": row["spending_score"],
                "frequency": row["purchase_frequency"],
                "segment": row["segment"],
                "segment_name": segment_names.get(row["segment"], row["segment_name"]),
            }
            for row in customers
        ],
        "age_buckets": age_buckets,
        "frequency_buckets": frequency_buckets,
        "income_buckets": income_buckets,
    }


@app.get("/api/model", tags=["model"])
def model_info() -> dict:
    return {
        **service.status(),
        "segments": list(service.segment_info().values()),
    }


# ----------------------------------------------------------------- customers
@app.get(
    "/api/customers",
    response_model=CustomerListResponse,
    tags=["customers"],
)
def list_customers(
    segment: int | None = Query(default=None, ge=0, description="Filter by segment id"),
    search: str | None = Query(default=None, max_length=80, description="Name or id"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
) -> dict:
    return repository.list_customers(
        segment=segment, search=search, page=page, page_size=page_size
    )


@app.post(
    "/api/customers",
    response_model=CustomerResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["customers"],
)
def create_customer(payload: CustomerCreate) -> dict:
    return repository.create(
        name=payload.name,
        age=payload.age,
        annual_income=payload.annual_income,
        spending_score=payload.spending_score,
        purchase_frequency=payload.purchase_frequency,
    )


@app.get(
    "/api/customers/{customer_id}",
    response_model=CustomerResponse,
    tags=["customers"],
)
def get_customer(customer_id: int) -> dict:
    customer = repository.get(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@app.put(
    "/api/customers/{customer_id}",
    response_model=CustomerResponse,
    tags=["customers"],
)
def update_customer(customer_id: int, payload: CustomerUpdate) -> dict:
    changes = payload.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(status_code=400, detail="No fields to update")

    customer = repository.update(customer_id, changes)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@app.delete(
    "/api/customers/{customer_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["customers"],
)
def delete_customer(customer_id: int) -> Response:
    if not repository.delete(customer_id):
        raise HTTPException(status_code=404, detail="Customer not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --------------------------------------------------------------- predictions
@app.post("/api/predict", tags=["predictions"])
def predict(payload: PredictRequest) -> dict:
    prediction = service.predict(
        payload.age,
        payload.annual_income,
        payload.spending_score,
        payload.purchase_frequency,
    )
    return {
        **prediction,
        "input": {
            "age": payload.age,
            "annual_income": payload.annual_income,
            "spending_score": payload.spending_score,
            "purchase_frequency": payload.purchase_frequency,
        },
    }


@app.post("/predict", tags=["predictions"], include_in_schema=False)
def legacy_predict(
    age: int = Query(..., ge=18, le=100),
    income: float = Query(..., ge=1_000, le=1_000_000),
    spending_score: float = Query(..., ge=0, le=100),
    frequency: int = Query(..., ge=0, le=365),
) -> dict:
    prediction = service.predict(age, income, spending_score, frequency)
    return {
        "customer_segment": prediction["segment"],
        "segment_name": prediction["segment_name"],
    }


# ------------------------------------------------------------------- reports
@app.get("/api/drift-report/status", tags=["monitoring"])
def drift_report_status() -> dict:
    path = config.DRIFT_REPORT_PATH
    if not path.is_file():
        return {"available": False, "generated_at": None, "size_bytes": 0}

    stat = path.stat()
    return {
        "available": True,
        "generated_at": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
        "size_bytes": stat.st_size,
    }


@app.get("/api/drift-report", tags=["monitoring"], response_class=FileResponse)
def drift_report() -> FileResponse:
    if not config.DRIFT_REPORT_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="Drift report not generated yet. Run python -m monitoring.monitor",
        )
    return FileResponse(config.DRIFT_REPORT_PATH, media_type="text/html")


@app.get("/drift-report", include_in_schema=False, response_class=FileResponse)
def legacy_drift_report() -> FileResponse:
    return drift_report()


def _bucketize(values: list[float], upper_bounds: list[float], labels: list[str]) -> list[dict]:
    counts = [0] * len(upper_bounds)
    for value in values:
        for index, upper in enumerate(upper_bounds):
            if value <= upper:
                counts[index] += 1
                break
        else:
            counts[-1] += 1
    return [{"label": label, "count": count} for label, count in zip(labels, counts)]
