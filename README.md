# Customer Segmentation Pipeline (SegmentIQ)

End-to-end MLOps project that segments customers with K-Means clustering and serves the model through a FastAPI backend with a responsive analytics dashboard.

The project follows a production-style lifecycle: data versioning with DVC, experiment tracking with MLflow, model training with scikit-learn, automated tests, containerisation with Docker, CI with GitHub Actions, data-drift monitoring with Evidently, and public deployment on Render.

> **Live demo:** see [Deployment](#deployment) — the public Render URL is added after the first deployment.

## Features

- **Dashboard** — KPI cards, segment distribution, income-vs-spending scatter plot and live segment insights.
- **Customer management** — paginated directory with search and segment filters, create, update and delete. Every write re-scores the customer with the model.
- **Segmentation** — score a new customer, see the assigned segment, closest-centroid score, distance to every centroid and the typical profile of that segment.
- **Analytics** — age, income and purchase-frequency distributions plus a segment comparison table.
- **Model & drift** — live model metadata (inertia, silhouette, versions, training date), cluster centroid table and the embedded Evidently drift report.
- **REST API** — documented with OpenAPI/Swagger at `/docs`, plus a health endpoint for platform probes.
- **Resilient inference** — if the persisted model cannot be loaded, the service retrains from `data/customers.csv` at startup.

## Architecture

```text
                    ┌─────────────────────────────┐
                    │   Browser dashboard (SPA)   │
                    │  HTML · CSS · JS · Chart.js │
                    └──────────────┬──────────────┘
                                   │ fetch /api/*
                    ┌──────────────▼──────────────┐
                    │        FastAPI service      │
                    │  routers · validation · CORS│
                    ├──────────────┬──────────────┤
                    │  SQLite CRUD │ K-Means model │
                    │  (customers) │ + StandardScaler
                    └──────┬───────┴──────┬───────┘
                           │              │
              ┌────────────▼───┐   ┌──────▼────────┐
              │ data/app.db    │   │ models/*.pkl  │
              │ (auto-seeded)  │   │ + metadata    │
              └────────────────┘   └───────────────┘

 Training pipeline: data/customers.csv → train.py → MLflow run → models + segmented CSV
 MLOps tooling:     DVC (data versioning) · Evidently (drift) · GitHub Actions (CI) · Docker
```

## Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11 |
| ML | scikit-learn K-Means, StandardScaler |
| Data | pandas, NumPy |
| API | FastAPI, Uvicorn, Pydantic |
| Storage | SQLite (stdlib `sqlite3`) |
| Frontend | HTML5, CSS3, vanilla JavaScript, Chart.js 4 |
| Experiment tracking | MLflow |
| Data versioning | DVC |
| Monitoring | Evidently AI |
| CI/CD | GitHub Actions, Render |
| Container | Docker |

## Project Structure

```text
.
├── app/
│   ├── main.py              # FastAPI app, routes, startup seeding
│   ├── config.py            # paths and environment configuration
│   ├── ml.py                # model loading, fallback training, prediction
│   ├── db.py                # SQLite customer repository
│   ├── schemas.py           # Pydantic request/response models
│   ├── web.py               # optional legacy Streamlit client
│   └── static/              # dashboard (index.html, css, js, Chart.js)
├── src/
│   ├── train.py             # training pipeline + MLflow logging
│   └── segmentation.py      # shared feature list, profiles, segment naming
├── monitoring/
│   ├── monitor.py           # Evidently drift report generator
│   └── data_drift_report.html
├── data/
│   ├── customers.csv        # reference dataset
│   ├── current_customers.csv
│   └── segmented_customers.csv.dvc
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   └── model_metadata.json  # metrics, versions, centroid profiles
├── tests/test_api.py        # 13 API/CRUD/prediction tests
├── Dockerfile
├── render.yaml
├── requirements.txt         # runtime dependencies (Render/Docker)
├── requirements-dev.txt     # MLflow, DVC, Evidently, Streamlit, pytest
├── .github/workflows/ci.yml
└── pytest.ini
```

## Quickstart

```bash
# 1. Create an environment (Python 3.11)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install runtime + MLOps tooling
pip install -r requirements-dev.txt

# 3. Train the model (writes models/*.pkl, metadata, segmented CSV, MLflow run)
python src/train.py

# 4. Generate the drift report (optional)
python -m monitoring.monitor

# 5. Run the API and dashboard
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Interactive API docs are at <http://127.0.0.1:8000/docs>.

The database (`data/app.db`) is created and seeded from `data/customers.csv` on first start using the trained model, so no manual setup is required.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8000` | Port used by Uvicorn (Render sets this automatically) |
| `APP_ENV` | `development` | Reported by `/health` |
| `DATA_DIR` | `./data` | Dataset and SQLite location |
| `MODELS_DIR` | `./models` | Model artifact location |
| `CUSTOMER_DB_PATH` | `./data/app.db` | Override the SQLite file |
| `DRIFT_REPORT_PATH` | `./monitoring/data_drift_report.html` | Drift report served by the API |
| `N_CLUSTERS` | `3` | Cluster count used by `src/train.py` |

## Model

- **Algorithm:** K-Means (k = 3, `random_state=42`, `n_init=10`) on four features: `Age`, `AnnualIncome`, `SpendingScore`, `PurchaseFrequency`.
- **Preprocessing:** `StandardScaler` fitted on the reference dataset; the same scaler is used at inference.
- **Metrics (current artifacts):** inertia ≈ 6.86, silhouette ≈ 0.572 (20 reference customers).
- **Segment names** are derived from centroid percentiles rather than hard-coded cluster IDs, so they stay meaningful if clusters are relabelled:
  - `Young Enthusiasts` — low income, high spending and frequency
  - `Affluent Conservatives` — high income, low spending and frequency
  - `Balanced Mainstream` — moderate income and spending
- **Prediction output:** assigned segment, readable name, description, scaled distance to the closest centroid, a 0–1 similarity score and distances to every centroid.

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Liveness/readiness with model and DB status |
| GET | `/api/summary` | KPIs, segment overview, model metrics |
| GET | `/api/segments` | Segment sizes and averages |
| GET | `/api/analytics` | Chart-ready distributions and scatter data |
| GET | `/api/customers` | Paginated list (`page`, `page_size`, `search`, `segment`) |
| POST | `/api/customers` | Create a customer and assign a segment |
| GET | `/api/customers/{id}` | Fetch one customer |
| PUT | `/api/customers/{id}` | Update a customer and re-score |
| DELETE | `/api/customers/{id}` | Delete a customer |
| POST | `/api/predict` | Score without persisting |
| GET | `/api/model` | Model metadata and centroid profiles |
| GET | `/api/drift-report` | Evidently HTML report |
| GET | `/docs` | Swagger UI |

Legacy endpoints kept for compatibility: `POST /predict` (query parameters) and `GET /drift-report`.

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 28, "annual_income": 32000, "spending_score": 82, "purchase_frequency": 15}'
```

## Testing

```bash
python -m pytest
```

The suite covers health checks, static assets, pagination, search/filtering, summary consistency, prediction validation, legacy compatibility, full customer CRUD, model metadata and drift-report availability. Current result: **13 passed**.

## MLOps Workflow

- **Training + MLflow:** `python src/train.py` logs parameters, metrics (inertia, silhouette) and artifacts to `mlflow.db`; inspect with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
- **Data versioning (DVC):** `data/segmented_customers.csv` is tracked with `dvc add`. Re-run after retraining.
- **Data drift (Evidently):** `python -m monitoring.monitor` compares `customers.csv` with `current_customers.csv` and writes the report embedded in the dashboard.
- **CI:** `.github/workflows/ci.yml` installs the runtime requirements, runs the test suite and retrains the model on every push/PR.
- **Docker:** `docker build -t customer-segmentation . && docker run -p 8000:8000 customer-segmentation`.
- **Streamlit (optional):** `streamlit run app/web.py` provides the earlier lightweight client.

## Deployment (Render)

The repository includes a [render.yaml](render.yaml) blueprint for a free Render Web Service.

**Blueprint (recommended)**

1. Push the repository to GitHub.
2. In Render: **New → Blueprint**, select the repository, and apply `render.yaml`.
3. Render installs `requirements.txt` and starts `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
4. The health check path `/health` gates traffic until the model and database are ready.

**Manual Web Service**

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health check | `/health` |
| Env vars | `PYTHON_VERSION=3.11.9`, `APP_ENV=production` |

Notes for the free tier:

- Instances spin down after inactivity; the first request can take ~50 seconds.
- The disk is ephemeral. SQLite data resets on redeploy/restart and is automatically re-seeded from `data/customers.csv` with model predictions, so the app always starts in a consistent state.
- No secrets are required; the application reads only the committed model artifacts and dataset.

## Demonstration Script (for faculty)

1. **Dashboard** — explain the KPI cards, the three segments and the income-vs-spending clusters.
2. **Customers** — add a customer (e.g. age 24, income 28,000, spending 88, frequency 16) and show that the model instantly assigns *Young Enthusiasts*. Then edit and delete it.
3. **Segmentation** — score a new profile, walk through the closest-centroid score, the distance-to-every-centroid chart and the segment profile; optionally save it as a customer.
4. **Analytics** — show the age/income/frequency distributions and the segment comparison table updating with new records.
5. **Model & Drift** — show inertia, silhouette, scikit-learn version, the centroid table and load the Evidently data-drift report.
6. **API docs** — open `/docs`, call `POST /api/predict` from Swagger and show the `/health` probe.
7. **MLOps story** — training script → MLflow experiment → DVC-tracked dataset → GitHub Actions tests → Docker image → Render deployment.
