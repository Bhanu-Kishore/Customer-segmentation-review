"""API, CRUD and prediction tests."""

from __future__ import annotations

VALID_CUSTOMER = {
    "name": "Test Customer",
    "age": 29,
    "annual_income": 33000,
    "spending_score": 81,
    "purchase_frequency": 14,
}


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["database"] == "ok"
    assert payload["customers"] >= 20


def test_frontend_and_static_assets(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "SegmentIQ" in page.text

    script = client.get("/static/js/app.js")
    assert script.status_code == 200
    assert "loadDashboard" in script.text

    styles = client.get("/static/css/styles.css")
    assert styles.status_code == 200


def test_list_customers_pagination(client):
    response = client.get("/api/customers", params={"page": 1, "page_size": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] >= 20
    assert len(payload["items"]) == 5
    assert payload["pages"] >= 4
    assert {"id", "name", "segment", "segment_name"} <= set(payload["items"][0])


def test_customer_filters_and_search(client):
    segments = client.get("/api/segments").json()["segments"]
    segment_id = segments[0]["segment"]

    filtered = client.get("/api/customers", params={"segment": segment_id, "page_size": 100})
    assert filtered.status_code == 200
    assert all(item["segment"] == segment_id for item in filtered.json()["items"])

    searched = client.get("/api/customers", params={"search": "Customer 001"})
    assert searched.status_code == 200
    assert any(item["name"] == "Customer 001" for item in searched.json()["items"])


def test_segment_summary_matches_customer_total(client):
    summary = client.get("/api/summary").json()
    analytics = client.get("/api/analytics").json()

    assert summary["total_customers"] == analytics["total_customers"]
    assert sum(segment["size"] for segment in summary["segments"]) == summary["total_customers"]
    assert len(analytics["scatter"]) == summary["total_customers"]
    assert summary["model"]["n_clusters"] == 3


def test_prediction_endpoint(client):
    response = client.post(
        "/api/predict",
        json={
            "age": 28,
            "annual_income": 32000,
            "spending_score": 82,
            "purchase_frequency": 15,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert 0 <= payload["segment"] <= 2
    assert payload["segment_name"]
    assert payload["description"]
    assert 0 < payload["similarity"] <= 1
    assert len(payload["centroid_distances"]) == 3
    assert payload["input"]["age"] == 28


def test_prediction_validation(client):
    response = client.post(
        "/api/predict",
        json={
            "age": 10,
            "annual_income": 32000,
            "spending_score": 82,
            "purchase_frequency": 15,
        },
    )
    assert response.status_code == 422


def test_legacy_predict_endpoint(client):
    response = client.post(
        "/predict",
        params={"age": 50, "income": 90000, "spending_score": 20, "frequency": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "customer_segment" in payload
    assert payload["segment_name"]


def test_customer_crud_lifecycle(client):
    created = client.post("/api/customers", json=VALID_CUSTOMER)
    assert created.status_code == 201
    customer = created.json()
    customer_id = customer["id"]
    try:
        assert customer["name"] == "Test Customer"
        assert customer["segment_name"]

        fetched = client.get(f"/api/customers/{customer_id}")
        assert fetched.status_code == 200

        updated = client.put(
            f"/api/customers/{customer_id}",
            json={"spending_score": 12, "annual_income": 120000, "purchase_frequency": 1},
        )
        assert updated.status_code == 200
        assert updated.json()["spending_score"] == 12
        assert updated.json()["annual_income"] == 120000
    finally:
        deleted = client.delete(f"/api/customers/{customer_id}")
        assert deleted.status_code == 204

    assert client.get(f"/api/customers/{customer_id}").status_code == 404
    assert client.delete(f"/api/customers/{customer_id}").status_code == 404


def test_customer_create_validation(client):
    response = client.post(
        "/api/customers",
        json={**VALID_CUSTOMER, "annual_income": -5},
    )
    assert response.status_code == 422


def test_update_without_fields_rejected(client):
    created = client.post("/api/customers", json=VALID_CUSTOMER).json()
    try:
        response = client.put(f"/api/customers/{created['id']}", json={})
        assert response.status_code == 400
    finally:
        client.delete(f"/api/customers/{created['id']}")


def test_model_info(client):
    response = client.get("/api/model")
    assert response.status_code == 200
    payload = response.json()
    assert payload["loaded"] is True
    assert payload["n_clusters"] == 3
    assert payload["algorithm"] == "KMeans"
    assert len(payload["segments"]) == 3
    assert len(payload["features"]) == 4


def test_drift_report_status(client):
    status = client.get("/api/drift-report/status")
    assert status.status_code == 200
    payload = status.json()
    assert "available" in payload

    if payload["available"]:
        report = client.get("/api/drift-report")
        assert report.status_code == 200
        assert "text/html" in report.headers["content-type"]
