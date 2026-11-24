"""Shared customer segmentation domain logic.

Used by the training pipeline (src/train.py) and the FastAPI service
(app/ml.py) so cluster naming and profiling never drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["Age", "AnnualIncome", "SpendingScore", "PurchaseFrequency"]

FEATURE_LABELS = {
    "Age": "Age",
    "AnnualIncome": "Annual income",
    "SpendingScore": "Spending score",
    "PurchaseFrequency": "Purchase frequency",
}

SEGMENT_DESCRIPTIONS = {
    "Young Enthusiasts": (
        "Younger, lower-income shoppers who spend heavily and buy often. "
        "Engage them with loyalty perks, trend-led campaigns and mobile offers."
    ),
    "Affluent Conservatives": (
        "High-income customers who spend carefully and rarely. "
        "Nurture them with premium value propositions and personal outreach."
    ),
    "Premium Loyalists": (
        "High-income, high-spending customers. "
        "Retain them with VIP programmes, exclusive launches and concierge service."
    ),
    "Budget Conscious": (
        "Price-sensitive customer with low spend and low frequency. "
        "Re-engage them with discounts, bundles and entry-level products."
    ),
    "Balanced Mainstream": (
        "Moderate income and moderate spending. "
        "Grow them with cross-sell, seasonal promotions and loyalty tiers."
    ),
}


def describe_segment(name: str) -> str:
    return SEGMENT_DESCRIPTIONS.get(
        name, "Customer group discovered by the K-Means clustering model."
    )


def _percentile(values: pd.Series, threshold: float) -> float:
    if values.empty:
        return 0.5
    return float((values < threshold).mean())


def name_clusters(profiles: pd.DataFrame, customers: pd.DataFrame) -> dict[int, str]:
    """Give each cluster a human readable name based on its income/spending profile.

    ``profiles`` must contain one row per cluster with the mean feature values in
    original units. Percentiles are computed against the customer population so
    the names stay stable regardless of the arbitrary K-Means label order.
    """

    used: set[str] = set()
    names: dict[int, str] = {}

    for cluster_id, row in profiles.iterrows():
        income_pct = _percentile(customers["AnnualIncome"], row["AnnualIncome"])
        spend_pct = _percentile(customers["SpendingScore"], row["SpendingScore"])

        if income_pct >= 0.6 and spend_pct >= 0.6:
            name = "Premium Loyalists"
        elif income_pct >= 0.6 and spend_pct < 0.4:
            name = "Affluent Conservatives"
        elif income_pct < 0.4 and spend_pct >= 0.6:
            name = "Young Enthusiasts"
        elif income_pct < 0.4 and spend_pct < 0.4:
            name = "Budget Conscious"
        else:
            name = "Balanced Mainstream"

        if name in used:
            name = f"{name} {int(cluster_id) + 1}"
        used.add(name)
        names[int(cluster_id)] = name

    return names


def build_cluster_profiles(
    customers: pd.DataFrame,
    model,
    scaler,
) -> list[dict]:
    """Build per-cluster profile records with original-unit averages."""

    centers_scaled = np.asarray(model.cluster_centers_, dtype=float)
    centers = scaler.inverse_transform(centers_scaled)
    profiles = pd.DataFrame(centers, columns=FEATURES)
    profiles.index = list(range(len(centers)))

    names = name_clusters(profiles, customers)
    total = max(len(customers), 1)

    result: list[dict] = []
    for cluster_id in range(len(centers)):
        row = profiles.loc[cluster_id]
        size = int((customers["Segment"] == cluster_id).sum()) if "Segment" in customers else 0
        result.append(
            {
                "segment": cluster_id,
                "name": names[cluster_id],
                "description": describe_segment(names[cluster_id]),
                "size": size,
                "share": round(size / total, 4),
                "avg_age": round(float(row["Age"]), 2),
                "avg_income": round(float(row["AnnualIncome"]), 2),
                "avg_spending_score": round(float(row["SpendingScore"]), 2),
                "avg_purchase_frequency": round(float(row["PurchaseFrequency"]), 2),
            }
        )

    return sorted(result, key=lambda item: item["segment"])


def assign_segment_labels(
    customers: pd.DataFrame, profiles: list[dict]
) -> pd.Series:
    """Map numeric segments to their readable names."""

    mapping = {profile["segment"]: profile["name"] for profile in profiles}
    return customers["Segment"].map(mapping)
