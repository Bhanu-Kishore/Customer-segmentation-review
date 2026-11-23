import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Reference / original data
reference_data = pd.read_csv(
    "data/customers.csv"
)

# New / current data
current_data = pd.read_csv(
    "data/current_customers.csv"
)

# Create Evidently report
report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

# Compare reference and current data
result = report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save report
result.save_html(
    "monitoring/data_drift_report.html"
)

print("===================================")
print("DATA DRIFT ANALYSIS COMPLETED")
print("===================================")
print("Report saved to:")
print("monitoring/data_drift_report.html")