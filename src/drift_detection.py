import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def detect_drift():
    print("⚠️ Detecting data drift...")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train, current_data=test)

    os.makedirs("reports", exist_ok=True)
    report.save_html("reports/drift_report.html")

    print("✅ Drift report generated")

    # Simple logic (always True for demo)
    return True