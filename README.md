# MLOps Pipeline with Data Drift Detection

## Overview
This project implements an end-to-end MLOps pipeline including:
- Data ingestion
- Preprocessing
- Model training & evaluation
- Data drift detection (Evidently AI)
- Automated retraining
- Model versioning

## Workflow
1. Load dataset
2. Train model (v1)
3. Evaluate performance
4. Simulate data drift
5. Detect drift
6. Retrain model (v2)
7. Compare performance

## Outputs
- Models: models/model_v1.pkl, model_v2.pkl
- Drift Report: reports/drift_report.html

## Tech Stack
- Python
- Scikit-learn
- Pandas
- Evidently AI
