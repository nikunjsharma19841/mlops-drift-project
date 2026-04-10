from src.data_ingestion import load_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.drift_detection import detect_drift
from src.retrain import retrain_model
import pandas as pd

def simulate_drift():
    print("⚠️ Simulating data drift...")
    df = pd.read_csv("data/test.csv")
    df.iloc[:, 0] = df.iloc[:, 0] * 1.5
    df.to_csv("data/test.csv", index=False)

def run_pipeline():
    print("🚀 Pipeline started")

    load_data()
    train_model("v1")
    acc1 = evaluate_model("v1")

    simulate_drift()

    drift = detect_drift()

    if drift:
        retrain_model()
        acc2 = evaluate_model("v2")

        print("\n📊 Comparison:")
        print("v1:", acc1)
        print("v2:", acc2)

    print("🎯 Done")

if __name__ == "__main__":
    run_pipeline()  