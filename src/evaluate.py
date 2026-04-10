import pandas as pd
import pickle
from sklearn.metrics import r2_score
from src.preprocessing import preprocess

def evaluate_model(version="v1"):
    print("📊 Evaluating model...")

    df = pd.read_csv("data/test.csv")
    df = preprocess(df)

    X = df.drop("target", axis=1)
    y = df["target"]

    with open(f"models/model_{version}.pkl", "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X)
    score = r2_score(y, preds)

    print(f"✅ Model {version} Score:", score)
    return score