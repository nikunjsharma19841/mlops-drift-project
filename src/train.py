import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from src.preprocessing import preprocess

def train_model(version="v1"):
    print("🤖 Training model...")

    df = pd.read_csv("data/train.csv")
    df = preprocess(df)

    X = df.drop("target", axis=1)
    y = df["target"]

    model = RandomForestRegressor()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)

    with open(f"models/model_{version}.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model {version} saved")