import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    print("📥 Loading data...")

    # Add column names manually
    columns = [
        "id", "area", "bedrooms", "bathrooms", "floors",
        "year_built", "location", "condition", "garage", "price"
    ]

    df = pd.read_csv("data/raw.csv", header=None, names=columns)

    # Set target column
    df.rename(columns={"price": "target"}, inplace=True)

    print("Columns:", df.columns)

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    print("✅ Data split done")