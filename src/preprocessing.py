from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    print("⚙️ Preprocessing data...")

    df = df.copy()

    # Fill missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    return df