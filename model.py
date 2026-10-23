
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "preprocessed_healthcare_records.csv"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_FILE = MODEL_DIR / "condition_model.joblib"

TARGET = "synthetic_condition_label"

NUMERIC_FEATURES = ["age"]

CATEGORICAL_FEATURES = [
    "sex",
    "region",
    "visit_type",
    "reported_symptoms",
    "symptom_duration",
    "reported_severity",
]


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run preprocessing.py first."
        )

    return pd.read_csv(DATA_FILE)


def train_model(df):
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[required].copy()
    data = data.dropna(subset=[TARGET])
    data[TARGET] = data[TARGET].astype(str).str.strip()
    data = data[data[TARGET] != ""]

    if data[TARGET].nunique() < 2:
        raise ValueError("At least two target classes are required.")

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]

    class_counts = y.value_counts()
    stratify_target = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_target,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])

    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print("MODEL TRAINING SUMMARY")
    print("-" * 35)
    print(f"Total usable records: {len(data)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Target classes: {y.nunique()}")
    print(f"Model saved to: {MODEL_FILE}")

    return model, X_test, y_test


def main():
    df = load_data()
    train_model(df)


if __name__ == "__main__":
    main()
