
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/preprocessed_healthcare_records.csv")

NUMERIC_FEATURES = ["age"]
CATEGORICAL_FEATURES = [
    "sex",
    "region",
    "visit_type",
    "reported_symptoms",
    "symptom_duration",
    "reported_severity",
]
TARGET = "synthetic_condition_label"


def build_pipeline(classifier):
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

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
    }

    results = []

    for name, classifier in models.items():
        model = build_pipeline(classifier)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Macro F1": f1_score(
                y_test, predictions, average="macro", zero_division=0
            ),
        })

    results_df = pd.DataFrame(results).sort_values(
        "Macro F1", ascending=False
    )
    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False, formatters={
        "Accuracy": "{:.3f}".format,
        "Macro F1": "{:.3f}".format,
    }))

    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nResults saved to model_comparison_results.csv")


if __name__ == "__main__":
    main()
