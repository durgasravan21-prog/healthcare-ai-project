
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "preprocessed_healthcare_records.csv"
MODEL_FILE = PROJECT_DIR / "models" / "condition_model.joblib"

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


def evaluate_model():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Preprocessed dataset not found. Run preprocessing.py first."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Trained model not found. Run model.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET] != ""]

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    class_counts = y.value_counts()
    stratify_target = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify_target,
    )

    model = joblib.load(MODEL_FILE)

    # Evaluate the saved model on the held-out test set.
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nMODEL EVALUATION")
    print("=" * 45)
    print(f"Test records: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f}")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )

    print("\nCLASSIFICATION REPORT")
    print(report)

    labels = model.named_steps["classifier"].classes_

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    print("\nCONFUSION MATRIX")
    print(matrix_df)

    # Cross-validation uses fresh model fits on the full dataset.
    cv_scores = None

    if class_counts.min() >= 2:
        folds = min(5, int(class_counts.min()))

        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=42,
        )

        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
        )

        print("\nCROSS-VALIDATION ACCURACY")
        print("Fold scores:", cv_scores.round(3))
        print(f"Mean accuracy: {cv_scores.mean():.3f}")
        print(f"Standard deviation: {cv_scores.std():.3f}")
    else:
        print(
            "\nCross-validation skipped because at least one class "
            "has fewer than two records."
        )

    output_file = PROJECT_DIR / "evaluation_report.txt"

    with output_file.open("w", encoding="utf-8") as file:
        file.write("SYNTHETIC DATASET MODEL EVALUATION\n")
        file.write("=" * 45 + "\n")
        file.write(f"Test records: {len(y_test)}\n")
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write("CLASSIFICATION REPORT\n")
        file.write(report)
        file.write("\nCONFUSION MATRIX\n")
        file.write(matrix_df.to_string())

        if cv_scores is not None:
            file.write("\n\nCROSS-VALIDATION ACCURACY\n")
            file.write(f"Fold scores: {cv_scores.round(3)}\n")
            file.write(f"Mean accuracy: {cv_scores.mean():.3f}\n")
            file.write(f"Standard deviation: {cv_scores.std():.3f}\n")

        file.write(
            "\n\nLimitation: Results describe performance on a synthetic "
            "educational dataset only. They do not establish clinical validity."
        )

    print(f"\nEvaluation report saved to: {output_file}")


if __name__ == "__main__":
    evaluate_model()
