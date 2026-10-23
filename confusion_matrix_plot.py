
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/preprocessed_healthcare_records.csv")
MODEL_PATH = Path("models/condition_model.joblib")
OUTPUT_PATH = Path("confusion_matrix.png")


def main():
    df = pd.read_csv(DATA_PATH)

    target = "synthetic_condition_label"
    features = [
        "age",
        "sex",
        "region",
        "visit_type",
        "reported_symptoms",
        "symptom_duration",
        "reported_severity",
    ]

    X = df[features]
    y = df[target]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)
    predictions = model.predict(X_test)

    labels = model.named_steps["classifier"].classes_
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(15, 13))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )
    display.plot(ax=ax, xticks_rotation=90, colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix — Synthetic Healthcare Dataset")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Confusion matrix saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
