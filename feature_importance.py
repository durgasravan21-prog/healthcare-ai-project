
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

MODEL_PATH = Path("models/condition_model.joblib")
OUTPUT_PATH = Path("feature_importance.png")


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run model.py first.")
        return

    model = joblib.load(MODEL_PATH)
    preprocessing = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessing.get_feature_names_out()
    importance = abs(classifier.coef_).mean(axis=0)

    results = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance,
    })

    results = results.nlargest(15, "Importance").sort_values("Importance")

    plt.figure(figsize=(12, 7))
    plt.barh(results["Feature"], results["Importance"])
    plt.xlabel("Mean absolute coefficient")
    plt.title("Top 15 Model Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Chart saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
