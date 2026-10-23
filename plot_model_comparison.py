
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = Path("model_comparison_results.csv")
OUTPUT_PATH = Path("model_comparison.png")


def main():
    if not RESULTS_PATH.exists():
        print("Results file not found. Run model_comparison.py first.")
        return

    df = pd.read_csv(RESULTS_PATH)

    ax = df.set_index("Model")[["Accuracy", "Macro F1"]].plot(
        kind="bar",
        figsize=(9, 5),
        ylim=(0, 1),
        rot=0,
    )

    ax.set_title("Machine Learning Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Chart saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
