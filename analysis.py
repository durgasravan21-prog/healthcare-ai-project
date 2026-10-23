
from pathlib import Path
import pandas as pd



def load_healthcare_data():
    project_dir = Path(__file__).resolve().parent

    csv_file = (
        project_dir
        / "data"
        / "synthetic_healthcare_records_500.csv"
    )

    print("Dataset path:", csv_file)

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_file}"
        )

    return pd.read_csv(csv_file)


def analyze_age_group_by_category(df):
    return pd.crosstab(
        df["age_group"],
        df["category"]
    )


def analyze_category_by_treatment(df):
    return pd.crosstab(
        df["category"],
        df["synthetic_treatment_category"]
    )


def analyze_treatment_distribution(df):
    return (
        df["synthetic_treatment_category"]
        .value_counts()
        .rename_axis("Treatment Category")
        .reset_index(name="Record Count")
    )

def plot_age_category_heatmap(df):
    import matplotlib.pyplot as plt

    table = pd.crosstab(
        df["age_group"],
        df["category"]
    )

    plt.figure(figsize=(14, 5))

    plt.imshow(table, aspect="auto")
    plt.colorbar(label="Number of Records")

    plt.xticks(
        range(len(table.columns)),
        table.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(table.index)),
        table.index
    )

    plt.title("Age Group vs Healthcare Category")
    plt.xlabel("Healthcare Category")
    plt.ylabel("Age Group")

    plt.tight_layout()
    plt.savefig(
        "age_category_heatmap.png",
        dpi=300
    )
    plt.show()



if __name__ == "__main__":
    df = load_healthcare_data()

    print("=== Age Group vs Healthcare Category ===")
    print(analyze_age_group_by_category(df).to_string())

    print("\n=== Healthcare Category vs Synthetic Treatment ===")
    print(analyze_category_by_treatment(df).to_string())

    print("\n=== Synthetic Treatment Distribution ===")
    print(analyze_treatment_distribution(df).to_string(index=False))

    print(
        "\nNote: These are descriptive patterns in synthetic "
        "records, not clinical conclusions."
    )
    plot_age_category_heatmap(df)