from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_healthcare_data():
    project_dir = Path(__file__).resolve().parent.parent
    print("Project folder:", project_dir)

    csv_file = (
        project_dir
        / "data"
        / "synthetic_healthcare_records_500.csv"
    )

    print("Looking for dataset at:", csv_file)

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {csv_file}"
        )

    return pd.read_csv(csv_file)



def plot_age_group_distribution(df):
    counts = df["age_group"].value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("Synthetic Healthcare Records by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("age_group_distribution.png", dpi=300)
    plt.show()


def plot_category_distribution(df):
    counts = df["category"].value_counts().sort_values()

    plt.figure(figsize=(10, 7))
    counts.plot(kind="barh")

    plt.title("Synthetic Healthcare Records by Category")
    plt.xlabel("Number of Records")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig("category_distribution.png", dpi=300)
    plt.show()


def plot_treatment_distribution(df):
    counts = (
        df["synthetic_treatment_category"]
        .value_counts()
        .sort_values()
    )

    plt.figure(figsize=(10, 8))
    counts.plot(kind="barh")

    plt.title("Synthetic Treatment-Category Distribution")
    plt.xlabel("Number of Records")
    plt.ylabel("Synthetic Treatment Category")
    plt.tight_layout()
    plt.savefig("treatment_distribution.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    print("Loading synthetic healthcare dataset...")

    df = load_healthcare_data()

    print("Dataset loaded successfully.")
    print("Total records:", len(df))

    print("\nGenerating age-group chart...")
    plot_age_group_distribution(df)

    print("Generating category chart...")
    plot_category_distribution(df)

    print("Generating treatment-category chart...")
    plot_treatment_distribution(df)

    print("\nAll charts have been generated.")
    print("Check the main project folder for PNG files.")
