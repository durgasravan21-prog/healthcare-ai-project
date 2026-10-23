
from pathlib import Path
import pandas as pd


def load_dataset():
    project_dir = Path(__file__).resolve().parent
    csv_file = (
        project_dir
        / "data"
        / "synthetic_healthcare_records_500.csv"
    )

    if not csv_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_file}"
        )

    return pd.read_csv(csv_file)


def generate_report(df):
    total = len(df)

    print("\n" + "=" * 55)
    print("HEALTHCARE RECORDS ANALYSIS — PROJECT REPORT")
    print("=" * 55)

    print("\n1. DATASET OVERVIEW")
    print("Total synthetic records:", total)
    print("Total columns:", len(df.columns))

    if total == 0:
        print("Dataset is empty; further analysis is unavailable.")
        return

    print("\n2. DEMOGRAPHIC SUMMARY")
    print("Average age:", round(df["age"].mean(), 2))

    print("\nAge-group distribution:")
    for group, count in df["age_group"].value_counts().items():
        percentage = count / total * 100
        print(
            f"  {group}: {count} records "
            f"({percentage:.1f}%)"
        )

    print("\n3. HEALTHCARE CATEGORY SUMMARY")
    for category, count in df["category"].value_counts().items():
        percentage = count / total * 100
        print(
            f"  {category}: {count} records "
            f"({percentage:.1f}%)"
        )

    print("\n4. SYNTHETIC TREATMENT-CATEGORY SUMMARY")
    treatment_counts = (
        df["synthetic_treatment_category"].value_counts()
    )

    print("Distinct treatment categories:", len(treatment_counts))

    for treatment, count in treatment_counts.items():
        print(f"  {treatment}: {count} records")

    print("\n5. DATA LIMITATIONS")
    print("- The dataset is synthetic, not real patient data.")
    print("- Treatment labels are illustrative.")
    print("- Results describe this dataset only.")
    print("- Findings are not medical advice.")

    print("\n" + "=" * 55)
    print("END OF REPORT")
    print("=" * 55)


if __name__ == "__main__":
    dataset = load_dataset()
    generate_report(dataset)
