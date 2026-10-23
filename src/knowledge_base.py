
from pathlib import Path
import pandas as pd


def load_healthcare_data():
    project_dir = Path(__file__).resolve().parent.parent
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


def build_knowledge_base(df):
    knowledge = {
        "total_records": len(df),
        "average_age": round(df["age"].mean(), 2),
        "age_group_counts": (
            df["age_group"].value_counts().to_dict()
        ),
        "category_counts": (
            df["category"].value_counts().to_dict()
        ),
        "treatment_category_counts": (
            df["synthetic_treatment_category"]
            .value_counts()
            .to_dict()
        ),
    }

    return knowledge


if __name__ == "__main__":
    df = load_healthcare_data()
    knowledge = build_knowledge_base(df)

    print("=== Healthcare Knowledge Base ===")
    print("Total records:", knowledge["total_records"])
    print("Average age:", knowledge["average_age"])

    print("\nRecords by age group:")
    for group, count in knowledge["age_group_counts"].items():
        print(f"{group}: {count}")

    print("\nRecords by category:")
    for category, count in knowledge["category_counts"].items():
        print(f"{category}: {count}")

    print("\nRecords by treatment category:")
    for treatment, count in knowledge["treatment_category_counts"].items():
        print(f"{treatment}: {count}")
