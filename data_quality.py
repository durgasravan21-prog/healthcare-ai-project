
from pathlib import Path
import pandas as pd


EXPECTED_RECORDS = 500

EXPECTED_AGE_GROUPS = {
    "Child",
    "Adult",
    "Senior",
}

EXPECTED_CATEGORIES = {
    "Respiratory",
    "Dermatological",
    "Gastrointestinal",
    "Allergy",
    "Musculoskeletal",
    "Endocrine",
    "General",
    "Routine",
    "Urinary",
    "Mental wellness",
    "Cardiovascular",
    "Ear",
    "Neurological",
    "Eye",
    "Dental",
}



def load_data():
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



def check_data_quality(df):
    report = {}

    report["record_count"] = len(df)
    report["expected_record_count"] = EXPECTED_RECORDS
    report["record_count_ok"] = len(df) == EXPECTED_RECORDS

    report["missing_values"] = (
        df.isnull().sum()
        .loc[lambda counts: counts > 0]
        .to_dict()
    )

    report["duplicate_rows"] = int(
        df.duplicated().sum()
    )

    report["duplicate_patient_ids"] = int(
        df["patient_id"].duplicated().sum()
    ) if "patient_id" in df.columns else None

    if "age_group" in df.columns:
        found_age_groups = set(
            df["age_group"].dropna().unique()
        )
        report["unexpected_age_groups"] = sorted(
            found_age_groups - EXPECTED_AGE_GROUPS
        )
    else:
        report["unexpected_age_groups"] = [
            "Missing age_group column"
        ]

    if "category" in df.columns:
        found_categories = set(
            df["category"].dropna().unique()
        )
        report["unexpected_categories"] = sorted(
            found_categories - EXPECTED_CATEGORIES
        )
    else:
        report["unexpected_categories"] = [
            "Missing category column"
        ]

    report["required_columns_missing"] = [
        column
        for column in [
            "patient_id",
            "age",
            "age_group",
            "category",
            "reported_symptoms",
            "synthetic_treatment_category",
        ]
        if column not in df.columns
    ]

    return report


def print_quality_report(report):
    print("\n=== Synthetic Healthcare Data Quality Report ===")

    print(
        f"Record count: {report['record_count']} "
        f"(expected {report['expected_record_count']})"
    )
    print("Record count correct:", report["record_count_ok"])

    print("\nMissing values:")
    if report["missing_values"]:
        for column, count in report["missing_values"].items():
            print(f"  {column}: {count}")
    else:
        print("  None found")

    print("\nDuplicate rows:", report["duplicate_rows"])
    print(
        "Duplicate patient IDs:",
        report["duplicate_patient_ids"]
    )

    print("\nUnexpected age groups:")
    print(report["unexpected_age_groups"] or "None found")

    print("\nUnexpected categories:")
    print(report["unexpected_categories"] or "None found")

    print("\nMissing required columns:")
    print(report["required_columns_missing"] or "None")

    print(
        "\nNote: This report checks data structure and consistency; "
        "it does not validate medical accuracy."
    )


if __name__ == "__main__":
    data = load_data()
    quality_report = check_data_quality(data)
    print_quality_report(quality_report)
