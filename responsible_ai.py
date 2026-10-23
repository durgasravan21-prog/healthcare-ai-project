
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "data" / "preprocessed_healthcare_records.csv"
REPORT_FILE = PROJECT_DIR / "responsible_ai_report.txt"


def check_dataset(df):
    findings = []

    findings.append(
        f"Dataset size: {len(df)} records and {len(df.columns)} columns."
    )

    if "sex" in df.columns:
        findings.append("\nRecords by sex:")
        findings.append(df["sex"].value_counts(dropna=False).to_string())

    if "age_group" in df.columns:
        findings.append("\nRecords by age group:")
        findings.append(
            df["age_group"].value_counts(dropna=False).to_string()
        )

    if "region" in df.columns:
        findings.append("\nRecords by region:")
        findings.append(df["region"].value_counts(dropna=False).to_string())

    findings.append("\nMissing values by column:")
    findings.append(df.isna().sum().to_string())

    if "patient_id" in df.columns:
        findings.append(
            "\nDuplicate patient IDs: "
            f"{df['patient_id'].duplicated().sum()}"
        )

    return findings


def generate_responsible_ai_report(df):
    report = [
        "RESPONSIBLE AI AND DATA LIMITATIONS REPORT",
        "=" * 50,
        "",
        "1. DATASET TRANSPARENCY",
        "This project uses synthetic healthcare records for educational analysis.",
        "The records do not represent verified real-world patients.",
        "",
        "2. PRIVACY",
        "Patient identifiers are synthetic identifiers, not verified identities.",
        "If real data is introduced later, access controls, data minimization,",
        "and appropriate authorization must be implemented.",
        "",
        "3. BIAS AND REPRESENTATION",
        "Review demographic distributions and missing values before interpreting",
        "results. Distribution checks alone cannot establish that a dataset is fair.",
        "",
        "4. MODEL TRANSPARENCY",
        "The model uses logistic regression with preprocessing pipelines.",
        "Its predictions reflect patterns in the synthetic training data.",
        "Predictions must not be treated as medical diagnoses.",
        "",
        "5. EVALUATION LIMITATIONS",
        "A held-out test split estimates performance on this dataset only.",
        "It does not establish performance on real patients or other populations.",
        "",
        "6. SAFE USE",
        "Do not use this project for diagnosis, treatment selection, or clinical",
        "decision-making. Seek qualified healthcare professionals for medical care.",
        "",
        "7. DATASET CHECKS",
        "",
    ]

    report.extend(check_dataset(df))

    report.extend([
        "",
        "8. CONCLUSION",
        "Responsible use requires transparency, careful interpretation, privacy",
        "protection, and further validation before any real-world application.",
    ])

    return "\n".join(report)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run preprocessing.py first."
        )

    df = pd.read_csv(DATA_FILE)
    report = generate_responsible_ai_report(df)

    REPORT_FILE.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
