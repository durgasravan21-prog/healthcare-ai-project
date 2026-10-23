
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/preprocessed_healthcare_records.csv")
EVALUATION_PATH = Path("evaluation_report.txt")
RESPONSIBLE_AI_PATH = Path("responsible_ai_report.txt")
COMPARISON_PATH = Path("model_comparison_results.csv")
OUTPUT_PATH = Path("final_project_report.txt")


def read_text(path):
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "Not available. Run the corresponding project module first."


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    lines = [
        "HEALTHCARE RECORDS ANALYSIS — FINAL PROJECT REPORT",
        "=" * 55,
        "",
        "1. PROJECT OVERVIEW",
        "This project analyzes synthetic healthcare records to explore",
        "patient demographics, reported symptoms, and visit patterns.",
        "It is an educational demonstration, not a clinical tool.",
        "",
        "2. DATASET SUMMARY",
        f"Total records: {len(df)}",
        f"Total columns: {len(df.columns)}",
        f"Missing cells: {int(df.isna().sum().sum())}",
        f"Duplicate rows: {int(df.duplicated().sum())}",
        "",
        "Condition-label distribution:",
        df["synthetic_condition_label"].value_counts().to_string(),
        "",
        "3. MODEL COMPARISON",
        read_text(COMPARISON_PATH).strip()
        if False else (
            pd.read_csv(COMPARISON_PATH).to_string(index=False)
            if COMPARISON_PATH.exists()
            else "Model comparison results are not available yet."
        ),
        "",
        "4. MODEL EVALUATION",
        read_text(EVALUATION_PATH),
        "",
        "5. RESPONSIBLE AI AND LIMITATIONS",
        read_text(RESPONSIBLE_AI_PATH),
        "",
        "6. CONCLUSION",
        "The project demonstrates data processing, visualization,",
        "search, classification, evaluation, and responsible-AI reporting.",
        "All findings are limited to the synthetic dataset and should not",
        "be interpreted as evidence of real-world medical performance.",
    ]

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Final report saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
