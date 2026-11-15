"""Basic explainability, fairness and data-quality diagnostics."""
import pandas as pd


def data_quality_summary(df):
    return {
        "rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "columns": len(df.columns),
        "duplicate_patient_ids": int(df["patient_id"].duplicated().sum())
        if "patient_id" in df else None,
    }


def group_distribution(df, group_column, target_column="category"):
    if group_column not in df or target_column not in df:
        return pd.DataFrame()
    return pd.crosstab(df[group_column].fillna("Missing"),
                       df[target_column].fillna("Missing"))


def simple_explanation(row):
    fields = ["age_group", "category", "visit_type", "reported_severity"]
    return [f"{field}: {row.get(field, 'Not available')}"
            for field in fields if field in row]


def responsible_ai_checklist():
    return [
        ("Privacy", "Use synthetic records; never upload identifiable real patient data."),
        ("Fairness", "Compare group distributions; differences alone do not prove discrimination."),
        ("Transparency", "Show the fields, rules and methods behind each output."),
        ("Safety", "Do not use this educational model for diagnosis, triage or treatment."),
        ("Accountability", "Humans remain responsible for real-world healthcare decisions."),
        ("Security", "Restrict data access and do not store credentials or personal identifiers."),
    ]
