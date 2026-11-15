"""Data quality, explainability and group-distribution checks."""
import pandas as pd

def quality(df):
    return {"rows":len(df),"columns":len(df.columns),
    "missing cells":int(df.isna().sum().sum()),
    "duplicate rows":int(df.duplicated().sum()),
    "duplicate patient IDs":int(df.patient_id.duplicated().sum())}


def group_table(df,group):
    return pd.crosstab(df[group].fillna("Missing"),df.category.fillna("Missing"))


def explain(row):
    return [f"{key}: {row.get(key,'N/A')}"
            for key in ["age_group","category","visit_type","reported_severity"]]


CHECKLIST=[("Privacy","Use synthetic records; never expose identifiable health data."),
("Fairness","Group differences are descriptive, not proof of discrimination."),
("Transparency","Show input fields, rules and model limitations."),
("Safety","Never use this classroom model for diagnosis or treatment."),
("Accountability","Humans remain responsible for real-world decisions."),
("Security","Restrict access and protect stored data.")]
