from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Find the CSV from the project folder, not from the terminal's current folder.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "synthetic_healthcare_records_500.csv"

st.set_page_config(
    page_title="Healthcare Records Dashboard",
    page_icon="🏥",
    layout="wide",
)

st.title("Healthcare Records: Demographic and Treatment Patterns")
st.caption(
    "Educational analysis of 500 synthetic records. "
    "This dashboard does not provide medical diagnosis or treatment advice."
)


@st.cache_data
def load_data():
    """Read and prepare the synthetic healthcare CSV."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "patient_id", "age", "age_group", "category",
        "synthetic_treatment_category", "region", "visit_type",
        "reported_severity",
    ]
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing_columns)}")

    data["age"] = pd.to_numeric(data["age"], errors="coerce")
    data = data.dropna(subset=["age", "age_group", "category"])
    return data


try:
    df = load_data()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filter records")
age_groups = ["All"] + sorted(df["age_group"].dropna().unique().tolist())
selected_age = st.sidebar.selectbox("Age group", age_groups)

categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Healthcare category", categories)

regions = ["All"] + sorted(df["region"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Region", regions)

filtered = df.copy()
if selected_age != "All":
    filtered = filtered[filtered["age_group"] == selected_age]
if selected_category != "All":
    filtered = filtered[filtered["category"] == selected_category]
if selected_region != "All":
    filtered = filtered[filtered["region"] == selected_region]

# Summary metrics
st.subheader("Dataset overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total records", f"{len(filtered):,}")
col2.metric("Average age", f"{filtered['age'].mean():.1f}" if not filtered.empty else "N/A")
col3.metric("Healthcare categories", filtered["category"].nunique())
col4.metric("Age groups", filtered["age_group"].nunique())

if filtered.empty:
    st.warning("No records match these filters. Change the filters in the sidebar.")
    st.stop()

# Three simple charts that directly support the project title.
left, right = st.columns(2)

with left:
    st.subheader("Patient age-group distribution")
    age_counts = filtered["age_group"].value_counts().reindex(
        ["Child", "Adult", "Senior"], fill_value=0
    )
    st.bar_chart(age_counts)
    st.caption("Shows how records are distributed across age groups.")

with right:
    st.subheader("Healthcare category distribution")
    category_counts = filtered["category"].value_counts().sort_values(ascending=False)
    st.bar_chart(category_counts)
    st.caption("Shows the most frequently represented healthcare categories.")

st.subheader("Illustrative treatment-label distribution")
treatment_counts = (
    filtered["synthetic_treatment_category"]
    .value_counts()
    .sort_values(ascending=False)
)
st.bar_chart(treatment_counts)
st.caption(
    "These are synthetic educational labels, not verified treatments received "
    "by real patients."
)

# A small, explainable record search
st.subheader("Search patient records")
search_text = st.text_input(
    "Search by patient ID, symptom, category or synthetic label",
    placeholder="Example: SYN0001 or headache",
)
if search_text.strip():
    text_columns = [
        "patient_id", "reported_symptoms", "category",
        "synthetic_condition_label", "synthetic_treatment_category",
    ]
    available_columns = [col for col in text_columns if col in filtered.columns]
    matches = pd.Series(False, index=filtered.index)
    for column in available_columns:
        matches = matches | filtered[column].astype(str).str.contains(
            search_text.strip(), case=False, na=False
        )
    results = filtered[matches]
    st.write(f"Matching records: {len(results)}")
    st.dataframe(results, use_container_width=True, hide_index=True)

with st.expander("View filtered dataset"):
    st.dataframe(filtered, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Dataset: synthetic_healthcare_records_500.csv | "
    "Purpose: demographic and descriptive pattern analysis only."
)
