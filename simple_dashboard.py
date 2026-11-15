from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "synthetic_healthcare_records_500.csv"

st.set_page_config(page_title="Healthcare Records Dashboard", page_icon="🏥", layout="wide")
st.title("Healthcare Records: Demographic and Treatment Patterns")
st.caption("Educational analysis of synthetic records only. Not for diagnosis or treatment advice.")

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    required = ["patient_id", "age", "age_group", "category", "synthetic_treatment_category",
                "region", "reported_symptoms", "synthetic_condition_label"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    data["age"] = pd.to_numeric(data["age"], errors="coerce")
    return data.dropna(subset=["age", "age_group", "category"])

try:
    df = load_data()
except Exception as error:
    st.error(f"Unable to load dataset: {error}")
    st.stop()

st.sidebar.header("Filter records")
age_options = ["All"] + sorted(df["age_group"].unique().tolist())
category_options = ["All"] + sorted(df["category"].unique().tolist())
region_options = ["All"] + sorted(df["region"].unique().tolist())
age_choice = st.sidebar.selectbox("Age group", age_options)
category_choice = st.sidebar.selectbox("Healthcare category", category_options)
region_choice = st.sidebar.selectbox("Region", region_options)

filtered = df.copy()
if age_choice != "All":
    filtered = filtered[filtered["age_group"] == age_choice]
if category_choice != "All":
    filtered = filtered[filtered["category"] == category_choice]
if region_choice != "All":
    filtered = filtered[filtered["region"] == region_choice]

st.subheader("Dataset overview")
a, b, c, d = st.columns(4)
a.metric("Records", len(filtered))
b.metric("Average age", f"{filtered['age'].mean():.1f}" if not filtered.empty else "N/A")
c.metric("Categories", filtered["category"].nunique())
d.metric("Age groups", filtered["age_group"].nunique())

if filtered.empty:
    st.warning("No records match these filters. Change the filters in the sidebar.")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Patient age-group distribution")
    st.bar_chart(filtered["age_group"].value_counts().reindex(["Child", "Adult", "Senior"], fill_value=0))
with right:
    st.subheader("Healthcare category distribution")
    st.bar_chart(filtered["category"].value_counts())

st.subheader("Illustrative treatment-label distribution")
st.bar_chart(filtered["synthetic_treatment_category"].value_counts())
st.caption("Treatment labels are synthetic educational examples, not verified treatment records.")

st.subheader("Search patient records")
query = st.text_input("Search patient ID, symptom, category or label", placeholder="Example: SYN0001 or headache")
if query.strip():
    searchable = ["patient_id", "reported_symptoms", "category", "synthetic_condition_label",
                  "synthetic_treatment_category"]
    found = pd.Series(False, index=filtered.index)
    for column in searchable:
        found |= filtered[column].astype(str).str.contains(query.strip(), case=False, na=False)
    results = filtered[found]
    st.write(f"Matching records: {len(results)}")
    st.dataframe(results, use_container_width=True, hide_index=True)

with st.expander("View filtered dataset"):
    st.dataframe(filtered, use_container_width=True, hide_index=True)

st.divider()
st.caption("500-record synthetic dataset | Descriptive analysis only")
