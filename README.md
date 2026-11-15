# Healthcare Records Analysis Dashboard

## Project title
**Visualization and Analysis of Healthcare Records for Understanding Patient Demographic and Treatment Patterns**

## What it does
This project explores 500 synthetic healthcare records. It displays:
- Patient age-group distribution
- Healthcare category distribution
- Distribution of illustrative treatment labels
- Summary metrics, filters, and simple record search

It is a descriptive data-analysis dashboard. It does not diagnose diseases, predict conditions, or recommend treatments.

## Workflow
```
CSV dataset -> Pandas loads and prepares data -> User applies filters
            -> Counts and averages are calculated -> Streamlit displays charts and records
```

## Simple project structure
```
healthcare-ai-project/
├── data/
│   └── synthetic_healthcare_records_500.csv
├── simple_dashboard.py
├── requirements.txt
└── README.md
```
Older experimental scripts and outputs remain in the repository for reference. The simplified dashboard does not depend on them.

## Run
Install Python 3.10 or newer, then run from the repository root:

```bash
python -m pip install -r requirements.txt
streamlit run simple_dashboard.py
```

The dashboard opens in your browser. Use the sidebar to filter by age group, healthcare category, and region.

## Technologies
- Python: application logic
- Pandas: CSV reading, filtering, and calculations
- Streamlit: interactive dashboard and charts

## Dataset and limitations
The CSV contains 500 synthetic educational records, not real patient records. Condition and treatment fields are illustrative labels, not verified diagnoses or treatment histories. Results describe only this generated dataset and must not be used for clinical decisions.
