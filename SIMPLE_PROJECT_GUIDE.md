# Simple project guide and faculty explanation

## One-minute explanation
"Our project analyzes 500 synthetic healthcare records to understand patient demographics and healthcare patterns. We use Python and Pandas to load and organize the CSV dataset. The Streamlit dashboard displays summary values and charts for age groups, healthcare categories, and illustrative treatment labels. Users can filter records and search by patient ID or symptom. This is an educational data-analysis tool, not a medical diagnosis or treatment system."

## Workflow
1. Dataset: a CSV file stores 500 synthetic records.
2. Loading: Pandas reads the CSV into a DataFrame.
3. Preparation: required columns are checked and age is converted to a numeric value.
4. Filtering: sidebar controls filter by age group, category, and region.
5. Analysis: Pandas calculates counts, unique values, and average age.
6. Visualization: Streamlit displays bar charts and tables.
7. Search: text matching finds records containing the entered ID, symptom, category, or label.

## Functions to explain
- load_data(): reads the CSV, checks required columns, and prepares age values.
- value_counts(): counts occurrences of each distinct value.
- mean(): calculates average age.
- nunique(): counts distinct values.
- st.selectbox(): creates a filter control.
- st.bar_chart(): displays a bar chart.
- st.dataframe(): displays records in a table.

## Faculty demonstration
1. Run `streamlit run simple_dashboard.py`.
2. Explain the four summary metrics.
3. Select Adult in the age-group filter and show the charts updating.
4. Select a healthcare category and region.
5. Search for `headache` and show matching synthetic records.
6. Expand "View filtered dataset".

## Viva questions
**1. What is the objective?**  
To explore demographic and healthcare-category patterns in synthetic records using charts and filters.

**2. Why Pandas?**  
It makes reading CSV files, filtering rows, calculating averages, and counting categories straightforward.

**3. What is a DataFrame?**  
A two-dimensional table of rows and columns provided by Pandas.

**4. What does value_counts() do?**  
It counts how often each distinct value occurs in a column.

**5. What is data visualization?**  
Presenting data with charts so patterns are easier to understand.

**6. Why Streamlit?**  
It creates an interactive Python dashboard without requiring separate frontend code.

**7. Is this real patient data?**  
No. It is a synthetic educational dataset of 500 generated records.

**8. Does it diagnose or recommend treatment?**  
No. It only describes patterns in the synthetic dataset.

**9. What happens when a filter changes?**  
The selected conditions filter the rows, and the metrics and charts are recalculated from those rows.

**10. What are the limitations?**  
Synthetic records do not necessarily represent real patient populations or clinical practice.

## Be accurate during presentation
- Do not call synthetic labels verified diagnoses.
- Do not call illustrative treatment labels actual treatments received.
- Do not claim the dashboard predicts disease or provides medical advice.
- This simplified dashboard does not train or use a machine-learning model.
