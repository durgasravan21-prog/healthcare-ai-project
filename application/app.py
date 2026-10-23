
from pathlib import Path
from collections import deque
import heapq
import subprocess
import sys

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Project root: C:\Users\durga\Desktop\ai project
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "preprocessed_healthcare_records.csv"
ORIGINAL_DATA = BASE_DIR / "data" / "synthetic_healthcare_records_500.csv"
MODEL_PATH = BASE_DIR / "models" / "condition_model.joblib"

st.set_page_config(
    page_title="Healthcare Records AI",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Healthcare Records AI")
st.caption("AI Foundations Project • Synthetic Healthcare Data Analysis")

# ---------- Helpers ----------

def load_data():
    path = DATA_PATH if DATA_PATH.exists() else ORIGINAL_DATA
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read dataset: {exc}")
        return None


def run_script(filename):
    """Run an existing project script without modifying it."""
    script_path = BASE_DIR / filename

    if not script_path.exists():
        st.error(f"Script not found: {script_path}")
        return

    with st.spinner(f"Running {filename}..."):
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.stdout:
                st.code(result.stdout)
            if result.stderr:
                st.text_area(
                    "Messages / errors",
                    result.stderr,
                    height=180,
                )
            if result.returncode == 0:
                st.success(f"{filename} completed.")
            else:
                st.error(
                    f"{filename} exited with code {result.returncode}."
                )
        except subprocess.TimeoutExpired:
            st.error("The operation took too long and was stopped.")
        except Exception as exc:
            st.error(f"Could not run script: {exc}")


def show_report(filename):
    path = BASE_DIR / filename
    if path.exists():
        st.text(path.read_text(encoding="utf-8", errors="replace"))
        st.download_button(
            f"Download {filename}",
            data=path.read_bytes(),
            file_name=filename,
        )
    else:
        st.info(
            f"{filename} has not been created yet. "
            "Run the corresponding report option first."
        )


def graph_search(graph, start, goal, method):
    """BFS, DFS, or A* on the project's illustrative workflow graph."""
    if start not in graph or goal not in graph:
        return None

    if method == "BFS":
        queue = deque([(start, [start])])
        visited = set()

        while queue:
            node, path = queue.popleft()
            if node == goal:
                return path
            if node in visited:
                continue
            visited.add(node)
            for neighbour in graph.get(node, []):
                if neighbour not in visited:
                    queue.append((neighbour, path + [neighbour]))
        return None

    if method == "DFS":
        stack = [(start, [start])]
        visited = set()

        while stack:
            node, path = stack.pop()
            if node == goal:
                return path
            if node in visited:
                continue
            visited.add(node)
            for neighbour in reversed(graph.get(node, [])):
                if neighbour not in visited:
                    stack.append((neighbour, path + [neighbour]))
        return None

    # A*: unit edge costs and shortest remaining distance as heuristic.
    reverse_graph = {node: [] for node in graph}
    for node, neighbours in graph.items():
        for neighbour in neighbours:
            reverse_graph.setdefault(neighbour, []).append(node)

    distance = {goal: 0}
    queue = deque([goal])
    while queue:
        node = queue.popleft()
        for neighbour in reverse_graph.get(node, []):
            if neighbour not in distance:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)

    if start not in distance:
        return None

    frontier = [(distance[start], 0, start, [start])]
    best_cost = {start: 0}

    while frontier:
        _, cost, node, path = heapq.heappop(frontier)
        if node == goal:
            return path
        if cost > best_cost.get(node, float("inf")):
            continue

        for neighbour in graph.get(node, []):
            new_cost = cost + 1
            if new_cost < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = new_cost
                heapq.heappush(
                    frontier,
                    (
                        new_cost + distance.get(neighbour, 10**6),
                        new_cost,
                        neighbour,
                        path + [neighbour],
                    ),
                )
    return None


df = load_data()

# ---------- Sidebar navigation ----------

menu = [
    "1. Dataset Summary",
    "2. Knowledge-Base Reasoning",
    "3. Search and Filter Records",
    "4. Display Charts",
    "5. Data-Quality Checks",
    "6. Age-Category Heatmap",
    "7. Generate Project Report",
    "8. Preprocess Dataset",
    "9. Train Machine-Learning Model",
    "10. Evaluate Model",
    "11. Responsible AI Report",
    "12. BFS, DFS and A*",
    "13. Feature Importance",
    "14. Confusion Matrix",
    "15. Model Comparison",
    "16. Model Comparison Chart",
    "17. Final Project Report",
]

st.sidebar.title("Project Menu")
page = st.sidebar.radio("Select a function", menu)
st.sidebar.divider()
st.sidebar.caption("All paths are resolved from your project root.")

if df is None:
    st.error(
        "Dataset not found. Expected one of these files:\n\n"
        f"- {DATA_PATH}\n- {ORIGINAL_DATA}"
    )
    st.info(
        "Check that the CSV files are in your existing project's data folder."
    )
    st.stop()

target = "synthetic_condition_label"

# ---------- 1. Dataset summary ----------

if page.startswith("1."):
    st.header("Dataset Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    c4.metric(
        "Condition Labels",
        int(df[target].nunique()) if target in df.columns else "N/A",
    )

    if "age" in df.columns:
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Age", f"{df['age'].mean():.1f}")
        c2.metric("Minimum Age", f"{df['age'].min():.0f}")
        c3.metric("Maximum Age", f"{df['age'].max():.0f}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Column information"):
        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [str(x) for x in df.dtypes],
            "Missing Values": [int(df[c].isna().sum()) for c in df.columns],
            "Unique Values": [int(df[c].nunique()) for c in df.columns],
        })
        st.dataframe(info, use_container_width=True)

# ---------- 2. Knowledge-base reasoning ----------

elif page.startswith("2."):
    st.header("Knowledge-Base Reasoning")
    st.write("Summary observations generated from the available dataset.")

    if "age" in df.columns:
        st.write(f"- Average recorded age: **{df['age'].mean():.1f}**")
    if target in df.columns:
        counts = df[target].value_counts()
        st.write(f"- Unique synthetic labels: **{counts.size}**")
        st.write(f"- Most frequent label: **{counts.index[0]} ({counts.iloc[0]} records)**")
        st.bar_chart(counts.head(10))

    for col in ["sex", "region", "visit_type", "reported_severity"]:
        if col in df.columns:
            with st.expander(f"Distribution: {col}"):
                st.dataframe(
                    df[col].value_counts(dropna=False).rename("Count"),
                    use_container_width=True,
                )

    st.caption(
        "These observations describe synthetic records, not clinical facts."
    )

# ---------- 3. Search and filter records ----------

elif page.startswith("3."):
    st.header("Search and Filter Records")
    filtered = df.copy()

    search_text = st.text_input(
        "Search across text fields",
        placeholder="Enter a symptom, region, or label",
    )
    if search_text:
        text_cols = filtered.select_dtypes(include=["object", "string"]).columns
        mask = pd.Series(False, index=filtered.index)
        for col in text_cols:
            mask |= filtered[col].astype(str).str.contains(
                search_text, case=False, na=False
            )
        filtered = filtered[mask]

    if "age" in filtered.columns:
        low = int(df["age"].min())
        high = int(df["age"].max())
        if low < high:
            age_range = st.slider(
                "Age range",
                min_value=low,
                max_value=high,
                value=(low, high),
            )
            filtered = filtered[
                filtered["age"].between(age_range[0], age_range[1])
            ]

    for col in [target, "visit_type", "region", "sex", "reported_severity"]:
        if col in filtered.columns:
            options = sorted(filtered[col].dropna().astype(str).unique())
            selected = st.selectbox(
                f"Filter by {col}",
                ["All"] + options,
                key=f"filter_{col}",
            )
            if selected != "All":
                filtered = filtered[filtered[col].astype(str) == selected]

    st.write(f"**Matching records: {len(filtered)}**")
    st.dataframe(filtered, use_container_width=True, height=400)
    st.download_button(
        "Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_records.csv",
        mime="text/csv",
    )

# ---------- 4. Display charts ----------

elif page.startswith("4."):
    st.header("Dataset Charts")

    for col, title in [
        ("age_group", "Age Group Distribution"),
        (target, "Synthetic Condition Distribution"),
        ("visit_type", "Visit Type Distribution"),
        ("reported_severity", "Reported Severity Distribution"),
        ("region", "Region Distribution"),
    ]:
        if col in df.columns:
            if col == "age_group":
                series = df[col].value_counts().sort_index()
            else:
                series = df[col].value_counts()
            st.subheader(title)
            st.bar_chart(series)

    if "age" in df.columns and target in df.columns:
        st.subheader("Average Age by Synthetic Condition Label")
        age_summary = df.groupby(target, observed=False)["age"].mean()
        st.bar_chart(age_summary)

# ---------- 5. Data quality ----------

elif page.startswith("5."):
    st.header("Data-Quality Checks")
    quality = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(x) for x in df.dtypes],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Unique Values": [int(df[c].nunique()) for c in df.columns],
    })
    st.metric("Duplicate Rows", int(df.duplicated().sum()))
    st.metric("Missing Cells", int(df.isna().sum().sum()))
    st.dataframe(quality, use_container_width=True)

    if st.button("Run existing data_quality.py"):
        run_script("data_quality.py")

# ---------- 6. Heatmap ----------

elif page.startswith("6."):
    st.header("Age-Category Heatmap")

    if "age" not in df.columns or target not in df.columns:
        st.warning("The dataset needs age and synthetic condition label columns.")
    else:
        plot_df = df.copy()
        if "age_group" not in plot_df.columns:
            plot_df["age_group"] = pd.cut(
                plot_df["age"],
                bins=[0, 18, 35, 50, 65, 120],
                labels=["0–18", "19–35", "36–50", "51–65", "66+"],
                include_lowest=True,
            )

        pivot = pd.crosstab(plot_df["age_group"], plot_df[target])
        fig, ax = plt.subplots(figsize=(14, 5))
        image = ax.imshow(pivot.values, aspect="auto")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(str))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90, fontsize=8)
        ax.set_xlabel("Synthetic Condition Label")
        ax.set_ylabel("Age Group")
        ax.set_title("Age Group vs Synthetic Condition Label")
        fig.colorbar(image, ax=ax, label="Record Count")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        if st.button("Run existing analysis.py"):
            run_script("analysis.py")

# ---------- 7. Project report ----------

elif page.startswith("7."):
    st.header("Generate Project Report")
    st.write("Runs the project's existing report generator.")
    if st.button("Generate report", type="primary"):
        run_script("report.py")
    st.divider()
    show_report("project_report.txt")

# ---------- 8. Preprocess ----------

elif page.startswith("8."):
    st.header("Preprocess Dataset")
    st.warning(
        "This runs your existing preprocessing script, which may regenerate "
        "the preprocessed CSV used by other project functions."
    )
    if st.button("Run preprocessing.py", type="primary"):
        run_script("preprocessing.py")

# ---------- 9. Train model ----------

elif page.startswith("9."):
    st.header("Train Machine-Learning Model")
    st.warning(
        "Training can replace the saved model file. This model uses synthetic "
        "labels and is not a medical diagnostic system."
    )
    if st.button("Train model", type="primary"):
        run_script("model.py")

    if MODEL_PATH.exists():
        st.success(f"Saved model found: {MODEL_PATH.relative_to(BASE_DIR)}")

# ---------- 10. Evaluation ----------

elif page.startswith("10."):
    st.header("Evaluate Machine-Learning Model")
    if st.button("Run evaluation.py", type="primary"):
        run_script("evaluation.py")
    st.divider()
    show_report("evaluation_report.txt")

# ---------- 11. Responsible AI ----------

elif page.startswith("11."):
    st.header("Responsible AI")
    if st.button("Generate Responsible AI report", type="primary"):
        run_script("responsible_ai.py")
    st.divider()
    show_report("responsible_ai_report.txt")

# ---------- 12. BFS, DFS, A* ----------

elif page.startswith("12."):
    st.header("Search Algorithms Demonstration")
    st.write("Explore an illustrative healthcare workflow graph.")

    graph = {
        "Assessment": ["Treatment_A", "Treatment_B"],
        "Treatment_A": ["Followup_A", "Followup_B"],
        "Treatment_B": ["Followup_B"],
        "Followup_A": ["Outcome"],
        "Followup_B": ["Outcome"],
        "Outcome": [],
    }

    start = st.selectbox("Start node", list(graph.keys()), index=0)
    goal = st.selectbox(
        "Goal node",
        list(graph.keys()),
        index=list(graph.keys()).index("Outcome"),
    )

    if st.button("Run BFS, DFS and A*", type="primary"):
        for algorithm in ["BFS", "DFS", "A*"]:
            path = graph_search(graph, start, goal, algorithm)
            st.subheader(algorithm)
            if path:
                st.code(" → ".join(path))
            else:
                st.write("No path found.")

    with st.expander("View workflow graph"):
        for node, neighbours in graph.items():
            st.write(f"**{node}** → {', '.join(neighbours) if neighbours else 'None'}")

    st.caption("This graph demonstrates search algorithms, not medical decisions.")

# ---------- 13. Feature importance ----------

elif page.startswith("13."):
    st.header("Model Feature Importance")

    if not MODEL_PATH.exists():
        st.warning("Train the model first using option 9.")
    else:
        try:
            import joblib

            model = joblib.load(MODEL_PATH)
            prep = model.named_steps["preprocessing"]
            clf = model.named_steps["classifier"]
            names = prep.get_feature_names_out()
            importance = abs(clf.coef_).mean(axis=0)

            result = pd.DataFrame({
                "Feature": names,
                "Mean Absolute Coefficient": importance,
            }).nlargest(15, "Mean Absolute Coefficient")

            st.dataframe(result, use_container_width=True)
            chart = result.set_index("Feature")["Mean Absolute Coefficient"]
            st.bar_chart(chart.sort_values())
            st.caption(
                "Coefficient magnitude is a model-specific association, "
                "not medical importance or causation."
            )
            st.download_button(
                "Download feature importance CSV",
                result.to_csv(index=False).encode("utf-8"),
                file_name="feature_importance.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Could not inspect the model: {exc}")

# ---------- 14. Confusion matrix ----------

elif page.startswith("14."):
    st.header("Confusion Matrix")

    if not MODEL_PATH.exists():
        st.warning("Train the model first using option 9.")
    else:
        try:
            import joblib
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            from sklearn.model_selection import train_test_split

            if target not in df.columns:
                st.error(f"Target column not found: {target}")
            else:
                features = [
                    "age", "sex", "region", "visit_type",
                    "reported_symptoms", "symptom_duration",
                    "reported_severity",
                ]
                missing = [c for c in features if c not in df.columns]
                if missing:
                    st.error(f"Required columns missing: {missing}")
                else:
                    X = df[features]
                    y = df[target]
                    _, X_test, _, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    model = joblib.load(MODEL_PATH)
                    predictions = model.predict(X_test)
                    labels = model.named_steps["classifier"].classes_
                    matrix = confusion_matrix(
                        y_test, predictions, labels=labels
                    )
                    fig, ax = plt.subplots(figsize=(14, 12))
                    display = ConfusionMatrixDisplay(
                        confusion_matrix=matrix,
                        display_labels=labels,
                    )
                    display.plot(
                        ax=ax, xticks_rotation=90,
                        colorbar=False, values_format="d",
                    )
                    ax.set_title("Confusion Matrix — Synthetic Data")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
        except Exception as exc:
            st.error(f"Could not create confusion matrix: {exc}")

# ---------- 15. Model comparison ----------

elif page.startswith("15."):
    st.header("Model Comparison")
    results_path = BASE_DIR / "model_comparison_results.csv"

    if st.button("Run model_comparison.py", type="primary"):
        run_script("model_comparison.py")

    if results_path.exists():
        results = pd.read_csv(results_path)
        st.dataframe(results, use_container_width=True)
        metrics = [
            col for col in ["Accuracy", "Macro F1"]
            if col in results.columns
        ]
        if "Model" in results.columns and metrics:
            st.bar_chart(results.set_index("Model")[metrics])
    else:
        st.info("Run the comparison script to create the results file.")

# ---------- 16. Comparison chart ----------

elif page.startswith("16."):
    st.header("Model Comparison Chart")
    results_path = BASE_DIR / "model_comparison_results.csv"

    if not results_path.exists():
        st.warning("Run option 15 to generate comparison results first.")
    else:
        results = pd.read_csv(results_path)
        metrics = [
            col for col in ["Accuracy", "Macro F1"]
            if col in results.columns
        ]
        if "Model" in results.columns and metrics:
            st.bar_chart(results.set_index("Model")[metrics])
        if st.button("Run existing plot_model_comparison.py"):
            run_script("plot_model_comparison.py")

# ---------- 17. Final project report ----------

elif page.startswith("17."):
    st.header("Final Project Report")
    if st.button("Generate final report", type="primary"):
        run_script("final_project_report.py")
    st.divider()
    show_report("final_project_report.txt")

st.divider()
st.caption(
    "Educational project only. All records and condition labels are synthetic. "
    "Do not use this application for diagnosis or treatment."
)
