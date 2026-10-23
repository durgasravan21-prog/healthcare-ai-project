
# Healthcare Records Analysis

## Project Title
Visualization and Analysis of Healthcare Records for Understanding Patient Demographic and Treatment Patterns

## Overview
This project analyzes a synthetic healthcare dataset containing 500 records. It explores demographic distributions, healthcare categories, and illustrative treatment categories using Python.

## Objectives
- Search records by reported symptoms.
- Demonstrate BFS, DFS, and A* search algorithms.
- Build a knowledge base from dataset summaries.
- Generate rule-based observations.
- Check dataset quality.
- Visualize age-group and category distributions.
- Analyze relationships between age groups, categories, and synthetic treatment labels.

## Technologies Used
- Python
- Pandas
- Matplotlib
- CSV dataset
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- A* Search

## Project Structure
- `main.py` — main application menu
- `search.py` — record search and search algorithms
- `src/knowledge_base.py` — dataset summaries
- `src/reasoning.py` — rule-based observations
- `src/visualization.py` — charts
- `src/local_data.py` — dataset inspection utility
- `analysis.py` — cross-tabulation and heatmap
- `data_quality.py` — dataset quality checks
- `report.py` — project summary report
- `data/` — synthetic healthcare CSV dataset
- `requirements.txt` — Python dependencies

## Setup
1. Install Python 3.11 or a compatible version.
2. Create and activate a virtual environment.
3. Install dependencies:

   `python -m pip install -r requirements.txt`

4. Run the application:

   `python main.py`

## Dataset
The project uses 500 synthetic records. The dataset is intended for educational demonstrations and does not contain real patient records.

## Limitations
- The dataset is synthetic.
- Treatment labels are illustrative, not verified treatments received by patients.
- Search-path costs are illustrative.
- Results describe only the generated dataset.
- The project is not intended for clinical diagnosis or treatment decisions.

## Future Improvements
- Add an interactive dashboard.
- Improve data validation and error handling.
- Add more filters and visualizations.
- Integrate all modules into a unified interface.

# Healthcare Records Visualization and Analysis

## Project Overview
This project analyzes a synthetic healthcare dataset containing 500 records to explore patient demographics, healthcare categories, and illustrative treatment patterns.

It combines data analysis, visualization, search algorithms, and rule-based reasoning in a single Python application.

## Objectives
- Analyze patient age-group distributions.
- Explore healthcare category and illustrative treatment-label distributions.
- Visualize demographic patterns using charts and a heatmap.
- Search and filter synthetic records.
- Demonstrate BFS, DFS, and A* search algorithms.
- Check dataset structure and consistency.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Visual Studio Code

## Project Structure
```text
ai project/
├── data/
│   └── synthetic_healthcare_records_500.csv
├── src/
│   ├── knowledge_base.py
│   ├── reasoning.py
│   └── visualization.py
├── main.py
├── search.py
├── analysis.py
├── data_quality.py
├── report.py
├── requirements.txt
└── README.md
```

## Installation
Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

## Running the Project
Run the main application from the project root:

```bash
python main.py
```

Run individual analysis modules when needed:

```bash
python analysis.py
python data_quality.py
python report.py
```

## Dataset
The project uses a synthetic dataset of 500 records. It is intended for educational analysis and does not represent verified real-world patient data.

## Limitations
- The dataset is synthetic.
- Treatment labels are illustrative, not clinical recommendations.
- Search algorithms demonstrate pathfinding on an illustrative workflow graph.
- Findings must not be used for diagnosis or medical decision-making.

## Conclusion
The project demonstrates how Python-based data processing, visualization, search algorithms, and rule-based reasoning can be combined to explore healthcare-related data patterns for educational purposes.
