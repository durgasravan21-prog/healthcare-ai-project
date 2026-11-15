# Healthcare Records — CSE276 AI Foundations Project

**Project title:** Visualization and Analysis of Healthcare Records Using AI Foundations Concepts

An educational project mapped to all six units of LPU CSE276 Artificial Intelligence Foundations, using a 500-row synthetic healthcare-style dataset.

## Modules
- **Unit I:** AI foundations, AI/ML/DL/Data Science comparison, lifecycle and state-space formulation.
- **Unit II:** BFS, DFS, Greedy Best-First, Hill Climbing, A*, semantic networks, frames, production rules, logic examples and Naive Bayes.
- **Unit III:** PEAS, reflex/model-based/goal-based/utility-based/learning agents and cooperative multi-agent simulation.
- **Unit IV:** Decision Tree supervised learning, K-Means unsupervised learning and modern AI paradigms.
- **Unit V:** Prompt engineering, local TF-IDF retrieval, explainability, group comparisons and responsible-AI checks.
- **Unit VI:** Python AI ecosystem, frameworks, model hubs, LLM tools, vector databases and future trends.
- **Dataset dashboard:** demographic/category charts, filters and record search.

## Project structure
~~~text
healthcare-ai-project/
├── data/
│   └── synthetic_healthcare_records_500.csv
├── ai/
│   ├── search_algorithms.py
│   ├── knowledge.py
│   ├── agents.py
│   ├── ml_models.py
│   ├── generative.py
│   └── responsible_ai.py
├── app.py
├── requirements.txt
├── SYLLABUS_MAPPING.md
└── FACULTY_GUIDE.md
~~~

Older experimental scripts and outputs are retained on this working branch for reference. The new integrated dashboard starts from app.py.

## Run locally
Python 3.10 or newer is recommended.

~~~bash
python -m venv .venv
~~~

Windows PowerShell:
~~~powershell
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
~~~

macOS/Linux:
~~~bash
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
~~~

## Technologies
Python, Pandas, NumPy, scikit-learn and Streamlit.

## Dataset and safety
The CSV contains synthetic educational records, not real patient records. Condition and treatment columns are illustrative labels. The project is not clinically validated and must not be used for diagnosis, triage or treatment decisions.

## Documentation
- SYLLABUS_MAPPING.md — coverage by unit
- FACULTY_GUIDE.md — demonstration sequence, algorithms and viva questions
