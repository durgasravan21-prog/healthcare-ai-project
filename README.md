# CSE276 Healthcare AI Foundations Project

One Streamlit application demonstrates all six CSE276 units using 500 synthetic healthcare-style records.

## Modules
- **Unit I:** AI history, types, AI/ML/DL/Data Science, lifecycle, applications and problem formulation.
- **Unit II:** BFS, DFS, Greedy Best-First, Hill Climbing, A*, knowledge representation, logic, rules and Naive Bayes.
- **Unit III:** PEAS, reflex/model/goal/utility/learning-agent concepts and multi-agent simulation.
- **Unit IV:** Decision Tree, K-Means and modern AI paradigms.
- **Unit V:** Prompt engineering, local TF-IDF retrieval/RAG, optional Ollama LLM, explainability, fairness and responsible AI.
- **Unit VI:** AI ecosystem and future trends.
- **Dashboard:** Dataset filters, metrics, charts and record search.

## Run
Python 3.10+ recommended.

~~~bash
python -m venv .venv
~~~
Windows:
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

## Tests
~~~bash
python -m unittest discover -s tests
~~~

## Optional LLM
Install Ollama separately, start its service, download a model such as llama3.2, then use the Unit V generation button. All other features work without Ollama.

## Structure
~~~text
data/synthetic_healthcare_records_500.csv
ai/search_algorithms.py
ai/knowledge.py
ai/agents.py
ai/ml_models.py
ai/generative.py
ai/responsible_ai.py
tests/test_ai_modules.py
app.py
requirements.txt
SYLLABUS_MAPPING.md
FACULTY_GUIDE.md
~~~

All records and labels are synthetic. Workflow paths are illustrative. Model metrics describe generated labels and have no clinical meaning. Never use this project for diagnosis, triage or treatment decisions.
