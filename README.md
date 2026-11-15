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
├── tests/
│   └── test_ai_modules.py
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
Python, Pandas, NumPy, scikit-learn, Streamlit and Requests. Optional: Ollama with a locally downloaded language model.

## Dataset and safety
Optional local LLM: install Ollama separately, download a model such as llama3.2, start the Ollama service, and use the Unit V button. Without Ollama, the local retrieval and prompt-design demonstrations still work.\n\nThe CSV contains synthetic educational records, not real patient records. Condition and treatment columns are illustrative labels. The project is not clinically validated and must not be used for diagnosis, triage or treatment decisions.

## Documentation
- SYLLABUS_MAPPING.md — coverage by unit
- FACULTY_GUIDE.md — demonstration sequence, algorithms and viva questions


## Run the built-in tests
From the project root, after installing the requirements:

~~~bash
python -m unittest discover -s tests
~~~

## Optional local LLM
To try the Unit V local LLM button:
1. Install Ollama separately from its official website.
2. Start the Ollama service.
3. Download a supported model, for example: ollama pull llama3.2
4. Keep Ollama running and enter the same model name in the dashboard.

Without Ollama, all search, reasoning, machine-learning, retrieval and dashboard demonstrations remain available.
