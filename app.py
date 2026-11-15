from pathlib import Path
import pandas as pd
import streamlit as st

from ai.search_algorithms import GRAPH, run_all
from ai.knowledge import (
    FRAMES, SEMANTIC_NETWORK, PRODUCTION_RULES, apply_rules,
    propositional_examples, predicate_examples, naive_bayes_category,
)
from ai.agents import (
    peas_description, simple_reflex_agent, model_based_agent,
    goal_based_agent, utility_based_agent, learning_agent_demo, multi_agent_demo,
)
from ai.ml_models import supervised_demo, unsupervised_demo
from ai.responsible_ai import (
    data_quality_summary, group_distribution, simple_explanation,
    responsible_ai_checklist,
)
from ai.generative import (
    build_prompt, retrieve_context, rag_answer, prompt_experiments, ollama_generate,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "synthetic_healthcare_records_500.csv"

st.set_page_config(page_title="CSE276 Healthcare AI Foundations", page_icon="🏥", layout="wide")
st.title("Healthcare Records — AI Foundations Laboratory")
st.caption("CSE276 | Six-unit syllabus-aligned project | Synthetic educational data only")


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    required = [
        "patient_id", "age", "age_group", "sex", "region", "visit_type",
        "reported_symptoms", "reported_severity", "category",
        "synthetic_condition_label", "synthetic_treatment_category",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError("Missing dataset columns: " + ", ".join(missing))
    data["age"] = pd.to_numeric(data["age"], errors="coerce")
    return data.dropna(subset=["age", "age_group", "category"])


try:
    df = load_data()
except Exception as error:
    st.error(f"Dataset loading failed: {error}")
    st.stop()

st.sidebar.title("CSE276 Navigation")
page = st.sidebar.radio("Choose a module", [
    "Unit I — AI Foundations",
    "Unit II — Search and Knowledge",
    "Unit III — Intelligent Agents",
    "Unit IV — ML and Modern AI",
    "Unit V — Generative and Responsible AI",
    "Unit VI — AI Ecosystem",
    "Dataset Dashboard",
])
st.sidebar.caption(f"Dataset loaded: {len(df)} records")
st.sidebar.warning("This project is not a medical diagnosis or treatment tool.")

if page.startswith("Unit I"):
    st.header("Unit I — Foundations of AI and Problem Solving")
    st.write("Artificial Intelligence studies computational systems that perceive, reason, learn and act toward goals.")
    one, two, three = st.columns(3)
    one.metric("Synthetic records", len(df))
    two.metric("Dataset fields", len(df.columns))
    three.metric("Healthcare categories", df["category"].nunique())

    st.subheader("AI, ML, DL and Data Science")
    st.table(pd.DataFrame([
        ["Artificial Intelligence", "Broad field of intelligent behaviour", "Search, rules and agents"],
        ["Machine Learning", "Learns patterns from examples", "Decision Tree experiment"],
        ["Deep Learning", "Machine learning with multi-layer neural networks", "Explained conceptually"],
        ["Data Science", "Extracts insights from data", "Pandas summaries and charts"],
    ], columns=["Field", "Meaning", "Project connection"]))

    st.subheader("AI history and evolution")
    st.table(pd.DataFrame([
        ["1950", "Alan Turing proposes the imitation game"],
        ["1956", "Dartmouth workshop helps establish AI as a research field"],
        ["1980s", "Expert systems and symbolic AI become prominent"],
        ["2010s", "Large datasets and deep learning drive major advances"],
        ["2020s", "Foundation models and generative AI become widely used"],
    ], columns=["Period", "Milestone"]))
    st.subheader("Types and applications of AI")
    st.table(pd.DataFrame([
        ["Narrow AI", "Designed for a limited task", "Image classification or search"],
        ["General AI (AGI)", "Hypothetical broad human-level capability", "Not currently implemented"],
        ["Superintelligence", "Hypothetical intelligence beyond human capability", "Future concept"],
        ["Healthcare", "Record analysis and imaging support", "This educational dataset project"],
        ["Agriculture", "Crop monitoring and disease detection", "Example application"],
        ["Finance", "Fraud detection and risk analysis", "Example application"],
        ["Education", "Tutoring and adaptive learning", "Example application"],
        ["Manufacturing / Robotics", "Quality inspection and robot control", "Example application"],
        ["Smart cities", "Traffic and resource optimization", "Example application"],
    ], columns=["Type / Domain", "Description", "Example"]))

    st.subheader("AI lifecycle")
    st.write("Problem definition → Data collection → Preparation → Method selection → Evaluation → Deployment → Monitoring and governance")

    st.subheader("Formal state-space problem formulation")
    st.markdown("""
    - **Initial state:** Start
    - **Actions:** Move to a connected workflow node
    - **Transition model:** Follow a directed graph edge
    - **Goal test:** Check whether the current node is End
    - **Path cost:** One unit per edge
    """)
    st.graphviz_chart("""
    digraph G {
      Start -> Registration; Start -> Emergency;
      Registration -> Review; Emergency -> Review; Emergency -> Priority;
      Review -> Specialist; Review -> Information;
      Priority -> Specialist; Information -> Specialist;
      Specialist -> Followup; Followup -> End;
    }
    """)
    st.subheader("Environment characteristics")
    st.table(pd.DataFrame([
        ["Observability", "Fully observable toy graph"],
        ["Determinism", "Deterministic transitions"],
        ["Episodicity", "Sequential: each action changes the next state"],
        ["Dynamics", "Static during one search run"],
        ["State/action space", "Finite and discrete"],
        ["Agents", "Single-agent search example"],
    ], columns=["Dimension", "Workflow example"]))
    st.info("The graph models a fictional information workflow, not a clinical care pathway.")

elif page.startswith("Unit II"):
    st.header("Unit II — Search, Knowledge Representation and Reasoning")
    st.subheader("Classical search laboratory")
    nodes = list(GRAPH.keys())
    col1, col2 = st.columns(2)
    start = col1.selectbox("Initial node", nodes, index=nodes.index("Start"))
    goal = col2.selectbox("Goal node", nodes, index=nodes.index("End"))
    results = run_all(start, goal)
    rows = []
    for algorithm, result in results.items():
        rows.append({
            "Algorithm": algorithm,
            "Path": " → ".join(result["path"]) if result["path"] else "No path / local optimum",
            "Path steps": result["steps"],
            "Nodes explored": len(result["explored"]),
            "Time (ms)": result["elapsed_ms"],
            "Cost": result["cost"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with st.expander("Algorithm explanations and comparison"):
        st.markdown("""
        - **BFS:** FIFO queue; explores level by level. Complete for finite branching and optimal for equal step costs.
        - **DFS:** LIFO stack; explores deeply then backtracks. Not generally optimal.
        - **Greedy Best-First:** chooses the frontier node with the smallest h(n). Not generally optimal.
        - **Hill Climbing:** moves to a neighbour with a better heuristic; can get stuck at a local optimum or plateau.
        - **A\*:** uses f(n) = g(n) + h(n). With an admissible heuristic and suitable graph conditions, it finds an optimal path.
        """)
    st.subheader("Knowledge representation")
    with st.expander("Semantic network"):
        st.json(SEMANTIC_NETWORK)
    with st.expander("Frames"):
        st.json(FRAMES)
    st.write("Production rules (IF–THEN)")
    st.table(pd.DataFrame(PRODUCTION_RULES, columns=["IF", "THEN"]))
    st.write("Propositional and predicate logic examples")
    logic_rows = [{"Expression": a, "Meaning": b} for a, b in propositional_examples()]
    logic_rows += [{"Expression": a, "Meaning": "Predicate-logic example"} for a in predicate_examples()]
    st.table(pd.DataFrame(logic_rows))
    st.subheader("Rule-based expert-system demonstration")
    selected_id = st.selectbox("Choose a synthetic record", df["patient_id"].tolist())
    record = df[df["patient_id"] == selected_id].iloc[0].to_dict()
    for note in apply_rules(record):
        st.info(note)
    st.caption("Rules summarize dataset fields; they do not diagnose a person.")
    st.subheader("Bayes theorem / Naive Bayes demonstration")
    symptom_query = st.text_input("Enter symptom words", "headache")
    if symptom_query.strip():
        probabilities = naive_bayes_category(df.to_dict("records"), symptom_query)
        st.dataframe(pd.DataFrame(probabilities, columns=["Synthetic category", "Illustrative probability"]).head(8),
                     use_container_width=True, hide_index=True)
        st.caption("Probabilities are a classroom estimate of synthetic category labels, not medical probabilities.")

elif page.startswith("Unit III"):
    st.header("Unit III — Intelligent Agents")
    st.subheader("PEAS framework")
    st.table(pd.DataFrame(list(peas_description().items()), columns=["PEAS element", "Example"]))
    selected_id = st.selectbox("Select a record for agent simulation", df["patient_id"].tolist(), key="agent_record")
    record = df[df["patient_id"] == selected_id].iloc[0].to_dict()
    agents = [
        simple_reflex_agent(record),
        model_based_agent(record, {"records_seen": 12}),
        goal_based_agent(record, "find_followup"),
        utility_based_agent(record),
        learning_agent_demo(),
    ]
    st.subheader("Rational agents and environment")
    st.write("A rational agent selects an action expected to perform well against its performance measure, given its percept history and available knowledge.")
    st.write("Agent environments may be fully/partially observable, deterministic/stochastic, episodic/sequential, static/dynamic, discrete/continuous, and single-/multi-agent.")
    st.subheader("Agent architecture comparison")
    st.table(pd.DataFrame([[a.agent_name, a.action, a.explanation] for a in agents],
                          columns=["Agent type", "Action", "Decision basis"]))
    st.subheader("Cooperative multi-agent simulation")
    st.write("Separate agents check data structure, summarize a field and apply a safety warning.")
    st.table(pd.DataFrame([[a.agent_name, a.action, a.explanation] for a in multi_agent_demo(record)],
                          columns=["Agent", "Contribution", "Purpose"]))
    st.caption("These are software demonstrations, not autonomous clinical agents.")

elif page.startswith("Unit IV"):
    st.header("Unit IV — Evolution of Modern AI")
    st.subheader("Supervised learning: Decision Tree classifier")
    st.write("The model predicts the synthetic category field from selected demographic and visit fields.")
    if st.button("Run supervised learning experiment"):
        try:
            result = supervised_demo(df)
            st.metric("Test accuracy", f"{result['accuracy']:.1%}")
            st.write(f"Training rows: {result['train_size']} | Test rows: {result['test_size']}")
            st.dataframe(result["predictions"].head(30), use_container_width=True, hide_index=True)
            st.caption("This metric measures performance on synthetic labels only; it is not clinical accuracy.")
        except Exception as error:
            st.error(f"Experiment could not run: {error}")
    st.divider()
    st.subheader("Unsupervised learning: K-Means clustering")
    cluster_count = st.slider("Number of clusters", 2, 5, 3)
    if st.button("Run clustering experiment"):
        try:
            clustered, score = unsupervised_demo(df, cluster_count)
            st.metric("Silhouette score", f"{score:.3f}")
            st.bar_chart(clustered["cluster"].value_counts())
            st.dataframe(clustered.head(25), use_container_width=True, hide_index=True)
            st.caption("Clusters are mathematical groupings, not disease groups.")
        except Exception as error:
            st.error(f"Experiment could not run: {error}")
    st.subheader("Other modern AI paradigms")
    st.table(pd.DataFrame([
        ["Symbolic AI", "Uses explicit symbols, facts and rules", "Production rules and knowledge representation"],
        ["Statistical AI", "Learns patterns from data", "Decision Tree and K-Means"],
        ["Reinforcement learning", "Learns actions from rewards and penalties", "Conceptual; no policy trained here"],
        ["Deep learning", "Uses neural networks with multiple layers", "Conceptual overview"],
        ["Foundation models", "Large pretrained models adapted to tasks", "Prompt/RAG module"],
        ["Multimodal AI", "Combines text, images, audio or other inputs", "Conceptual extension"],
        ["Agentic AI", "Plans and uses tools to achieve goals", "Agent simulation"],
        ["Edge/Tiny AI", "Runs compact models near data sources", "Conceptual overview"],
    ], columns=["Paradigm", "Meaning", "Project connection"]))

elif page.startswith("Unit V"):
    st.header("Unit V — Generative AI and Responsible AI")
    st.subheader("Prompt engineering laboratory")
    question = st.text_input("Enter a project question", "Explain A star search for my faculty viva")
    style = st.selectbox("Prompt style", ["Clear and concise", "Step by step", "Faculty viva"])
    st.code(build_prompt(question, style))
    st.write("Compare prompt designs")
    for name, prompt in prompt_experiments(question).items():
        with st.expander(name):
            st.code(prompt)

    st.subheader("Retrieval-Augmented Generation (RAG) demonstration")
    rag_question = st.text_input("Ask the local project knowledge base", "What does A star search do?")
    if rag_question.strip():
        retrieved = retrieve_context(rag_question)
        st.write("Retrieved context")
        for note, score in retrieved:
            st.write(f"- Similarity {score:.2f}: {note}")
        st.write("Grounded retrieval output")
        st.info(rag_answer(rag_question))
    st.caption("Local TF-IDF retrieval is always available. Optional LLM generation below requires Ollama installed and running on this computer.")
    model_name = st.text_input("Local Ollama model name", "llama3.2")
    if st.button("Generate grounded answer with local Ollama"):
        try:
            context = "\\n".join(note for note, _ in retrieve_context(rag_question))
            prompt = build_prompt(rag_question, style="Faculty viva") + "\\nRETRIEVED CONTEXT:\\n" + context
            answer = ollama_generate(prompt, model=model_name)
            st.write(answer)
            st.caption("Verify generated statements against the source notes. LLM output can still hallucinate.")
        except Exception as error:
            st.warning("Ollama is not available. Install and start Ollama, pull a model, and try again. Details: " + str(error))

    st.subheader("LLM and hallucination awareness")
    st.write("A Large Language Model (LLM) generates text by predicting likely token sequences from context. It can produce fluent but unsupported statements called hallucinations.")
    st.write("Mitigation: retrieve trusted context, constrain the prompt, ask the model to state uncertainty, check sources, and require human review. A prompt alone cannot guarantee correctness.")
    st.subheader("Explainability")
    explain_id = st.selectbox("Select a record to explain", df["patient_id"].tolist(), key="explain_record")
    row = df[df["patient_id"] == explain_id].iloc[0].to_dict()
    st.write("Fields visible in this record summary:")
    for item in simple_explanation(row):
        st.write("- " + item)

    st.subheader("Fairness / group distribution audit")
    group_column = st.selectbox("Compare categories across", ["sex", "age_group", "region"])
    st.dataframe(group_distribution(df, group_column), use_container_width=True)
    st.caption("Differences are descriptive and do not prove bias, discrimination or causation.")
    st.subheader("Responsible AI checklist")
    st.table(pd.DataFrame(responsible_ai_checklist(), columns=["Principle", "Project practice"]))
    st.write("Data quality summary")
    st.json(data_quality_summary(df))

elif page.startswith("Unit VI"):
    st.header("Unit VI — AI Ecosystem and Future Trends")
    st.table(pd.DataFrame([
        ["Python", "General-purpose programming language", "Used"],
        ["Pandas / NumPy", "Data and numerical processing", "Used"],
        ["scikit-learn", "Classical machine-learning tools", "Used"],
        ["TensorFlow / PyTorch", "Deep-learning frameworks", "Explained; not required for this small dataset"],
        ["Hugging Face", "Model and dataset ecosystem", "Potential model source"],
        ["Ollama", "Local model runner for supported LLMs", "Optional future integration"],
        ["LangChain", "Framework for LLM and retrieval workflows", "Conceptual connection"],
        ["Vector databases", "Store and search embedding vectors", "Conceptual; TF-IDF retrieval used here"],
    ], columns=["Tool", "Purpose", "Project use"]))
    st.subheader("AI development workflow")
    st.write("Problem → Data → Preparation → Model/method → Evaluation → Deployment → Monitoring → Governance")
    st.subheader("Future trends")
    st.markdown("""
    - **AGI:** hypothetical general intelligence across a broad range of tasks.
    - **Sustainable AI:** reducing compute, energy and resource costs.
    - **Multimodal AI:** processing multiple kinds of information.
    - **Edge AI:** running inference near the source of data.
    - **Human-AI collaboration:** AI supports people; humans retain accountability.
    """)
    st.info("The current project is a lightweight local Python application; no cloud API or large language model is required.")

else:
    st.header("Dataset Dashboard")
    age_filter = st.selectbox("Age group", ["All"] + sorted(df["age_group"].unique()))
    category_filter = st.selectbox("Healthcare category", ["All"] + sorted(df["category"].unique()))
    region_filter = st.selectbox("Region", ["All"] + sorted(df["region"].unique()))
    view = df.copy()
    if age_filter != "All":
        view = view[view["age_group"] == age_filter]
    if category_filter != "All":
        view = view[view["category"] == category_filter]
    if region_filter != "All":
        view = view[view["region"] == region_filter]
    a, b, c, d = st.columns(4)
    a.metric("Records", len(view))
    b.metric("Average age", f"{view['age'].mean():.1f}" if len(view) else "N/A")
    c.metric("Categories", view["category"].nunique())
    d.metric("Age groups", view["age_group"].nunique())
    if len(view):
        left, right = st.columns(2)
        left.subheader("Age groups")
        left.bar_chart(view["age_group"].value_counts())
        right.subheader("Healthcare categories")
        right.bar_chart(view["category"].value_counts())
        st.subheader("Illustrative treatment labels")
        st.bar_chart(view["synthetic_treatment_category"].value_counts())
        term = st.text_input("Search by patient ID or symptom", "")
        if term:
            matches = view["reported_symptoms"].astype(str).str.contains(term, case=False, na=False)
            matches |= view["patient_id"].astype(str).str.contains(term, case=False, na=False)
            st.dataframe(view[matches], use_container_width=True, hide_index=True)
        else:
            st.dataframe(view.head(50), use_container_width=True, hide_index=True)
    else:
        st.warning("No records match these filters.")

st.divider()
st.caption("CSE276 learning demonstration. Confirm practical requirements with your course faculty.")
