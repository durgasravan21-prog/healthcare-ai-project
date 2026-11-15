from pathlib import Path
import pandas as pd
import streamlit as st
from ai.search_algorithms import GRAPH,run_all
from ai.knowledge import SEMANTIC_NETWORK,FRAMES,RULES,apply_rules,logic_examples,naive_bayes
from ai.agents import peas,agent_examples,multi_agent
from ai.ml_models import supervised,unsupervised
from ai.responsible_ai import quality,group_table,explain,CHECKLIST
from ai.generative import prompt,retrieve,rag,ollama

DATA=Path(__file__).resolve().parent/"data"/"synthetic_healthcare_records_500.csv"
st.set_page_config(page_title="CSE276 Healthcare AI",page_icon="🏥",layout="wide")
st.title("Healthcare Records — AI Foundations Laboratory")
st.caption("CSE276 | Six-unit project | Synthetic educational data only")


@st.cache_data
def load_data():
    df=pd.read_csv(DATA)
    required=["patient_id","age","age_group","sex","region","visit_type",
    "reported_symptoms","reported_severity","category","synthetic_treatment_category"]
    missing=[c for c in required if c not in df]
    if missing: raise ValueError("Missing columns: "+", ".join(missing))
    df["age"]=pd.to_numeric(df["age"],errors="coerce")
    return df.dropna(subset=["age","age_group","category"])


try: df=load_data()
except Exception as error:
    st.error(f"Dataset loading failed: {error}"); st.stop()

page=st.sidebar.radio("CSE276 Modules",[
"Unit I — AI Foundations","Unit II — Search and Knowledge",
"Unit III — Intelligent Agents","Unit IV — ML and Modern AI",
"Unit V — Generative and Responsible AI","Unit VI — AI Ecosystem","Dataset Dashboard"])
st.sidebar.metric("Synthetic records",len(df))
st.sidebar.warning("Not for diagnosis, triage or treatment.")

if page.startswith("Unit I"):
    st.header("Unit I — AI Foundations and Problem Solving")
    a,b,c=st.columns(3)
    a.metric("Records",len(df)); b.metric("Fields",len(df.columns)); c.metric("Categories",df.category.nunique())
    st.subheader("AI history and evolution")
    st.table(pd.DataFrame([
    ["1950","Turing proposes the imitation game"],["1956","Dartmouth workshop establishes AI as a research field"],
    ["1980s","Expert systems become prominent"],["2010s","Deep learning advances"],["2020s","Foundation and generative models expand"]],
    columns=["Period","Milestone"]))
    st.subheader("AI, ML, DL and Data Science")
    st.table(pd.DataFrame([
    ["AI","Broad field of intelligent behaviour","Search, rules, agents"],
    ["ML","Learns patterns from examples","Decision Tree"],
    ["DL","ML using multi-layer neural networks","Conceptual"],
    ["Data Science","Extracts insights from data","Pandas analysis"]],
    columns=["Field","Meaning","Project example"]))
    st.write("**Types:** Narrow AI performs specific tasks; AGI and superintelligence are hypothetical broader forms.")
    st.write("**Applications:** healthcare, agriculture, finance, education, manufacturing, robotics and smart cities.")
    st.subheader("AI lifecycle")
    st.write("Problem → Data → Preparation → Method → Evaluation → Deployment → Monitoring → Governance")
    st.subheader("Problem formulation and state space")
    st.markdown("**Initial:** Start | **Actions:** follow graph edges | **Transition:** move to connected node | **Goal:** End | **Path cost:** one per edge")
    st.write("State space = all possible states; search space = states explored; solution space = goal-reaching paths.")
    st.write("Search performance: completeness, optimality, time and memory.")
    st.graphviz_chart("""
    digraph G {Start->Registration; Start->Emergency; Registration->Review;
    Emergency->Review; Emergency->Priority; Review->Specialist; Review->Information;
    Priority->Specialist; Information->Specialist; Specialist->Followup; Followup->End;}""")
    st.caption("This is a fictional information workflow, not a clinical pathway.")

elif page.startswith("Unit II"):
    st.header("Unit II — Search, Knowledge Representation and Reasoning")
    a,b=st.columns(2)
    start=a.selectbox("Start node",list(GRAPH))
    goal=b.selectbox("Goal node",list(GRAPH),index=len(GRAPH)-1)
    results=run_all(start,goal)
    st.dataframe(pd.DataFrame([{
    "Algorithm":name,"Path":" → ".join(r["path"]) if r["path"] else "No path/local optimum",
    "Steps":r["steps"],"Explored":len(r["explored"]),"Cost":r["cost"],"Time ms":r["elapsed_ms"]}
    for name,r in results.items()]),use_container_width=True,hide_index=True)
    st.markdown("""
    - **BFS:** FIFO queue; level by level; shortest path for equal edge costs.
    - **DFS:** LIFO stack; deep exploration and backtracking; not generally optimal.
    - **Greedy Best-First:** chooses lowest h(n); not generally optimal.
    - **Hill Climbing:** moves to a better neighbour; may stop at a local optimum.
    - **A\*:** f(n)=g(n)+h(n); suitable admissible heuristics support optimality.
    """)
    with st.expander("Knowledge representation"):
        st.write("Semantic network"); st.json(SEMANTIC_NETWORK)
        st.write("Frames"); st.json(FRAMES)
        st.write("Production rules"); st.table(pd.DataFrame(RULES,columns=["IF","THEN"]))
        st.write("Propositional and predicate logic"); st.table(pd.DataFrame(logic_examples(),columns=["Expression","Meaning"]))
    patient=st.selectbox("Select a record for rule-based reasoning",df.patient_id.tolist())
    row=df[df.patient_id==patient].iloc[0].to_dict()
    for note in apply_rules(row): st.info(note)
    st.write("Expert system = knowledge base + inference engine + user interface.")
    st.write("Bayes theorem: P(A|B) = P(B|A) × P(A) / P(B)")
    query=st.text_input("Naive Bayes: enter symptom words","headache")
    if query:
        st.dataframe(pd.DataFrame(naive_bayes(df.to_dict("records"),query),
        columns=["Synthetic category","Illustrative probability"]).head(8),use_container_width=True,hide_index=True)
        st.caption("Synthetic category estimates only, not medical probabilities.")

elif page.startswith("Unit III"):
    st.header("Unit III — Intelligent Agents")
    st.table(pd.DataFrame(peas().items(),columns=["PEAS","Project example"]))
    patient=st.selectbox("Select a record for agent simulation",df.patient_id.tolist(),key="agent")
    row=df[df.patient_id==patient].iloc[0].to_dict()
    st.subheader("Agent architecture comparison")
    st.table(pd.DataFrame([[x.name,x.action,x.reason] for x in agent_examples(row,{"seen":12})],
    columns=["Agent","Action","Decision basis"]))
    st.write("A rational agent selects actions expected to meet its performance measure, based on percepts and knowledge.")
    st.write("Environment properties: observable/partially observable, deterministic/stochastic, episodic/sequential, static/dynamic, discrete/continuous and single-/multi-agent.")
    st.subheader("Cooperative multi-agent example")
    st.table(pd.DataFrame([[x.name,x.action,x.reason] for x in multi_agent(row)],
    columns=["Agent","Contribution","Purpose"]))
    st.caption("Software demonstrations, not autonomous clinical agents.")

elif page.startswith("Unit IV"):
    st.header("Unit IV — Machine Learning and Modern AI")
    st.subheader("Supervised learning — Decision Tree")
    st.write("Predicts a synthetic category from demographic and visit fields.")
    if st.button("Run Decision Tree"):
        try:
            score,predictions=supervised(df)
            st.metric("Test accuracy on synthetic labels",f"{score:.1%}")
            st.dataframe(predictions.head(25),use_container_width=True,hide_index=True)
        except Exception as error: st.error(str(error))
    st.subheader("Unsupervised learning — K-Means")
    k=st.slider("Number of clusters",2,5,3)
    if st.button("Run K-Means"):
        try:
            clusters,score=unsupervised(df,k)
            st.metric("Silhouette score",f"{score:.3f}")
            st.bar_chart(clusters.cluster.value_counts())
            st.dataframe(clusters.head(20),use_container_width=True,hide_index=True)
        except Exception as error: st.error(str(error))
    st.table(pd.DataFrame([
    ["Symbolic AI","Facts and explicit rules","Production rules"],
    ["Statistical AI","Learns patterns from data","Decision Tree and K-Means"],
    ["Reinforcement learning","Learns actions from rewards","Conceptual"],
    ["Deep learning","Multi-layer neural networks","Conceptual"],
    ["Foundation models","Large pretrained models","Unit V prompt/RAG"],
    ["Multimodal AI","Combines input types","Conceptual"],
    ["Agentic AI","Plans and uses tools","Agent examples"],
    ["Edge/Tiny AI","Compact models near data sources","Conceptual"]],
    columns=["Paradigm","Meaning","Project coverage"]))

elif page.startswith("Unit V"):
    st.header("Unit V — Generative and Responsible AI")
    question=st.text_input("Prompt engineering question","Explain A* search for my viva")
    style=st.selectbox("Prompt style",["Faculty viva","Steps","Brief"])
    st.code(prompt(question,style))
    st.subheader("Local RAG demonstration")
    query=st.text_input("Search project knowledge notes","What does A star do?")
    for note,score in retrieve(query): st.write(f"{score:.2f} — {note}")
    st.info(rag(query))
    st.caption("TF-IDF retrieval is not itself an LLM-generated answer.")
    model=st.text_input("Optional local Ollama model","llama3.2")
    if st.button("Generate with local Ollama"):
        try:
            context="\n".join(note for note,_ in retrieve(query))
            st.write(ollama(prompt(query,style)+"\nContext:\n"+context,model))
            st.caption("Verify output; LLMs can hallucinate.")
        except Exception as error: st.warning(f"Start Ollama and download the model first: {error}")
    st.write("LLMs generate likely token sequences. Hallucination risks can be reduced through trusted retrieval, constrained prompts, source checks and human review.")
    patient=st.selectbox("Explain a record",df.patient_id.tolist(),key="explain")
    row=df[df.patient_id==patient].iloc[0].to_dict()
    st.write(explain(row))
    group=st.selectbox("Fairness comparison group",["sex","age_group","region"])
    st.dataframe(group_table(df,group),use_container_width=True)
    st.caption("Group differences are descriptive, not proof of bias or causation.")
    st.subheader("Responsible AI checklist")
    st.table(pd.DataFrame(CHECKLIST,columns=["Principle","Project practice"]))
    st.json(quality(df))

elif page.startswith("Unit VI"):
    st.header("Unit VI — AI Ecosystem and Future Trends")
    st.table(pd.DataFrame([
    ["Python / Pandas / NumPy","Programming and data processing","Used"],
    ["scikit-learn","Classical ML","Used"],
    ["TensorFlow / PyTorch","Deep learning frameworks","Conceptual"],
    ["Hugging Face","Models and datasets","Potential model source"],
    ["Ollama","Local LLM runner","Optional"],
    ["LangChain","LLM application framework","Conceptual"],
    ["Vector databases","Embedding storage and similarity search","Conceptual; TF-IDF used"]],
    columns=["Tool","Purpose","Project use"]))
    st.write("AI workflow: problem → data → preparation → method → evaluation → deployment → monitoring → governance.")
    st.write("Future trends: AGI (hypothetical), sustainable AI, multimodal AI, edge AI and human-AI collaboration.")
    st.info("Only listed dependencies are used at runtime; conceptual tools are not falsely claimed as implemented.")

else:
    st.header("Healthcare Dataset Dashboard")
    age=st.selectbox("Age group",["All"]+sorted(df.age_group.unique()))
    category=st.selectbox("Category",["All"]+sorted(df.category.unique()))
    region=st.selectbox("Region",["All"]+sorted(df.region.unique()))
    view=df.copy()
    if age!="All": view=view[view.age_group==age]
    if category!="All": view=view[view.category==category]
    if region!="All": view=view[view.region==region]
    a,b,c,d=st.columns(4)
    a.metric("Records",len(view)); b.metric("Average age",f"{view.age.mean():.1f}" if len(view) else "N/A")
    c.metric("Categories",view.category.nunique()); d.metric("Age groups",view.age_group.nunique())
    if len(view):
        left,right=st.columns(2)
        left.bar_chart(view.age_group.value_counts()); right.bar_chart(view.category.value_counts())
        st.bar_chart(view.synthetic_treatment_category.value_counts())
        term=st.text_input("Search patient ID or symptom")
        if term:
            mask=view.patient_id.astype(str).str.contains(term,case=False,na=False)
            mask|=view.reported_symptoms.astype(str).str.contains(term,case=False,na=False)
            st.dataframe(view[mask],use_container_width=True,hide_index=True)
        else: st.dataframe(view.head(50),use_container_width=True,hide_index=True)
    else: st.warning("No records match the filters.")

st.divider()
st.caption("CSE276 educational project using synthetic data. Never use for clinical decisions.")
