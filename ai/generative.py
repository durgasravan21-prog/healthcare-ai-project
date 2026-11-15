"""Prompt engineering, local retrieval and optional Ollama."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NOTES=[
"The 500 healthcare-style records are synthetic and do not represent real patients.",
"Age groups are Child, Adult and Senior; counts describe only this generated dataset.",
"Category fields group synthetic records such as Respiratory and Dermatological.",
"Treatment labels are illustrative information, not verified treatments.",
"BFS explores level by level and finds shortest edge-count path when costs are equal.",
"DFS explores deeply before backtracking and does not generally find shortest path.",
"A star uses f(n)=g(n)+h(n), combining path cost and estimated remaining cost.",
"Responsible AI includes privacy, fairness, transparency, safety and human oversight."]


def prompt(question,style="Faculty viva"):
    instructions={"Faculty viva":"Answer clearly and accurately for a student viva.",
    "Steps":"Explain in numbered steps with an example.",
    "Brief":"Answer in at most five short bullet points."}
    return ("You are an educational AI tutor. Never diagnose or prescribe. "
    "Say when context is insufficient.\n"+instructions.get(style,"Answer clearly.")+
    "\nQuestion: "+question)


def retrieve(question,k=3):
    vector=TfidfVectorizer(stop_words="english")
    matrix=vector.fit_transform(NOTES+[question])
    scores=cosine_similarity(matrix[-1],matrix[:-1]).ravel()
    return [(NOTES[i],float(scores[i])) for i in scores.argsort()[::-1][:k] if scores[i]>0]


def rag(question):
    found=retrieve(question)
    return "\n".join("- "+note for note,_ in found) if found else "No matching local note found."


def ollama(prompt_text,model="llama3.2"):
    import requests
    response=requests.post("http://localhost:11434/api/generate",
        json={"model":model,"prompt":prompt_text,"stream":False},timeout=90)
    response.raise_for_status()
    return response.json().get("response","")
