"""Prompt engineering and local retrieval-augmented generation teaching demo."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_NOTES = [
    "The dataset contains synthetic healthcare records for education and not real patients.",
    "Age group labels include Child, Adult, and Senior. Counts describe only this generated dataset.",
    "Category values group records, such as Respiratory and Dermatological.",
    "Synthetic treatment labels are illustrative information, not verified treatments received.",
    "Breadth First Search explores a graph level by level and finds a shortest edge-count path when costs are equal.",
    "Depth First Search explores deeply before backtracking and does not generally find the shortest path.",
    "A star search uses f(n) = g(n) + h(n), where g is path cost and h estimates remaining cost.",
    "Responsible AI includes privacy, fairness, transparency, safety and human oversight.",
]


def build_prompt(question, style="Clear and concise"):
    system = ("You are an educational assistant for a synthetic healthcare data project. "
              "Do not diagnose, prescribe, or present synthetic labels as real medical facts. "
              "Say when the provided context is insufficient.")
    styles = {
        "Clear and concise": "Answer in simple language using up to five bullet points.",
        "Step by step": "Explain in numbered steps and define technical terms.",
        "Faculty viva": "Answer as a student in a faculty viva: precise and easy to say aloud.",
    }
    return (f"SYSTEM: {system}\nSTYLE: {styles.get(style, styles['Clear and concise'])}"
            f"\nQUESTION: {question}\nCONTEXT: [Insert retrieved project notes here]")


def prompt_experiments(question):
    return {
        "Basic": f"Explain: {question}",
        "Role-based": f"You are an AI Foundations tutor. Explain {question} for a beginner.",
        "Constrained": f"Explain {question} in four bullet points, include one example and one limitation.",
    }


def retrieve_context(question, top_k=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(KNOWLEDGE_NOTES + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = scores.argsort()[::-1][:top_k]
    return [(KNOWLEDGE_NOTES[i], float(scores[i])) for i in ranked if scores[i] > 0]


def rag_answer(question, top_k=3):
    retrieved = retrieve_context(question, top_k)
    if not retrieved:
        return "No relevant note was found in the local project knowledge base."
    return "Retrieved project notes:\n" + "\n".join(
        f"- {note}" for note, _ in retrieved
    ) + "\n\nThis is retrieval output, not an LLM-generated medical answer."



def ollama_generate(prompt, model="llama3.2", host="http://localhost:11434"):
    """Optional local LLM inference through an Ollama server."""
    import requests
    response = requests.post(
        host.rstrip("/") + "/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=90,
    )
    response.raise_for_status()
    return response.json().get("response", "")
