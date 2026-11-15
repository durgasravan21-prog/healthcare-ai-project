"""Knowledge representation and probabilistic reasoning examples."""
import math
import re
from collections import Counter

SEMANTIC_NETWORK = {
    "HealthcareRecord": ["Patient", "Visit", "ReportedSymptom", "SyntheticLabel"],
    "Patient": ["AgeGroup", "Region"],
    "Visit": ["VisitType", "VisitDate"],
    "ReportedSymptom": ["SymptomDuration", "ReportedSeverity"],
    "SyntheticLabel": ["Category", "IllustrativeTreatmentLabel"],
}
FRAMES = {
    "PatientFrame": {"patient_id": "synthetic identifier", "age": "numeric age",
                     "age_group": "Child / Adult / Senior", "region": "synthetic region"},
    "VisitFrame": {"visit_type": "visit category", "reported_symptoms": "reported text",
                   "reported_severity": "synthetic severity label"},
}
PRODUCTION_RULES = [
    ("IF age_group is Senior", "THEN state that the record belongs to Senior age group"),
    ("IF reported_severity is Severe", "THEN state that the dataset marks it Severe"),
    ("IF visit_type is Follow-up", "THEN state that it is labelled Follow-up"),
]


def apply_rules(record):
    notes = []
    if record.get("age_group") == "Senior":
        notes.append("Age-group rule matched: Senior.")
    if record.get("reported_severity") == "Severe":
        notes.append("Severity rule matched: the synthetic dataset says Severe.")
    if record.get("visit_type") == "Follow-up":
        notes.append("Visit rule matched: Follow-up.")
    return notes or ["No demonstration rule matched."]


def propositional_examples():
    return [
        ("S AND V", "A record has a symptom label AND a visit entry."),
        ("S OR F", "A record has a symptom label OR a follow-up label."),
        ("NOT M", "The selected record is not marked Moderate."),
    ]


def predicate_examples():
    return [
        "Patient(x) -> HasSyntheticRecord(x)",
        "Senior(x) -> BelongsToAgeGroup(x, 'Senior')",
        "For every x, HealthcareRecord(x) is true for each dataset record.",
    ]


def naive_bayes_category(records, query):
    """Multinomial Naive Bayes estimate for synthetic categories, not diagnosis."""
    query_tokens = re.findall(r"[a-z]+", query.lower())
    if not query_tokens:
        return []
    documents = []
    vocabulary = set()
    for row in records:
        tokens = re.findall(r"[a-z]+", str(row.get("reported_symptoms", "")).lower())
        label = str(row.get("category", "Unknown"))
        if tokens:
            documents.append((tokens, label))
            vocabulary.update(tokens)
    if not documents or not vocabulary:
        return []
    labels = sorted({label for _, label in documents})
    scores = []
    for label in labels:
        class_docs = [tokens for tokens, y in documents if y == label]
        token_counts = Counter(token for tokens in class_docs for token in tokens)
        total_tokens = sum(token_counts.values())
        denominator = total_tokens + len(vocabulary)
        score = math.log(len(class_docs) / len(documents))
        query_counts = Counter(query_tokens)
        for word, frequency in query_counts.items():
            score += frequency * math.log((token_counts[word] + 1) / denominator)
        scores.append((label, score))
    maximum = max(score for _, score in scores)
    exponentials = [(label, math.exp(score - maximum)) for label, score in scores]
    normalizer = sum(value for _, value in exponentials)
    return sorted([(label, round(value / normalizer, 4))
                   for label, value in exponentials],
                  key=lambda item: item[1], reverse=True)
