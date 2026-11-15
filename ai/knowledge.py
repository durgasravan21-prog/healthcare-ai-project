"""Knowledge representation, IF-THEN rules and Naive Bayes."""
from collections import Counter
from math import exp,log
import re

SEMANTIC_NETWORK={"HealthcareRecord":["Patient","Visit","Symptom","SyntheticLabel"],
"Patient":["AgeGroup","Region"],"Visit":["VisitType","VisitDate"],
"Symptom":["Duration","Severity"],"SyntheticLabel":["Category","TreatmentLabel"]}
FRAMES={"Patient":{"id":"synthetic ID","age":"number","age_group":"Child/Adult/Senior"},
"Visit":{"type":"visit type","symptoms":"reported text","severity":"synthetic label"}}
RULES=[("age_group == Senior","Record belongs to Senior age group"),
("reported_severity == Severe","Dataset marks this record Severe"),
("visit_type == Follow-up","Record is labelled as a follow-up")]


def apply_rules(row):
    checks=[("age_group","Senior"),("reported_severity","Severe"),("visit_type","Follow-up")]
    found=[f"IF {key} is {value}: {message}"
           for (key,value),(_,message) in zip(checks,RULES) if row.get(key)==value]
    return found or ["No rule matched this record."]


def logic_examples():
    return [("S AND V","Symptom entry AND visit entry"),
    ("S OR F","Symptom entry OR follow-up visit"),("NOT M","Record is not Moderate"),
    ("Patient(x) -> HasRecord(x)","Predicate logic: patient has a record"),
    ("Senior(x) -> AgeGroup(x, Senior)","Predicate logic: senior group membership")]


def naive_bayes(records,query):
    """Multinomial Naive Bayes for synthetic categories, never diagnosis."""
    words=Counter(re.findall(r"[a-z]+",query.lower()))
    docs=[(re.findall(r"[a-z]+",str(r.get("reported_symptoms","")).lower()),
           str(r.get("category","Unknown"))) for r in records]
    docs=[(tokens,label) for tokens,label in docs if tokens]
    vocab={w for tokens,_ in docs for w in tokens}
    if not words or not docs or not vocab: return []
    scores=[]
    for label in sorted({y for _,y in docs}):
        group=[tokens for tokens,y in docs if y==label]
        counts=Counter(w for tokens in group for w in tokens)
        denom=sum(counts.values())+len(vocab)
        score=log(len(group)/len(docs))+sum(
            n*log((counts[w]+1)/denom) for w,n in words.items())
        scores.append((label,score))
    top=max(s for _,s in scores)
    probabilities=[(label,exp(score-top)) for label,score in scores]
    total=sum(p for _,p in probabilities)
    return sorted([(label,round(p/total,4)) for label,p in probabilities],
                  key=lambda item:item[1],reverse=True)
