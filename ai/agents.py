"""PEAS and simple intelligent-agent demonstrations."""
from dataclasses import dataclass

@dataclass
class Agent:
    name:str
    action:str
    reason:str


def peas():
    return {"Performance":"Clear summaries and safe outputs",
    "Environment":"Local dashboard and synthetic CSV",
    "Actuators":"Display charts, filters and warnings",
    "Sensors":"CSV fields and user selections"}


def agent_examples(row,memory=None):
    severity=str(row.get("reported_severity","")).lower()
    score={"mild":10,"moderate":20,"severe":30}.get(severity,0)
    return [
    Agent("Simple reflex","Flag for human review" if score==30 else "Show summary",
          "Uses only the current severity field."),
    Agent("Model-based","Update stored summary",
          "Uses current record and memory ("+str((memory or {}).get("seen",0))+" earlier records)."),
    Agent("Goal-based","Include in follow-up subset" if row.get("visit_type")=="Follow-up" else "Continue search",
          "Chooses action to reach the follow-up goal."),
    Agent("Utility-based","Illustrative score: "+str(score),
          "Uses a classroom score, not clinical triage."),
    Agent("Learning-agent concept","Update policy from feedback",
          "Learns from experience; no clinical policy is trained here.")]


def multi_agent(row):
    return [Agent("Data agent","Check record fields","Checks structure."),
    Agent("Pattern agent","Read category: "+str(row.get("category","Unknown")),"Summarizes a dataset label."),
    Agent("Safety agent","Attach synthetic-data warning","Prevents clinical misinterpretation.")]
