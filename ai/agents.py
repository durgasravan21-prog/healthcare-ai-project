"""Explainable agent architecture and cooperation demonstrations."""
from dataclasses import dataclass


@dataclass
class AgentResult:
    agent_name: str
    action: str
    explanation: str


def simple_reflex_agent(record):
    if str(record.get("reported_severity", "")).lower() == "severe":
        return AgentResult("Simple reflex", "Flag record for human review",
                           "Uses only the current severity percept.")
    return AgentResult("Simple reflex", "Display record summary",
                       "Uses only the current severity percept.")


def model_based_agent(record, memory=None):
    memory = memory or {}
    count = memory.get("records_seen", 0)
    return AgentResult("Model-based", "Update internal summary",
                       f"Uses current record plus stored memory ({count} earlier records).")


def goal_based_agent(record, goal="find_followup"):
    if goal == "find_followup" and record.get("visit_type") == "Follow-up":
        action = "Include in follow-up subset"
    elif goal == "find_followup":
        action = "Continue searching"
    else:
        action = "Add to descriptive summary"
    return AgentResult("Goal-based", action, f"Action selected to achieve goal: {goal}.")


def utility_based_agent(record):
    score = {"mild": 1, "moderate": 2, "severe": 3}.get(
        str(record.get("reported_severity", "")).lower(), 0)
    return AgentResult("Utility-based", f"Illustrative review score: {score * 10}",
                       "Uses a simple utility score for a classroom example, not clinical triage.")


def learning_agent_demo():
    return AgentResult("Learning-agent concept", "Update policy after feedback",
                       "A learning agent improves its decision policy using feedback; this project explains the cycle rather than training a clinical policy.")


def multi_agent_demo(record):
    return [
        AgentResult("Data agent", "Check required fields", "Checks the record structure."),
        AgentResult("Pattern agent", f"Read category: {record.get('category', 'Unknown')}",
                    "Reports the synthetic category field."),
        AgentResult("Safety agent", "Attach synthetic-data warning",
                    "Prevents the output being mistaken for medical advice."),
    ]


def peas_description():
    return {
        "Performance measure": "Accurate summaries and no unsupported clinical claims.",
        "Environment": "Local dashboard and synthetic CSV records.",
        "Actuators": "Display summaries, filters, warnings and charts.",
        "Sensors": "CSV fields and user-selected filters or search text.",
    }
