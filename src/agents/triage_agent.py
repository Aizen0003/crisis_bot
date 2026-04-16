"""
Triage Agent: Classifies incoming reports by severity and disaster type.
Uses keyword matching for disaster classification and heuristic rules
for severity assessment.
"""

from src.config import DISASTER_KEYWORDS, SEVERITY_LEVELS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Keywords that indicate higher severity
CRITICAL_INDICATORS = [
    "casualty", "casualties", "trapped", "missing", "dead", "fatal",
    "collapse", "collapsed", "derailment", "explosion", "tsunami",
    "stranded", "rescue", "medevac", "extrication", "mass casualty",
    "mcm", "danger mark", "breach", "critical",
]

HIGH_INDICATORS = [
    "evacuation", "evacuated", "blocked", "suspended", "submerged",
    "overflow", "surge", "hazmat", "toxic", "contamination",
    "embankment", "alert", "warning", "advisory",
]

MEDIUM_INDICATORS = [
    "damage", "affected", "disrupted", "monitoring", "assessment",
    "deployed", "mobilised", "dispatched", "inspection",
]


def classify_disaster_type(text: str) -> str:
    """
    Classify the disaster type from text using keyword matching.
    Returns the most relevant disaster category.
    """
    text_lower = text.lower()
    scores = {}
    
    for dtype, keywords in DISASTER_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[dtype] = count
    
    if scores:
        return max(scores, key=scores.get)
    return "other"


def assess_severity(text: str) -> str:
    """
    Assess the severity level of a disaster report.
    Returns: "CRITICAL", "HIGH", "MEDIUM", or "LOW"
    """
    text_lower = text.lower()
    
    critical_count = sum(1 for kw in CRITICAL_INDICATORS if kw in text_lower)
    high_count = sum(1 for kw in HIGH_INDICATORS if kw in text_lower)
    medium_count = sum(1 for kw in MEDIUM_INDICATORS if kw in text_lower)
    
    if critical_count >= 2:
        return "CRITICAL"
    elif critical_count >= 1 or high_count >= 2:
        return "HIGH"
    elif high_count >= 1 or medium_count >= 2:
        return "MEDIUM"
    else:
        return "LOW"


def extract_source_agency(text: str) -> str:
    """Extract the reporting agency from a log entry."""
    if ":" in text:
        return text.split(":")[0].strip()
    return "Unknown Agency"


def triage_report(text: str) -> dict:
    """
    Full triage pipeline for a single report.
    Returns classification results with all metadata.
    """
    disaster_type = classify_disaster_type(text)
    severity = assess_severity(text)
    source = extract_source_agency(text)
    severity_config = SEVERITY_LEVELS.get(severity, SEVERITY_LEVELS["LOW"])
    
    result = {
        "disaster_type": disaster_type,
        "severity": severity,
        "severity_color": severity_config["color"],
        "severity_icon": severity_config["icon"],
        "severity_priority": severity_config["priority"],
        "source_agency": source,
    }
    
    logger.info(
        f"Triage: [{severity}] {disaster_type} — {source}"
    )
    return result
