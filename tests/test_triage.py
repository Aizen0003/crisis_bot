"""Tests for heuristic triage: disaster classification and severity assessment."""

from src.agents.triage_agent import (
    classify_disaster_type,
    assess_severity,
    extract_source_agency,
    triage_report,
)


def test_classify_flood():
    text = "Severe flooding submerged the embankment; boats deployed for rescue."
    assert classify_disaster_type(text) == "flood"


def test_classify_earthquake():
    assert classify_disaster_type("Strong seismic tremor, aftershock recorded.") == "earthquake"


def test_classify_unknown_returns_other():
    assert classify_disaster_type("Routine status update, all clear.") == "other"


def test_classify_picks_highest_scoring_type():
    # Mentions both fire and flood, but more flood keywords.
    text = "Flood waters rising, river overflow, submerged homes; small fire reported."
    assert classify_disaster_type(text) == "flood"


def test_assess_severity_critical():
    text = "Mass casualty: building collapsed, several trapped and missing."
    assert assess_severity(text) == "CRITICAL"


def test_assess_severity_high_single_critical():
    assert assess_severity("One person trapped under debris.") == "HIGH"


def test_assess_severity_medium():
    assert assess_severity("Roads disrupted; teams deployed for assessment.") == "MEDIUM"


def test_assess_severity_low():
    assert assess_severity("Weather is calm and stable today.") == "LOW"


def test_extract_source_agency():
    assert extract_source_agency("NDRF: flooding in Assam") == "NDRF"
    assert extract_source_agency("no colon here") == "Unknown Agency"


def test_triage_report_shape():
    result = triage_report("NDRF: building collapsed, several trapped and missing")
    assert result["disaster_type"] == "infrastructure"
    assert result["severity"] == "CRITICAL"
    assert result["source_agency"] == "NDRF"
    assert "severity_color" in result and "severity_icon" in result
