"""Tests for location extraction / geocoding heuristics."""

from src.utils.location_extractor import extract_location, extract_all_locations


def test_extract_specific_city_preferred_over_state():
    # "Guwahati" is more specific than "Assam"; both present.
    name, coords = extract_location("Flooding reported in Guwahati, Assam.")
    assert name == "Guwahati"
    assert coords is not None and len(coords) == 2


def test_extract_returns_none_when_no_location():
    name, coords = extract_location("All quiet, no incidents to report.")
    assert name is None
    assert coords is None


def test_extract_coordinates_match_config():
    name, coords = extract_location("Landslide reported near Yellapur village.")
    assert name == "Yellapur"
    assert coords == (14.9640, 74.7620)


def test_extract_all_locations_dedupes():
    locs = extract_all_locations("Kolkata and Kolkata again, plus Patna.")
    names = {loc["name"] for loc in locs}
    assert "Kolkata" in names
    assert "Patna" in names
    # No duplicate Kolkata entries.
    kolkata_count = sum(1 for loc in locs if loc["name"] == "Kolkata")
    assert kolkata_count == 1
