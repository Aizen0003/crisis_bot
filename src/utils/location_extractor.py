"""
Location extraction and geocoding utilities.
Extracts location names from disaster text logs and maps them to coordinates.

Geocoding is intentionally backed by a static `LOCATION_COORDINATES` dictionary
(in src/config.py) rather than an online geocoder (e.g. geopy/Nominatim). This
is a deliberate choice for this demo: it has no network dependency, is fully
deterministic and fast, and never hangs ingestion on a flaky lookup. The dataset
is a fixed set of Pan-India locations, so the dictionary covers it completely.
"""

from src.config import LOCATION_COORDINATES


def extract_location(text: str) -> tuple[str | None, tuple[float, float] | None]:
    """
    Extract the most specific location mention from a text log.
    Returns (location_name, (lat, lon)) or (None, None).
    
    Strategy: Check for specific cities first (more specific = higher priority),
    then fall back to state-level matches.
    """
    text_lower = text.lower()
    
    # Sort by specificity — longer names are typically more specific
    sorted_locations = sorted(
        LOCATION_COORDINATES.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    )
    
    for loc_name, coords in sorted_locations:
        if loc_name in text_lower:
            return loc_name.title(), coords
    
    return None, None


def extract_all_locations(text: str) -> list[dict]:
    """Extract all location mentions from text with their coordinates."""
    text_lower = text.lower()
    locations = []
    found_names = set()
    
    sorted_locations = sorted(
        LOCATION_COORDINATES.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    )
    
    for loc_name, coords in sorted_locations:
        if loc_name in text_lower and loc_name not in found_names:
            # Avoid duplicate state-level matches when city is already found
            found_names.add(loc_name)
            locations.append({
                "name": loc_name.title(),
                "lat": coords[0],
                "lon": coords[1],
            })
    
    return locations
