"""
Shared pytest fixtures.

Tests never use real API keys or hit the network: Qdrant search, embedding
models, and Gemini are all mocked. Dummy env vars are set so any incidental
config read is harmless.
"""

import os
import sys
from types import SimpleNamespace

import pytest

# Ensure project root is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Harmless placeholders so config validation never needs real secrets.
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-qdrant-key")


class FakeHit(SimpleNamespace):
    """Mimics a Qdrant ScoredPoint: has .id, .score, .payload."""


def make_text_hit(text, score, **payload):
    payload = {"chat_text": text, "role": "system_report", **payload}
    return FakeHit(id="t-" + str(abs(hash(text)) % 10_000), score=score, payload=payload)


def make_image_hit(filename, score, description="", **payload):
    payload = {"filename": filename, "description": description, "type": "photo", **payload}
    return FakeHit(id="i-" + str(abs(hash(filename)) % 10_000), score=score, payload=payload)


@pytest.fixture
def fake_text_hits():
    return [
        make_text_hit("NDRF: Severe flooding submerged Assam, boats deployed.", 0.81,
                      severity="HIGH", disaster_type="flood", region="Assam",
                      source_agency="NDRF"),
        make_text_hit("IMD: Heavy rainfall warning issued for Guwahati.", 0.62,
                      severity="MEDIUM", disaster_type="flood", region="Guwahati",
                      source_agency="IMD"),
    ]


@pytest.fixture
def fake_image_hits():
    return [
        make_image_hit("data_images/flood.jpg", 0.55, description="flood"),
        make_image_hit("data_images/flood.jpg", 0.40, description="flood"),  # dup
        make_image_hit("data_images/guwahati-flood.jpg", 0.48,
                       description="guwahati flood ndrf rescue"),
    ]
