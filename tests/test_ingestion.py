"""Tests for deterministic IDs (idempotent ingestion helper)."""

import uuid

from src.qdrant_manager import deterministic_id


def test_deterministic_id_is_stable():
    assert deterministic_id("hello") == deterministic_id("hello")


def test_deterministic_id_differs_by_content():
    assert deterministic_id("a") != deterministic_id("b")


def test_deterministic_id_is_valid_uuid():
    value = deterministic_id("NDRF: flooding in Assam")
    # Should parse as a UUID (Qdrant accepts UUID strings as point IDs).
    parsed = uuid.UUID(value)
    assert str(parsed) == value


def test_same_log_line_maps_to_same_point_id():
    line = "IMD: Heavy rainfall warning issued for Guwahati."
    # Two ingestion runs over the same line → identical IDs → upsert overwrites.
    assert deterministic_id(line) == deterministic_id(line)
