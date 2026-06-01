"""Tests for the LangGraph workflow: routing, no-image behavior, compile smoke."""

import pytest

import src.langchain_adapters.retrievers as retrievers_mod
import src.langchain_adapters.llm as llm_mod
from src.graph.crisis_graph import (
    parse_query,
    route_after_text,
    build_crisis_graph,
    run_crisis_graph,
)


# ── Pure routing / parsing ────────────────────────────────────────────────────
def test_parse_query_detects_text_only():
    out = parse_query({"query": "text only please, no images", "filters": {}})
    assert out["show_images"] is False


def test_parse_query_normalizes_filters():
    out = parse_query({
        "query": "flood",
        "filters": {"disaster": "Flood", "severity": "HIGH", "region": "All"},
    })
    assert out["qdrant_filters"] == {"disaster_type": "flood", "severity": "HIGH"}


def test_route_after_text_skips_images_for_text_only():
    assert route_after_text({"show_images": False}) == "rerank_context"


def test_route_after_text_goes_to_images_by_default():
    assert route_after_text({"show_images": True}) == "retrieve_images"


def test_route_after_text_short_circuits_on_error():
    assert route_after_text({"show_images": True, "error": "boom"}) == "synthesize_response"


# ── Compile smoke test ─────────────────────────────────────────────────────────
def test_graph_compiles():
    graph = build_crisis_graph()
    assert hasattr(graph, "invoke")


# ── End-to-end with mocked embeddings / Qdrant / Gemini ─────────────────────────
@pytest.fixture
def mock_backends(monkeypatch, fake_text_hits, fake_image_hits):
    """Mock embeddings, Qdrant search, and Gemini so no network/keys are needed."""
    monkeypatch.setattr(retrievers_mod, "encode_text", lambda q: [0.0] * 384)
    monkeypatch.setattr(retrievers_mod, "encode_query_for_images", lambda q: [0.0] * 512)
    monkeypatch.setattr(retrievers_mod, "search_text",
                        lambda *a, **k: fake_text_hits)

    calls = {"images": 0}

    def fake_search_images(*a, **k):
        calls["images"] += 1
        return fake_image_hits

    monkeypatch.setattr(retrievers_mod, "search_images", fake_search_images)
    monkeypatch.setattr(llm_mod, "synthesize",
                        lambda context, visual_context, query: "GROUNDED ANSWER [Source 1]")
    return calls


def test_run_graph_full_pipeline(mock_backends):
    result = run_crisis_graph("What is the flood situation in Assam?",
                              filters={}, persist=False)
    assert result["response"] == "GROUNDED ANSWER [Source 1]"
    assert len(result["text_results"]) == 2
    # Duplicate flood.jpg deduped → 2 unique images.
    assert len(result["image_results"]) == 2
    assert result["show_images"] is True
    assert mock_backends["images"] == 1  # image retrieval ran


def test_run_graph_text_only_skips_image_search(mock_backends):
    result = run_crisis_graph("Give me a text only summary, no images",
                              filters={}, persist=False)
    assert result["show_images"] is False
    assert result["image_results"] == []
    assert mock_backends["images"] == 0  # image retrieval was skipped entirely


def test_run_graph_handles_text_search_failure(monkeypatch):
    from src.qdrant_manager import QdrantSearchError

    monkeypatch.setattr(retrievers_mod, "encode_text", lambda q: [0.0] * 384)

    def boom(*a, **k):
        raise QdrantSearchError("connection refused")

    monkeypatch.setattr(retrievers_mod, "search_text", boom)
    # Gemini should not even be called when retrieval fails.
    monkeypatch.setattr(llm_mod, "synthesize",
                        lambda **k: pytest.fail("LLM should not run on retrieval failure"))

    result = run_crisis_graph("flood?", filters={}, persist=False)
    assert result["error"] is not None
    assert "Retrieval Failed" in result["response"]
