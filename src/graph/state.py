"""
Typed state for the Crisis Intelligence LangGraph workflow.

`CrisisState` is the single mutable object passed between graph nodes. Each node
reads the fields it needs and writes the fields it produces. Using a TypedDict
keeps the contract explicit and interview-explainable: you can point at the
state and say exactly what flows through the pipeline.
"""

from typing import Any, TypedDict

from langchain_core.documents import Document


class CrisisState(TypedDict, total=False):
    # ── Inputs ───────────────────────────────────────────────────────────
    query: str                      # The commander's natural-language question
    filters: dict[str, Any]         # Raw UI filters: {disaster, severity, region}
    persist: bool                   # Whether to write conversation memory

    # ── parse_query ──────────────────────────────────────────────────────
    show_images: bool               # False when user asks for text-only / no images
    qdrant_filters: dict[str, Any]  # Normalized metadata filters for Qdrant

    # ── triage_query ─────────────────────────────────────────────────────
    triage: dict[str, Any]          # Heuristic disaster_type / severity of the query

    # ── retrieve_text / retrieve_images ──────────────────────────────────
    text_docs: list[Document]       # Raw text evidence Documents from Qdrant
    image_docs: list[Document]      # Raw image evidence Documents from Qdrant

    # ── rerank_context ───────────────────────────────────────────────────
    text_results: list[dict]        # UI-shaped text results (text, score, metadata)
    image_results: list[dict]       # UI-shaped image results (filename, score, ...)
    context_string: str             # Text context block for the LLM prompt
    visual_context: str             # Visual-evidence block for the LLM prompt
    reasoning_trace: str            # Human-readable trace of what was retrieved

    # ── synthesize_response ──────────────────────────────────────────────
    response: str                   # Final grounded, cited answer

    # ── Error channel (surfaced to UI; keeps mocked tests deterministic) ─
    error: str | None               # Actionable error message, if a node failed
