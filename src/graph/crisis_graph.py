"""
Crisis Intelligence LangGraph workflow.

Models the multimodal RAG pipeline as an explicit, typed state machine using
`langgraph.graph.StateGraph`. Nodes:

    parse_query        → detect text-only intent, normalize metadata filters
    triage_query       → heuristic disaster-type / severity of the query
    retrieve_text      → MiniLM 384d search over base reports (role=system_report)
    retrieve_images    → CLIP 512d cross-modal search (skipped on text-only queries)
    rerank_context     → recency decay + dedupe + build LLM context blocks
    synthesize_response→ grounded, cited answer via LangChain ChatGoogleGenerativeAI
    persist_memory     → store the turn in episodic memory (non-fatal)

Conditional routing skips image retrieval for text-only queries and short-circuits
to synthesis if core (text) retrieval fails, so failures degrade gracefully.

Public entry point: `run_crisis_graph(query, filters) -> dict`.
"""

from langgraph.graph import StateGraph, START, END

from src.agents.retrieval_agent import should_show_images
from src.agents.triage_agent import triage_report
from src.graph.state import CrisisState
from src.langchain_adapters import llm
from src.langchain_adapters.retrievers import (
    TextEvidenceRetriever,
    ImageEvidenceRetriever,
)
from src.qdrant_manager import QdrantSearchError
from src.retrieval import apply_recency_decay
from src.config import TEXT_SEARCH_LIMIT, IMAGE_SEARCH_LIMIT
from src.memory import store_user_message, store_assistant_response
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Nodes ────────────────────────────────────────────────────────────────────
def parse_query(state: CrisisState) -> dict:
    """Detect text-only intent and normalize UI filters into Qdrant filters."""
    query = state["query"]
    raw = state.get("filters") or {}

    qdrant_filters: dict = {}
    disaster = raw.get("disaster")
    severity = raw.get("severity")
    region = raw.get("region")
    if disaster and disaster != "All":
        qdrant_filters["disaster_type"] = disaster.lower()
    if severity and severity != "All":
        qdrant_filters["severity"] = severity.upper()
    if region and region != "All":
        qdrant_filters["region"] = region

    return {
        "show_images": should_show_images(query),
        "qdrant_filters": qdrant_filters,
        "error": None,
    }


def triage_query(state: CrisisState) -> dict:
    """Heuristically classify the query itself (disaster type + severity)."""
    return {"triage": triage_report(state["query"])}


def retrieve_text(state: CrisisState) -> dict:
    """Retrieve grounded text evidence (MiniLM 384d, base reports only)."""
    retriever = TextEvidenceRetriever(
        limit=TEXT_SEARCH_LIMIT,
        filters=state.get("qdrant_filters") or None,
    )
    try:
        docs = retriever.invoke(state["query"])
        return {"text_docs": docs}
    except QdrantSearchError as e:
        # Text is the core evidence — failure here is fatal for grounding.
        logger.error(f"retrieve_text failed: {e}")
        return {"text_docs": [], "error": str(e)}


def retrieve_images(state: CrisisState) -> dict:
    """Retrieve image evidence (CLIP 512d, cross-modal). Non-fatal on failure."""
    retriever = ImageEvidenceRetriever(limit=IMAGE_SEARCH_LIMIT)
    try:
        docs = retriever.invoke(state["query"])
        return {"image_docs": docs}
    except QdrantSearchError as e:
        # Images are supplementary — degrade gracefully, keep text answer.
        logger.warning(f"retrieve_images failed (continuing without images): {e}")
        return {"image_docs": []}


def rerank_context(state: CrisisState) -> dict:
    """Apply recency decay, dedupe images, and build LLM context blocks."""
    text_docs = state.get("text_docs") or []
    image_docs = state.get("image_docs") or []

    # ── Text: recency re-rank ────────────────────────────────────────────
    scored = []
    for doc in text_docs:
        base = doc.metadata.get("score", 0.0)
        adj = apply_recency_decay(base, doc.metadata.get("timestamp"))
        scored.append((doc, adj))
    scored.sort(key=lambda x: x[1], reverse=True)

    text_results = []
    reasoning_parts = []
    for doc, adj in scored:
        meta = {k: v for k, v in doc.metadata.items() if k not in ("score", "id")}
        text_results.append({
            "text": doc.page_content,
            "score": round(adj, 4),
            "original_score": round(doc.metadata.get("score", 0.0), 4),
            "role": meta.get("role", "unknown"),
            "metadata": {k: v for k, v in meta.items() if k != "role"},
        })
        source = meta.get("source_agency", "Unknown Source")
        region = meta.get("region", "Unknown Region")
        reasoning_parts.append(
            f"[Score: {adj:.3f}] {source} — {region}: "
            f"\"{doc.page_content[:80]}...\""
        )

    # ── Images: dedupe by filename (keep highest score first) ────────────
    image_results = []
    seen = set()
    for doc in image_docs:
        filename = doc.metadata.get("filename", "")
        if filename in seen:
            continue
        seen.add(filename)
        image_results.append({
            "filename": filename,
            "description": doc.metadata.get("description", doc.page_content),
            "score": round(doc.metadata.get("score", 0.0), 4),
            "metadata": {
                k: v for k, v in doc.metadata.items()
                if k not in ("filename", "description", "type", "score", "id")
            },
        })
        reasoning_parts.append(
            f"[Visual: {doc.metadata.get('score', 0.0):.3f}] "
            f"Retrieved image: {doc.metadata.get('description', 'N/A')}"
        )

    # ── Context blocks (same format the old pipeline produced) ───────────
    if text_results:
        lines = []
        for i, r in enumerate(text_results, 1):
            severity = r["metadata"].get("severity", "")
            tag = f" [{severity.upper()}]" if severity else ""
            lines.append(f"[Source {i}]{tag}: {r['text']}")
        context_string = "\n".join(lines)
    else:
        context_string = "NO RELEVANT DATA FOUND IN DATABASE."

    if image_results:
        visual_context = "\n".join(
            f"Image found (confidence: {img['score']:.2f}): {img['description']}"
            for img in image_results
        )
    else:
        visual_context = "No relevant images found."

    if reasoning_parts:
        reasoning_trace = (
            f"Query: \"{state['query']}\"\n"
            f"Text matches: {len(text_results)} | "
            f"Image matches: {len(image_results)}\n"
            + "\n".join(reasoning_parts)
        )
    else:
        reasoning_trace = "No relevant data found in any collection."

    logger.info(
        f"Retrieval complete: {len(text_results)} text, {len(image_results)} images"
    )

    return {
        "text_results": text_results,
        "image_results": image_results,
        "context_string": context_string,
        "visual_context": visual_context,
        "reasoning_trace": reasoning_trace,
    }


def synthesize_response(state: CrisisState) -> dict:
    """Generate the grounded, cited answer via LangChain Gemini."""
    if state.get("error"):
        return {
            "response": (
                "⚠️ **Retrieval Failed**\n\n"
                f"{state['error']}\n\n"
                "Could not query the vector database. Check the Qdrant URL/API key "
                "and connectivity, then try again."
            )
        }

    try:
        response = llm.synthesize(
            context=state.get("context_string", ""),
            visual_context=state.get("visual_context", ""),
            query=state["query"],
        )
        return {"response": response}
    except Exception as e:  # noqa: BLE001 — surface any LLM failure as actionable UI text
        logger.error(f"Gemini synthesis failed: {e}")
        return {
            "response": (
                "⚠️ **Response Generation Failed**\n\n"
                f"Error: {e}\n\n"
                "The retrieval system found data, but the AI synthesis engine "
                "encountered an error. Please check your Gemini API key and try again."
            ),
            "error": str(e),
        }


def persist_memory(state: CrisisState) -> dict:
    """Store the user/assistant turn in episodic memory (non-fatal)."""
    if not state.get("persist", True) or state.get("error"):
        return {}
    try:
        store_user_message(state["query"])
        store_assistant_response(state.get("response", ""), query_context=state["query"])
    except Exception as e:  # noqa: BLE001 — memory write must not break the answer
        logger.warning(f"persist_memory failed (answer still returned): {e}")
    return {}


# ── Routing ──────────────────────────────────────────────────────────────────
def route_after_text(state: CrisisState) -> str:
    """Skip images on text-only queries; short-circuit to synthesis on error."""
    if state.get("error"):
        return "synthesize_response"
    if state.get("show_images", True):
        return "retrieve_images"
    return "rerank_context"


# ── Graph construction ─────────────────────────────────────────────────────────
def build_crisis_graph():
    """Build and compile the LangGraph StateGraph for the RAG pipeline."""
    graph = StateGraph(CrisisState)

    graph.add_node("parse_query", parse_query)
    graph.add_node("triage_query", triage_query)
    graph.add_node("retrieve_text", retrieve_text)
    graph.add_node("retrieve_images", retrieve_images)
    graph.add_node("rerank_context", rerank_context)
    graph.add_node("synthesize_response", synthesize_response)
    graph.add_node("persist_memory", persist_memory)

    graph.add_edge(START, "parse_query")
    graph.add_edge("parse_query", "triage_query")
    graph.add_edge("triage_query", "retrieve_text")
    graph.add_conditional_edges(
        "retrieve_text",
        route_after_text,
        {
            "retrieve_images": "retrieve_images",
            "rerank_context": "rerank_context",
            "synthesize_response": "synthesize_response",
        },
    )
    graph.add_edge("retrieve_images", "rerank_context")
    graph.add_edge("rerank_context", "synthesize_response")
    graph.add_edge("synthesize_response", "persist_memory")
    graph.add_edge("persist_memory", END)

    return graph.compile()


# Lazily-built, process-wide compiled graph (compilation is pure/stateless).
_COMPILED_GRAPH = None


def get_compiled_graph():
    """Return the cached compiled graph, building it on first use."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = build_crisis_graph()
    return _COMPILED_GRAPH


def run_crisis_graph(query: str, filters: dict | None = None,
                     persist: bool = True) -> dict:
    """
    Run the full crisis RAG workflow for one query.

    Args:
        query: the commander's natural-language question.
        filters: optional raw UI filters {disaster, severity, region}.
        persist: whether to write the turn to episodic memory.

    Returns:
        dict with: response, text_results, image_results, context_string,
        visual_context, reasoning_trace, show_images, filters_applied, error.
    """
    initial: CrisisState = {
        "query": query,
        "filters": filters or {},
        "persist": persist,
    }
    final = get_compiled_graph().invoke(initial)

    return {
        "response": final.get("response", ""),
        "text_results": final.get("text_results", []),
        "image_results": final.get("image_results", []),
        "context_string": final.get("context_string", ""),
        "visual_context": final.get("visual_context", ""),
        "reasoning_trace": final.get("reasoning_trace", ""),
        "show_images": final.get("show_images", True),
        "filters_applied": final.get("qdrant_filters", {}),
        "error": final.get("error"),
    }
