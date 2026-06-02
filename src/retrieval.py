"""
Shared recency-decay helper.

This module now holds ONLY `apply_recency_decay`, the small scoring helper used
by the LangGraph rerank node (`src.graph.crisis_graph.rerank_context`). The
former standalone retrieval orchestrator (`retrieve_context`) and its private
`_apply_recency_boost` re-ranker were removed: the live retrieval path is the
LangChain `BaseRetriever` adapters + the LangGraph workflow, which already
reimplement that logic. Keeping a second, parallel implementation here was dead
code that told a confusing second story.
"""

from datetime import datetime, timezone

from src.config import MEMORY_DECAY_HOURS, MEMORY_DECAY_FACTOR


def apply_recency_decay(score: float, timestamp_str: str | None) -> float:
    """
    Apply recency-based decay to a relevance score.

    Reports older than MEMORY_DECAY_HOURS get their score multiplied by
    MEMORY_DECAY_FACTOR. Returns the (possibly) adjusted score.

    NOTE: decay only differentiates items that carry *distinct* timestamps —
    i.e. stored conversation-memory turns (user/assistant), which are timestamped
    per turn. Base reports all share a single ingest timestamp, so decay is a
    no-op among them (see `rerank_context`).
    """
    if not timestamp_str:
        return score
    try:
        ts = datetime.fromisoformat(timestamp_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        hours_old = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
        if hours_old > MEMORY_DECAY_HOURS:
            return score * MEMORY_DECAY_FACTOR
    except (ValueError, TypeError):
        pass
    return score
