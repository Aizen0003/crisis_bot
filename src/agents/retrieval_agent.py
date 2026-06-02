"""
Query-intent helper (not an orchestrator).

Originally a retrieval orchestrator, this module now holds ONLY
`should_show_images` — a small query-intent heuristic used by the LangGraph
`parse_query` node to honor explicit "text only / no images" requests. The
former `execute_retrieval` orchestrator was removed: the live retrieval path is
the LangGraph workflow + LangChain retriever adapters, not this module.

Kept here (rather than moved to src/utils/) to avoid import churn across the
graph and README, since this is the only remaining symbol.
"""

# Keywords that signal the user wants images suppressed.
NEGATIVE_IMAGE_KEYWORDS = [
    "don't show", "dont show", "no photo", "no image",
    "stop showing", "hide image", "text only",
]


def should_show_images(query: str) -> bool:
    """Return False when the user explicitly asks to suppress images."""
    query_lower = query.lower()
    return not any(kw in query_lower for kw in NEGATIVE_IMAGE_KEYWORDS)
