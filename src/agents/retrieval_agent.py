"""
Retrieval Agent: Orchestrates multi-collection search and builds
a unified context package for the synthesis agent.
"""

from src.retrieval import retrieve_context
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Keywords that suppress image display
NEGATIVE_IMAGE_KEYWORDS = [
    "don't show", "dont show", "no photo", "no image",
    "stop showing", "hide image", "text only",
]


def should_show_images(query: str) -> bool:
    """Check if the user explicitly wants to suppress images."""
    query_lower = query.lower()
    return not any(kw in query_lower for kw in NEGATIVE_IMAGE_KEYWORDS)


def execute_retrieval(query: str, disaster_filter: str = None,
                      severity_filter: str = None,
                      region_filter: str = None) -> dict:
    """
    Execute the full retrieval pipeline with optional metadata filters.
    
    Returns a context package ready for the synthesis agent:
    {
        "text_results": [...],
        "image_results": [...],
        "context_string": str,
        "visual_context": str,
        "reasoning_trace": str,
        "show_images": bool,
        "filters_applied": dict,
    }
    """
    # Build metadata filters
    filters = {}
    if disaster_filter and disaster_filter != "All":
        filters["disaster_type"] = disaster_filter.lower()
    if severity_filter and severity_filter != "All":
        filters["severity"] = severity_filter.upper()
    if region_filter and region_filter != "All":
        filters["region"] = region_filter
    
    # Execute retrieval
    results = retrieve_context(query, filters=filters if filters else None)
    
    # Add image display decision
    results["show_images"] = should_show_images(query)
    results["filters_applied"] = filters
    
    logger.info(
        f"Retrieval agent: query=\"{query[:50]}...\" "
        f"filters={filters} "
        f"text_hits={len(results['text_results'])} "
        f"img_hits={len(results['image_results'])}"
    )
    
    return results
