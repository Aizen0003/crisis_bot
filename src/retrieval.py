"""
Retrieval engine: orchestrates parallel text + image search,
applies threshold filtering, re-ranking by recency, and
returns traceable reasoning paths.
"""

from datetime import datetime, timezone
from src.embeddings import encode_text, encode_query_for_images
from src.qdrant_manager import search_text, search_images
from src.config import (
    TEXT_RELEVANCE_THRESHOLD, IMAGE_RELEVANCE_THRESHOLD,
    TEXT_SEARCH_LIMIT, IMAGE_SEARCH_LIMIT,
    MEMORY_DECAY_HOURS, MEMORY_DECAY_FACTOR,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _apply_recency_boost(results: list) -> list:
    """
    Re-rank results by boosting recent entries.
    Points with a timestamp payload get a recency multiplier.
    """
    now = datetime.now(timezone.utc)
    boosted = []
    
    for hit in results:
        score = hit.score
        ts_str = hit.payload.get("timestamp")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                hours_old = (now - ts).total_seconds() / 3600
                if hours_old > MEMORY_DECAY_HOURS:
                    score *= MEMORY_DECAY_FACTOR
            except (ValueError, TypeError):
                pass
        boosted.append((hit, score))
    
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted


def retrieve_context(query: str, filters: dict = None) -> dict:
    """
    Main retrieval function. Performs parallel text + image search.
    
    Returns:
        {
            "text_results": [{"text": str, "score": float, "metadata": dict}, ...],
            "image_results": [{"filename": str, "description": str, "score": float}, ...],
            "context_string": str,      # Formatted for LLM prompt
            "visual_context": str,      # Image descriptions for LLM
            "reasoning_trace": str,     # What was retrieved and why
        }
    """
    # ── Parallel Encoding ────────────────────────────────────────────
    text_vector = encode_text(query)
    image_vector = encode_query_for_images(query)
    
    # ── Text Search ──────────────────────────────────────────────────
    raw_text_hits = search_text(
        text_vector,
        limit=TEXT_SEARCH_LIMIT,
        score_threshold=TEXT_RELEVANCE_THRESHOLD,
        filters=filters,
    )
    
    # Apply recency re-ranking
    boosted_text = _apply_recency_boost(raw_text_hits)
    
    text_results = []
    reasoning_parts = []
    
    for hit, adj_score in boosted_text:
        text_results.append({
            "text": hit.payload.get("chat_text", ""),
            "score": round(adj_score, 4),
            "original_score": round(hit.score, 4),
            "role": hit.payload.get("role", "unknown"),
            "metadata": {
                k: v for k, v in hit.payload.items()
                if k not in ("chat_text", "role")
            },
        })
        # Build reasoning trace
        source = hit.payload.get("source_agency", "Unknown Source")
        region = hit.payload.get("region", "Unknown Region")
        reasoning_parts.append(
            f"[Score: {adj_score:.3f}] {source} — {region}: "
            f"\"{hit.payload.get('chat_text', '')[:80]}...\""
        )
    
    # ── Image Search ─────────────────────────────────────────────────
    raw_img_hits = search_images(
        image_vector,
        limit=IMAGE_SEARCH_LIMIT,
        score_threshold=IMAGE_RELEVANCE_THRESHOLD,
    )
    
    image_results = []
    seen_filenames = set()
    for hit in raw_img_hits:
        filename = hit.payload.get("filename", "")
        # Deduplicate: skip if we already have this image (keep highest score first)
        if filename in seen_filenames:
            continue
        seen_filenames.add(filename)
        
        image_results.append({
            "filename": filename,
            "description": hit.payload.get("description", ""),
            "score": round(hit.score, 4),
            "metadata": {
                k: v for k, v in hit.payload.items()
                if k not in ("filename", "description", "type")
            },
        })
        reasoning_parts.append(
            f"[Visual: {hit.score:.3f}] Retrieved image: {hit.payload.get('description', 'N/A')}"
        )
    
    # ── Build Context Strings ────────────────────────────────────────
    if text_results:
        context_lines = []
        for i, r in enumerate(text_results, 1):
            severity = r["metadata"].get("severity", "")
            severity_tag = f" [{severity.upper()}]" if severity else ""
            context_lines.append(f"[Source {i}]{severity_tag}: {r['text']}")
        context_string = "\n".join(context_lines)
    else:
        context_string = "NO RELEVANT DATA FOUND IN DATABASE."
    
    if image_results:
        visual_parts = []
        for img in image_results:
            visual_parts.append(
                f"Image found (confidence: {img['score']:.2f}): {img['description']}"
            )
        visual_context = "\n".join(visual_parts)
    else:
        visual_context = "No relevant images found."
    
    # ── Reasoning Trace ──────────────────────────────────────────────
    reasoning_trace = (
        f"Query: \"{query}\"\n"
        f"Text matches: {len(text_results)} | Image matches: {len(image_results)}\n"
        + "\n".join(reasoning_parts) if reasoning_parts
        else "No relevant data found in any collection."
    )
    
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
