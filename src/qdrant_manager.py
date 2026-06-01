"""
Qdrant database manager.
Handles collection creation, point upsert/delete, and search operations
using proper Qdrant SDK patterns (PointStruct, VectorParams, etc.).
"""

import uuid
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from src.config import (
    QDRANT_URL, QDRANT_API_KEY,
    COLLECTION_EPISODIC, COLLECTION_MULTIMODAL,
    TEXT_VECTOR_DIM, CLIP_VECTOR_DIM,
    IMAGE_RELEVANCE_THRESHOLD,
    validate_config,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantSearchError(RuntimeError):
    """Raised when a Qdrant vector search fails, so callers can surface it."""


# Stable namespace so the same content always maps to the same point ID.
# This makes ingestion idempotent: re-running upserts overwrite rather than
# duplicate, because Qdrant upsert is keyed on point ID.
_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def deterministic_id(content: str) -> str:
    """
    Derive a stable UUID5 point ID from stable content.

    Identical content (a text log line, or an image's stable key) always
    produces the same ID, so re-ingesting the same data updates the existing
    point instead of inserting a duplicate.
    """
    return str(uuid.uuid5(_ID_NAMESPACE, content))


@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    """Create and cache a Qdrant client connection."""
    validate_config(require=("QDRANT_URL", "QDRANT_API_KEY"))
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
        logger.info("Connected to Qdrant Cloud")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise


def ensure_collections():
    """Create required collections if they don't exist, and ensure payload indexes."""
    client = get_qdrant_client()
    
    collections_config = {
        COLLECTION_EPISODIC: TEXT_VECTOR_DIM,
        COLLECTION_MULTIMODAL: CLIP_VECTOR_DIM,
    }
    
    existing = {c.name for c in client.get_collections().collections}
    
    for name, dim in collections_config.items():
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {name} (dim={dim})")
        else:
            logger.info(f"Collection already exists: {name}")
    
    # Ensure payload indexes for filtered queries
    # Qdrant requires explicit keyword indexes for filter operations.
    episodic_indexes = ["role", "disaster_type", "severity", "region", "source_agency"]
    multimodal_indexes = ["type", "description", "region"]
    _ensure_payload_indexes(client, COLLECTION_EPISODIC, episodic_indexes)
    _ensure_payload_indexes(client, COLLECTION_MULTIMODAL, multimodal_indexes)

    logger.info("Payload indexes ensured for all collections")


def _ensure_payload_indexes(client: QdrantClient, collection: str, fields: list[str]):
    """
    Create keyword payload indexes, tolerating the "already exists" case.

    An index that already exists raises an UnexpectedResponse (HTTP 4xx),
    which is expected and harmless. Any other error is logged so genuine
    indexing problems are not silently swallowed.
    """
    for field in fields:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse:
            # Index already exists — expected on re-runs.
            logger.debug(f"Payload index already present: {collection}.{field}")
        except Exception as e:
            logger.warning(
                f"Could not create payload index {collection}.{field}: {e}"
            )


def upsert_text_point(text: str, vector: list[float], role: str, metadata: dict = None):
    """Insert a text point into the episodic memory collection."""
    client = get_qdrant_client()
    payload = {"chat_text": text, "role": role}
    if metadata:
        payload.update(metadata)
    
    point = models.PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=payload,
    )
    
    client.upsert(
        collection_name=COLLECTION_EPISODIC,
        points=[point],
    )


def upsert_text_points_batch(points_data: list[dict]):
    """
    Batch insert text points.
    Each item: {"text": str, "vector": list, "role": str, "metadata": dict}
    """
    client = get_qdrant_client()
    points = []
    for item in points_data:
        payload = {"chat_text": item["text"], "role": item["role"]}
        if item.get("metadata"):
            payload.update(item["metadata"])
        # Deterministic ID keyed on the report content → idempotent re-ingest.
        point_id = item.get("id") or deterministic_id(item["text"])
        points.append(models.PointStruct(
            id=point_id,
            vector=item["vector"],
            payload=payload,
        ))

    # Batch in chunks of 100
    for i in range(0, len(points), 100):
        chunk = points[i:i+100]
        client.upsert(collection_name=COLLECTION_EPISODIC, points=chunk)

    logger.info(f"Batch upserted {len(points)} text points")


def upsert_image_points_batch(points_data: list[dict]):
    """
    Batch insert image points.
    Each item: {"vector": list, "filename": str, "description": str, "metadata": dict}
    """
    client = get_qdrant_client()
    points = []
    for item in points_data:
        payload = {
            "filename": item["filename"],
            "description": item["description"],
            "type": "photo",
        }
        if item.get("metadata"):
            payload.update(item["metadata"])
        # Deterministic ID keyed on the image's stable key (filename) → idempotent.
        point_id = item.get("id") or deterministic_id(item.get("id_key", item["filename"]))
        points.append(models.PointStruct(
            id=point_id,
            vector=item["vector"],
            payload=payload,
        ))
    
    for i in range(0, len(points), 100):
        chunk = points[i:i+100]
        client.upsert(collection_name=COLLECTION_MULTIMODAL, points=chunk)
    
    logger.info(f"Batch upserted {len(points)} image points")


def search_text(query_vector: list[float], limit: int = 5, score_threshold: float = 0.35,
                filters: dict = None) -> list:
    """
    Search the episodic memory collection with optional metadata filtering.
    Returns list of ScoredPoint objects.
    """
    client = get_qdrant_client()
    
    query_filter = None
    if filters:
        must_conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                must_conditions.append(
                    models.FieldCondition(key=key, match=models.MatchAny(any=value))
                )
            else:
                must_conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
        query_filter = models.Filter(must=must_conditions)
    
    try:
        results = client.query_points(
            collection_name=COLLECTION_EPISODIC,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
        ).points
        return results
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        raise QdrantSearchError(f"Text vector search failed: {e}") from e


def search_images(query_vector: list[float], limit: int = 3,
                  score_threshold: float = IMAGE_RELEVANCE_THRESHOLD) -> list:
    """Search the multimodal collection for matching images."""
    client = get_qdrant_client()
    
    try:
        results = client.query_points(
            collection_name=COLLECTION_MULTIMODAL,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        ).points
        return results
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        raise QdrantSearchError(f"Image vector search failed: {e}") from e


def clear_conversation_memory():
    """
    Safe Reset: Delete ONLY user/assistant entries, preserving system_report data.
    This is the 'memory consolidation' mechanism.
    """
    client = get_qdrant_client()
    try:
        client.delete(
            collection_name=COLLECTION_EPISODIC,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="role",
                            match=models.MatchAny(any=["user", "assistant"]),
                        )
                    ]
                )
            ),
        )
        logger.info("Conversation memory cleared (system reports preserved)")
        return True
    except Exception as e:
        logger.error(f"Failed to clear memory: {e}")
        return False


def get_collection_stats() -> dict:
    """Get statistics for all collections."""
    client = get_qdrant_client()
    stats = {}
    for name in [COLLECTION_EPISODIC, COLLECTION_MULTIMODAL]:
        try:
            info = client.get_collection(name)
            # Try multiple attributes — newer qdrant-client versions vary
            count = info.points_count or info.vectors_count or 0
            if count == 0:
                # Fallback: use count() API
                try:
                    count = client.count(collection_name=name).count
                except Exception:
                    count = 0
            stats[name] = {
                "vectors_count": count,
                "points_count": count,
                "status": info.status.value if info.status else "unknown",
            }
        except Exception:
            stats[name] = {"vectors_count": 0, "points_count": 0, "status": "not_found"}
    return stats

