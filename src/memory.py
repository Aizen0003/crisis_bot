"""
Memory management module.
Handles episodic memory (conversation) storage, retrieval reinforcement,
and session lifecycle management.
"""

from src.embeddings import encode_text
from src.qdrant_manager import upsert_text_point, clear_conversation_memory
from src.utils.logger import get_logger
from datetime import datetime, timezone

logger = get_logger(__name__)


def store_user_message(text: str):
    """Store a user's message in episodic memory with timestamp."""
    vector = encode_text(text)
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "user_input",
    }
    upsert_text_point(text, vector, role="user", metadata=metadata)
    logger.info("Stored user message in episodic memory")


def store_assistant_response(text: str, query_context: str = ""):
    """Store an AI response in episodic memory with context linkage."""
    vector = encode_text(text)
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "ai_response",
        "triggered_by": query_context[:200] if query_context else "",
    }
    upsert_text_point(text, vector, role="assistant", metadata=metadata)
    logger.info("Stored assistant response in episodic memory")


def reset_scenario():
    """
    Safe Reset Protocol:
    - Clears user/assistant conversation memories
    - Preserves system_report entries (base disaster data)
    """
    success = clear_conversation_memory()
    if success:
        logger.info("Scenario reset: episodic memory cleared, base data preserved")
    return success
