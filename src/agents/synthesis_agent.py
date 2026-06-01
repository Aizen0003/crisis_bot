"""
Synthesis Agent (compatibility shim).

The live chat path runs synthesis inside the LangGraph workflow's
`synthesize_response` node, which calls the LangChain Gemini adapter
(`src.langchain_adapters.llm`). This module is kept for backward compatibility
and now delegates to that same LangChain adapter, so there is a single Gemini
code path going through `ChatGoogleGenerativeAI`.
"""

from src.langchain_adapters import llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_response(retrieval_results: dict, query: str) -> str:
    """
    Generate a grounded response from a retrieval-results package via LangChain.

    Args:
        retrieval_results: must contain "context_string" and "visual_context".
        query: the user's original question.

    Returns:
        The AI-generated response text (or an actionable error message).
    """
    try:
        return llm.synthesize(
            context=retrieval_results.get("context_string", ""),
            visual_context=retrieval_results.get("visual_context", ""),
            query=query,
        )
    except Exception as e:  # noqa: BLE001 — surface as actionable message
        logger.error(f"Gemini generation failed: {e}")
        return (
            "⚠️ **Response Generation Failed**\n\n"
            f"Error: {e}\n\n"
            "The retrieval system found data, but the AI synthesis engine "
            "encountered an error. Please check your Gemini API key and try again."
        )
