"""
LangChain LLM adapter for Gemini.

Standardizes Gemini access through LangChain's `ChatGoogleGenerativeAI` model
and `ChatPromptTemplate`, replacing direct `google.genai` calls. The synthesis
prompt is expressed as a reusable chat-prompt template so prompting is
declarative and testable.
"""

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_API_KEY, GEMINI_MODEL, validate_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── System / Human prompt as a reusable LangChain template ───────────────────
_SYSTEM_PROMPT = """You are a **Disaster Response Intelligence Assistant** operating in a real-time crisis command center.

═══ RESPONSE PROTOCOL ═══
1. **Grounded Analysis**: Base your answer ONLY on the retrieved evidence provided. Cite sources as [Source 1], [Source 2], etc.
2. **If NO data found**: Clearly state "No matching records found in the database" but provide general guidance based on standard disaster protocols.
3. **Visual Evidence**: If images were retrieved, describe what they show and how they correlate with the text logs.
4. **Actionable Output**: End with specific, actionable recommendations for the field commander.
5. **Concise Format**: Use bullet points. Be professional and direct — lives depend on clarity.
6. **Severity Assessment**: If applicable, indicate the assessed severity level (CRITICAL/HIGH/MEDIUM/LOW)."""

_HUMAN_PROMPT = """═══ RETRIEVED DATABASE CONTEXT ═══
{context}

═══ VISUAL EVIDENCE ANALYSIS ═══
{visual_context}

═══ COMMANDER'S QUERY ═══
{query}"""


def get_synthesis_prompt() -> ChatPromptTemplate:
    """Return the reusable ChatPromptTemplate used for grounded synthesis."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", _HUMAN_PROMPT),
        ]
    )


@st.cache_resource
def get_chat_model() -> ChatGoogleGenerativeAI:
    """Create and cache the LangChain Gemini chat model."""
    validate_config(require=("GEMINI_API_KEY",))
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )


def synthesize(context: str, visual_context: str, query: str) -> str:
    """
    Generate a grounded, cited response via the LangChain Gemini chain.

    Builds `prompt | model` and invokes it. Raises on failure so callers can
    surface an actionable error (the LangGraph synthesize node does this).
    """
    chain = get_synthesis_prompt() | get_chat_model()
    response = chain.invoke(
        {"context": context, "visual_context": visual_context, "query": query}
    )
    text = (response.content or "").strip()
    if not text:
        return "Unable to generate response. Please try again."
    logger.info(f"Gemini (LangChain) response generated ({len(text)} chars)")
    return text
