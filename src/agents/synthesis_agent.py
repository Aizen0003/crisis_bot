"""
Synthesis Agent: Takes retrieved context and generates a grounded,
evidence-cited response using Google Gemini.
"""

import streamlit as st
from google import genai
from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)


@st.cache_resource
def get_gemini_client():
    """Create and cache the Gemini client."""
    return genai.Client(api_key=GEMINI_API_KEY)


def build_system_prompt(context: str, visual_context: str, query: str) -> str:
    """Build the structured prompt for Gemini with citation instructions."""
    return f"""You are a **Disaster Response Intelligence Assistant** operating in a real-time crisis command center.

═══ RETRIEVED DATABASE CONTEXT ═══
{context}

═══ VISUAL EVIDENCE ANALYSIS ═══
{visual_context}

═══ RESPONSE PROTOCOL ═══
1. **Grounded Analysis**: Base your answer ONLY on the retrieved evidence above. Cite sources as [Source 1], [Source 2], etc.
2. **If NO data found**: Clearly state "No matching records found in the database" but provide general guidance based on standard disaster protocols.
3. **Visual Evidence**: If images were retrieved, describe what they show and how they correlate with text logs.
4. **Actionable Output**: End with specific, actionable recommendations for the field commander.
5. **Concise Format**: Use bullet points. Be professional and direct — lives depend on clarity.
6. **Severity Assessment**: If applicable, indicate the assessed severity level (CRITICAL/HIGH/MEDIUM/LOW).

═══ COMMANDER'S QUERY ═══
{query}
"""


def generate_response(retrieval_results: dict, query: str) -> str:
    """
    Generate a grounded response using Gemini.
    
    Args:
        retrieval_results: Output from the retrieval agent
        query: The user's original question
    
    Returns:
        The AI-generated response text
    """
    prompt = build_system_prompt(
        context=retrieval_results["context_string"],
        visual_context=retrieval_results["visual_context"],
        query=query,
    )
    
    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        
        response_text = response.text if response.text else "Unable to generate response. Please try again."
        logger.info(f"Gemini response generated ({len(response_text)} chars)")
        return response_text
        
    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return (
            "⚠️ **Response Generation Failed**\n\n"
            f"Error: {str(e)}\n\n"
            "The retrieval system found data, but the AI synthesis engine "
            "encountered an error. Please check your Gemini API key and try again."
        )
