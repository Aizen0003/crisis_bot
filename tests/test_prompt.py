"""Tests for prompt construction and context formatting."""

from langchain_core.documents import Document

from src.langchain_adapters.llm import get_synthesis_prompt
from src.graph.crisis_graph import rerank_context


def test_synthesis_prompt_injects_fields():
    prompt = get_synthesis_prompt()
    messages = prompt.format_messages(
        context="[Source 1]: flooding in Assam",
        visual_context="Image found: flood",
        query="What is the flood situation?",
    )
    # System message carries the response protocol; human message carries the data.
    system_text = messages[0].content
    human_text = messages[1].content
    assert "Disaster Response Intelligence Assistant" in system_text
    assert "[Source 1]: flooding in Assam" in human_text
    assert "Image found: flood" in human_text
    assert "What is the flood situation?" in human_text


def test_rerank_builds_cited_context_and_dedupes_images():
    state = {
        "query": "flood in Assam",
        "text_docs": [
            Document(page_content="Flooding submerged Assam.",
                     metadata={"score": 0.8, "severity": "HIGH", "region": "Assam",
                               "source_agency": "NDRF", "role": "system_report"}),
            Document(page_content="Rainfall warning for Guwahati.",
                     metadata={"score": 0.6, "severity": "MEDIUM", "region": "Guwahati",
                               "source_agency": "IMD", "role": "system_report"}),
        ],
        "image_docs": [
            Document(page_content="flood",
                     metadata={"score": 0.55, "filename": "data_images/flood.jpg",
                               "description": "flood", "type": "photo"}),
            Document(page_content="flood",
                     metadata={"score": 0.40, "filename": "data_images/flood.jpg",
                               "description": "flood", "type": "photo"}),  # duplicate
        ],
    }
    out = rerank_context(state)

    assert "[Source 1] [HIGH]: Flooding submerged Assam." in out["context_string"]
    assert "[Source 2] [MEDIUM]: Rainfall warning for Guwahati." in out["context_string"]
    # Duplicate image filename collapsed to one.
    assert len(out["image_results"]) == 1
    assert out["image_results"][0]["filename"] == "data_images/flood.jpg"
    # Higher score retained.
    assert out["image_results"][0]["score"] == 0.55
    assert "confidence" in out["visual_context"]


def test_rerank_no_data_message():
    out = rerank_context({"query": "nothing", "text_docs": [], "image_docs": []})
    assert out["context_string"] == "NO RELEVANT DATA FOUND IN DATABASE."
    assert out["visual_context"] == "No relevant images found."
