"""
Crisis Intelligence Command Center
===================================
A Multimodal RAG System for National Disaster Response

Entry point for the Streamlit application.
The chat pipeline is orchestrated by a LangGraph workflow (see src/graph),
with Gemini and Qdrant accessed through LangChain adapters (see src/langchain_adapters).

Usage:
    streamlit run app.py
"""

from src.ui.dashboard import setup_page, render_header, render_sidebar, render_main_content


def main():
    # 1. Configure page and initialize resources
    setup_page()
    
    # 2. Render the header
    render_header()
    
    # 3. Render sidebar (returns active filters)
    filters = render_sidebar()
    
    # 4. Render tabbed main content
    render_main_content(filters)


if __name__ == "__main__":
    main()
