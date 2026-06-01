"""
LangChain adapters for the Crisis Intelligence Command Center.

This package standardizes external model access behind LangChain interfaces:
  - `llm`        : Gemini chat model via ChatGoogleGenerativeAI + ChatPromptTemplate
  - `retrievers` : LangChain-compatible retrievers over Qdrant text/image vectors

Low-level multimodal vector search stays in `src.qdrant_manager`; these adapters
wrap it in LangChain abstractions without degrading vector-search correctness.
"""
