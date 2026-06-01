"""
LangChain-compatible retrievers over Qdrant.

These wrap the low-level multimodal vector search in `src.qdrant_manager` behind
LangChain's `BaseRetriever` interface, returning `langchain_core.documents.Document`
objects. Two separate vector spaces are preserved:

  - TextEvidenceRetriever : MiniLM 384d query vector → text collection (base reports)
  - ImageEvidenceRetriever: CLIP   512d text-query vector → image collection

We deliberately keep the raw Qdrant calls (rather than `langchain-qdrant`'s single
`QdrantVectorStore`) because text and image use different embedding models and
vector dimensions, and image retrieval is cross-modal (text query → image vectors).
Forcing both into one vector-store abstraction would degrade that. The retriever
interface is the clean LangChain seam; the vector math stays correct underneath.
"""

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import (
    TEXT_RELEVANCE_THRESHOLD,
    IMAGE_RELEVANCE_THRESHOLD,
    TEXT_SEARCH_LIMIT,
    IMAGE_SEARCH_LIMIT,
    SYSTEM_REPORT_ROLE,
)
from src.embeddings import encode_text, encode_query_for_images
from src.qdrant_manager import search_text, search_images


class TextEvidenceRetriever(BaseRetriever):
    """Retrieve grounded text evidence from the MiniLM (384d) collection.

    Defaults to filtering on ``role="system_report"`` so conversation memory
    does not pollute base disaster retrieval, unless ``restrict_to_reports``
    is set to False.
    """

    limit: int = TEXT_SEARCH_LIMIT
    score_threshold: float = TEXT_RELEVANCE_THRESHOLD
    filters: dict[str, Any] | None = None
    restrict_to_reports: bool = True

    def _build_filters(self) -> dict[str, Any] | None:
        merged: dict[str, Any] = dict(self.filters or {})
        if self.restrict_to_reports:
            merged.setdefault("role", SYSTEM_REPORT_ROLE)
        return merged or None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        vector = encode_text(query)
        hits = search_text(
            vector,
            limit=self.limit,
            score_threshold=self.score_threshold,
            filters=self._build_filters(),
        )
        docs = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("chat_text", "")
            docs.append(
                Document(
                    page_content=text,
                    metadata={**payload, "score": hit.score, "id": hit.id},
                )
            )
        return docs


class ImageEvidenceRetriever(BaseRetriever):
    """Retrieve disaster images from the CLIP (512d) collection via a text query.

    CLIP aligns text and image embeddings in a shared space, so the text query
    is encoded into CLIP space and matched against stored image vectors.
    """

    limit: int = IMAGE_SEARCH_LIMIT
    score_threshold: float = IMAGE_RELEVANCE_THRESHOLD

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        vector = encode_query_for_images(query)
        hits = search_images(
            vector, limit=self.limit, score_threshold=self.score_threshold
        )
        docs = []
        for hit in hits:
            payload = dict(hit.payload or {})
            description = payload.get("description", "")
            docs.append(
                Document(
                    page_content=description,
                    metadata={**payload, "score": hit.score, "id": hit.id},
                )
            )
        return docs
