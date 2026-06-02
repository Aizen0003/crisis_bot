"""
Embedding model wrappers for text and image encoding.
Provides a unified interface for both MiniLM (text) and CLIP (vision) encoders.

Dimension map (see src/config.py):
  - MiniLM `all-MiniLM-L6-v2`  → 384-d  (TEXT_VECTOR_DIM)  — text collection
  - CLIP   `clip-ViT-B-32`     → 512-d  (CLIP_VECTOR_DIM)  — image collection

Note the text/image split: `encode_text` uses MiniLM (384-d), but BOTH
`encode_query_for_images` (text→CLIP) and `encode_image` (image→CLIP) use CLIP
and produce the SAME 512-d vector space — that shared space is exactly what
makes cross-modal (text-query → image) retrieval possible.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
from PIL import Image
from src.config import TEXT_ENCODER_MODEL, CLIP_MODEL


@st.cache_resource
def load_text_encoder():
    """Load and cache the text embedding model."""
    return SentenceTransformer(TEXT_ENCODER_MODEL)


@st.cache_resource
def load_clip_model():
    """Load and cache the CLIP vision-language model."""
    return SentenceTransformer(CLIP_MODEL)


def encode_text(text: str) -> list[float]:
    """Encode a text string into a 384-dimensional vector."""
    encoder = load_text_encoder()
    return encoder.encode(text).tolist()


def encode_text_batch(texts: list[str]) -> list[list[float]]:
    """Encode a batch of text strings efficiently."""
    encoder = load_text_encoder()
    return encoder.encode(texts).tolist()


def encode_image(image_path: str) -> list[float]:
    """Encode an image file into a 512-dimensional CLIP vector."""
    model = load_clip_model()
    img = Image.open(image_path).convert("RGB")
    return model.encode(img).tolist()


def encode_query_for_images(query: str) -> list[float]:
    """
    Encode a text query into CLIP space for cross-modal image search.
    CLIP aligns text and image embeddings in the same space,
    so a text query can retrieve semantically matching images.

    Returns a 512-d CLIP vector (CLIP_VECTOR_DIM) — the SAME dimension and space
    as `encode_image`, which is why a text query can match stored image vectors.
    """
    model = load_clip_model()
    return model.encode(query).tolist()
