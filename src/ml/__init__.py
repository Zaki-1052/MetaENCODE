# src/ml/__init__.py
"""Machine learning module for embeddings and similarity computation."""

from .embeddings import EmbeddingGenerator
from .similarity import SimilarityEngine

__all__ = ["EmbeddingGenerator", "SimilarityEngine"]
