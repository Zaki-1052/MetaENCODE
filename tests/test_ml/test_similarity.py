# tests/test_ml/test_similarity.py
"""Tests for similarity computation engine."""

import numpy as np
import pytest

from src.ml.similarity import SimilarityEngine


class TestSimilarityEngine:
    """Test suite for SimilarityEngine."""

    def test_engine_initialization(self):
        """Test that engine initializes correctly."""
        engine = SimilarityEngine()
        assert engine.metric == "cosine"
        assert engine._fitted is False

        engine_euclidean = SimilarityEngine(metric="euclidean")
        assert engine_euclidean.metric == "euclidean"

    @pytest.mark.skip(reason="Not implemented yet")
    def test_fit_sets_fitted_flag(self, sample_embeddings):
        """Test that fit() sets the fitted flag."""
        engine = SimilarityEngine()
        engine.fit(sample_embeddings)
        assert engine._fitted is True

    @pytest.mark.skip(reason="Not implemented yet")
    def test_find_similar_returns_correct_count(
        self, sample_embeddings, sample_embedding_single
    ):
        """Test that find_similar returns requested number of results."""
        engine = SimilarityEngine()
        engine.fit(sample_embeddings)
        results = engine.find_similar(sample_embedding_single, n=5)
        assert len(results) == 5

    @pytest.mark.skip(reason="Not implemented yet")
    def test_compute_similarity_range(self, sample_embeddings):
        """Test that cosine similarity is between 0 and 1."""
        engine = SimilarityEngine()
        similarity = engine.compute_similarity(
            sample_embeddings[0], sample_embeddings[1]
        )
        assert 0 <= similarity <= 1

    @pytest.mark.skip(reason="Not implemented yet")
    def test_similarity_matrix_is_symmetric(self, sample_embeddings):
        """Test that similarity matrix is symmetric."""
        engine = SimilarityEngine()
        engine.fit(sample_embeddings)
        matrix = engine.compute_similarity_matrix()
        assert np.allclose(matrix, matrix.T)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_find_similar_raises_if_not_fitted(self, sample_embedding_single):
        """Test that find_similar raises error if not fitted."""
        engine = SimilarityEngine()
        with pytest.raises(ValueError):
            engine.find_similar(sample_embedding_single, n=5)
