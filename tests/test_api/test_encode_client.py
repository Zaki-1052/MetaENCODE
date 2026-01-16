# tests/test_api/test_encode_client.py
"""Tests for the ENCODE API client."""

import pytest

from src.api.encode_client import EncodeClient


class TestEncodeClient:
    """Test suite for EncodeClient."""

    def test_client_initialization(self):
        """Test that client initializes correctly."""
        client = EncodeClient()
        assert client.BASE_URL == "https://www.encodeproject.org"
        assert client.RATE_LIMIT == 10

    @pytest.mark.skip(reason="Not implemented yet")
    def test_fetch_experiments_returns_dataframe(self):
        """Test that fetch_experiments returns a DataFrame."""
        client = EncodeClient()
        df = client.fetch_experiments(limit=5)
        assert hasattr(df, "columns")
        assert "accession" in df.columns

    @pytest.mark.skip(reason="Not implemented yet")
    def test_fetch_experiment_by_accession(self, sample_experiment_data):
        """Test fetching single experiment by accession."""
        client = EncodeClient()
        result = client.fetch_experiment_by_accession("ENCSR000AAA")
        assert "accession" in result

    @pytest.mark.skip(reason="Not implemented yet")
    def test_search_returns_results(self):
        """Test that search returns matching results."""
        client = EncodeClient()
        results = client.search("K562", limit=5)
        assert len(results) > 0
