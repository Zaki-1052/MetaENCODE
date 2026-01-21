# tests/test_api/test_encode_client.py
"""Tests for the ENCODE API client."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.api.encode_client import EncodeClient, RateLimiter


class TestRateLimiter:
    """Test suite for RateLimiter."""

    def test_rate_limiter_initialization(self):
        """Test that rate limiter initializes correctly."""
        limiter = RateLimiter(max_requests=10, window_seconds=1.0)
        assert limiter.max_requests == 10
        assert limiter.window == 1.0

    def test_rate_limiter_allows_requests_under_limit(self):
        """Test that rate limiter allows requests under the limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=1.0)
        # Should not block for fewer than max_requests
        for _ in range(4):
            limiter.wait_if_needed()
        # If we get here without blocking too long, test passes


class TestEncodeClient:
    """Test suite for EncodeClient."""

    def test_client_initialization(self):
        """Test that client initializes correctly."""
        client = EncodeClient()
        assert client.BASE_URL == "https://www.encodeproject.org"
        assert client.RATE_LIMIT == 10
        assert client._rate_limiter is not None

    def test_build_search_url(self):
        """Test URL building."""
        client = EncodeClient()
        url = client._build_search_url({"type": "Experiment", "limit": 10})
        assert "https://www.encodeproject.org/search/?" in url
        assert "type=Experiment" in url
        assert "limit=10" in url

    def test_parse_experiment(self, sample_experiment_data):
        """Test experiment parsing."""
        client = EncodeClient()
        parsed = client._parse_experiment(sample_experiment_data)
        assert parsed["accession"] == "ENCSR000AAA"
        assert parsed["assay_term_name"] == "ChIP-seq"
        assert parsed["replicate_count"] == 2
        assert parsed["file_count"] == 2

    @patch("requests.Session.get")
    def test_fetch_experiments_returns_dataframe(
        self, mock_get, sample_experiment_data
    ):
        """Test that fetch_experiments returns a DataFrame."""
        mock_response = Mock()
        mock_response.json.return_value = {"@graph": [sample_experiment_data]}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EncodeClient()
        df = client.fetch_experiments(limit=5)

        assert isinstance(df, pd.DataFrame)
        assert "accession" in df.columns
        assert len(df) == 1
        assert df.iloc[0]["accession"] == "ENCSR000AAA"

    @patch("requests.Session.get")
    def test_fetch_experiment_by_accession(self, mock_get, sample_experiment_data):
        """Test fetching single experiment by accession."""
        mock_response = Mock()
        mock_response.json.return_value = sample_experiment_data
        mock_response.raise_for_status = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        client = EncodeClient()
        result = client.fetch_experiment_by_accession("ENCSR000AAA")

        assert "accession" in result
        assert result["accession"] == "ENCSR000AAA"

    @patch("requests.Session.get")
    def test_search_returns_results(self, mock_get, sample_experiment_data):
        """Test that search returns matching results."""
        mock_response = Mock()
        mock_response.json.return_value = {"@graph": [sample_experiment_data]}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        client = EncodeClient()
        results = client.search("K562", limit=5)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1

    @patch("requests.Session.get")
    def test_fetch_experiment_not_found(self, mock_get):
        """Test that ValueError is raised for non-existent accession."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        client = EncodeClient()
        with pytest.raises(ValueError, match="not found"):
            client.fetch_experiment_by_accession("NONEXISTENT")
