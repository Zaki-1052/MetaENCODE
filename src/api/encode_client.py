# src/api/encode_client.py
"""ENCODE REST API client for fetching experiment metadata.

This module provides a client for interacting with the ENCODE Data Coordination
Center (DCC) REST API. It handles rate limiting, pagination, and conversion of
JSON responses to pandas DataFrames.

Reference: https://www.encodeproject.org/help/rest-api/
"""

from typing import Optional

import pandas as pd
import requests


class EncodeClient:
    """Client for interacting with the ENCODE REST API.

    Attributes:
        BASE_URL: Base URL for the ENCODE API.
        RATE_LIMIT: Maximum requests per second (API limit is 10).

    Example:
        >>> client = EncodeClient()
        >>> experiments = client.fetch_experiments(assay_type="ChIP-seq", limit=100)
        >>> print(experiments.head())
    """

    BASE_URL = "https://www.encodeproject.org"
    RATE_LIMIT = 10  # requests per second
    HEADERS = {"accept": "application/json"}

    def __init__(self) -> None:
        """Initialize the ENCODE API client."""
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)

    def fetch_experiments(
        self,
        assay_type: Optional[str] = None,
        organism: Optional[str] = None,
        biosample: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch experiment metadata from ENCODE API.

        Args:
            assay_type: Filter by assay type (e.g., "ChIP-seq", "RNA-seq").
            organism: Filter by organism (e.g., "human", "mouse").
            biosample: Filter by biosample term name (e.g., "K562").
            limit: Maximum number of experiments to fetch (default 100, use 0 for all).

        Returns:
            DataFrame containing experiment metadata with columns:
            accession, description, assay_term_name, biosample_ontology, lab, status.

        Raises:
            requests.RequestException: If the API request fails.
        """
        raise NotImplementedError("fetch_experiments not yet implemented")

    def fetch_experiment_by_accession(self, accession: str) -> dict:
        """Fetch a single experiment by its accession number.

        Args:
            accession: ENCODE accession number (e.g., "ENCSR000AKS").

        Returns:
            Dictionary containing the full experiment metadata.

        Raises:
            requests.RequestException: If the API request fails.
            ValueError: If the accession is not found.
        """
        raise NotImplementedError("fetch_experiment_by_accession not yet implemented")

    def search(
        self,
        search_term: str,
        object_type: str = "Experiment",
        limit: int = 25,
    ) -> pd.DataFrame:
        """Search ENCODE for datasets matching a search term.

        Args:
            search_term: Text to search for.
            object_type: Type of object to search (default "Experiment").
            limit: Maximum results to return.

        Returns:
            DataFrame containing search results.
        """
        raise NotImplementedError("search not yet implemented")

    def _build_search_url(self, params: dict) -> str:
        """Build a search URL with the given parameters.

        Args:
            params: Dictionary of query parameters.

        Returns:
            Fully formed search URL.
        """
        raise NotImplementedError("_build_search_url not yet implemented")

    def _parse_experiment(self, data: dict) -> dict:
        """Parse raw experiment JSON into standardized format.

        Args:
            data: Raw JSON data from API response.

        Returns:
            Dictionary with standardized field names.
        """
        raise NotImplementedError("_parse_experiment not yet implemented")
