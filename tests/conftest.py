# tests/conftest.py
"""Pytest fixtures and configuration for MetaENCODE tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_experiment_data() -> dict:
    """Sample experiment data from ENCODE API response."""
    return {
        "accession": "ENCSR000AAA",
        "description": "ChIP-seq on human K562 cells targeting H3K27ac",
        "assay_term_name": "ChIP-seq",
        "biosample_ontology": {"term_name": "K562"},
        "lab": "/labs/encode-consortium/",
        "status": "released",
        "files": ["/files/ENCFF001AAA/", "/files/ENCFF002AAA/"],
        "replicates": ["/replicates/1/", "/replicates/2/"],
    }


@pytest.fixture
def sample_experiments_df() -> pd.DataFrame:
    """Sample DataFrame of experiments for testing."""
    return pd.DataFrame(
        {
            "accession": ["ENCSR000AAA", "ENCSR000BBB", "ENCSR000CCC"],
            "description": [
                "ChIP-seq on human K562 cells targeting H3K27ac",
                "RNA-seq of mouse liver tissue",
                "ATAC-seq on human HepG2 cells",
            ],
            "assay_term_name": ["ChIP-seq", "RNA-seq", "ATAC-seq"],
            "organism": ["human", "mouse", "human"],
            "biosample": ["K562", "liver", "HepG2"],
            "lab": ["encode-consortium", "encode-consortium", "encode-consortium"],
        }
    )


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Sample embeddings array for testing."""
    np.random.seed(42)
    return np.random.randn(10, 384)  # 10 samples, 384 dimensions (MiniLM)


@pytest.fixture
def sample_embedding_single() -> np.ndarray:
    """Single sample embedding for testing."""
    np.random.seed(42)
    return np.random.randn(384)
