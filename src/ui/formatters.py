# src/ui/formatters.py
"""Display formatting utilities for MetaENCODE UI.

This module provides formatting functions for displaying data in the UI,
such as organism names with genome assemblies and ENCODE URL generation.
"""

import pandas as pd
import streamlit as st

from src.ui.vocabularies import get_organism_display

# ENCODE Portal base URL
ENCODE_BASE_URL = "https://www.encodeproject.org"


def get_encode_experiment_url(accession: str) -> str:
    """Generate ENCODE experiment URL from accession.

    Args:
        accession: ENCODE accession ID (e.g., ENCSR000AKS).

    Returns:
        Full URL to the experiment page on ENCODE portal.
    """
    if not accession:
        return ""
    return f"{ENCODE_BASE_URL}/experiments/{accession}/"


def format_accession_as_link(accession: str) -> str:
    """Format accession as markdown hyperlink to ENCODE portal.

    Args:
        accession: ENCODE accession ID (e.g., ENCSR000AKS).

    Returns:
        Markdown link string (e.g., "[ENCSR000AKS](https://...)").
    """
    if not accession:
        return "N/A"
    url = get_encode_experiment_url(accession)
    return f"[{accession}]({url})"


def format_organism_display(organism: str) -> str:
    """Format organism name with genome assembly label.

    Delegates to get_organism_display for consistent formatting
    across all organisms, including those not in the known list.

    Args:
        organism: Organism name (common or scientific).

    Returns:
        Formatted string with assembly (e.g., "Human [hg38]") or
        just the organism name if no assembly info available.
    """
    if not organism:
        return "N/A"
    return get_organism_display(organism)


def truncate_text(text: str, max_length: int = 80) -> str:
    """Truncate text to specified length with ellipsis.

    Args:
        text: Text to truncate.
        max_length: Maximum length before truncation.

    Returns:
        Truncated text with '...' suffix if needed.
    """
    text_str = str(text) if text else ""
    if len(text_str) > max_length:
        return text_str[:max_length] + "..."
    return text_str


# Shared column display labels for results DataFrames
DISPLAY_COLUMN_LABELS = {
    "similarity_score": "Similarity",
    "accession": "Accession",
    "assay_term_name": "Assay",
    "organism": "Organism [Assembly]",
    "biosample_term_name": "Biosample",
    "description": "Description",
}


def format_results_for_display(
    df: pd.DataFrame,
    display_columns: list[str],
    description_max_length: int = 80,
) -> pd.DataFrame:
    """Prepare a DataFrame of experiments for st.dataframe display.

    Applies organism formatting, description truncation, accession URL
    linkification, and column renaming.

    Args:
        df: Source DataFrame with experiment metadata.
        display_columns: Ordered list of columns to include.
        description_max_length: Max chars before description truncation.

    Returns:
        Formatted DataFrame with renamed columns ready for display.
    """
    cols = [c for c in display_columns if c in df.columns]
    display_df = df[cols].copy()

    if "similarity_score" in display_df.columns:
        display_df["similarity_score"] = display_df["similarity_score"].apply(
            lambda x: f"{x:.3f}"
        )
    if "organism" in display_df.columns:
        display_df["organism"] = display_df["organism"].apply(format_organism_display)
    if "description" in display_df.columns:
        display_df["description"] = display_df["description"].apply(
            lambda x: truncate_text(str(x), description_max_length)
        )
    display_df["accession"] = df["accession"].apply(get_encode_experiment_url)

    display_df = display_df.rename(
        columns={
            k: v for k, v in DISPLAY_COLUMN_LABELS.items() if k in display_df.columns
        }
    )
    return display_df


def get_accession_link_column_config() -> dict:
    """Return the Accession LinkColumn configuration for st.dataframe.

    Returns:
        Dict suitable for st.dataframe column_config parameter.
    """
    return {
        "Accession": st.column_config.LinkColumn(
            "Accession",
            display_text=r"experiments/(ENC[^/]+)/",
            help="Click to open on ENCODE Portal",
        ),
    }
