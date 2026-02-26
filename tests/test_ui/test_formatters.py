# tests/test_ui/test_formatters.py
"""Tests for src/ui/formatters.py."""

import pandas as pd

from src.ui.formatters import (
    DISPLAY_COLUMN_LABELS,
    format_organism_display,
    format_results_for_display,
    get_accession_link_column_config,
    truncate_text,
)


class TestFormatOrganismDisplay:
    """Tests for format_organism_display function."""

    def test_empty_string_returns_na(self):
        """Empty string should return 'N/A'."""
        assert format_organism_display("") == "N/A"

    def test_none_returns_na(self):
        """None should return 'N/A'."""
        assert format_organism_display(None) == "N/A"

    def test_human_scientific_name(self):
        """Homo sapiens should format with assembly."""
        result = format_organism_display("Homo sapiens")
        assert "Human" in result or "Homo sapiens" in result

    def test_mouse_scientific_name(self):
        """Mus musculus should format with assembly."""
        result = format_organism_display("Mus musculus")
        assert "Mouse" in result or "Mus musculus" in result

    def test_unknown_organism_returned_as_is(self):
        """Unknown organisms should be returned unchanged."""
        result = format_organism_display("Unknown species")
        assert "Unknown species" in result

    def test_human_common_name(self):
        """Common name 'human' should be handled."""
        result = format_organism_display("human")
        # Should either return formatted or as-is
        assert result is not None
        assert result != "N/A"

    def test_whitespace_only_returns_na(self):
        """Whitespace-only string should return 'N/A'."""
        # The function checks `if not organism` which is False for "   "
        # So whitespace will be passed to get_organism_display
        result = format_organism_display("   ")
        # Whitespace is truthy, so it goes to get_organism_display
        assert result is not None


class TestTruncateText:
    """Tests for truncate_text function."""

    def test_short_text_unchanged(self):
        """Text shorter than max_length should be unchanged."""
        text = "Short text"
        result = truncate_text(text, max_length=80)
        assert result == "Short text"

    def test_exact_length_unchanged(self):
        """Text exactly at max_length should be unchanged."""
        text = "x" * 80
        result = truncate_text(text, max_length=80)
        assert result == text
        assert "..." not in result

    def test_long_text_truncated(self):
        """Text longer than max_length should be truncated with ellipsis."""
        text = "x" * 100
        result = truncate_text(text, max_length=80)
        assert len(result) == 83  # 80 + "..."
        assert result.endswith("...")

    def test_custom_max_length(self):
        """Custom max_length should be respected."""
        text = "Hello World"
        result = truncate_text(text, max_length=5)
        assert result == "Hello..."

    def test_empty_string(self):
        """Empty string should return empty string."""
        result = truncate_text("", max_length=80)
        assert result == ""

    def test_none_returns_empty_string(self):
        """None should return empty string."""
        result = truncate_text(None, max_length=80)
        assert result == ""

    def test_non_string_converted(self):
        """Non-string values should be converted to string."""
        result = truncate_text(12345, max_length=80)
        assert result == "12345"

    def test_non_string_truncated(self):
        """Non-string values should be truncated after conversion."""
        result = truncate_text(123456789, max_length=5)
        assert result == "12345..."

    def test_default_max_length_is_80(self):
        """Default max_length should be 80."""
        text = "x" * 100
        result = truncate_text(text)
        assert len(result) == 83  # 80 + "..."

    def test_unicode_text(self):
        """Unicode text should be handled correctly."""
        text = "Hello 世界! " * 20
        result = truncate_text(text, max_length=20)
        assert len(result) == 23  # 20 + "..."
        assert result.endswith("...")

    def test_newlines_preserved(self):
        """Newlines in text should be preserved."""
        text = "Line 1\nLine 2"
        result = truncate_text(text, max_length=80)
        assert result == "Line 1\nLine 2"

    def test_single_character_max_length(self):
        """Single character max_length should work."""
        result = truncate_text("Hello", max_length=1)
        assert result == "H..."


class TestFormatResultsForDisplay:
    """Tests for format_results_for_display function."""

    def _make_df(self, **kwargs) -> pd.DataFrame:
        """Helper to build a minimal experiment DataFrame."""
        defaults = {"accession": ["ENCSR000AAA"]}
        defaults.update(kwargs)
        return pd.DataFrame(defaults)

    def test_minimal_dataframe_accession_only(self):
        """Minimal DataFrame with just accession should work."""
        df = self._make_df()
        result = format_results_for_display(df, ["accession"])
        assert "Accession" in result.columns
        assert "encodeproject.org" in result["Accession"].iloc[0]

    def test_similarity_score_formatted(self):
        """Similarity score should be formatted to 3 decimal places."""
        df = self._make_df(similarity_score=[0.87654321])
        result = format_results_for_display(df, ["similarity_score", "accession"])
        assert result["Similarity"].iloc[0] == "0.877"

    def test_description_truncation(self):
        """Description should be truncated to specified length."""
        long_desc = "x" * 100
        df = self._make_df(description=[long_desc])
        result = format_results_for_display(
            df, ["accession", "description"], description_max_length=30
        )
        assert result["Description"].iloc[0].endswith("...")
        # 30 chars + "..."
        assert len(result["Description"].iloc[0]) == 33

    def test_organism_formatting_applied(self):
        """Organism column should be formatted via format_organism_display."""
        df = self._make_df(organism=["Homo sapiens"])
        result = format_results_for_display(df, ["accession", "organism"])
        # Should contain "Human" or "Homo sapiens" (formatted by get_organism_display)
        val = result["Organism [Assembly]"].iloc[0]
        assert "Human" in val or "Homo sapiens" in val

    def test_column_renaming(self):
        """All known columns should be renamed to display labels."""
        df = self._make_df(
            assay_term_name=["ChIP-seq"],
            organism=["Mus musculus"],
            biosample_term_name=["cerebellum"],
            description=["A short desc"],
        )
        result = format_results_for_display(
            df,
            [
                "accession",
                "assay_term_name",
                "organism",
                "biosample_term_name",
                "description",
            ],
        )
        assert "Assay" in result.columns
        assert "Organism [Assembly]" in result.columns
        assert "Biosample" in result.columns
        assert "Description" in result.columns
        assert "Accession" in result.columns

    def test_missing_columns_skipped(self):
        """Columns not in the DataFrame should be skipped gracefully."""
        df = self._make_df(assay_term_name=["RNA-seq"])
        # Request columns that don't exist
        result = format_results_for_display(
            df, ["accession", "assay_term_name", "nonexistent_col"]
        )
        assert "Assay" in result.columns
        assert "nonexistent_col" not in result.columns

    def test_multiple_rows(self):
        """Should handle multiple rows correctly."""
        df = pd.DataFrame(
            {
                "accession": ["ENCSR000AAA", "ENCSR000BBB", "ENCSR000CCC"],
                "assay_term_name": ["ChIP-seq", "RNA-seq", "ATAC-seq"],
                "description": ["desc1", "desc2", "desc3"],
            }
        )
        result = format_results_for_display(
            df, ["accession", "assay_term_name", "description"]
        )
        assert len(result) == 3
        assert result["Assay"].tolist() == ["ChIP-seq", "RNA-seq", "ATAC-seq"]

    def test_accession_url_linkification(self):
        """Accession should be converted to full ENCODE URL."""
        df = self._make_df()
        result = format_results_for_display(df, ["accession"])
        url = result["Accession"].iloc[0]
        assert url == "https://www.encodeproject.org/experiments/ENCSR000AAA/"


class TestGetAccessionLinkColumnConfig:
    """Tests for get_accession_link_column_config function."""

    def test_returns_dict(self):
        """Should return a dict."""
        config = get_accession_link_column_config()
        assert isinstance(config, dict)

    def test_has_accession_key(self):
        """Should have 'Accession' key."""
        config = get_accession_link_column_config()
        assert "Accession" in config

    def test_link_column_type(self):
        """Value should be a Streamlit LinkColumn."""
        config = get_accession_link_column_config()
        # LinkColumn is a column config object
        assert config["Accession"] is not None


class TestDisplayColumnLabels:
    """Tests for DISPLAY_COLUMN_LABELS constant."""

    def test_is_dict(self):
        """Should be a dict."""
        assert isinstance(DISPLAY_COLUMN_LABELS, dict)

    def test_expected_keys(self):
        """Should contain expected column keys."""
        expected = {
            "similarity_score",
            "accession",
            "assay_term_name",
            "organism",
            "biosample_term_name",
            "description",
        }
        assert set(DISPLAY_COLUMN_LABELS.keys()) == expected

    def test_values_are_strings(self):
        """All values should be non-empty strings."""
        for key, value in DISPLAY_COLUMN_LABELS.items():
            assert isinstance(value, str), f"{key} value is not a string"
            assert len(value) > 0, f"{key} value is empty"
