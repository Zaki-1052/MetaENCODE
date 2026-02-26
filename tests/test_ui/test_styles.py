# tests/test_ui/test_styles.py
"""Tests for src/ui/styles.py."""

from unittest.mock import patch

import pytest

from src.ui.styles import (
    ACTIVE_TAB_INDICATOR,
    COLOR_DARK_TEXT,
    COLOR_LIGHT_GREEN,
    COLOR_OLIVE,
    COLOR_PRIMARY_GREEN,
    COLOR_TAN,
    LAYOUT_STYLES,
    SECTION_HEADER_STYLES,
    SIDEBAR_STYLES,
    TAB_NAV_STYLES,
    VIZ_OPTIONS_STYLES,
)


class TestColorConstants:
    """Verify color constants are valid hex codes."""

    @pytest.mark.parametrize(
        "color",
        [
            COLOR_PRIMARY_GREEN,
            COLOR_LIGHT_GREEN,
            COLOR_TAN,
            COLOR_DARK_TEXT,
            COLOR_OLIVE,
        ],
    )
    def test_valid_hex_color(self, color):
        """Color constants should be valid 7-char hex strings."""
        assert color.startswith("#")
        assert len(color) == 7
        # Should not raise ValueError if valid hex
        int(color[1:], 16)


class TestStyleBlocks:
    """Verify style blocks are well-formed."""

    @pytest.mark.parametrize(
        "style_block",
        [
            LAYOUT_STYLES,
            TAB_NAV_STYLES,
            SIDEBAR_STYLES,
            SECTION_HEADER_STYLES,
            VIZ_OPTIONS_STYLES,
        ],
    )
    def test_contains_style_tags(self, style_block):
        """All style blocks should contain opening and closing <style> tags."""
        assert "<style>" in style_block
        assert "</style>" in style_block

    def test_layout_styles_has_block_container(self):
        """Layout styles should define .block-container."""
        assert ".block-container" in LAYOUT_STYLES

    def test_layout_styles_has_title(self):
        """Layout styles should define .title."""
        assert ".title" in LAYOUT_STYLES

    def test_tab_nav_styles_has_card_container(self):
        """Tab nav styles should reference card-container."""
        assert "card-container" in TAB_NAV_STYLES

    def test_sidebar_styles_has_sidebar_selector(self):
        """Sidebar styles should target stSidebar."""
        assert "stSidebar" in SIDEBAR_STYLES

    def test_section_header_has_header_class(self):
        """Section header styles should define .section-header."""
        assert ".section-header" in SECTION_HEADER_STYLES

    def test_section_header_has_subtitle_class(self):
        """Section header styles should define .section-subtitle."""
        assert ".section-subtitle" in SECTION_HEADER_STYLES

    def test_viz_options_has_marker(self):
        """Viz options styles should reference options-container-marker."""
        assert "options-container-marker" in VIZ_OPTIONS_STYLES


class TestActiveTabIndicator:
    """Verify active tab indicator HTML."""

    def test_contains_primary_green(self):
        """Active tab indicator should use the primary green color."""
        assert COLOR_PRIMARY_GREEN in ACTIVE_TAB_INDICATOR

    def test_is_div_element(self):
        """Active tab indicator should be a div element."""
        assert ACTIVE_TAB_INDICATOR.startswith("<div")
        assert ACTIVE_TAB_INDICATOR.endswith("</div>")

    def test_has_border_bottom(self):
        """Active tab indicator should use border-bottom styling."""
        assert "border-bottom" in ACTIVE_TAB_INDICATOR


class TestStyleInjectionFunctions:
    """Test that style injection functions call st.markdown correctly."""

    def test_apply_global_styles(self):
        """apply_global_styles should inject layout and section header CSS."""
        with patch("src.ui.styles.st") as mock_st:
            from src.ui.styles import apply_global_styles

            apply_global_styles()
            assert mock_st.markdown.call_count == 2

    def test_apply_sidebar_styles(self):
        """apply_sidebar_styles should inject sidebar CSS."""
        with patch("src.ui.styles.st") as mock_st:
            from src.ui.styles import apply_sidebar_styles

            apply_sidebar_styles()
            mock_st.markdown.assert_called_once_with(
                SIDEBAR_STYLES, unsafe_allow_html=True
            )

    def test_apply_viz_options_styles(self):
        """apply_viz_options_styles should inject visualization CSS."""
        with patch("src.ui.styles.st") as mock_st:
            from src.ui.styles import apply_viz_options_styles

            apply_viz_options_styles()
            mock_st.markdown.assert_called_once_with(
                VIZ_OPTIONS_STYLES, unsafe_allow_html=True
            )
