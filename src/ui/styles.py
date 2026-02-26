# src/ui/styles.py
"""Centralized CSS styles for MetaENCODE Streamlit UI.

All inline CSS blocks are consolidated here as named constants.
Each constant is a complete <style>...</style> block ready to be
injected via st.markdown(..., unsafe_allow_html=True).
"""

import streamlit as st

# --- Color Palette ---
COLOR_PRIMARY_GREEN = "#618B4A"
COLOR_LIGHT_GREEN = "#C6DEB4"
COLOR_TAN = "#afbc88"
COLOR_DARK_TEXT = "#31333F"
COLOR_OLIVE = "#8e9a6a"

# --- Global Styles (layout and title) ---
LAYOUT_STYLES = """
<style>
    /* Block container padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 4rem;
        padding-right: 4rem;
    }

    /* App title */
    .title {
        font-size: 3.0rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
</style>
"""

# --- Tab Navigation Styles (card buttons) ---
TAB_NAV_STYLES = """
<style>
    [data-testid="stVerticalBlock"] > div:has(div.card-container) {
        margin-top: -10px !important;
    }

    /* Base Card */
    [data-testid="stVerticalBlock"] > div:has(div.card-container) button {
        height: 4.3rem;
        border-radius: 8px;
        border: 2px solid #afbc88;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.2s ease-in-out;
        color: #31333F;
    }

    hr {
        margin-top: 3px !important;
        margin-bottom: 5px !important;
    }

    /* Button Text */
    [data-testid="stVerticalBlock"] > div:has(div.card-container) button p {
        font-size: 1.4rem;
        font-weight: 400;
    }

    /* Hover Effect */
    [data-testid="stVerticalBlock"] > div:has(div.card-container) button:hover {
        border-color: #618B4A;
        color: #618B4A;
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
</style>
"""

# --- Sidebar Styles ---
SIDEBAR_STYLES = """
<style>
    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #C6DEB4;
    }

    /* Adjust padding in sidebar */
    [data-testid="stSidebar"] [data-testid="stHeading"] h1{
        padding-top: 0px;
        margin-top: -5px;
        font-size: 1.8rem;
    }

    [data-testid="stSidebarContent"] hr {
        border-top: 1px solid #8e9a6a
    }

    [data-testid="stSidebarContent"] div[role="slider"] {
        background-color: #000000 !important;
        border: 2px solid #000000;
    }
</style>
"""

# --- Shared Section Header (deduplicated from search, similar, visualize tabs) ---
SECTION_HEADER_STYLES = """
<style>
    .section-header {
        font-size: 1.9rem;
        font-weight: 650;
        margin-bottom: 0.25rem;
    }

    .section-subtitle {
        font-size: 1.6rem;
        font-weight: 550;
        margin-top: 1.0rem;
        margin-bottom: 0.4rem;
    }
</style>
"""

# --- Visualize Tab Options Panel ---
VIZ_OPTIONS_STYLES = """
<style>
    .viz-subtitle {
        font-size: 1.4rem;
        font-weight: 550;
        margin-bottom: 0.4rem;
    }

    /* Options panel */
    [data-testid="stColumn"]:has(.options-container-marker) {
        background-color: #C6DEB4 !important;
        padding: 25px !important;
        border-radius: 15px !important;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05) !important;
    }

    [data-testid="stColumn"]:has(.options-container-marker) [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-radius: 8px !important;
    }
</style>
"""

# --- Active Tab Indicator (inline HTML, not a <style> block) ---
ACTIVE_TAB_INDICATOR = (
    "<div style='border-bottom: 5px solid #618B4A; margin-top: -15px;'></div>"
)


def apply_global_styles() -> None:
    """Inject all app-wide CSS once at the top of the render cycle."""
    st.markdown(LAYOUT_STYLES, unsafe_allow_html=True)
    st.markdown(SECTION_HEADER_STYLES, unsafe_allow_html=True)


def apply_sidebar_styles() -> None:
    """Inject sidebar-scoped CSS."""
    st.markdown(SIDEBAR_STYLES, unsafe_allow_html=True)


def apply_viz_options_styles() -> None:
    """Inject visualization options panel CSS."""
    st.markdown(VIZ_OPTIONS_STYLES, unsafe_allow_html=True)
