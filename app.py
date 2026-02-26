# app.py
"""MetaENCODE: ENCODE Dataset Similarity Search Application.

This Streamlit application enables researchers to discover related ENCODE
datasets through metadata-driven similarity scoring. Users can search for
datasets, select a seed dataset, and explore similar experiments through
interactive visualizations.

Run with: streamlit run app.py
"""

import streamlit as st

from src.ui.components.session import (
    init_session_state,
    load_cached_data_into_session,
    load_selection_history_into_session,
)
from src.ui.sidebar import render_sidebar
from src.ui.styles import ACTIVE_TAB_INDICATOR, TAB_NAV_STYLES, apply_global_styles
from src.ui.tabs import render_search_tab, render_similar_tab, render_visualize_tab

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="MetaENCODE",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()

# Tab definitions: (button_label, state_key, widget_key)
_TABS = [
    ("Search & Select", "Search", "btn_search"),
    ("Similar Datasets", "Similar", "btn_similar"),
    ("Visualize", "Visualize", "btn_visualize"),
]


def _render_tab_bar() -> None:
    """Render the tab navigation bar with active tab indicator."""
    with st.container():
        st.markdown('<div class="card-container"></div>', unsafe_allow_html=True)
        cols = st.columns(len(_TABS))

        for col, (label, tab_key, btn_key) in zip(cols, _TABS):
            with col:
                if st.button(label, use_container_width=True, key=btn_key):
                    st.session_state.active_tab = tab_key
                    st.rerun()
                if st.session_state.active_tab == tab_key:
                    st.markdown(ACTIVE_TAB_INDICATOR, unsafe_allow_html=True)


def render_main_content() -> None:
    """Render main content area with tab navigation and selected tab."""
    st.markdown(
        "<div class='title'>MetaENCODE</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "**Discover related ENCODE datasets and visualize dataset similarity.**"
    )

    st.markdown(TAB_NAV_STYLES, unsafe_allow_html=True)

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Search"

    _render_tab_bar()
    st.divider()

    # Display content for the active tab
    if st.session_state.active_tab == "Search":
        render_search_tab()
    elif st.session_state.active_tab == "Similar":
        render_similar_tab()
    elif st.session_state.active_tab == "Visualize":
        render_visualize_tab()


def main() -> None:
    """Main application entry point."""
    init_session_state()
    load_cached_data_into_session()
    load_selection_history_into_session()
    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
