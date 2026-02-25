# src/ui/tabs/similar.py
"""Similar datasets tab for MetaENCODE."""

import pandas as pd
import streamlit as st

from src.ui.components.initializers import get_embedding_generator
from src.ui.formatters import (
    format_accession_as_link,
    format_results_for_display,
    get_accession_link_column_config,
)


def render_similar_tab() -> None:
    """Render the similar datasets tab."""

    st.markdown(
        "<div class='section-header'>Similar Datasets</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.selected_dataset is None:
        st.info("Select a dataset first to find similar experiments.")
        st.session_state.similar_datasets = None
        st.session_state.last_computed_accession = None
        return

    # Check if we have loaded data
    if st.session_state.metadata_df is None or st.session_state.embeddings is None:
        st.warning(
            "No data loaded. Please ensure the precomputed cache files exist in data/cache/."
        )
        return

    selected = st.session_state.selected_dataset
    accession = selected.get("accession", "Unknown")

    last_acc = st.session_state.get("last_computed_accession")
    if accession != last_acc:
        with st.spinner(f"Computing datasets similar to {accession}..."):
            try:
                embedder = get_embedding_generator()
                similarity_engine = st.session_state.similarity_engine
                feature_combiner = st.session_state.feature_combiner

                if similarity_engine is None:
                    st.error("Similarity engine not initialized.")
                    return

                # --- Core Computation Logic ---
                text = f"{selected.get('description', '')} {selected.get('title', '')}"
                text_embedding = embedder.encode_single(text)

                if feature_combiner is not None and feature_combiner.is_fitted:
                    query_vector = feature_combiner.transform_single(
                        selected, text_embedding
                    )
                else:
                    query_vector = text_embedding

                # Find similar
                top_n = st.session_state.filter_state.max_results
                fetch_n = max(top_n * 3, 30)
                similar_df = similarity_engine.find_similar(
                    query_vector, n=fetch_n, exclude_self=True
                )

                # Get metadata
                metadata_df = st.session_state.metadata_df
                results = []
                for _, row in similar_df.iterrows():
                    idx = int(row["index"])
                    if idx < len(metadata_df):
                        meta = metadata_df.iloc[idx].to_dict()
                        meta["similarity_score"] = row["similarity_score"]
                        results.append(meta)

                # 4. Save to session state to prevent re-computation on every click
                st.session_state.similar_datasets = pd.DataFrame(results)
                st.session_state.last_computed_accession = accession

            except Exception as e:
                st.error(f"Error computing similarities: {e}")
                return

    accession_link = format_accession_as_link(accession)
    st.markdown(f"Finding datasets similar to: **{accession_link}**")

    # Get filter state
    filter_state = st.session_state.filter_state
    top_n = filter_state.max_results

    # Display similar datasets
    if st.session_state.similar_datasets is not None:
        similar = st.session_state.similar_datasets

        if not similar.empty:

            # Limit display to max_results; similarity ranking already applied (no additional filters)
            display_similar = similar.head(top_n)

            similar_display_cols = [
                "similarity_score",
                "accession",
                "assay_term_name",
                "organism",
                "biosample_term_name",
                "description",
            ]
            display_df = format_results_for_display(
                display_similar, similar_display_cols, description_max_length=60
            )
            column_config = get_accession_link_column_config()

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config=column_config,
            )
