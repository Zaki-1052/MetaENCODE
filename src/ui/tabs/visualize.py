# src/ui/tabs/visualize.py
"""Visualization tab for MetaENCODE.

This module can be easily commented out or replaced with
a teammate's implementation.
"""

import streamlit as st

from src.ui.components.initializers import get_cache_manager
from src.ui.styles import apply_viz_options_styles
from src.visualization.plots import (
    DimensionalityReducer,
    PlotGenerator,
    percentile_range_filtering,
)

# Cache key prefix must match scripts/precompute_visualizations.py
_VIZ_CACHE_PREFIX = "viz_coords"


def _store_viz_state(
    coords_2d,
    metadata,
    method: str,
    mode: str,
    variance_ratio,
    slider_value,
    color_by: str | None = None,
) -> None:
    """Store visualization results in session state and invalidate cached figure."""
    st.session_state.coords_2d = coords_2d
    st.session_state.viz_metadata = metadata
    st.session_state.viz_reduction_method = method
    st.session_state.viz_mode = mode
    st.session_state.viz_variance_ratio = variance_ratio
    st.session_state.viz_slider_value = slider_value
    if color_by is not None:
        st.session_state.viz_color_by = color_by
    st.session_state.pop("viz_fig_key", None)


def _viz_cache_key(method: str, filtered: bool) -> str:
    """Build cache key for precomputed visualization coordinates."""
    suffix = "filtered" if filtered else "unfiltered"
    return f"{_VIZ_CACHE_PREFIX}_{method.replace('-', '_')}_{suffix}"


def _try_load_precomputed(method: str, filter_outliers: bool) -> bool:
    """Attempt to load precomputed visualization coordinates from cache.

    If precomputed data exists for the given method and filter setting,
    loads the 2D coordinates and associated metadata into session state.

    Args:
        method: Dimensionality reduction method ('pca', 'umap', or 't-sne').
        filter_outliers: Whether the outlier-filtered variant is requested.

    Returns:
        True if precomputed data was loaded, False otherwise.
    """
    cache_mgr = get_cache_manager()
    key = _viz_cache_key(method, filter_outliers)
    cached = cache_mgr.load(key)

    if cached is None:
        return False

    coords_2d = cached["coords_2d"]
    variance_ratio = cached.get("variance_ratio")
    metadata_df = st.session_state.metadata_df

    if filter_outliers:
        filter_mask = cache_mgr.load(f"{_VIZ_CACHE_PREFIX}_filter_mask")
        if filter_mask is not None:
            if len(filter_mask) != len(metadata_df):
                st.warning(
                    f"Precomputed filter mask ({len(filter_mask)} entries) doesn't match "
                    f"metadata ({len(metadata_df)} entries). Recomputing..."
                )
                return False
            filtered_metadata = metadata_df[filter_mask].reset_index(drop=True)
        else:
            _, mask = percentile_range_filtering(st.session_state.embeddings)
            filtered_metadata = metadata_df[mask].reset_index(drop=True)
    else:
        filtered_metadata = metadata_df

    if len(coords_2d) != len(filtered_metadata):
        st.warning(
            f"Precomputed coords ({len(coords_2d)} points) don't match "
            f"metadata ({len(filtered_metadata)} entries). Recomputing..."
        )
        return False

    _store_viz_state(
        coords_2d, filtered_metadata, method, "all_datasets", variance_ratio, None
    )
    return True


def generate_visualization(
    method: str, color_by: str, filter_outliers: bool = True
) -> None:
    """Generate 2D visualization of embeddings.

    For all-datasets mode, tries to load precomputed coordinates first.
    Falls back to on-the-fly computation if precomputed data is unavailable.

    Args:
        method: Dimensionality reduction method ('pca', 'umap', or 't-sne').
        color_by: Column to color points by.
        filter_outliers: Whether to filter outliers using percentile range.
    """
    # Try precomputed cache first (instant load)
    if _try_load_precomputed(method, filter_outliers):
        return

    # Fall back to on-the-fly computation
    with st.spinner(f"Computing {method.upper()} projection..."):
        try:
            embeddings = st.session_state.embeddings
            metadata_df = st.session_state.metadata_df

            if filter_outliers:
                filtered_embeddings, mask = percentile_range_filtering(embeddings)
                filtered_metadata = metadata_df[mask].reset_index(drop=True)
            else:
                filtered_embeddings = embeddings
                filtered_metadata = metadata_df

            reducer = DimensionalityReducer(method=method)
            coords_2d = reducer.fit_transform(filtered_embeddings)

            _store_viz_state(
                coords_2d,
                filtered_metadata,
                method,
                "all_datasets",
                reducer.variance_ratio_,
                None,
                color_by,
            )
        except Exception as e:
            st.error(f"Error generating visualization: {e}")
            st.info(
                "Tip: Try using PCA instead of UMAP, or ensure data is loaded first."
            )


def generate_similar_only_visualization(method: str, color_by: str) -> None:
    """Generate visualization of only the similar datasets.

    Args:
        method: Dimensionality reduction method ('pca', 'umap', or 't-sne').
        color_by: Column to color points by.
    """
    with st.spinner(f"Computing {method.upper()} projection for similar datasets..."):
        try:
            similar_df = st.session_state.similar_datasets
            if similar_df is None or similar_df.empty:
                st.error("No similar datasets found. Run a similarity search first.")
                return

            # Get the top N from filter state
            filter_state = st.session_state.filter_state
            top_n = filter_state.max_results
            similar_df = similar_df.head(top_n)

            # Get embeddings for only the similar datasets
            full_metadata = st.session_state.metadata_df
            full_embeddings = st.session_state.embeddings

            # Find indices of similar datasets in full metadata
            similar_accs = set(similar_df["accession"].tolist())
            indices = [
                i
                for i, acc in enumerate(full_metadata["accession"])
                if acc in similar_accs
            ]

            if not indices:
                st.error("Could not find embeddings for similar datasets.")
                return

            # Extract embeddings for similar datasets only
            similar_embeddings = full_embeddings[indices]
            similar_metadata = full_metadata.iloc[indices].reset_index(drop=True)

            # Add similarity scores to metadata for coloring option
            score_map = dict(
                zip(similar_df["accession"], similar_df["similarity_score"])
            )
            similar_metadata = similar_metadata.copy()
            similar_metadata["similarity_score"] = similar_metadata["accession"].map(
                score_map
            )

            # Run dimensionality reduction on just the similar embeddings
            reducer = DimensionalityReducer(method=method)
            coords_2d = reducer.fit_transform(similar_embeddings)

            _store_viz_state(
                coords_2d,
                similar_metadata,
                method,
                "similar_only",
                reducer.variance_ratio_,
                top_n,
                color_by,
            )

        except Exception as e:
            st.error(f"Error generating visualization: {e}")


def _auto_regenerate_similar_viz(current_method: str, current_color: str) -> None:
    """Auto-regenerate similar_only visualization when slider value changes.

    Detects when the sidebar slider has changed since the last viz was
    generated and re-runs generate_similar_only_visualization with the
    stored settings. Only triggers for similar_only mode when a viz
    already exists.
    """
    if st.session_state.get("viz_mode") != "similar_only":
        return
    if st.session_state.coords_2d is None:
        return
    if st.session_state.similar_datasets is None:
        return

    current_slider = st.session_state.filter_state.max_results
    last_slider = st.session_state.get("viz_slider_value")

    if last_slider is None or current_slider == last_slider:
        return

    # Use stored settings so slider change doesn't accidentally switch method
    method = st.session_state.get("viz_reduction_method", current_method)
    color = st.session_state.get("viz_color_by", current_color)
    generate_similar_only_visualization(method, color)


def _build_chart_figure(
    coords, viz_metadata, color_option, title, highlight_idx, variance, actual_method
):
    """Build a Plotly figure for the visualization chart."""
    plotter = PlotGenerator(reduction_method=actual_method)
    return plotter.scatter_plot(
        coords,
        viz_metadata,
        color_by=color_option,
        title=title,
        highlight_indices=highlight_idx,
        variance_ratio=variance,
    )


def render_visualize_tab() -> None:
    """Render the visualization tab."""
    apply_viz_options_styles()

    st.markdown(
        "<div class='section-header'>Dataset Visualization</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.metadata_df is None or st.session_state.embeddings is None:
        st.info(
            "No data loaded. Please ensure the precomputed cache files exist in data/cache/."
        )
        return

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown(
            '<div class="options-container-marker"></div>', unsafe_allow_html=True
        )
        st.markdown(
            "<div class='viz-subtitle'>Options</div>",
            unsafe_allow_html=True,
        )

        # View mode selector
        similar_available = st.session_state.similar_datasets is not None
        view_mode = st.radio(
            "View Mode",
            options=["similar_only", "all_datasets"],
            format_func=lambda x: {
                "similar_only": "Similar Datasets",
                "all_datasets": "Global Datasets",
            }.get(x, x),
            help="Show similar datasets from your search, or all datasets in the database",
            disabled=False,
        )

        # Warn if similar-only selected but no similar datasets
        if view_mode == "similar_only" and not similar_available:
            st.error(
                "Search for a dataset and run a similarity search first to use this view."
            )

        reduction_method = st.selectbox(
            "Reduction Method",
            options=["pca", "umap", "t-sne"],
            index=0,
            help="PCA is faster; UMAP/t-SNE preserve local structure better",
        )

        # Determine available color options based on metadata columns
        available_colors = ["assay_term_name", "organism"]
        if st.session_state.metadata_df is not None:
            # Add slim type color options if columns exist
            slim_color_columns = [
                "organ",
                "cell_type",
                "developmental_layer",
                "body_system",
            ]
            for col in slim_color_columns:
                if col in st.session_state.metadata_df.columns:
                    available_colors.append(col)
            # Add lab at the end if it exists
            if "lab" in st.session_state.metadata_df.columns:
                available_colors.append("lab")

        # Add similarity_score option if in similar-only mode with available data
        if view_mode == "similar_only" and similar_available:
            available_colors.insert(0, "similarity_score")

        color_display_names = {
            "similarity_score": "Similarity Score",
            "assay_term_name": "Assay Type",
            "organism": "Organism",
            "organ": "Organ System",
            "cell_type": "Cell Type",
            "developmental_layer": "Germ Layer",
            "body_system": "Body System",
            "lab": "Lab",
        }
        color_option = st.selectbox(
            "Color By",
            options=available_colors,
            format_func=lambda x: color_display_names.get(
                x, x.replace("_", " ").title()
            ),
        )

        # Outlier filtering option (only for all datasets mode)
        filter_outliers = False
        if view_mode == "all_datasets":
            filter_outliers = st.checkbox(
                "Filter Outliers",
                value=False,
                help="Remove points outside 5th-95th percentile range. "
                "Disable to show all datasets.",
            )

        # Generate button - different function based on view mode
        can_generate = view_mode == "all_datasets" or similar_available
        if st.button(
            "Generate Visualization", type="primary", disabled=not can_generate
        ):
            if view_mode == "similar_only":
                generate_similar_only_visualization(reduction_method, color_option)
            else:
                generate_visualization(reduction_method, color_option, filter_outliers)

    with col1:
        # Auto-regenerate similar_only viz when slider changes
        _auto_regenerate_similar_viz(reduction_method, color_option)

        # Auto-load visualization on first tab visit
        if (
            st.session_state.coords_2d is None
            and st.session_state.metadata_df is not None
        ):
            if view_mode == "similar_only" and similar_available:
                generate_similar_only_visualization(reduction_method, color_option)
            elif view_mode == "all_datasets":
                _try_load_precomputed(reduction_method, filter_outliers)

        if st.session_state.coords_2d is not None:
            viz_metadata = getattr(
                st.session_state, "viz_metadata", st.session_state.metadata_df
            )
            coords = st.session_state.coords_2d
            actual_method = st.session_state.get(
                "viz_reduction_method", reduction_method
            )
            stored_mode = st.session_state.get("viz_mode", "all_datasets")

            # Only highlight similar datasets in "all datasets" mode
            highlight_idx = None
            if (
                stored_mode == "all_datasets"
                and st.session_state.similar_datasets is not None
            ):
                similar_accs = set(
                    st.session_state.similar_datasets["accession"].tolist()
                )
                highlight_idx = [
                    i
                    for i, acc in enumerate(viz_metadata["accession"])
                    if acc in similar_accs
                ]

            # Build content-based cache key for the figure
            n_highlights = len(highlight_idx) if highlight_idx else 0
            fig_cache_key = (
                len(coords),
                float(coords[0, 0]) if len(coords) > 0 else None,
                color_option,
                n_highlights,
                actual_method,
                stored_mode,
            )

            # Only rebuild the figure when inputs actually change
            if st.session_state.get("viz_fig_key") != fig_cache_key:
                title = (
                    "Similar Datasets"
                    if stored_mode == "similar_only"
                    else "Dataset Similarity Map"
                )
                variance = st.session_state.get("viz_variance_ratio", None)
                st.session_state.viz_fig = _build_chart_figure(
                    coords,
                    viz_metadata,
                    color_option,
                    title,
                    highlight_idx,
                    variance,
                    actual_method,
                )
                st.session_state.viz_fig_key = fig_cache_key

            st.plotly_chart(
                st.session_state.viz_fig,
                use_container_width=True,
            )

            st.caption(
                "Hover over points to see dataset details. "
                "Copy accession ID to visit encodeproject.org/experiments/{accession}/"
            )
        else:
            st.info(
                "Click 'Generate Visualization' to create the embedding plot. "
                "This may take a moment for UMAP/t-SNE."
            )
