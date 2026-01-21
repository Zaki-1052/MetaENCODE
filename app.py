# app.py
"""MetaENCODE: ENCODE Dataset Similarity Search Application.

This Streamlit application enables researchers to discover related ENCODE
datasets through metadata-driven similarity scoring. Users can search for
datasets, select a seed dataset, and explore similar experiments through
interactive visualizations.

Run with: streamlit run app.py
"""

import pandas as pd
import streamlit as st

from src.api.encode_client import EncodeClient
from src.ml.embeddings import EmbeddingGenerator
from src.ml.similarity import SimilarityEngine
from src.processing.metadata import MetadataProcessor
from src.utils.cache import CacheManager
from src.visualization.plots import DimensionalityReducer, PlotGenerator

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="MetaENCODE",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Component Initialization with Caching ---


@st.cache_resource
def get_cache_manager() -> CacheManager:
    """Get or create the cache manager instance."""
    return CacheManager()


@st.cache_resource
def get_api_client() -> EncodeClient:
    """Get or create the API client instance."""
    return EncodeClient()


@st.cache_resource
def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create the embedding generator instance."""
    return EmbeddingGenerator()


@st.cache_resource
def get_metadata_processor() -> MetadataProcessor:
    """Get or create the metadata processor instance."""
    return MetadataProcessor()


@st.cache_data
def load_cached_data(
    _cache_mgr: CacheManager,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load precomputed metadata and embeddings from cache.

    Args:
        _cache_mgr: Cache manager instance (prefixed with _ to avoid hashing).

    Returns:
        Tuple of (metadata_df, embeddings) or (None, None) if not cached.
    """
    if _cache_mgr.exists("metadata") and _cache_mgr.exists("embeddings"):
        metadata = _cache_mgr.load("metadata")
        embeddings = _cache_mgr.load("embeddings")
        return metadata, embeddings
    return None, None


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "selected_dataset": None,
        "selected_index": None,
        "search_results": None,
        "similar_datasets": None,
        "metadata_df": None,
        "embeddings": None,
        "similarity_engine": None,
        "coords_2d": None,
        "filter_settings": {
            "organism": None,
            "assay_type": None,
            "top_n": 10,
        },
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> dict:
    """Render sidebar with search and filter controls.

    Returns:
        Dictionary containing current filter settings.
    """
    st.sidebar.title("MetaENCODE")
    st.sidebar.markdown("*ENCODE Dataset Similarity Search*")
    st.sidebar.divider()

    # Search section
    st.sidebar.subheader("Search")
    search_query = st.sidebar.text_input(
        "Search datasets",
        placeholder="e.g., ChIP-seq K562 H3K27ac",
        help="Search ENCODE for datasets by keyword",
        key="search_query_input",
    )

    if st.sidebar.button("Search", type="primary", use_container_width=True):
        if search_query.strip():
            with st.spinner("Searching ENCODE..."):
                try:
                    client = get_api_client()
                    results = client.search(search_query, limit=50)
                    st.session_state.search_results = results
                    st.sidebar.success(f"Found {len(results)} results")
                except Exception as e:
                    st.sidebar.error(f"Search failed: {e}")
        else:
            st.sidebar.warning("Please enter a search term")

    st.sidebar.divider()

    # Filter section
    st.sidebar.subheader("Filters")

    organism = st.sidebar.selectbox(
        "Organism",
        options=[None, "human", "mouse"],
        format_func=lambda x: "All organisms" if x is None else x.capitalize(),
    )

    assay_type = st.sidebar.selectbox(
        "Assay Type",
        options=[None, "ChIP-seq", "RNA-seq", "ATAC-seq", "DNase-seq"],
        format_func=lambda x: "All assay types" if x is None else x,
    )

    top_n = st.sidebar.slider(
        "Number of similar datasets",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )

    st.sidebar.divider()

    # Data loading section
    st.sidebar.subheader("Data")
    if st.sidebar.button("Load Sample Data", use_container_width=True):
        load_sample_data()

    # About section
    st.sidebar.divider()
    st.sidebar.subheader("About")
    st.sidebar.markdown(
        """
        MetaENCODE uses machine learning to find similar datasets based on
        metadata embeddings. Built with SBERT and Streamlit.

        [ENCODE Portal](https://www.encodeproject.org/)
        """
    )

    return {
        "search_query": search_query,
        "organism": organism,
        "assay_type": assay_type,
        "top_n": top_n,
    }


def load_sample_data() -> None:
    """Load a sample of ENCODE experiments for demonstration."""
    with st.spinner("Loading sample data from ENCODE API..."):
        try:
            client = get_api_client()
            processor = get_metadata_processor()
            embedder = get_embedding_generator()

            # Fetch a small sample of experiments
            raw_df = client.fetch_experiments(limit=100)

            if raw_df.empty:
                st.error("No experiments found")
                return

            # Process metadata
            processed_df = processor.process(raw_df)

            # Generate embeddings
            st.info("Generating embeddings (this may take a moment)...")
            texts = processed_df["combined_text"].tolist()
            embeddings = embedder.encode(texts, show_progress=False)

            # Fit similarity engine
            similarity_engine = SimilarityEngine()
            similarity_engine.fit(embeddings)

            # Store in session state
            st.session_state.metadata_df = processed_df
            st.session_state.embeddings = embeddings
            st.session_state.similarity_engine = similarity_engine

            # Cache the data
            cache_mgr = get_cache_manager()
            cache_mgr.save("metadata", processed_df)
            cache_mgr.save("embeddings", embeddings)

            st.success(f"Loaded {len(processed_df)} experiments")

        except Exception as e:
            st.error(f"Failed to load data: {e}")


def render_main_content() -> None:
    """Render main content area."""
    st.title("MetaENCODE")
    st.markdown(
        "**Discover related ENCODE datasets through "
        "metadata-driven similarity scoring**"
    )

    # Tabs for different views
    tab_search, tab_similar, tab_visualize = st.tabs(
        ["Search & Select", "Similar Datasets", "Visualize"]
    )

    with tab_search:
        render_search_tab()

    with tab_similar:
        render_similar_tab()

    with tab_visualize:
        render_visualize_tab()


def render_search_tab() -> None:
    """Render the search and selection tab."""
    st.header("Search & Select Dataset")

    # Display search results if available
    if st.session_state.search_results is not None:
        results_df = st.session_state.search_results

        if not results_df.empty:
            st.subheader("Search Results")

            # Display as interactive table
            display_cols = ["accession", "assay_term_name", "organism", "description"]
            display_cols = [c for c in display_cols if c in results_df.columns]

            # Truncate descriptions for display
            display_df = results_df[display_cols].copy()
            if "description" in display_df.columns:
                display_df["description"] = display_df["description"].apply(
                    lambda x: (str(x)[:80] + "...") if len(str(x)) > 80 else str(x)
                )

            # Let user select a row
            selection = st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
            )

            # Handle selection
            if selection and selection.selection.rows:
                selected_idx = selection.selection.rows[0]
                selected_row = results_df.iloc[selected_idx]
                st.session_state.selected_dataset = selected_row.to_dict()
                st.success(f"Selected: {selected_row['accession']}")
        else:
            st.info("No results found. Try a different search term.")
    else:
        st.info(
            "Use the search bar in the sidebar to find datasets, "
            "or enter an accession number below."
        )

    st.divider()

    # Manual accession input
    st.subheader("Or enter an accession directly")
    accession = st.text_input(
        "ENCODE Accession",
        placeholder="e.g., ENCSR000AKS",
        help="Enter an ENCODE experiment accession number",
    )

    if st.button("Load Dataset"):
        if accession.strip():
            with st.spinner(f"Loading {accession}..."):
                try:
                    client = get_api_client()
                    dataset = client.fetch_experiment_by_accession(accession.strip())
                    st.session_state.selected_dataset = dataset
                    st.success(f"Loaded dataset: {accession}")
                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")
        else:
            st.warning("Please enter an accession number")

    # Display selected dataset
    if st.session_state.selected_dataset is not None:
        st.divider()
        st.subheader("Selected Dataset")
        dataset = st.session_state.selected_dataset

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accession", dataset.get("accession", "N/A"))
            st.metric("Assay", dataset.get("assay_term_name", "N/A"))
        with col2:
            st.metric("Organism", dataset.get("organism", "N/A"))
            st.metric("Biosample", dataset.get("biosample_term_name", "N/A"))

        with st.expander("Full Metadata"):
            st.json(dataset)


def render_similar_tab() -> None:
    """Render the similar datasets tab."""
    st.header("Similar Datasets")

    if st.session_state.selected_dataset is None:
        st.info("Select a dataset first to find similar experiments.")
        return

    # Check if we have loaded data
    if st.session_state.metadata_df is None or st.session_state.embeddings is None:
        st.warning(
            "Please load sample data first using the 'Load Sample Data' button "
            "in the sidebar."
        )
        return

    selected = st.session_state.selected_dataset
    st.write(f"Finding datasets similar to: **{selected.get('accession', 'Unknown')}**")

    top_n = st.session_state.filter_settings.get("top_n", 10)

    if st.button("Find Similar Datasets", type="primary"):
        with st.spinner("Computing similarities..."):
            try:
                embedder = get_embedding_generator()
                similarity_engine = st.session_state.similarity_engine

                if similarity_engine is None:
                    st.error(
                        "Similarity engine not initialized. Please load data first."
                    )
                    return

                # Generate embedding for selected dataset
                text = f"{selected.get('description', '')} {selected.get('title', '')}"
                query_embedding = embedder.encode_single(text)

                # Find similar datasets
                similar_df = similarity_engine.find_similar(
                    query_embedding, n=top_n, exclude_self=True
                )

                # Get metadata for similar datasets
                metadata_df = st.session_state.metadata_df
                results = []
                for _, row in similar_df.iterrows():
                    idx = int(row["index"])
                    if idx < len(metadata_df):
                        meta = metadata_df.iloc[idx].to_dict()
                        meta["similarity_score"] = row["similarity_score"]
                        results.append(meta)

                st.session_state.similar_datasets = pd.DataFrame(results)

            except Exception as e:
                st.error(f"Error computing similarities: {e}")

    # Display similar datasets
    if st.session_state.similar_datasets is not None:
        similar = st.session_state.similar_datasets

        if not similar.empty:
            st.subheader("Most Similar Datasets")

            # Display columns
            display_cols = [
                "similarity_score",
                "accession",
                "assay_term_name",
                "organism",
                "description",
            ]
            display_cols = [c for c in display_cols if c in similar.columns]

            display_df = similar[display_cols].copy()
            display_df["similarity_score"] = display_df["similarity_score"].apply(
                lambda x: f"{x:.3f}"
            )
            if "description" in display_df.columns:
                display_df["description"] = display_df["description"].apply(
                    lambda x: (str(x)[:60] + "...") if len(str(x)) > 60 else str(x)
                )

            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Link to ENCODE
            st.markdown("Click accession numbers to view on ENCODE portal:")
            for _, row in similar.head(5).iterrows():
                acc = row.get("accession", "")
                if acc:
                    url = f"https://www.encodeproject.org/experiments/{acc}/"
                    st.markdown(f"- [{acc}]({url})")


def render_visualize_tab() -> None:
    """Render the visualization tab."""
    st.header("Dataset Visualization")

    if st.session_state.metadata_df is None or st.session_state.embeddings is None:
        st.info(
            "Load sample data first using the 'Load Sample Data' button "
            "in the sidebar to visualize datasets."
        )
        return

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Options")

        reduction_method = st.selectbox(
            "Reduction Method",
            options=["pca", "umap"],
            index=0,
            help="PCA is faster, UMAP preserves local structure better",
        )

        color_option = st.selectbox(
            "Color By",
            options=["assay_term_name", "organism", "lab"],
            format_func=lambda x: x.replace("_", " ").title(),
        )

        if st.button("Generate Visualization", type="primary"):
            generate_visualization(reduction_method, color_option)

    with col1:
        st.subheader("Embedding Space")

        if st.session_state.coords_2d is not None:
            metadata_df = st.session_state.metadata_df
            coords = st.session_state.coords_2d

            # Get highlight indices if we have similar datasets
            highlight_idx = None
            if st.session_state.similar_datasets is not None:
                # Find indices of similar datasets in the full metadata
                similar_accs = set(
                    st.session_state.similar_datasets["accession"].tolist()
                )
                highlight_idx = [
                    i
                    for i, acc in enumerate(metadata_df["accession"])
                    if acc in similar_accs
                ]

            # Generate plot
            plotter = PlotGenerator(reduction_method=reduction_method)
            fig = plotter.scatter_plot(
                coords,
                metadata_df,
                color_by=color_option,
                title="Dataset Similarity Map",
                highlight_indices=highlight_idx,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "Click 'Generate Visualization' to create the embedding plot. "
                "This may take a moment for UMAP."
            )


def generate_visualization(method: str, color_by: str) -> None:
    """Generate 2D visualization of embeddings.

    Args:
        method: Dimensionality reduction method ('pca' or 'umap').
        color_by: Column to color points by.
    """
    with st.spinner(f"Computing {method.upper()} projection..."):
        try:
            embeddings = st.session_state.embeddings
            reducer = DimensionalityReducer(method=method)
            coords_2d = reducer.fit_transform(embeddings)
            st.session_state.coords_2d = coords_2d
        except Exception as e:
            st.error(f"Error generating visualization: {e}")


def main() -> None:
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Try to load cached data on startup
    cache_mgr = get_cache_manager()
    if st.session_state.metadata_df is None:
        cached_meta, cached_emb = load_cached_data(cache_mgr)
        if cached_meta is not None and cached_emb is not None:
            st.session_state.metadata_df = cached_meta
            st.session_state.embeddings = cached_emb
            # Fit similarity engine
            similarity_engine = SimilarityEngine()
            similarity_engine.fit(cached_emb)
            st.session_state.similarity_engine = similarity_engine

    # Render sidebar and get filter settings
    filters = render_sidebar()

    # Update session state with filter settings
    st.session_state.filter_settings.update(filters)

    # Render main content
    render_main_content()


if __name__ == "__main__":
    main()
