# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MetaENCODE is a Streamlit web application that helps researchers discover related ENCODE biological datasets through metadata-driven similarity scoring. It uses SBERT text embeddings, categorical/numeric feature encoding, and cosine similarity to rank and recommend datasets from the [ENCODE project](https://www.encodeproject.org/).

**DS3 x UBIC Collaborative Project** at UCSD.

## Commands

### Run the Application
```bash
streamlit run app.py
```
App opens at `http://localhost:8501`.

### Precompute Embeddings (required before first run)
```bash
python scripts/precompute_embeddings.py --limit 100    # Quick test
python scripts/precompute_embeddings.py --limit 1000   # Medium
python scripts/precompute_embeddings.py --limit all --batch-size 64  # Full
python scripts/precompute_embeddings.py --limit 1000 --refresh  # Force recompute
```

### Precompute Visualizations
```bash
python scripts/precompute_visualizations.py
```

### Testing
```bash
python -m pytest tests/ -v                              # All tests (~708)
python -m pytest tests/ --cov=src --cov-report=term-missing  # With coverage
python -m pytest tests/test_ml/ -v                      # Single module
python -m pytest tests/test_ml/test_embeddings.py -v    # Single file
python -m pytest tests/test_ml/test_embeddings.py::test_name -v  # Single test
```

Note: Some tests require `sentence-transformers` (PyTorch) and `umap-learn`. Without these, ~30 tests are skipped.

### Code Quality
```bash
black src/ tests/ scripts/ app.py    # Format
isort src/ tests/ scripts/ app.py    # Sort imports
flake8 src/ tests/                   # Lint
mypy src/                            # Type check
```

## Architecture

```
ENCODE REST API
    ↓
EncodeClient (rate-limited 10 req/sec, pagination, nested JSON parsing)
    ↓
MetadataProcessor (text cleaning, missing value imputation, ontology mapping)
    ↓
EmbeddingGenerator (SBERT all-MiniLM-L6-v2 → 384-dim vectors, lazy-loaded)
    ↓
FeatureCombiner (weighted concatenation → ~437-dim combined vectors)
    ↓
CacheManager (pickle files in data/cache/ with atomic writes)
    ↓
SimilarityEngine (cosine similarity via scikit-learn NearestNeighbors)
    ↓
Streamlit UI (3 tabs: Search & Select, Similar Datasets, Visualize)
```

### Key Architectural Decisions

- **Feature weighting**: `FeatureCombiner` applies `sqrt(weight)` scaling to concatenated sub-vectors so that cosine similarity contributions are proportional to configured weights. Default weights: text(0.5), assay_type(0.2), organism(0.15), cell_type(0.1), lab(0.03), numeric(0.02).

- **Vocabulary single source of truth**: All ENCODE vocabularies (assay types, organisms, biosamples, hierarchical organ/cell mappings) are loaded from `data/encode_facets_raw.json` via `src/ui/vocabularies.py`. Never hardcode vocabulary lists — regenerate via `scripts/fetch_encode_facets.py`.

- **Precomputation pipeline**: The app loads precomputed pickle files from `data/cache/` at startup. The pipeline is: `scripts/precompute_embeddings.py` → fetch API data → generate embeddings → combine features → cache as pickle. Visualization coordinates are precomputed separately by `scripts/precompute_visualizations.py`.

- **Session state management**: `src/ui/components/session.py` defines `SESSION_DEFAULTS` dict. `init_session_state()` runs once per session. `load_cached_data_into_session()` loads pickle cache and fits the `SimilarityEngine`. Tab navigation uses `st.session_state.active_tab`.

- **Caching pattern**: `CacheManager` uses atomic writes (write to `.tmp` file, then `os.rename`) to prevent partial-write corruption. Cache keys map to filenames like `experiments_metadata_*.pkl`.

## Module Layout

| Directory | Purpose |
|-----------|---------|
| `src/api/` | ENCODE REST API client with rate limiting |
| `src/ml/` | Embeddings (SBERT), feature combination, similarity engine |
| `src/processing/` | Text cleaning, categorical/numeric encoding |
| `src/ui/` | Streamlit UI: sidebar, tabs, session state, vocabularies, autocomplete |
| `src/utils/` | Cache manager, spell check, selection history, user ID |
| `src/visualization/` | Dimensionality reduction (UMAP/PCA/t-SNE) and Plotly plots |
| `scripts/` | Precomputation and data pipeline scripts |
| `data/` | `encode_facets_raw.json` (vocab source) + `cache/` (precomputed pickle files) |
| `tests/` | pytest suite mirroring `src/` structure, with shared fixtures in `conftest.py` |

## Key Files

- **`app.py`** — Entry point. Initializes session, renders sidebar + 3 tab buttons.
- **`src/ml/feature_combiner.py`** — `DEFAULT_WEIGHTS`, `CATEGORICAL_COLUMNS`, `NUMERIC_COLUMNS` constants define the similarity model.
- **`src/ui/vocabularies.py`** (~1600 lines) — All vocabulary accessors with `@lru_cache`. Loads from `encode_facets_raw.json`.
- **`src/ui/components/session.py`** — `SESSION_DEFAULTS` dict, session initialization and cache loading.
- **`src/ui/search_filters.py`** — `FilterState` dataclass for filter management.
- **`data/encode_facets_raw.json`** — 27,398 experiments, field counts, hierarchical biosample mappings.
- **`tests/conftest.py`** — Shared fixtures (`sample_experiments_df`, `sample_embeddings`, etc.).

## Tech Stack

- **Python 3.10+**, Streamlit, pandas, numpy
- **ML**: sentence-transformers (all-MiniLM-L6-v2, 384-dim), scikit-learn, umap-learn
- **Viz**: Plotly
- **Spell correction**: symspellpy + jellyfish
- **API**: ENCODE REST API (no auth, 10 req/sec rate limit, `frame=embedded` responses)

---

## Review



**Language > Specialist**: Python/JS > Bioinformatics & Data Science

Solid project. Here are my thoughts:

## Strengths

**Well-architected pipeline.** The flow from ENCODE API → preprocessing → SBERT embeddings → feature combination → cosine similarity is clean and well-separated. Each stage has a single responsibility, and the module layout reflects that.

**Smart feature weighting design.** Using `sqrt(weight)` scaling on concatenated sub-vectors so cosine similarity contributions are proportional — that's a thoughtful detail. The default weights (text 0.5, assay 0.2, organism 0.15, etc.) are reasonable priors for biological dataset similarity.

**Production-quality infrastructure.** Atomic cache writes, rate-limited API calls, precomputation scripts, comprehensive test suite (~708 tests) — this isn't a prototype. The caching pattern with pickle + `.tmp` rename is a good guard against corruption.

**Vocabulary management done right.** Single source of truth from `encode_facets_raw.json` with `@lru_cache` accessors. Avoids the common trap of hardcoded biological vocabularies drifting out of sync.

## Areas I'd Watch

**Embedding model choice.** `all-MiniLM-L6-v2` is a solid general-purpose model, but biomedical text has domain-specific semantics. Models like `PubMedBERT` or `BioSentVec` could improve similarity quality for experiment descriptions, biosample terms, and assay nomenclature. Worth benchmarking if you haven't.

**Scalability.** ENCODE has ~27K experiments now. Brute-force cosine similarity with scikit-learn's `NearestNeighbors` works fine at that scale, but if you ever expand to GEO or other repositories, you'd want approximate nearest neighbors (FAISS, Annoy).

**The 384-dim + categorical = ~437-dim combined vector.** That's a modest dimensionality — good for speed, but the categorical features only get ~53 dims. Depending on how many categories you're encoding, that could be sparse. Worth checking if the categorical signal is actually contributing meaningfully to recommendations vs. just the text embedding dominating.

**Pickle for caching.** Works, but it's version-fragile and not human-inspectable. Parquet for DataFrames and `.npy` for arrays would be more robust long-term, especially across Python/library version upgrades.

## Overall

This is a well-engineered bioinformatics tool — clean separation of concerns, defensive coding, and a clear precomputation strategy. The CLAUDE.md and test coverage suggest a mature development process. The core idea (metadata-driven similarity for ENCODE datasets) fills a real gap — ENCODE's own search is keyword-based, not semantic.

The `viz-points` branch you're on suggests you're working on visualization improvements. Happy to dig into anything specific.
