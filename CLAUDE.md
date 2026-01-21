# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MetaENCODE is a Streamlit web application for discovering related biological datasets from ENCODE (Encyclopedia of DNA Elements) through ML-based metadata similarity. It combines SBERT text embeddings with categorical/numeric feature encoding to compute dataset similarity.

## Common Commands

```bash
# Setup & Installation
./scripts/setup.sh           # Full setup: venv + deps + pre-commit hooks
make dev                     # Install all dependencies (prod + dev)

# Running the Application
make run                     # streamlit run app.py
streamlit run app.py         # Direct run (port 8501)

# Testing
make test                    # pytest -v --cov=src --cov-report=term-missing
pytest tests/test_ml/ -v     # Run specific test directory
pytest tests/test_ml/test_embeddings.py -v  # Run single test file

# Code Quality
make lint                    # black --check, isort --check, flake8, mypy
make format                  # Auto-format with black and isort

# Precompute Embeddings (required before first run)
python scripts/precompute_embeddings.py --limit 1000
python scripts/precompute_embeddings.py --limit all --batch-size 64
python scripts/precompute_embeddings.py --refresh  # Force cache refresh

# HPC (SLURM)
sbatch scripts/precompute.sb  # Submit batch job for full dataset
```

## Architecture

### Data Flow
1. **Fetch**: ENCODE REST API → raw experiment metadata (rate-limited: 10 req/sec)
2. **Process**: Clean text, one-hot encode categoricals, normalize numerics
3. **Embed**: SBERT text embeddings (all-MiniLM-L6-v2, 384-dim)
4. **Combine**: Weighted concatenation → ~437-dim combined vectors
5. **Search**: Cosine similarity to find top-N similar datasets
6. **Visualize**: UMAP/PCA projection + Plotly interactive plots

### Feature Weights (in FeatureCombiner)
- Text embeddings: 50%
- Assay type: 20%
- Organism: 15%
- Biosample: 10%
- Lab: 3%
- Numeric features: 2%

### Module Structure
```
src/
├── api/encode_client.py      # ENCODE API client with rate limiting
├── ml/
│   ├── embeddings.py         # SBERT text embedding (lazy model loading)
│   ├── similarity.py         # Cosine similarity & NearestNeighbors
│   └── feature_combiner.py   # Orchestrates feature combination
├── processing/
│   ├── metadata.py           # Metadata extraction and cleaning
│   └── encoders.py           # One-hot (categorical) + MinMax (numeric)
├── utils/cache.py            # Pickle-based caching with atomic writes
└── visualization/plots.py    # UMAP/PCA + Plotly scatter plots
```

### Key Files
- `app.py`: Main Streamlit application with session state management
- `scripts/precompute_embeddings.py`: Batch precomputation CLI
- `data/cache/`: Precomputed embeddings, metadata, and feature combiner

## ENCODE API

- **Base URL**: `https://www.encodeproject.org/`
- **No authentication** required for public data
- **Rate limit**: 10 requests/second (enforced by sliding window in EncodeClient)
- **Query pattern**: `type=Experiment&frame=object&limit=N`

## Caching

Precomputed data is stored in `data/cache/` as pickle files:
- `metadata.pkl`: Processed DataFrame with experiment metadata
- `embeddings.pkl`: SBERT text embeddings (n_experiments × 384)
- `combined_vectors.pkl`: Final feature vectors (n_experiments × ~437)
- `feature_combiner.pkl`: Fitted FeatureCombiner for consistent encoding

The app loads from cache on startup via `@st.cache_resource` for fast response times.

## Testing

Tests are in `tests/` with fixtures in `conftest.py`. Key test categories:
- `test_api/`: API client and rate limiting
- `test_ml/`: Embeddings, similarity, feature combination
- `test_processing/`: Metadata processing, encoders
- `test_integration.py`: End-to-end workflows

## HPC Deployment

For full dataset precomputation on HPC clusters:
```bash
# Create conda environment
conda env create -f encode.yml
conda activate encode

# Submit SLURM job
sbatch scripts/precompute.sb
```

The SLURM script requests 8 CPUs, 40GB RAM, and 6 hours runtime.
