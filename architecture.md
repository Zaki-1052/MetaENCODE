# MetaENCODE: Technical Architecture & System Design

## 1. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE (Streamlit)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Search    │  │   Filters   │  │  Results    │  │    Visualization        │ │
│  │    Input    │  │   Panel     │  │   Table     │  │   (UMAP/PCA Plot)       │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘ │
└─────────┼────────────────┼────────────────┼─────────────────────┼───────────────┘
          │                │                │                     │
          ▼                ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SESSION STATE MANAGER                                  │
│         (selected_dataset, filters, search_history, cached_results)             │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            APPLICATION LOGIC LAYER                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐  │
│  │  Search Engine   │  │ Similarity Engine │  │   Visualization Engine      │  │
│  │  - keyword match │  │ - cosine similarity│  │   - UMAP projection        │  │
│  │  - filter apply  │  │ - top-N retrieval │  │   - PCA projection         │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────────┬───────────────┘  │
└───────────┼─────────────────────┼───────────────────────────┼───────────────────┘
            │                     │                           │
            ▼                     ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                          │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                    EMBEDDINGS CACHE (precomputed)                        │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐   │   │
│  │   │    Text     │  │ Categorical │  │   Numeric   │  │   Combined    │   │   │
│  │   │  Embeddings │  │   Vectors   │  │   Vectors   │  │    Vectors    │   │   │
│  │   │   (SBERT)   │  │ (one-hot)   │  │ (normalized)│  │  (concatenated)│  │   │
│  │   └─────────────┘  └─────────────┘  └─────────────┘  └───────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                    METADATA STORE (pandas DataFrame)                     │   │
│  │        accession | description | assay | organism | lab | ...            │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION LAYER                                   │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────────────┐ │
│  │   API Client       │  │   JSON Parser      │  │   Data Validator           │ │
│  │   - rate limiting  │  │   - flatten nested │  │   - check required fields  │ │
│  │   - error handling │  │   - extract fields │  │   - filter incomplete      │ │
│  └─────────┬──────────┘  └─────────┬──────────┘  └──────────────┬─────────────┘ │
└────────────┼──────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ENCODE REST API                                     │
│                      https://www.encodeproject.org/                              │
│                         (External Data Source)                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack Overview

```
┌────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                       │
├────────────────────────────────────────────────────────────────┤
│  Streamlit          │ Web framework, UI components, session    │
│  Plotly/Altair      │ Interactive visualizations               │
│  Streamlit-aggrid   │ Advanced data tables (optional)          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                        PROCESSING LAYER                         │
├────────────────────────────────────────────────────────────────┤
│  sentence-transformers │ SBERT text embeddings                 │
│  scikit-learn          │ Cosine similarity, NearestNeighbors   │
│  umap-learn            │ UMAP dimensionality reduction         │
│  pandas                │ DataFrame operations                  │
│  numpy                 │ Numerical operations                  │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                             │
├────────────────────────────────────────────────────────────────┤
│  requests              │ HTTP client for ENCODE API            │
│  pickle/joblib         │ Serialize precomputed embeddings      │
│  JSON                  │ API response parsing                  │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                       EXTERNAL SERVICES                         │
├────────────────────────────────────────────────────────────────┤
│  ENCODE REST API       │ Source of experiment metadata         │
│  Streamlit Cloud       │ Deployment platform                   │
│  GitHub                │ Version control + CI/CD               │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Feature Engineering Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RAW ENCODE EXPERIMENT JSON                               │
│  {                                                                               │
│    "accession": "ENCSR000ABC",                                                   │
│    "description": "H3K4me3 ChIP-seq on human K562 cells",                       │
│    "assay_term_name": "ChIP-seq",                                               │
│    "biosample_ontology": {"term_name": "K562", "organism": "human"},            │
│    "lab": "/labs/john-stamatoyannopoulos/",                                     │
│    "replicates": [{...}, {...}]                                                 │
│  }                                                                               │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
         │  TEXT FIELDS │  │ CATEGORICAL  │  │   NUMERIC    │
         │              │  │    FIELDS    │  │    FIELDS    │
         │ • description│  │ • assay_type │  │ • replicate  │
         │ • title      │  │ • organism   │  │   count      │
         │              │  │ • cell_type  │  │ • file_count │
         │              │  │ • lab        │  │              │
         │              │  │ • antibody   │  │              │
         └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                │                 │                 │
                ▼                 ▼                 ▼
         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
         │    CLEAN     │  │   ENCODE     │  │  NORMALIZE   │
         │ • lowercase  │  │ • one-hot    │  │ • min-max or │
         │ • strip punct│  │   encoding   │  │   z-score    │
         │ • handle null│  │ • handle     │  │ • handle     │
         │              │  │   unknown    │  │   missing    │
         └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                │                 │                 │
                ▼                 ▼                 ▼
         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
         │    SBERT     │  │   SPARSE/    │  │   SCALAR     │
         │   EMBED      │  │   DENSE      │  │   VECTOR     │
         │              │  │   VECTOR     │  │              │
         │  dim: 384    │  │  dim: ~50    │  │  dim: ~3     │
         │  (MiniLM)    │  │  (variable)  │  │              │
         └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                │                 │                 │
                └────────────┬────┴─────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │     CONCATENATE     │
                  │   (with optional    │
                  │     weighting)      │
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │   COMBINED VECTOR   │
                  │    dim: ~437        │
                  │  (384 + 50 + 3)     │
                  └─────────────────────┘
```

---

## 4. Data Flow Diagram

```
┌─────────┐
│  START  │
└────┬────┘
     │
     ▼
┌─────────────────┐    No     ┌─────────────────────────────────────────┐
│ Cached data     ├──────────►│ FETCH: Query ENCODE API                 │
│ exists?         │           │ • type=Experiment&frame=object&limit=all│
└────────┬────────┘           └───────────────────┬─────────────────────┘
         │ Yes                                    │
         │                                        ▼
         │                    ┌─────────────────────────────────────────┐
         │                    │ PARSE: Extract fields from JSON         │
         │                    │ • Flatten nested objects                │
         │                    │ • Handle missing fields                 │
         │                    └───────────────────┬─────────────────────┘
         │                                        │
         │                                        ▼
         │                    ┌─────────────────────────────────────────┐
         │                    │ VALIDATE: Filter incomplete records     │
         │                    │ • Must have description OR title        │
         │                    │ • Must have assay_term_name             │
         │                    └───────────────────┬─────────────────────┘
         │                                        │
         │                                        ▼
         │                    ┌─────────────────────────────────────────┐
         │                    │ PREPROCESS: Clean and encode            │
         │                    │ • Text: clean → SBERT embed             │
         │                    │ • Categorical: one-hot encode           │
         │                    │ • Numeric: normalize                    │
         │                    └───────────────────┬─────────────────────┘
         │                                        │
         │                                        ▼
         │                    ┌─────────────────────────────────────────┐
         │                    │ CACHE: Save to disk                     │
         │                    │ • embeddings.pkl                        │
         │                    │ • metadata.parquet                      │
         └───────────────────►└───────────────────┬─────────────────────┘
                                                  │
                                                  ▼
                              ┌─────────────────────────────────────────┐
                              │ LOAD INTO SESSION STATE                 │
                              │ • df_metadata (pandas DataFrame)        │
                              │ • embeddings_matrix (numpy array)       │
                              └───────────────────┬─────────────────────┘
                                                  │
                                                  ▼
                              ┌─────────────────────────────────────────┐
                              │ READY FOR USER INTERACTION              │
                              └─────────────────────────────────────────┘
```

---

## 5. User Interaction Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT APP                                    │
└──────────────────────────────────────────────────────────────────────────────┘

     USER ACTION                    SYSTEM RESPONSE                    OUTPUT
     ───────────                    ───────────────                    ──────

  ┌─────────────────┐          ┌─────────────────────┐         ┌──────────────┐
  │ Enter search    │          │ Filter metadata     │         │ Display      │
  │ keyword         ├─────────►│ by keyword match    ├────────►│ matching     │
  │                 │          │ (title/description) │         │ datasets     │
  └─────────────────┘          └─────────────────────┘         └──────────────┘
           │
           ▼
  ┌─────────────────┐          ┌─────────────────────┐         ┌──────────────┐
  │ Select seed     │          │ Store in session    │         │ Show dataset │
  │ dataset         ├─────────►│ state; display      ├────────►│ details      │
  │                 │          │ metadata            │         │ panel        │
  └─────────────────┘          └─────────────────────┘         └──────────────┘
           │
           ▼
  ┌─────────────────┐          ┌─────────────────────┐         ┌──────────────┐
  │ Click "Find     │          │ Compute cosine sim  │         │ Ranked table │
  │ Similar"        ├─────────►│ vs all embeddings;  ├────────►│ with scores  │
  │                 │          │ sort by score       │         │              │
  └─────────────────┘          └─────────────────────┘         └──────────────┘
           │
           ▼
  ┌─────────────────┐          ┌─────────────────────┐         ┌──────────────┐
  │ Apply filters   │          │ Filter results by   │         │ Updated      │
  │ (organism,      ├─────────►│ selected criteria;  ├────────►│ results      │
  │  assay, etc.)   │          │ re-rank             │         │ table        │
  └─────────────────┘          └─────────────────────┘         └──────────────┘
           │
           ▼
  ┌─────────────────┐          ┌─────────────────────┐         ┌──────────────┐
  │ View            │          │ Run UMAP/PCA on     │         │ Interactive  │
  │ visualization   ├─────────►│ embeddings;         ├────────►│ scatter plot │
  │                 │          │ color by attribute  │         │ with hover   │
  └─────────────────┘          └─────────────────────┘         └──────────────┘
           │
           ▼
  ┌─────────────────┐          ┌─────────────────────┐         ┌──────────────┐
  │ Click dataset   │          │ Generate ENCODE URL │         │ Open in      │
  │ link            ├─────────►│ from accession      ├────────►│ new tab      │
  │                 │          │                     │         │              │
  └─────────────────┘          └─────────────────────┘         └──────────────┘
```

---

## 6. Project Directory Structure

```
metaencode/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions for testing
├── data/
│   ├── cache/
│   │   ├── embeddings.pkl         # Precomputed embedding vectors
│   │   └── metadata.parquet       # Processed metadata DataFrame
│   └── raw/                       # Raw API responses (for debugging)
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── client.py              # ENCODE API client with rate limiting
│   │   └── parser.py              # JSON response parsing utilities
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text.py                # Text cleaning functions
│   │   ├── categorical.py         # One-hot encoding utilities
│   │   ├── numeric.py             # Normalization functions
│   │   └── pipeline.py            # Combined preprocessing pipeline
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── sbert.py               # SBERT embedding generation
│   │   └── combined.py            # Feature concatenation logic
│   ├── similarity/
│   │   ├── __init__.py
│   │   ├── cosine.py              # Cosine similarity computation
│   │   └── search.py              # Top-N retrieval functions
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── reduction.py           # UMAP/PCA implementations
│   │   └── plots.py               # Plotly chart builders
│   └── utils/
│       ├── __init__.py
│       ├── cache.py               # Caching utilities
│       └── config.py              # Configuration constants
├── tests/
│   ├── __init__.py
│   ├── test_api.py                # API client tests
│   ├── test_preprocessing.py      # Preprocessing unit tests
│   ├── test_similarity.py         # Similarity computation tests
│   └── test_integration.py        # End-to-end tests
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── requirements-dev.txt           # Dev dependencies (pytest, etc.)
├── .env.example                   # Environment variable template
├── .gitignore
├── README.md                      # Project documentation
└── setup.py                       # Package configuration (optional)
```

---

## 7. Component Responsibility Matrix

| Component | Module | Responsibility | Key Functions |
|-----------|--------|----------------|---------------|
| **API Client** | `src/api/client.py` | Fetch data from ENCODE | `fetch_experiments()`, `fetch_by_accession()` |
| **JSON Parser** | `src/api/parser.py` | Extract/flatten JSON fields | `parse_experiment()`, `extract_metadata()` |
| **Text Preprocessor** | `src/preprocessing/text.py` | Clean text for embedding | `clean_text()`, `normalize_whitespace()` |
| **Categorical Encoder** | `src/preprocessing/categorical.py` | One-hot encode categories | `encode_assay()`, `encode_organism()` |
| **Numeric Normalizer** | `src/preprocessing/numeric.py` | Scale numeric features | `normalize_counts()` |
| **SBERT Embedder** | `src/embeddings/sbert.py` | Generate text embeddings | `embed_descriptions()`, `load_model()` |
| **Feature Combiner** | `src/embeddings/combined.py` | Concatenate all vectors | `combine_features()`, `weight_features()` |
| **Similarity Engine** | `src/similarity/cosine.py` | Compute similarities | `compute_similarity()`, `pairwise_cosine()` |
| **Search Engine** | `src/similarity/search.py` | Retrieve top-N results | `find_similar()`, `filter_results()` |
| **Dim. Reduction** | `src/visualization/reduction.py` | Project to 2D | `run_umap()`, `run_pca()` |
| **Plot Builder** | `src/visualization/plots.py` | Create interactive charts | `scatter_plot()`, `add_tooltips()` |
| **Cache Manager** | `src/utils/cache.py` | Save/load precomputed data | `save_embeddings()`, `load_cache()` |
| **Streamlit App** | `app.py` | UI orchestration | `main()`, UI components |

---

## 8. Build & Deployment Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DEVELOPMENT WORKFLOW                                │
└─────────────────────────────────────────────────────────────────────────────────┘

  LOCAL DEVELOPMENT              CI/CD PIPELINE                  DEPLOYMENT
  ─────────────────              ──────────────                  ──────────

  ┌─────────────────┐          ┌─────────────────────┐
  │ Code changes    │          │ GitHub Actions      │
  │ on feature      │          │ triggered on        │
  │ branch          │          │ push/PR             │
  └────────┬────────┘          └──────────┬──────────┘
           │                              │
           ▼                              ▼
  ┌─────────────────┐          ┌─────────────────────┐
  │ Run local tests │          │ Run pytest suite    │
  │ pytest tests/   │          │ + linting (flake8)  │
  └────────┬────────┘          └──────────┬──────────┘
           │                              │
           ▼                              ▼
  ┌─────────────────┐          ┌─────────────────────┐
  │ Test Streamlit  │          │ Build passes?       │
  │ locally         │          │                     │
  │ streamlit run   │          │   No → Fix errors   │
  │ app.py          │          │   Yes → Continue    │
  └────────┬────────┘          └──────────┬──────────┘
           │                              │
           ▼                              ▼
  ┌─────────────────┐          ┌─────────────────────┐         ┌──────────────┐
  │ Push to GitHub  ├─────────►│ Merge to main       ├────────►│ Streamlit    │
  │ Open PR         │          │ branch              │         │ Cloud auto-  │
  └─────────────────┘          └─────────────────────┘         │ deploys      │
                                                               └──────────────┘
```

---

## 9. Key Design Decisions & Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Frontend Framework** | Streamlit | Low barrier, Python-native, built-in session state, easy deployment |
| **Embedding Model** | `all-MiniLM-L6-v2` | Good balance of quality and speed; 384-dim vectors are manageable |
| **Similarity Metric** | Cosine similarity | Standard for comparing embeddings; robust to magnitude differences |
| **Dim. Reduction** | UMAP (primary), PCA (fallback) | UMAP preserves local structure better for exploration |
| **Caching Strategy** | Precompute all embeddings | Avoid recomputing on each request; essential for UX |
| **Data Format** | Parquet for metadata | Efficient columnar storage; fast filtering |
| **Deployment** | Streamlit Cloud | Free tier available; integrates with GitHub |
| **Testing** | pytest | Standard Python testing framework; good Streamlit support |

---

## 10. Week-by-Week Build Plan

```
WEEK 1-2: FOUNDATION
├── Set up repository structure
├── Implement API client (src/api/client.py)
├── Implement JSON parser (src/api/parser.py)
├── Write tests for API layer
└── Fetch and save sample data

WEEK 3-4: PREPROCESSING & EMBEDDINGS
├── Implement text cleaning (src/preprocessing/text.py)
├── Implement categorical encoding (src/preprocessing/categorical.py)
├── Implement SBERT embedding (src/embeddings/sbert.py)
├── Implement feature combination (src/embeddings/combined.py)
├── Write preprocessing tests
└── Generate and cache embeddings for full dataset

WEEK 5-6: SIMILARITY & BASIC UI
├── Implement cosine similarity (src/similarity/cosine.py)
├── Implement top-N search (src/similarity/search.py)
├── Build basic Streamlit UI (app.py)
│   ├── Search input
│   ├── Dataset selection
│   ├── Results table
│   └── Filter sidebar
└── Write similarity tests

WEEK 7-8: VISUALIZATION & OPTIMIZATION
├── Implement UMAP/PCA (src/visualization/reduction.py)
├── Build interactive plots (src/visualization/plots.py)
├── Add hover tooltips
├── Optimize caching strategy
├── Performance testing
└── UI polish

WEEK 9-10: TESTING, DOCS & DEPLOYMENT
├── Integration tests
├── User testing
├── Write README.md
├── Write code documentation
├── Deploy to Streamlit Cloud
└── Prepare presentation
```

---

## 11. API Rate Limiting Strategy

```python
# Implemented in src/api/client.py

import time
from functools import wraps

class RateLimiter:
    """Enforce ENCODE's 10 requests/second limit."""

    def __init__(self, max_requests=10, window_seconds=1):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = []

    def wait_if_needed(self):
        now = time.time()
        # Remove requests outside the window
        self.requests = [t for t in self.requests if now - t < self.window]

        if len(self.requests) >= self.max_requests:
            sleep_time = self.window - (now - self.requests[0])
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.requests.append(time.time())

# Usage: call rate_limiter.wait_if_needed() before each API request
```

---

## 12. Embedding Weight Configuration

The combined embedding vector can be weighted to emphasize different similarity aspects:

```python
# Implemented in src/embeddings/combined.py

DEFAULT_WEIGHTS = {
    'text_embedding': 0.5,      # Description/title similarity
    'assay_type': 0.2,          # Same assay type is important
    'organism': 0.15,           # Same species matters
    'cell_type': 0.1,           # Cell type similarity
    'lab': 0.03,                # Same lab = consistent methods
    'numeric_features': 0.02    # Replicate counts, etc.
}

def combine_features(text_emb, categorical_vecs, numeric_vec, weights=None):
    """Concatenate feature vectors with optional weighting."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    weighted = [
        text_emb * weights['text_embedding'],
        categorical_vecs['assay'] * weights['assay_type'],
        categorical_vecs['organism'] * weights['organism'],
        categorical_vecs['cell_type'] * weights['cell_type'],
        categorical_vecs['lab'] * weights['lab'],
        numeric_vec * weights['numeric_features']
    ]

    return np.concatenate(weighted)
```

This allows users or future iterations to tune what "similar" means (e.g., more weight on assay type for users who care about experimental consistency vs. more weight on description for conceptual similarity).
