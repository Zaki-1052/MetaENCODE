# MetaENCODE -- Technical Specification & Architecture Guide

> **Audience:** Team members, communications staff, and presentation preparation.
> This document describes what MetaENCODE is, how it works, how it was designed and programmed, and why each technical decision was made.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [User-Facing Features](#3-user-facing-features)
4. [System Architecture](#4-system-architecture)
5. [Data Pipeline](#5-data-pipeline)
6. [Module-by-Module Breakdown](#6-module-by-module-breakdown)
7. [Key Algorithms & Techniques](#7-key-algorithms--techniques)
8. [Data Schemas & Structures](#8-data-schemas--structures)
9. [Vocabulary & Ontology System](#9-vocabulary--ontology-system)
10. [User Interface Design](#10-user-interface-design)
11. [Caching & Performance Strategy](#11-caching--performance-strategy)
12. [Precomputation Scripts & Deployment](#12-precomputation-scripts--deployment)
13. [Testing & Code Quality](#13-testing--code-quality)
14. [Technology Stack](#14-technology-stack)
15. [Design Patterns & Engineering Principles](#15-design-patterns--engineering-principles)
16. [Glossary](#16-glossary)
17. [FAQ](#17-faq)

---

## 1. Project Overview

**MetaENCODE** is a web application that helps researchers discover related biological datasets from the [ENCODE (Encyclopedia of DNA Elements)](https://www.encodeproject.org/) project. ENCODE is a public database containing over 27,000 genomic experiments. MetaENCODE makes it possible to find experiments that are *similar* to one another based on their metadata -- not just keyword matching, but genuine semantic and structural similarity.

**In one sentence:** Given any ENCODE experiment, MetaENCODE finds and ranks the most similar experiments and shows them in an interactive 2D visualization.

- **Developed by:** DS3 x UBIC Collaborative Project at UC San Diego
- **License:** MIT (open source)
- **Tech:** Python, Streamlit, Sentence-BERT, scikit-learn, Plotly, UMAP

---

## 2. Problem Statement & Motivation

### The Problem

ENCODE hosts ~27,000 experiments spanning hundreds of assay types, organisms, cell lines, and labs. Researchers frequently need to find related datasets -- for example, "I'm studying ChIP-seq in human K562 cells; what similar experiments exist?" The ENCODE portal offers keyword search and faceted browsing, but it cannot answer the question *"which experiments are most similar to mine?"* in a principled, quantitative way.

### What MetaENCODE Does Differently

MetaENCODE goes beyond keyword search by:

1. **Understanding text semantically** -- Using a language AI model (Sentence-BERT) to convert experiment descriptions into numerical vectors that capture meaning, not just exact word matches. "Histone modification profiling" and "ChIP-seq for histone marks" are recognized as similar even though they share few words.

2. **Combining multiple signals** -- Similarity is computed from a weighted blend of text descriptions (50%), assay type (20%), organism (15%), cell type (10%), lab (3%), and numeric features like replicate counts (2%). This multi-modal approach mirrors how a domain expert would judge similarity.

3. **Visualizing the landscape** -- All ~27,000 experiments are projected onto a 2D scatter plot using dimensionality reduction (UMAP, PCA, or t-SNE), letting researchers visually explore clusters and relationships across the entire dataset.

---

## 3. User-Facing Features

MetaENCODE is organized into three tabs, accessed through a sidebar-driven workflow:

### Tab 1: Search & Select

| Feature | Description |
|---------|-------------|
| **Faceted search** | Filter by assay type, organism, biosample/tissue, histone mark/target, life stage, lab, and replicate counts |
| **Description search** | Free-text search with automatic spell correction for biological terms |
| **Hierarchical biosample browsing** | Browse tissues organized by organ system, cell type, germ layer, or body system |
| **Direct accession lookup** | Enter an ENCODE accession (e.g., `ENCSR000AKS`) to load a specific experiment |
| **Selection history** | Recently viewed experiments are saved and available in a dropdown |

### Tab 2: Similar Datasets

| Feature | Description |
|---------|-------------|
| **Similarity ranking** | Shows the top N most similar experiments, ranked by a multi-modal similarity score |
| **Score display** | Each result includes a similarity score from 0.0 (unrelated) to 1.0 (identical) |
| **Linked accessions** | Each result links directly to the ENCODE portal for that experiment |
| **Configurable results** | Sidebar slider controls how many results to display (5-50) |

### Tab 3: Visualize

| Feature | Description |
|---------|-------------|
| **2D scatter plot** | Interactive Plotly visualization of the embedding space |
| **Three reduction methods** | PCA, t-SNE, and UMAP projections available |
| **Color coding** | Color by assay type, organism, organ system, cell type, germ layer, body system, lab, or similarity score |
| **Two view modes** | "All Datasets" shows the full ~27K landscape; "Similar Only" shows just the similar datasets |
| **Outlier filtering** | Optional 5th-95th percentile filtering to remove extreme outliers |
| **Highlight overlay** | In "All Datasets" mode, similar datasets are highlighted with red star markers |

---

## 4. System Architecture

### High-Level Pipeline

```
ENCODE REST API (encodeproject.org)
        |
        v
   EncodeClient              Rate-limited HTTP client (10 req/sec)
        |
        v
   MetadataProcessor         Text cleaning, missing value imputation, ontology mapping
        |
        v
   EmbeddingGenerator        Sentence-BERT (all-MiniLM-L6-v2) -> 384-dim text vectors
        |
        v
   FeatureCombiner           Weighted concatenation -> ~437-dim combined vectors
        |
        v
   CacheManager              Atomic pickle I/O in data/cache/
        |
        v
   SimilarityEngine          Cosine similarity via scikit-learn NearestNeighbors
        |
        v
   DimensionalityReducer     PCA / t-SNE / UMAP -> 2D coordinates
        |
        v
   PlotGenerator             Interactive Plotly scatter plots
        |
        v
   Streamlit UI              Three tabs: Search & Select | Similar Datasets | Visualize
```

### Module Dependency Map

```
src/
 |-- api/                    ENCODE REST API communication
 |     encode_client.py         EncodeClient, RateLimiter
 |
 |-- processing/             Data transformation layer
 |     metadata.py              MetadataProcessor (text cleaning, enrichment)
 |     encoders.py              CategoricalEncoder, NumericEncoder
 |
 |-- ml/                     Machine learning layer
 |     embeddings.py            EmbeddingGenerator (SBERT wrapper)
 |     feature_combiner.py      FeatureCombiner (weighted vector fusion)
 |     similarity.py            SimilarityEngine (k-NN search)
 |
 |-- visualization/          Dimensionality reduction + plotting
 |     plots.py                 DimensionalityReducer, PlotGenerator
 |
 |-- ui/                     User interface layer
 |     sidebar.py               Sidebar filter controls
 |     search_filters.py        FilterState dataclass, SearchFilterManager
 |     handlers.py              Search execution logic
 |     formatters.py            Display formatting, ENCODE links
 |     autocomplete.py          Autocomplete suggestion engine
 |     vocabularies.py          Vocabulary loading from JSON
 |     styles.py                CSS styling (green color theme)
 |     components/
 |       session.py             Streamlit session state management
 |       initializers.py        Cached singleton factories
 |     tabs/
 |       search.py              Search & Select tab
 |       similar.py             Similar Datasets tab
 |       visualize.py           Visualization tab
 |
 |-- utils/                  Infrastructure utilities
       cache.py                 CacheManager (pickle-based persistence)
       spell_check.py           SymSpell + phonetic spell correction
       history.py               SelectionHistory (per-user MRU list)
       user_id.py               Per-browser user identification
       data_loader.py           (deprecated; loading handled by session.py)
```

### Information Flow Between Layers

```
     API Layer          Processing Layer          ML Layer            UI Layer
  +-----------+       +----------------+     +---------------+    +----------+
  |           |       |                |     |               |    |          |
  | Encode    | ----> | Metadata       | --> | Embedding     | -> | Search   |
  | Client    |       | Processor      |     | Generator     |    | Tab      |
  |           |       |                |     |               |    |          |
  +-----------+       | Categorical    | --> | Feature       | -> | Similar  |
                      | Encoder        |     | Combiner      |    | Tab      |
                      |                |     |               |    |          |
                      | Numeric        | --> | Similarity    | -> | Visualize|
                      | Encoder        |     | Engine        |    | Tab      |
                      +----------------+     +---------------+    +----------+
                                                    |
                                              +-----v------+
                                              | Dimension   |
                                              | Reducer     |
                                              | + Plot Gen  |
                                              +-------------+
```

---

## 5. Data Pipeline

MetaENCODE processes data in two phases: **offline precomputation** (run once before the app starts) and **online serving** (real-time user interactions).

### Phase 1: Offline Precomputation

This is the heavy lifting. Run once via `precompute_embeddings.py` and `precompute_visualizations.py`.

```
Step 1: FETCH
  - Query ENCODE REST API for all ~27,000 released experiments
  - Rate-limited at 10 requests/second
  - Extract 13 metadata fields per experiment (accession, description,
    assay type, organism, biosample, lab, replicate counts, etc.)
  - Output: Raw metadata DataFrame

Step 2: PROCESS
  - Clean text fields (lowercase, remove special characters, normalize whitespace)
  - Create combined_text field from description + title
  - Fill missing values (text -> "", categorical -> "unknown", numeric -> 0)
  - Enrich with ontology lookups (biosample -> organ, cell type, germ layer, body system)
  - Output: Processed metadata DataFrame with ~20 columns

Step 3: EMBED
  - Feed combined_text through Sentence-BERT (all-MiniLM-L6-v2)
  - Produces a 384-dimensional vector per experiment
  - Batch processing (64 texts per batch) for efficiency
  - Output: NumPy array of shape (27000, 384)

Step 4: COMBINE FEATURES
  - One-hot encode categorical columns (assay, organism, biosample, lab)
  - Min-max normalize numeric columns (replicate counts, file count)
  - Apply sqrt(weight) scaling to each feature group
  - Concatenate text embeddings + categorical + numeric vectors
  - Output: NumPy array of shape (27000, ~437)

Step 5: CACHE
  - Serialize metadata, embeddings, combined vectors, and fitted encoders
  - Atomic writes (write to temp file, then rename) to prevent corruption
  - Output: Pickle files in data/cache/

Step 6: PRECOMPUTE VISUALIZATIONS (optional but recommended)
  - Project combined vectors to 2D using PCA, t-SNE, and UMAP
  - Generate both unfiltered and outlier-filtered variants
  - Cache as 6 coordinate sets (3 methods x 2 filter modes)
  - Output: Pickle files with 2D coordinate arrays
```

### Phase 2: Online Serving (Runtime)

When a user interacts with the app:

```
1. APP STARTUP
   - Load precomputed data from cache into Streamlit session state
   - Initialize SimilarityEngine with precomputed combined vectors
   - Build NearestNeighbors index (brute-force cosine)

2. USER SEARCHES (Search Tab)
   - Apply sidebar filters to precomputed metadata DataFrame
   - OR fetch live from ENCODE API if filter-based search is used
   - Display matching experiments in interactive table

3. USER SELECTS A DATASET
   - Store selected experiment in session state
   - Save to persistent selection history

4. SIMILARITY COMPUTATION (Similar Tab)
   - Encode selected dataset: text -> SBERT -> combined vector
   - Query SimilarityEngine.find_similar() for top N neighbors
   - Convert distances to similarity scores (1 - cosine_distance)
   - Display ranked results with scores

5. VISUALIZATION (Visualize Tab)
   - Load precomputed 2D coordinates from cache (instant)
   - OR compute on-the-fly if cache miss (slower)
   - Render interactive Plotly scatter plot
   - Color by user-selected metadata field
   - Highlight similar datasets with star markers
```

---

## 6. Module-by-Module Breakdown

### 6.1 API Module (`src/api/`)

#### RateLimiter

Enforces ENCODE's strict 10-requests-per-second API limit using a sliding window algorithm.

- **How it works:** Keeps a list of timestamps for recent requests. Before each new request, removes timestamps older than 1 second. If 10 requests are already in the window, sleeps until the oldest one exits.
- **Why:** ENCODE will block clients that exceed rate limits. This is a responsible citizen pattern.

#### EncodeClient

HTTP client for the ENCODE REST API.

| Method | Purpose |
|--------|---------|
| `fetch_experiments()` | Fetch experiments with optional filters (assay, organism, biosample, target, life stage, search term). Returns up to `limit` results as a DataFrame. |
| `fetch_experiment_by_accession()` | Fetch a single experiment by its ENCODE accession ID. |
| `search()` | Free-text search across ENCODE objects. |

**Parsing complexity:** ENCODE's JSON responses have deeply nested structures (e.g., organism name is buried under `replicates[0].library.biosample.donor.organism.name`). The `_parse_experiment()` method handles 3+ format variations per field with defensive fallbacks at every level.

**Design decision -- no retry logic:** At 10 req/sec with proper rate limiting, transient failures are rare. Retries would add complexity for minimal benefit.

---

### 6.2 Processing Module (`src/processing/`)

#### MetadataProcessor

Cleans and enriches raw ENCODE metadata into a form suitable for machine learning.

**Processing steps:**
1. **Text cleaning:** Lowercase, strip special characters, normalize whitespace
2. **Combined text:** Joins cleaned description + title into a single field for embedding
3. **Missing value imputation:** Text fields get `""`, categorical fields get `"unknown"`, numeric fields get `0`
4. **Ontology enrichment:** Maps each biosample term to its organ system, cell type classification, germ layer, and body system using hierarchical mappings from `encode_facets_raw.json`

**Example transformation:**
```
Input:  biosample_term_name = "K562"
Output: organ = "blood", cell_type = "leukemia cell line",
        developmental_layer = "mesoderm", body_system = "immune system"
```

#### CategoricalEncoder

Converts categorical text fields into numerical vectors for the ML pipeline.

- **One-hot encoding** (default): Creates a binary vector with one 1 and the rest 0s. If there are 30 assay types, each experiment gets a 30-element vector with a 1 in the position of its assay type.
- **Label encoding** (alternative): Assigns each category an integer index.
- **Unknown handling:** Unknown values produce all-zero vectors (graceful degradation).

#### NumericEncoder

Normalizes numeric fields to comparable scales.

- **Min-max** (default): Scales values to [0, 1] range: `(value - min) / (max - min)`
- **Standardize** (alternative): Z-score normalization: `(value - mean) / std`
- **Safety:** Handles constant-value columns (zero variance) without division-by-zero errors.

---

### 6.3 Machine Learning Module (`src/ml/`)

#### EmbeddingGenerator

Wraps the Sentence-BERT (SBERT) language model to convert experiment descriptions into numerical vectors.

**Model:** `all-MiniLM-L6-v2`
- 22 million parameters
- Produces 384-dimensional vectors
- Optimized for semantic similarity tasks
- ~2ms per text on CPU

**Why SBERT?** Traditional keyword matching (TF-IDF, BM25) cannot capture that "ChIP-seq for histone H3K27ac" and "Histone acetylation profiling by chromatin immunoprecipitation" describe similar experiments. SBERT encodes meaning into vectors where semantically similar texts are geometrically close.

**Lazy loading:** The model (~100 MB) is only loaded into memory on first use, not at import time. This keeps app startup fast.

#### FeatureCombiner

The core innovation: combines text embeddings with structured metadata into a single similarity-ready vector.

**Default feature weights:**

| Feature | Weight | Dimensions | Rationale |
|---------|--------|------------|-----------|
| Text embedding | 50% | 384 | Captures semantic meaning of descriptions |
| Assay type | 20% | ~30 (one-hot) | Experiments with same assay are methodologically similar |
| Organism | 15% | ~2 (one-hot) | Human vs. mouse is a fundamental distinction |
| Cell type / biosample | 10% | ~50 (one-hot) | Same cell line implies biological relevance |
| Lab | 3% | ~15 (one-hot) | Same lab may use similar protocols |
| Numeric features | 2% | 4 (scaled) | Replicate counts, file counts are weak signals |

**Weight application:** Weights are applied as `sqrt(weight)` on the sub-vectors, not directly as multipliers. This is a mathematical necessity: cosine similarity is quadratic in vector magnitude, so `sqrt(weight)` ensures the contribution to the final similarity score matches the intended percentage.

**Total combined dimension:** ~437 (varies based on vocabulary cardinalities).

#### SimilarityEngine

Finds the most similar experiments to any query using scikit-learn's NearestNeighbors.

**How it works:**
1. **Index building:** Stores all ~27K combined vectors in a brute-force cosine similarity index.
2. **Querying:** Given a query vector, computes cosine similarity against all stored vectors and returns the top N.
3. **Self-exclusion:** If the query is already in the index, it filters out exact self-matches (similarity > 0.9999).
4. **Score conversion:** Converts cosine distance to similarity: `similarity = 1 - cosine_distance`.

**Why brute force?** Approximate nearest neighbor methods (HNSW, FAISS) are faster for very large datasets but introduce approximation error. At 27K vectors with 437 dimensions, brute-force cosine is fast enough (<100ms per query) and guarantees exact results.

---

### 6.4 Visualization Module (`src/visualization/`)

#### DimensionalityReducer

Projects high-dimensional vectors (~437D) down to 2D for visualization using one of three methods:

| Method | How It Works | Best For |
|--------|-------------|----------|
| **PCA** | Finds the two directions of maximum variance in the data. Linear, deterministic, fast. Reports how much variance each axis captures (e.g., "PC-1 explains 23% of variance"). | Quick overview; preserving global structure |
| **t-SNE** | Preserves local neighborhood structure. Non-linear, stochastic. Points that are close in high dimensions stay close in 2D. | Revealing tight clusters |
| **UMAP** | Similar to t-SNE but faster and better at preserving global structure. Uses topological data analysis. | Best default; balances local and global structure |

**Adaptive parameters:** UMAP `n_neighbors` and t-SNE `perplexity` automatically adjust based on dataset size to avoid errors with small datasets.

#### PlotGenerator

Renders interactive Plotly scatter plots with:

- **Hover tooltips** showing accession, description, assay type, organism, and organ
- **Color coding** by any metadata field or similarity score
- **Custom similarity colorscale** (orange-white-purple diverging)
- **Highlight markers** (red stars) for selected/similar datasets
- **PCA variance labels** on axes (e.g., "PC-1 (23.4% variance)")
- **Jitter** to separate overlapping points

---

### 6.5 UI Module (`src/ui/`)

#### Sidebar (`sidebar.py`)

The left panel containing all search and filter controls, organized hierarchically:

1. **Results count** -- Slider from 5 to 50
2. **Description search** -- Free text with spell correction
3. **Assay type** -- Dropdown with experiment counts
4. **Organism** -- Dropdown with genome assembly info (e.g., "Human [hg38]")
5. **Target / Histone mark** -- Curated histone modifications + popular targets
6. **Biosample hierarchy** -- Switchable taxonomy:
   - *Organ system:* brain, heart, liver, etc.
   - *Cell type:* B cell, T cell, fibroblast, etc.
   - *Germ layer:* ectoderm, mesoderm, endoderm
   - *Body system:* immune, nervous, cardiovascular, etc.
7. **Life stage** -- adult, embryonic, child, etc.
8. **Advanced options** (collapsed) -- Lab, minimum replicate counts
9. **Search button** with query preview
10. **About section** with project description

#### Search Filters (`search_filters.py`)

`FilterState` is a Python dataclass that captures the complete state of all filter controls:

```
FilterState:
  assay_type, organism, body_part, biosample, target, age_stage,
  lab, min_replicates, min_bio_replicates, min_tech_replicates,
  max_results, description_search
```

`SearchFilterManager` implements fuzzy matching and synonym expansion for filter values. Its matching algorithm scores candidates by:
1. **Exact match** -> score 1.0
2. **Prefix match** -> score 0.9-1.0
3. **Contains match** -> score 0.6-0.8
4. **Word boundary match** -> score 0.65
5. **Fuzzy (sequence matching)** -> score proportional to similarity ratio

Tissue synonyms are expanded (e.g., searching "cerebellum" also finds "hindbrain" experiments).

#### Autocomplete (`autocomplete.py`)

Provides real-time search suggestions as the user types, with:
- Vocabulary-aware prefix matching
- Alias resolution (e.g., "ChIP" -> "ChIP-seq")
- Spell correction integration (confidence >= 0.6 threshold)
- Experiment count display for popularity signals

#### Vocabularies (`vocabularies.py`)

**Single source of truth** for all ENCODE vocabulary values. Every dropdown option, every filter value, and every biosample-to-organ mapping comes from `data/encode_facets_raw.json`, which is generated directly from the ENCODE API.

Vocabulary categories:
- **Assay types** (~30): ChIP-seq, RNA-seq, ATAC-seq, Hi-C, etc.
- **Organisms** (~5): Homo sapiens, Mus musculus, etc.
- **Biosamples** (~500): K562, HeLa, liver, brain, etc.
- **Targets** (~1000): H3K27ac, CTCF, p53, etc.
- **Labs** (~50): ENCODE consortium labs
- **Life stages** (~10): adult, embryonic, child, etc.
- **Organ systems** (~60): brain, heart, kidney, etc.
- **Cell type classifications** (~25): B cell, T cell, etc.
- **Germ layers** (3): ectoderm, mesoderm, endoderm
- **Body systems** (~14): immune, nervous, endocrine, etc.

All lists are ordered by experiment count (popularity) to surface the most relevant options first.

#### Styles (`styles.py`)

Consistent green color theme throughout the application:
- **Primary green:** `#618B4A`
- **Light green (sidebar):** `#C6DEB4`
- **Tan accent:** `#afbc88`
- **Olive:** `#8e9a6a`

#### Session Management (`components/session.py`)

Manages Streamlit session state with ~20 keys covering:
- Selected dataset and search results
- Precomputed ML artifacts (embeddings, combined vectors, similarity engine)
- Visualization state (coordinates, method, color-by, view mode)
- Filter state and selection history

All data is lazy-loaded: heavy pickle files are loaded once on first access, then cached in session state for the duration of the browser session.

#### Component Initializers (`components/initializers.py`)

Factory functions decorated with `@st.cache_resource` (Streamlit's singleton cache) for expensive objects:
- `get_cache_manager()` -- Pickle I/O handler
- `get_api_client()` -- ENCODE API client
- `get_embedding_generator()` -- SBERT model wrapper
- `get_metadata_processor()` -- Data cleaning pipeline
- `get_feature_combiner()` -- Feature engineering pipeline
- `get_filter_manager()` -- Search filter logic

These are created once and shared across all reruns of the Streamlit app.

---

### 6.6 Utils Module (`src/utils/`)

#### CacheManager (`cache.py`)

File-based caching using Python's pickle serialization.

- **Atomic writes:** Data is written to a temporary file first, then atomically renamed. This prevents corruption if the process is interrupted mid-write.
- **Optional expiration:** Cache entries can expire after a configurable number of hours.
- **Corruption recovery:** If a pickle file is corrupted, it's automatically deleted and a fresh load returns `None` instead of crashing.
- **Default location:** `data/cache/`

#### SpellChecker (`spell_check.py`)

Vocabulary-aware spell correction for biological terms, combining two strategies:

1. **SymSpell** -- Edit-distance-based spelling correction. Finds terms within 2 character edits (insertions, deletions, substitutions, transpositions) of the query.
2. **Double Metaphone (phonetic)** -- Finds terms that *sound like* the query even if the edit distance is high. Uses the `jellyfish` library for phonetic code generation.

**Confidence scoring:**
- Base score: `1.0 - (edit_distance / max_length)`
- Phonetic match bonus: `+0.15`
- Prefix match bonus: `+0.10`
- Frequency bonus: `+min(0.1, log10(frequency) / 50)`
- Large length difference penalty: `*0.8`

Only suggestions with confidence >= 0.7 are auto-applied.

**Optional dependency:** If `symspellpy` or `jellyfish` are not installed, the spell checker is silently disabled. The app works without it.

#### SelectionHistory (`history.py`)

Maintains a most-recently-used (MRU) list of up to 10 dataset selections, persisted as JSON.

- **Deduplication:** Re-selecting an existing entry moves it to the top.
- **Atomic saves:** Uses temp-file-then-rename pattern for safety.
- **Display format:** `"ENCSR000AKS - ChIP-seq | K562 | Homo sapiens"`

#### User ID (`user_id.py`)

Assigns a persistent UUID to each browser session via cookies (SameSite=Lax, 1-year expiry). Enables per-user state isolation in multi-user deployments.

---

## 7. Key Algorithms & Techniques

### 7.1 Sentence-BERT Text Embedding

**What:** A neural network that converts text into a 384-dimensional numerical vector, where semantically similar texts produce geometrically close vectors.

**Model:** `all-MiniLM-L6-v2` (22M parameters, based on Microsoft's MiniLM architecture)

**How it works (simplified):**
1. Tokenizes the input text into subwords
2. Passes tokens through 6 transformer layers that learn contextual representations
3. Pools token representations into a single 384-dimensional sentence vector
4. The model was pre-trained on 1 billion+ sentence pairs to learn that similar meanings should map to similar vectors

**Why this model:** MiniLM-L6-v2 is the best balance of speed (~2ms per text on CPU), quality (state-of-the-art for its size), and dimension count (384 is compact enough for efficient similarity search).

### 7.2 Weighted Feature Concatenation with sqrt Scaling

**What:** Combines text embeddings (384D), one-hot categorical vectors (~50D), and normalized numeric features (4D) into a single ~437D vector.

**The sqrt trick:** When using cosine similarity, the contribution of a sub-vector to the final score is proportional to the *square* of its magnitude. If we want text features to contribute 50% of the similarity score, we scale them by `sqrt(0.50) = 0.707`, not `0.50`. This ensures the mathematical contribution matches the intended weight.

**Why concatenation?** It's simple, interpretable, and efficient. More sophisticated fusion methods (attention-based, learned weighting) would add complexity without clear benefit for this dataset size.

### 7.3 Cosine Similarity

**What:** Measures the angle between two vectors, ignoring their magnitude. Two vectors pointing in the same direction have cosine similarity 1.0; perpendicular vectors have 0.0.

**Why cosine over Euclidean?** Text embeddings naturally vary in magnitude based on text length and content. Cosine similarity is magnitude-invariant, so it compares *what* the text says, not *how much* it says.

**Formula:** `similarity(A, B) = (A dot B) / (|A| * |B|)`

### 7.4 k-Nearest Neighbors (Brute Force)

**What:** Given a query vector, compute its cosine similarity to every vector in the index and return the top k.

**Complexity:** O(n * d) per query, where n = dataset size (27K), d = vector dimension (437). At these scales, this takes <100ms.

**Why brute force?** Approximate methods (HNSW, IVF, LSH) trade accuracy for speed. With 27K vectors, brute force is fast enough and guarantees exact results. At 100K+ vectors, we'd consider approximate methods.

### 7.5 Dimensionality Reduction

Three methods reduce ~437 dimensions to 2 for visualization:

**PCA (Principal Component Analysis):**
- Finds the two orthogonal directions that capture the most variance
- Linear transformation (fast, deterministic)
- Reports variance explained per axis
- Good for understanding overall structure

**t-SNE (t-distributed Stochastic Neighbor Embedding):**
- Optimizes a probability distribution over pairs of points
- Preserves local structure (nearby points stay nearby)
- Non-linear, stochastic (different runs may look different)
- Good for revealing clusters

**UMAP (Uniform Manifold Approximation and Projection):**
- Based on topological data analysis
- Preserves both local and global structure
- Faster than t-SNE for large datasets
- Best default choice

### 7.6 Spell Correction (SymSpell + Phonetic)

**SymSpell:** Pre-computes all possible "delete edits" of vocabulary words. At query time, generates delete edits of the query and looks them up. This makes edit-distance search very fast (microseconds) at the cost of memory.

**Double Metaphone:** Converts words to a phonetic code (how they sound). "Cerebellum" and "Cerebelum" produce the same code, catching phonetic typos that edit distance might miss.

The two strategies are combined: SymSpell finds close edits, Metaphone finds sound-alikes, and a confidence formula ranks all candidates.

---

## 8. Data Schemas & Structures

### 8.1 Metadata DataFrame (per experiment)

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `accession` | str | ENCODE API | Unique experiment ID (e.g., "ENCSR000AKS") |
| `description` | str | ENCODE API | Free-text experiment description |
| `title` | str | ENCODE API | Short experiment title |
| `description_clean` | str | MetadataProcessor | Cleaned description (lowercase, no special chars) |
| `title_clean` | str | MetadataProcessor | Cleaned title |
| `combined_text` | str | MetadataProcessor | Joined cleaned text for embedding |
| `assay_term_name` | str | ENCODE API | Experiment type (e.g., "ChIP-seq", "RNA-seq") |
| `organism` | str | ENCODE API | Species (e.g., "Homo sapiens") |
| `biosample_term_name` | str | ENCODE API | Cell/tissue type (e.g., "K562", "liver") |
| `lab` | str | ENCODE API | Producing laboratory |
| `status` | str | ENCODE API | Release status |
| `life_stage` | str | ENCODE API | Developmental stage (e.g., "adult") |
| `replicate_count` | int | ENCODE API | Total replicates |
| `bio_replicate_count` | int | ENCODE API | Biological replicates |
| `tech_replicate_count` | int | ENCODE API | Technical replicates |
| `file_count` | int | ENCODE API | Associated data files |
| `organ` | str | Ontology lookup | Primary organ system |
| `cell_type` | str | Ontology lookup | Cell type classification |
| `developmental_layer` | str | Ontology lookup | Germ layer (ectoderm/mesoderm/endoderm) |
| `body_system` | str | Ontology lookup | Body system classification |

### 8.2 Numerical Arrays

| Array | Shape | Type | Description |
|-------|-------|------|-------------|
| Text embeddings | (N, 384) | float32 | SBERT output vectors |
| Combined vectors | (N, ~437) | float32 | Weighted multi-modal vectors |
| 2D coordinates | (N, 2) | float32 | UMAP/PCA/t-SNE projections |
| Similarity matrix | (N, N) | float32 | Pairwise cosine similarities |

Where N = number of experiments (~27,000 for full dataset).

### 8.3 Cache Files

| Cache Key | Contents | Size (approx.) |
|-----------|----------|----------------|
| `metadata` | Processed DataFrame | ~50 MB |
| `embeddings` | Text embedding array | ~40 MB |
| `combined_vectors` | Combined feature array | ~45 MB |
| `feature_combiner` | Fitted encoder state | ~1 MB |
| `viz_coords_pca_unfiltered` | PCA 2D coordinates | ~0.5 MB |
| `viz_coords_pca_filtered` | PCA 2D coordinates (filtered) | ~0.5 MB |
| `viz_coords_tsne_unfiltered` | t-SNE 2D coordinates | ~0.5 MB |
| `viz_coords_tsne_filtered` | t-SNE 2D coordinates (filtered) | ~0.5 MB |
| `viz_coords_umap_unfiltered` | UMAP 2D coordinates | ~0.5 MB |
| `viz_coords_umap_filtered` | UMAP 2D coordinates (filtered) | ~0.5 MB |

---

## 9. Vocabulary & Ontology System

### Data Source

All vocabulary values are extracted from the ENCODE API itself, ensuring the app always reflects the real contents of the database. The extraction is performed by `scripts/fetch_encode_facets.py`, which:

1. Fetches all ~27,000 experiments from the ENCODE API
2. Counts occurrences of each value for every metadata field
3. Builds hierarchical mappings (biosample -> organ, cell type, germ layer, body system) using ENCODE's slim annotations
4. Saves everything to `data/encode_facets_raw.json`

### Hierarchical Biosample Classification ("Slims")

ENCODE uses "slim" ontology categories to organize biosamples into higher-level groupings. MetaENCODE leverages four slim taxonomies:

```
                    biosample_term_name
                    (e.g., "K562")
                          |
        +---------+-------+-------+---------+
        |         |               |         |
    organ_slim  cell_slim  developmental_slim  system_slim
    (blood)     (leukemia    (mesoderm)         (immune
                 cell line)                      system)
```

| Slim Type | Categories | Example |
|-----------|------------|---------|
| **Organ** | ~62 | brain, heart, liver, kidney, blood |
| **Cell type** | ~25 | B cell, T cell, fibroblast, stem cell |
| **Developmental (germ layer)** | 3 | ectoderm, mesoderm, endoderm |
| **Body system** | ~14 | immune, nervous, cardiovascular, digestive |

Users can switch between these classification systems in the sidebar to browse biosamples from different perspectives.

### Tissue Synonyms

The vocabulary system includes synonym mappings for tissues that have multiple common names:

```
cerebellum <-> hindbrain
forebrain <-> cerebral cortex
bone marrow <-> hematopoietic system
```

When a user searches for "cerebellum", results for "hindbrain" experiments are also included.

---

## 10. User Interface Design

### Layout

```
+------------------+--------------------------------------------------+
|                  |                                                  |
|    SIDEBAR       |              MAIN CONTENT AREA                  |
|                  |                                                  |
|  [Filters]       |  MetaENCODE                                     |
|  - Description   |  Discover related ENCODE datasets...            |
|  - Assay Type    |                                                  |
|  - Organism      |  [Search & Select] [Similar Datasets] [Visualize]|
|  - Target        |  ------------------------------------------------|
|  - Biosample     |                                                  |
|    - Organ       |  (Tab content area)                              |
|    - Cell Type   |                                                  |
|    - Germ Layer  |                                                  |
|  - Life Stage    |                                                  |
|  - Lab           |                                                  |
|  - Replicates    |                                                  |
|                  |                                                  |
|  [Search]        |                                                  |
|  [Clear Filters] |                                                  |
|                  |                                                  |
|  About           |                                                  |
+------------------+--------------------------------------------------+
```

### Color Scheme

The green color scheme reflects the biological/life sciences context:
- Sidebar background: light green (`#C6DEB4`)
- Active buttons/accents: primary green (`#618B4A`)
- Tab indicators: green underline on active tab
- Visualization options panel: light green background

### Interaction Flow

```
1. User opens app
   -> Sidebar loads with filter options
   -> Search tab is shown (default)
   -> Precomputed data loads from cache in background

2. User sets filters and clicks "Search"
   -> Results appear in interactive table
   -> Filter summary shown below table

3. User clicks a row in the results table
   -> Dataset details appear below table
   -> ENCODE portal link provided
   -> Selection saved to history

4. User clicks "Similar Datasets" tab
   -> Similarity computation runs (~100ms)
   -> Ranked results appear with scores
   -> Each result links to ENCODE portal

5. User clicks "Visualize" tab
   -> 2D scatter plot loads (instant from cache)
   -> Color coding and view mode selectable
   -> Similar datasets highlighted with stars
```

---

## 11. Caching & Performance Strategy

### Why Cache?

The two most expensive operations are:
1. **SBERT embedding** of 27K texts: ~15 minutes on CPU
2. **UMAP/t-SNE** dimensionality reduction: ~5-10 minutes

These are computed once offline and cached to disk. At runtime, the app loads cached results in seconds.

### Cache Architecture

```
data/cache/
  |-- metadata.pkl              Processed experiment metadata
  |-- embeddings.pkl            384-dim text embedding vectors
  |-- combined_vectors.pkl      437-dim combined feature vectors
  |-- feature_combiner.pkl      Fitted encoder parameters
  |-- viz_coords_pca_*.pkl      PCA 2D coordinates
  |-- viz_coords_tsne_*.pkl     t-SNE 2D coordinates
  |-- viz_coords_umap_*.pkl     UMAP 2D coordinates
  |-- selection_history.json    Per-user selection history
```

### Safety Mechanisms

- **Atomic writes:** All cache writes go to a temporary file first, then are atomically renamed. This prevents corruption from crashes or interrupted writes.
- **Corruption recovery:** If a pickle file fails to load (corrupted bytes, version mismatch), it's automatically deleted and regenerated.
- **Optional expiration:** Cache entries can be configured with a time-to-live (TTL) in hours.

### Runtime Performance

| Operation | Time | How |
|-----------|------|-----|
| App startup (cache load) | ~3 seconds | Pickle deserialization |
| Search query | <1 second | DataFrame filtering |
| Similarity computation | ~100ms | Brute-force cosine over 27K vectors |
| Visualization (cached) | <1 second | Load precomputed coordinates |
| Visualization (on-the-fly UMAP) | ~5-10 minutes | Compute from scratch |

---

## 12. Precomputation Scripts & Deployment

### Script 1: `precompute_embeddings.py`

**Purpose:** Build the complete ML pipeline from scratch.

**Usage:**
```bash
# Quick test (~100 experiments)
python scripts/precompute_embeddings.py --limit 100

# Full dataset (~27,000 experiments)
python scripts/precompute_embeddings.py --limit all --batch-size 64

# Force rebuild from scratch
python scripts/precompute_embeddings.py --limit 1000 --refresh
```

**Pipeline:**
1. Check cache (skip if exists, unless `--refresh`)
2. Fetch experiments from ENCODE API
3. Process metadata (clean text, enrich with ontology)
4. Generate SBERT text embeddings (batched)
5. Fit FeatureCombiner and combine all features
6. Save metadata, embeddings, combined vectors, and fitted combiner to cache

### Script 2: `precompute_visualizations.py`

**Purpose:** Generate 2D coordinates for the Visualize tab so it loads instantly.

**Usage:**
```bash
python scripts/precompute_visualizations.py
python scripts/precompute_visualizations.py --methods pca umap  # Only specific methods
python scripts/precompute_visualizations.py --refresh            # Force rebuild
```

**Outputs:** 6 cache files (3 methods x 2 filter modes: filtered + unfiltered)

### Script 3: `fetch_encode_facets.py`

**Purpose:** Regenerate `data/encode_facets_raw.json` from the live ENCODE API. Run when ENCODE adds new experiments, assay types, or biosamples.

**Usage:**
```bash
python scripts/fetch_encode_facets.py
```

**Extracts:**
- Value counts for all metadata fields (assay types, organisms, biosamples, targets, labs)
- Hierarchical slim-to-biosample mappings (organ -> biosample list, cell -> biosample list, etc.)
- Nested field extraction (organism, life stage from deeply nested replicate structures)

### Script 4: `generate_ontology.py`

**Purpose:** Parse OWL/RDF ontology files (UBERON, EFO, OBI, CLO) to build structured ontology mappings.

**Outputs:** Ontology term relationships, synonyms, and slim assignments in JSON format.

### SLURM Deployment (`precompute.sb`)

For running on high-performance computing clusters (tested on SDSC Expanse):

```bash
#SBATCH --job-name=encode_data
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
```

The full pipeline (27K experiments) requires ~40 GB RAM and ~6 hours on 8 CPU cores.

### Application Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 13. Testing & Code Quality

### Test Suite

MetaENCODE has ~708 tests organized by module:

```
tests/
  conftest.py                  Shared fixtures (sample data, pre-fitted models)
  test_api/                    EncodeClient, RateLimiter tests
  test_ml/                     Embeddings, FeatureCombiner, SimilarityEngine tests
  test_processing/             MetadataProcessor, Encoder tests
  test_ui/                     UI component tests
  test_utils/                  Cache, spell check, history tests
  test_visualization/          DimensionalityReducer, PlotGenerator tests
  test_integration.py          End-to-end pipeline tests
```

### Integration Tests

Five integration test classes verify the full pipeline:

| Test Class | What It Verifies |
|------------|------------------|
| `TestFullPipelineTextOnly` | Text-only similarity works (backward compatibility) |
| `TestFullPipelineCombined` | Full multi-modal pipeline (text + categorical + numeric) |
| `TestQueryMatchesBatch` | Single-record transforms produce identical results to batch |
| `TestSimilarityWithDifferentWeights` | Weight changes produce different similarity rankings |
| `TestChipSeqSimilarity` | Same assay type experiments rank higher than different assays |

### Test Fixtures

Reusable test data defined in `conftest.py`:
- `sample_experiments_df()` -- DataFrame with 3 diverse experiments
- `sample_embeddings()` -- 10x384 embedding array
- `fitted_feature_combiner()` -- Pre-fitted FeatureCombiner instance
- `sample_similarity_matrix()` -- 5x5 symmetric similarity matrix

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing

# Specific module
python -m pytest tests/test_ml/ -v
```

~30 tests are skipped without optional dependencies (`sentence-transformers`, `umap-learn`).

### Code Quality Tools

| Tool | Purpose | Command |
|------|---------|---------|
| **black** | Code formatting (PEP 8 compliant) | `black src/ tests/ scripts/ app.py` |
| **isort** | Import sorting | `isort src/ tests/ scripts/ app.py` |
| **flake8** | Linting (style + error detection) | `flake8 src/ tests/` |
| **mypy** | Static type checking | `mypy src/` |

Type stubs are included: `pandas-stubs`, `types-requests` for full type coverage.

---

## 14. Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Web Framework** | Streamlit | >= 1.28 | Interactive web UI with Python |
| **Visualization** | Plotly | >= 5.18 | Interactive scatter plots, heatmaps |
| **Text Embeddings** | sentence-transformers | >= 2.2 | SBERT model wrapper (MiniLM) |
| **ML / Similarity** | scikit-learn | >= 1.3 | NearestNeighbors, PCA, t-SNE, encoders |
| **Dimensionality Reduction** | umap-learn | >= 0.5.4 | UMAP projections |
| **Data Manipulation** | pandas | >= 2.0 | DataFrames for metadata |
| **Numerical Computing** | NumPy | >= 1.24 | Vector operations, array storage |
| **HTTP Client** | requests | >= 2.31 | ENCODE API communication |
| **Spell Correction** | symspellpy | >= 6.7 | Edit-distance spell checking |
| **Phonetic Matching** | jellyfish | >= 0.9 | Double Metaphone phonetic codes |
| **Environment** | python-dotenv | >= 1.0 | Environment variable management |
| **Data Source** | ENCODE REST API | -- | 27,000+ genomic experiments (no auth required) |

### System Requirements

- **Python:** 3.10+
- **Disk:** ~2 GB for full precomputed cache
- **RAM:** ~4 GB at runtime (embedding arrays + NearestNeighbors index)
- **CPU:** Multi-core recommended for precomputation (UMAP, t-SNE)

---

## 15. Design Patterns & Engineering Principles

### Architectural Patterns

| Pattern | Where Used | Why |
|---------|-----------|-----|
| **Layered Architecture** | API -> Processing -> ML -> UI | Clean separation of concerns; each layer has a single responsibility |
| **Cache-Aside** | CacheManager | Check cache before computing; compute and store on miss |
| **Lazy Initialization** | EmbeddingGenerator, SpellChecker | Defer expensive operations (model loading) until first use |
| **Singleton/Caching** | `@st.cache_resource`, `@lru_cache` | Expensive objects created once and shared |
| **Factory Method** | `DimensionalityReducer._create_reducer()` | Select reduction method by string name |
| **Strategy Pattern** | PCA/t-SNE/UMAP, cosine/euclidean | Interchangeable algorithms behind a common interface |
| **Pipeline Pattern** | Precompute scripts | Chain: Fetch -> Process -> Embed -> Combine -> Cache |
| **Facade Pattern** | `src/ui/__init__.py` | Simple public API over complex internal modules |
| **Atomic Write** | CacheManager, SelectionHistory | Write temp file -> rename to prevent corruption |
| **Fluent API** | `encoder.fit(data).transform(data)` | Method chaining with `return self` |
| **Dataclass** | FilterState, SpellingSuggestion | Lightweight, typed data containers |

### Engineering Principles

| Principle | Application |
|-----------|-------------|
| **DRY (Don't Repeat Yourself)** | Vocabulary loading centralized in `vocabularies.py`; encoding logic in reusable `CategoricalEncoder`/`NumericEncoder` classes |
| **Single Responsibility** | Each class does one thing: `EncodeClient` fetches data, `MetadataProcessor` cleans it, `EmbeddingGenerator` embeds it |
| **Fail Fast, Fail Loud** | Encoders raise `ValueError` if used before fitting; empty results return early |
| **Graceful Degradation** | Missing spell-check libraries disable spell correction silently; corrupted cache files are auto-deleted |
| **Defensive Parsing** | `_parse_experiment()` handles 3+ JSON format variations per field with type checks at every level |
| **Separation of Concerns** | UI logic, state management, ML computation, and API communication are in separate modules |
| **No Hardcoded Values** | All vocabulary lists loaded from `encode_facets_raw.json`, never hardcoded |

---

## 16. Glossary

| Term | Definition |
|------|------------|
| **ENCODE** | Encyclopedia of DNA Elements; a public database of genomic experiments funded by the NIH |
| **Accession** | A unique identifier for an ENCODE experiment (e.g., ENCSR000AKS) |
| **Assay** | The experimental technique used (e.g., ChIP-seq, RNA-seq, ATAC-seq) |
| **Biosample** | The biological material tested (e.g., K562 cell line, liver tissue) |
| **SBERT** | Sentence-BERT; a neural network that converts text into numerical vectors capturing semantic meaning |
| **Embedding** | A numerical vector representation of data (text, categories, etc.) in a high-dimensional space |
| **Cosine Similarity** | A measure of similarity between two vectors based on the angle between them (0 = unrelated, 1 = identical) |
| **One-Hot Encoding** | Converting a categorical variable into a binary vector (e.g., "ChIP-seq" -> [1,0,0,...,0]) |
| **PCA** | Principal Component Analysis; a linear method for reducing data dimensions |
| **t-SNE** | t-distributed Stochastic Neighbor Embedding; a non-linear method emphasizing local clusters |
| **UMAP** | Uniform Manifold Approximation and Projection; a fast non-linear dimensionality reduction method |
| **k-NN** | k-Nearest Neighbors; finding the k most similar items to a query |
| **Slim** | A simplified ontology category in ENCODE (e.g., organ slims map biosamples to organs) |
| **Ontology** | A structured vocabulary defining relationships between biological terms |
| **Germ Layer** | One of three primary cell layers in embryonic development: ectoderm, mesoderm, endoderm |
| **Rate Limiting** | Restricting API request frequency to avoid overloading the server |
| **Pickle** | Python's binary serialization format for saving objects to disk |
| **Session State** | Per-browser-session data storage in Streamlit that persists across reruns |
| **HPC** | High-Performance Computing; cluster computing for resource-intensive tasks |
| **SLURM** | A job scheduler for HPC clusters (used on SDSC Expanse) |

---

## 17. FAQ

### General

**Q: What datasets does MetaENCODE cover?**
A: All ~27,000 released experiments from the ENCODE project, spanning ChIP-seq, RNA-seq, ATAC-seq, Hi-C, whole-genome bisulfite sequencing, and dozens of other assay types across human, mouse, and other organisms.

**Q: Does MetaENCODE require authentication to use?**
A: No. The ENCODE API is publicly accessible without authentication, and MetaENCODE requires no login.

**Q: How current is the data?**
A: The data is as current as the last time `precompute_embeddings.py` was run. To update, re-run the precomputation pipeline, which will fetch the latest experiments from ENCODE.

### Technical

**Q: Why was Streamlit chosen over Flask/Django/React?**
A: Streamlit allows rapid development of data-science web apps in pure Python. Since the team is Python-focused and the app is primarily data visualization (not CRUD), Streamlit's reactive model is a natural fit. It eliminates the need for separate frontend code.

**Q: Why SBERT instead of TF-IDF or BM25?**
A: SBERT captures semantic meaning, not just word overlap. "Histone modification profiling" and "ChIP-seq for histone marks" share few words but describe similar experiments. SBERT understands this; TF-IDF does not.

**Q: Why are feature weights applied as sqrt(weight)?**
A: Cosine similarity is computed as a dot product of normalized vectors. The contribution of a sub-vector to the dot product is proportional to the square of its magnitude. Applying sqrt(weight) as a scaling factor ensures that the contribution to the final similarity score is proportional to the intended weight (e.g., 50% for text).

**Q: Why brute-force similarity instead of an approximate index (FAISS, HNSW)?**
A: At 27K vectors with 437 dimensions, brute-force cosine search takes <100ms. Approximate methods introduce accuracy loss and complexity for negligible speed gain at this scale. If the dataset grew to 100K+, approximate indexing would be warranted.

**Q: Why pickle for caching instead of a database?**
A: The cached objects are large NumPy arrays and fitted sklearn models. Pickle serializes these natively and efficiently. A database would require serialization/deserialization overhead and schema management for minimal benefit (there's no need for SQL queries over embeddings).

**Q: How much memory does the full app require?**
A: ~4 GB at runtime. The main consumers are the metadata DataFrame (~50 MB), text embeddings (27K x 384 x 4 bytes = ~40 MB), combined vectors (~45 MB), and the NearestNeighbors index (a copy of the combined vectors).

**Q: Can the similarity weights be changed?**
A: Yes. The `FeatureCombiner` accepts a custom weight dictionary. However, changing weights requires recomputing combined vectors for the full dataset (running `precompute_embeddings.py` again with the new weights).

### Design

**Q: Why is the vocabulary loaded from a JSON file instead of queried live?**
A: Performance and reliability. Querying the ENCODE API for all vocabulary values on every page load would be slow and fragile. The JSON file is generated once from the API and provides instant, offline access.

**Q: Why four biosample classification systems (organ, cell, developmental, system)?**
A: Different researchers think about biosamples differently. A neuroscientist searches by organ ("brain"), an immunologist by cell type ("T cell"), a developmental biologist by germ layer ("mesoderm"), and a physiologist by body system ("nervous system"). Four views serve all perspectives.

**Q: Why is spell correction optional?**
A: The spell checker depends on `symspellpy` and `jellyfish`, which are not universally available. Making them optional means the app works in minimal environments (e.g., CI/CD, Docker with slim base images) while providing enhanced functionality when the libraries are installed.

---

*This document was generated from the MetaENCODE source code (commit `9915b5f` on `main`).*
