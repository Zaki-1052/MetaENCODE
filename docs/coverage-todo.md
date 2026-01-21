# MetaENCODE Test Coverage TODO

**Initial Coverage:** 81% (624 statements, 121 missing)
**Final Coverage:** 100% (624 statements, 0 missing)
**Date:** January 2026
**Status:** COMPLETE

---

## Progress Summary

| Module | Initial | Final | Status |
|--------|---------|-------|--------|
| `src/visualization/plots.py` | 0% | 100% | DONE |
| `src/visualization/__init__.py` | 0% | 100% | DONE |
| `src/ml/embeddings.py` | 68% | 100% | DONE |
| `src/api/encode_client.py` | 78% | 100% | DONE |
| `src/ml/similarity.py` | 91% | 100% | DONE |
| `src/processing/encoders.py` | 92% | 100% | DONE |
| `src/utils/cache.py` | 97% | 100% | DONE |
| `src/ml/feature_combiner.py` | 98% | 100% | DONE |
| `src/processing/metadata.py` | 99% | 100% | DONE |

### Completed Work

**tests/test_visualization/test_plots.py** (NEW - 45 tests)
- Created `tests/test_visualization/__init__.py`
- TestDimensionalityReducerInit (4 tests)
- TestDimensionalityReducerFit (7 tests)
- TestDimensionalityReducerTransform (4 tests)
- TestDimensionalityReducerFitTransform (5 tests)
- TestPlotGeneratorInit (2 tests)
- TestPlotGeneratorScatterPlot (13 tests)
- TestPlotGeneratorSimilarityHeatmap (8 tests)
- TestVisualizationImports (3 tests)

**tests/test_ml/test_embeddings.py** (NEW - 17 tests)
- TestEmbeddingGeneratorInit (2 tests)
- TestEmbeddingGeneratorEncode (5 tests) - covers line 77 empty list
- TestEmbeddingGeneratorEncodeSingle (4 tests) - covers lines 100-111
- TestEmbeddingGeneratorEmbeddingDim (3 tests) - covers lines 121-126
- TestEmbeddingGeneratorModelLoading (3 tests)

**tests/test_api/test_encode_client.py** (EXTENDED - 16 new tests)
- TestRateLimiterSleep (2 tests) - covers lines 50-52
- TestFetchExperimentsFilters (4 tests) - covers lines 110-120
- TestParseExperimentEdgeCases (10 tests) - covers lines 228-280

**tests/test_ml/test_similarity.py** (EXTENDED - 3 new tests)
- TestSimilarityEngineEdgeCases - covers lines 140-141, 161-162, 179

**tests/test_processing/test_encoders.py** (EXTENDED - 7 new tests)
- TestCategoricalEncoderCoverageEdgeCases (5 tests) - covers lines 112-115, 120, 137, 144
- TestNumericEncoderCoverageEdgeCases (2 tests) - covers lines 241, 246

**tests/test_utils/test_cache.py** (EXTENDED - 2 new tests)
- TestCacheManagerCoverageEdgeCases - covers lines 110, 167

**tests/test_ml/test_feature_combiner.py** (EXTENDED - 2 new tests)
- TestFeatureCombinerCoverageEdgeCases - covers lines 201, 258

**tests/test_processing/test_metadata.py** (EXTENDED - 2 new tests)
- TestMetadataProcessorCoverageEdgeCases - covers line 163

**tests/conftest.py** (MODIFIED)
- Added `sample_2d_coords` fixture
- Added `sample_metadata_for_plotting` fixture
- Added `sample_similarity_matrix` fixture
- Added `sample_small_embeddings` fixture

---

## Remaining Work

**No remaining work - all modules at 100% coverage.**

---

## Detailed TODO by File

### 1. `src/visualization/plots.py` (0% → 100%)

**File:** `tests/test_visualization/test_plots.py` (NEW)

This is the largest gap. The entire visualization module is untested.

#### 1.1 DimensionalityReducer Tests

| Lines | Code | Test Required |
|-------|------|---------------|
| 30-47 | `__init__()` | Test initialization with umap/pca methods |
| 49-86 | `fit()` | Test fitting with UMAP and PCA methods |
| 60-73 | `fit()` umap branch | Test UMAP reducer creation with various sample sizes |
| 75-79 | `fit()` pca branch | Test PCA reducer creation |
| 81-82 | `fit()` invalid method | Test `ValueError` for unknown method |
| 88-101 | `transform()` | Test transform after fit |
| 97-98 | `transform()` not fitted | Test `ValueError` when not fitted |
| 103-138 | `fit_transform()` | Test combined fit+transform for UMAP and PCA |
| 134-135 | `fit_transform()` invalid | Test `ValueError` for unknown method |

```python
# Example test structure
class TestDimensionalityReducer:
    def test_init_umap(self):
        reducer = DimensionalityReducer(method="umap")
        assert reducer.method == "umap"

    def test_init_pca(self):
        reducer = DimensionalityReducer(method="pca")
        assert reducer.method == "pca"

    def test_fit_transform_umap(self, sample_embeddings):
        reducer = DimensionalityReducer(method="umap")
        coords = reducer.fit_transform(sample_embeddings)
        assert coords.shape == (len(sample_embeddings), 2)

    def test_fit_transform_pca(self, sample_embeddings):
        reducer = DimensionalityReducer(method="pca")
        coords = reducer.fit_transform(sample_embeddings)
        assert coords.shape == (len(sample_embeddings), 2)

    def test_invalid_method_raises(self):
        reducer = DimensionalityReducer(method="invalid")
        with pytest.raises(ValueError, match="Unknown method"):
            reducer.fit(np.random.randn(10, 384))

    def test_transform_before_fit_raises(self):
        reducer = DimensionalityReducer()
        with pytest.raises(ValueError, match="not been fitted"):
            reducer.transform(np.random.randn(5, 384))
```

#### 1.2 PlotGenerator Tests

| Lines | Code | Test Required |
|-------|------|---------------|
| 157-163 | `__init__()` | Test initialization |
| 165-249 | `scatter_plot()` | Test scatter plot generation |
| 203-211 | `scatter_plot()` with color_by | Test colored scatter plot |
| 212-219 | `scatter_plot()` without color_by | Test plain scatter plot |
| 222-238 | `scatter_plot()` with highlights | Test highlighted points |
| 251-285 | `similarity_heatmap()` | Test heatmap generation |

```python
class TestPlotGenerator:
    def test_scatter_plot_basic(self, sample_coords, sample_metadata):
        plotter = PlotGenerator()
        fig = plotter.scatter_plot(sample_coords, sample_metadata)
        assert fig is not None

    def test_scatter_plot_with_color(self, sample_coords, sample_metadata):
        plotter = PlotGenerator()
        fig = plotter.scatter_plot(sample_coords, sample_metadata, color_by="organism")
        assert fig is not None

    def test_scatter_plot_with_highlights(self, sample_coords, sample_metadata):
        plotter = PlotGenerator()
        fig = plotter.scatter_plot(sample_coords, sample_metadata, highlight_indices=[0, 1])
        assert len(fig.data) == 2  # Main trace + highlight trace

    def test_similarity_heatmap(self):
        plotter = PlotGenerator()
        matrix = np.array([[1, 0.5], [0.5, 1]])
        fig = plotter.similarity_heatmap(matrix, labels=["A", "B"])
        assert fig is not None
```

---

### 2. `src/ml/embeddings.py` (68% → 100%)

**File:** `tests/test_ml/test_embeddings.py` (extend existing)

| Lines | Code | Test Required |
|-------|------|---------------|
| 77 | `encode()` empty list | Test `encode([])` returns empty array with correct shape |
| 100-111 | `encode_single()` | Test single text encoding |
| 103-104 | `encode_single()` empty string | Test empty/whitespace string handling |
| 121-126 | `embedding_dim` unknown model | Test dim lookup for unknown model (triggers model load) |

```python
def test_encode_empty_list(self):
    """Line 77: Empty list returns (0, dim) array."""
    embedder = EmbeddingGenerator()
    result = embedder.encode([])
    assert result.shape == (0, 384)

def test_encode_single(self):
    """Lines 100-111: Single text encoding."""
    embedder = EmbeddingGenerator()
    result = embedder.encode_single("ChIP-seq on K562 cells")
    assert result.shape == (384,)
    assert result.dtype == np.float32

def test_encode_single_empty_string(self):
    """Lines 103-104: Empty string handled."""
    embedder = EmbeddingGenerator()
    result = embedder.encode_single("   ")
    assert result.shape == (384,)

def test_embedding_dim_unknown_model(self):
    """Lines 121-126: Unknown model triggers model load."""
    embedder = EmbeddingGenerator(model_name="all-mpnet-base-v2")
    # all-mpnet-base-v2 IS in MODEL_DIMENSIONS, so use a fake one
    embedder.model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Full name
    # This will fall through to model load
    dim = embedder.embedding_dim
    assert dim == 384
```

---

### 3. `src/api/encode_client.py` (78% → 100%)

**File:** `tests/test_api/test_encode_client.py` (extend existing)

#### 3.1 RateLimiter Tests

| Lines | Code | Test Required |
|-------|------|---------------|
| 50-52 | `wait_if_needed()` sleep branch | Test rate limiting triggers sleep |

```python
def test_rate_limiter_triggers_sleep(self, mocker):
    """Lines 50-52: Sleep when rate limit exceeded."""
    mock_sleep = mocker.patch("time.sleep")
    limiter = RateLimiter(max_requests=2, window_seconds=1.0)

    # Make 3 rapid requests
    limiter.wait_if_needed()
    limiter.wait_if_needed()
    limiter.wait_if_needed()  # Should trigger sleep

    mock_sleep.assert_called()
```

#### 3.2 fetch_experiments() Filter Tests

| Lines | Code | Test Required |
|-------|------|---------------|
| 110 | `assay_type` param | Test with assay_type filter |
| 112-114 | `organism` param | Test with organism filter |
| 116 | `biosample` param | Test with biosample filter |
| 120 | `limit=0` (all) | Test limit=0 sets "all" |

```python
def test_fetch_with_assay_type_filter(self, mocker):
    """Line 110: assay_type parameter."""
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"@graph": []}
    mocker.patch.object(client._session, "get", return_value=mock_response)

    client.fetch_experiments(assay_type="ChIP-seq")
    # Verify URL contains assay_term_name=ChIP-seq

def test_fetch_with_organism_filter(self, mocker):
    """Lines 112-114: organism parameter."""
    # Similar to above

def test_fetch_with_biosample_filter(self, mocker):
    """Line 116: biosample parameter."""
    # Similar to above

def test_fetch_with_limit_zero(self, mocker):
    """Line 120: limit=0 sets 'all'."""
    # Verify URL contains limit=all
```

#### 3.3 _parse_experiment() Edge Cases

| Lines | Code | Test Required |
|-------|------|---------------|
| 228 | lab as dict | Test `{"lab": {"title": "Lab Name"}}` |
| 234 | lab as plain string | Test `{"lab": "Lab Name"}` (no "/") |
| 241 | biosample_ontology not dict | Test `{"biosample_ontology": "/some/path/"}` |
| 250-258 | organism from replicates | Test deeply nested organism extraction |
| 269-270 | organism as string in biosample_ontology | Test `{"biosample_ontology": {"organism": "human"}}` |
| 279-280 | organism as string top-level | Test `{"organism": "human"}` |

```python
def test_parse_lab_as_dict(self):
    """Line 228: Lab as dictionary."""
    data = {"lab": {"title": "Snyder Lab", "name": "snyder"}}
    result = client._parse_experiment(data)
    assert result["lab"] == "Snyder Lab"

def test_parse_lab_as_plain_string(self):
    """Line 234: Lab as plain string (no slash)."""
    data = {"lab": "Snyder Lab"}
    result = client._parse_experiment(data)
    assert result["lab"] == "Snyder Lab"

def test_parse_biosample_ontology_not_dict(self):
    """Line 241: biosample_ontology as string."""
    data = {"biosample_ontology": "/biosample-types/cell_line/"}
    result = client._parse_experiment(data)
    assert result["biosample_term_name"] == ""

def test_parse_organism_from_nested_replicates(self):
    """Lines 250-258: Organism from replicates structure."""
    data = {
        "replicates": [{
            "library": {
                "biosample": {
                    "donor": {
                        "organism": {"name": "human", "scientific_name": "Homo sapiens"}
                    }
                }
            }
        }]
    }
    result = client._parse_experiment(data)
    assert result["organism"] == "human"

def test_parse_organism_string_in_biosample_ontology(self):
    """Lines 269-270: Organism as string."""
    data = {"biosample_ontology": {"organism": "mouse"}}
    result = client._parse_experiment(data)
    assert result["organism"] == "mouse"

def test_parse_organism_string_toplevel(self):
    """Lines 279-280: Top-level organism as string."""
    data = {"organism": "human"}
    result = client._parse_experiment(data)
    assert result["organism"] == "human"
```

---

### 4. `src/ml/similarity.py` (91% → 100%)

**File:** `tests/test_ml/test_similarity.py` (extend existing)

| Lines | Code | Test Required |
|-------|------|---------------|
| 140-141 | `compute_pairwise_similarity()` euclidean | Test with metric="euclidean" |
| 161-162 | `compute_similarity_matrix()` euclidean | Test matrix with euclidean |
| 179 | `get_embedding()` not fitted | Test raises ValueError |

```python
def test_pairwise_similarity_euclidean(self):
    """Lines 140-141: Euclidean metric."""
    engine = SimilarityEngine(metric="euclidean")
    engine.fit(sample_embeddings)
    sim = engine.compute_pairwise_similarity(sample_embeddings[0], sample_embeddings[1])
    assert 0 <= sim <= 1

def test_similarity_matrix_euclidean(self):
    """Lines 161-162: Matrix with euclidean."""
    engine = SimilarityEngine(metric="euclidean")
    engine.fit(sample_embeddings)
    matrix = engine.compute_similarity_matrix()
    assert matrix.shape == (len(sample_embeddings), len(sample_embeddings))

def test_get_embedding_not_fitted(self):
    """Line 179: Not fitted raises."""
    engine = SimilarityEngine()
    with pytest.raises(ValueError, match="not been fitted"):
        engine.get_embedding(0)
```

---

### 5. `src/processing/encoders.py` (92% → 100%)

**File:** `tests/test_processing/test_encoders.py` (extend existing)

| Lines | Code | Test Required |
|-------|------|---------------|
| 112-115 | Label encoding unknown + error | Test unknown category with label encoding + handle_unknown="error" |
| 120 | Invalid encoding_type | Test unknown encoding_type raises |
| 137 | n_categories not fitted | Test raises ValueError |
| 144 | categories not fitted | Test raises ValueError |
| 241 | minmax constant column | Test all same values (range=0) |
| 246 | Invalid method | Test unknown normalization method |

```python
def test_label_encoding_unknown_error(self):
    """Lines 112-115: Label encoding with unknown + error."""
    encoder = CategoricalEncoder(encoding_type="label", handle_unknown="error")
    encoder.fit(pd.Series(["A", "B"]))
    with pytest.raises(ValueError, match="Unknown category"):
        encoder.transform(pd.Series(["A", "C"]))

def test_invalid_encoding_type(self):
    """Line 120: Invalid encoding type."""
    encoder = CategoricalEncoder(encoding_type="invalid")
    encoder.fit(pd.Series(["A", "B"]))
    with pytest.raises(ValueError, match="Unknown encoding type"):
        encoder.transform(pd.Series(["A"]))

def test_n_categories_not_fitted(self):
    """Line 137: n_categories before fit."""
    encoder = CategoricalEncoder()
    with pytest.raises(ValueError, match="not been fitted"):
        _ = encoder.n_categories

def test_categories_not_fitted(self):
    """Line 144: categories before fit."""
    encoder = CategoricalEncoder()
    with pytest.raises(ValueError, match="not been fitted"):
        _ = encoder.categories

def test_minmax_constant_column(self):
    """Line 241: Minmax with constant values (range=0)."""
    encoder = NumericEncoder(method="minmax")
    result = encoder.fit_transform(pd.Series([5.0, 5.0, 5.0]))
    assert not np.isnan(result).any()
    assert (result == 0).all()  # Should be zeros

def test_invalid_normalization_method(self):
    """Line 246: Invalid method."""
    encoder = NumericEncoder(method="invalid")
    with pytest.raises(ValueError, match="Unknown normalization method"):
        encoder.fit(pd.Series([1, 2, 3]))
```

---

### 6. `src/utils/cache.py` (97% → 100%)

**File:** `tests/test_utils/test_cache.py` (extend existing)

| Lines | Code | Test Required |
|-------|------|---------------|
| 110 | `exists()` expired | Test exists returns False for expired |
| 167 | `_is_expired()` missing file | Test missing file returns True |

```python
def test_exists_returns_false_when_expired(self, tmp_path):
    """Line 110: Expired cache returns False."""
    cache = CacheManager(cache_dir=str(tmp_path), expiry_hours=0.001)  # ~3.6 seconds
    cache.save("test", {"data": 1})
    time.sleep(5)  # Wait for expiry
    assert not cache.exists("test")

def test_is_expired_missing_file(self, tmp_path):
    """Line 167: Missing file is expired."""
    cache = CacheManager(cache_dir=str(tmp_path), expiry_hours=1)
    # Check a file that doesn't exist
    result = cache._is_expired(tmp_path / "nonexistent.pkl")
    assert result is True
```

---

### 7. `src/ml/feature_combiner.py` (98% → 100%)

**File:** `tests/test_ml/test_feature_combiner.py` (extend existing)

| Lines | Code | Test Required |
|-------|------|---------------|
| 201 | No segments error | Test transform with no valid features |
| 258 | text_embedding is None | Test transform_single without text embedding |

```python
def test_transform_no_segments_raises(self):
    """Line 201: No features raises ValueError."""
    combiner = FeatureCombiner(weights={
        "text_embedding": 0,
        "assay_type": 0,
        "organism": 0,
        "cell_type": 0,
        "lab": 0,
        "numeric_features": 0
    })
    # Fit with minimal data
    df = pd.DataFrame({"accession": ["A"]})
    combiner.fit(df, text_embedding_dim=384)
    with pytest.raises(ValueError, match="No features to combine"):
        combiner.transform(df, np.zeros((1, 384)))

def test_transform_single_no_text_embedding(self, fitted_feature_combiner, sample_combined_df):
    """Line 258: transform_single without text embedding."""
    record = sample_combined_df.iloc[0].to_dict()
    result = fitted_feature_combiner.transform_single(record, text_embedding=None)
    # Should work but have zeros for text portion
    assert result is not None
```

---

### 8. `src/processing/metadata.py` (99% → 100%)

**File:** `tests/test_processing/test_metadata.py` (extend existing)

| Lines | Code | Test Required |
|-------|------|---------------|
| 163 | `_get_nested_value()` non-dict | Test nested path hitting non-dict |

```python
def test_get_nested_value_non_dict_intermediate(self):
    """Line 163: Non-dict in path returns None."""
    processor = MetadataProcessor()
    data = {"a": {"b": "string_not_dict"}}
    result = processor._get_nested_value(data, "a.b.c")
    assert result is None
```

---

## Test File Structure

```
tests/
├── test_api/
│   └── test_encode_client.py  (extend)
├── test_ml/
│   ├── test_embeddings.py     (extend)
│   ├── test_feature_combiner.py (extend)
│   └── test_similarity.py     (extend)
├── test_processing/
│   ├── test_encoders.py       (extend)
│   └── test_metadata.py       (extend)
├── test_utils/
│   └── test_cache.py          (extend)
├── test_visualization/        (NEW)
│   ├── __init__.py
│   └── test_plots.py          (NEW - ~150 lines)
├── conftest.py                (add fixtures)
└── test_integration.py
```

---

## New Fixtures Needed

Add to `tests/conftest.py`:

```python
@pytest.fixture
def sample_coords() -> np.ndarray:
    """2D coordinates for plot testing."""
    np.random.seed(42)
    return np.random.randn(10, 2).astype(np.float32)

@pytest.fixture
def sample_metadata_for_plots() -> pd.DataFrame:
    """Metadata DataFrame for plot testing."""
    return pd.DataFrame({
        "accession": [f"ENCSR{i:03d}" for i in range(10)],
        "description": [f"Experiment {i}" for i in range(10)],
        "assay_term_name": ["ChIP-seq"] * 5 + ["RNA-seq"] * 5,
        "organism": ["human"] * 7 + ["mouse"] * 3,
    })

@pytest.fixture
def sample_similarity_matrix() -> np.ndarray:
    """Sample similarity matrix for heatmap testing."""
    n = 5
    matrix = np.eye(n)
    np.fill_diagonal(matrix, 1.0)
    matrix += np.random.rand(n, n) * 0.3
    matrix = (matrix + matrix.T) / 2  # Symmetric
    return matrix.astype(np.float32)
```

---

## Estimated Effort

| Priority | New Lines of Test Code | Files to Modify |
|----------|------------------------|-----------------|
| High | ~200 | 2 new files |
| Medium | ~100 | 2 existing files |
| Low | ~80 | 5 existing files |
| **Total** | **~380** | **9 files** |

---

## Running Coverage

```bash
# Full coverage report
pytest -v --cov=src --cov-report=term-missing --cov-report=html

# Open HTML report
open htmlcov/index.html

# Check specific module
pytest tests/test_visualization/ -v --cov=src/visualization --cov-report=term-missing
```

---

## Verification Checklist

All tests passing with 100% coverage:

- [x] `src/visualization/plots.py` reaches 100%
- [x] `src/visualization/__init__.py` reaches 100%
- [x] `src/ml/embeddings.py` reaches 100%
- [x] `src/api/encode_client.py` reaches 100%
- [x] `src/ml/similarity.py` reaches 100%
- [x] `src/processing/encoders.py` reaches 100%
- [x] `src/utils/cache.py` reaches 100%
- [x] `src/ml/feature_combiner.py` reaches 100%
- [x] `src/processing/metadata.py` reaches 100%
- [x] All 9 modules at 100% = **TOTAL 100%**

---

## Final Summary

**Test coverage implementation complete.**

- Initial coverage: 81% (624 statements, 121 missing)
- Final coverage: 100% (624 statements, 0 missing)
- Total tests: 218
- New tests added: 94

All tests pass with no regressions. The test suite uses:
- Class-based test organization
- Mocking for external dependencies (SentenceTransformer, time.sleep, requests.Session)
- `np.random.seed(42)` for determinism
- `tmp_path` fixtures for file operations
- `pytest.raises` for error case validation
