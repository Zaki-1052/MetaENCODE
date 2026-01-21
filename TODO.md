# MetaENCODE: Product Requirements Document

## Executive Summary

MetaENCODE is an interactive web application that helps users discover related biological datasets from the Encyclopedia of DNA Elements (ENCODE) through metadata-driven similarity scoring. The system transforms dataset metadata into numeric representations (embeddings for text, vectors for categorical/numeric fields) and uses ML-based similarity computation to rank and recommend datasets, reducing manual filtering and enabling exploratory science.

**Project Context:** DS3 x UBIC collaborative project led by Vanshika + Isha
**Team Size:** 4 students + DS3 mentor + UBIC mentor
**Timeline:** 10 weeks (sprint-based)
**Final Deliverable:** Deployed Streamlit web application + open-source GitHub repository

---

## 1. Problem Statement

### 1.1 Current Pain Points
- Searching ENCODE for related datasets requires extensive manual filtering
- Searches often return hundreds of results without meaningful ranking
- No existing tool provides similarity-based dataset recommendations
- Researchers must manually parse metadata to find conceptually related experiments

### 1.2 Target Users
- Researchers searching for publicly available genomic, transcriptomic, and regulatory activity datasets
- Scientists doing exploratory research or finding gaps in existing data
- Anyone working with ENCODE data who needs to find similar experiments

### 1.3 Value Proposition
MetaENCODE streamlines dataset discovery by:
- Providing ranked similarity scores based on multiple metadata attributes
- Enabling exploration through interactive visualizations
- Reducing time spent on manual filtering and comparison

---

## 2. Data Source: ENCODE REST API

### 2.1 API Overview
The ENCODE Data Coordination Center (DCC) provides a RESTful API returning JSON-formatted data. All objects are accessible via URLs containing unique identifiers (accession numbers).

**Base URL:** `https://www.encodeproject.org/`
**Rate Limit:** Maximum 10 requests/second

### 2.2 API Authentication
- Public data is accessible without authentication (GET requests)
- No API key required for read-only access to released objects

### 2.3 Core API Patterns

#### Basic Object Retrieval
```python
import requests, json

headers = {'accept': 'application/json'}
url = 'https://www.encodeproject.org/biosample/ENCBS000AAA/?frame=object'
response = requests.get(url, headers=headers)
biosample = response.json()
```

#### Programmatic Search
```python
url = 'https://www.encodeproject.org/search/?searchTerm=bone+chip&frame=object&format=json'
response = requests.get(url, headers=headers)
search_results = response.json()
# Results are in search_results['@graph']
```

### 2.4 Key URL Parameters

| Parameter | Example | Description |
|-----------|---------|-------------|
| `frame=object` | Returns all properties with embedded objects as identifiers |
| `frame=embedded` | Returns all properties with selected embedded objects expanded |
| `limit=all` | Returns all results (default is 25) |
| `limit=N` | Returns up to N results |
| `searchTerm=X` | Full-text search for string X |
| `type=X` | Filter by object type (Experiment, Biosample, File, etc.) |
| `format=json` | Force JSON response format |
| `!=` | Negation operator (e.g., `assay_title!=ChIP-seq`) |
| `*` | Wildcard for existence check (e.g., `treatments=*`) |
| `lt:`, `lte:`, `gt:`, `gte:` | Range operators for numeric fields |

### 2.5 Response Structure

Search responses have this structure:
```json
{
    "notification": "Success",
    "title": "Search",
    "@id": "/search/?...",
    "total": 180,
    "facets": [...],
    "@graph": [
        { /* first result object */ },
        { /* second result object */ },
        ...
    ]
}
```

The `@graph` array contains the actual search results.

### 2.6 Key ENCODE Object Types

**Experiment** - Core object containing:
- `accession`: Unique identifier (e.g., "ENCSR000AKS")
- `description`: Text description of the experiment
- `assay_term_name`: Type of assay (ChIP-seq, RNA-seq, etc.)
- `biosample_ontology`: Information about sample type
- `lab`: Laboratory that performed experiment
- `files`: Array of associated data files
- `replicates`: Array of replicate information
- `status`: Release status

**Biosample** - Sample information:
- `accession`: Unique identifier
- `organism`: Species (human, mouse, etc.)
- `biosample_ontology`: Cell type/tissue classification
- `life_stage`: adult, embryonic, etc.
- `age`, `age_units`: Sample age
- `treatments`: Any applied treatments
- `donor`: Reference to donor information

**File** - Data file metadata:
- `accession`: Unique identifier
- `file_format`: fastq, bigWig, bed, etc.
- `file_size`: Size in bytes
- `dataset`: Reference to parent experiment

### 2.7 Example API Queries

1. **All experiments of a specific assay type:**
   ```
   https://www.encodeproject.org/search/?type=Experiment&assay_term_name=ChIP-seq&frame=object&format=json&limit=all
   ```

2. **Files from a specific experiment:**
   ```
   https://www.encodeproject.org/search/?type=File&dataset=/experiments/ENCSR000AKS/&format=json&frame=object
   ```

3. **Biosamples with treatments:**
   ```
   https://www.encodeproject.org/search/?type=Biosample&treatments=*&frame=object
   ```

4. **Experiments NOT of a certain type:**
   ```
   https://www.encodeproject.org/search/?type=Experiment&assay_title!=ChIP-seq&frame=object
   ```

---

## 3. Metadata Attributes for Similarity

### 3.1 Text Fields (for SBERT Embeddings)
- `title` - Experiment/dataset title
- `description` - Free-text description of the experiment

### 3.2 Categorical Fields (for Encoding)
- `organism` - Species (human, mouse, etc.)
- `assay_term_name` / `assay_type` - Type of experiment
- `biosample_ontology.term_name` - Cell type or tissue
- `lab` - Laboratory identifier
- `cell` - Cell line name (if applicable)
- `antibody` - Target antibody (for ChIP experiments)

### 3.3 Numeric Fields
- Replicate count
- Sample count
- File count

### 3.4 Reference: Java Implementation Attribute Handling

From the existing Java implementation, these column headings were used:
```java
private static final String[] columnHeadings = {
    "cell", "dataType", "antibody", "view", "replicate", "type", "lab"
};
```

The `EncodeFileRecord` class shows how metadata was stored:
```java
public class EncodeFileRecord {
    private final String path;
    private final Map<String, String> attributes;
    // Key attributes accessed: cell, antibody, dataType, view, replicate
}
```

Track name generation logic (useful for display):
```java
public String getTrackName() {
    StringBuilder sb = new StringBuilder();
    if (attributes.containsKey("cell")) sb.append(attributes.get("cell")).append(" ");
    if (attributes.containsKey("antibody")) sb.append(attributes.get("antibody")).append(" ");
    if (attributes.containsKey("dataType")) sb.append(attributes.get("dataType")).append(" ");
    if (attributes.containsKey("view")) sb.append(attributes.get("view")).append(" ");
    if (attributes.containsKey("replicate")) sb.append("rep ").append(attributes.get("replicate"));
    return sb.toString().trim();
}
```

Metadata validation check:
```java
public boolean hasMetaData() {
    return (attributes.containsKey("cell")) ||
           (attributes.containsKey("antibody") ||
            attributes.containsKey("Biosample"));
}
```

---

## 4. Technical Requirements

### 4.1 Core Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Frontend | Streamlit | Interactive web UI with session state management |
| Text Embeddings | SBERT (Sentence Transformers) | Convert text metadata to vectors |
| Similarity | scikit-learn | Cosine similarity, nearest neighbor search |
| Visualization | UMAP/PCA | Dimensionality reduction for plots |
| Data Processing | pandas | DataFrame operations and data cleaning |
| API Interaction | requests | HTTP calls to ENCODE API |

### 4.2 SBERT Integration

**Library:** `sentence-transformers` (https://www.sbert.net/)

SBERT provides pre-trained models for generating semantic embeddings from text. These embeddings capture meaning, allowing similar descriptions to have similar vector representations.

**Typical Usage:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # or similar model
embeddings = model.encode(list_of_descriptions)
```

### 4.3 Similarity Computation

**Cosine Similarity:**
- Measures angle between two vectors
- Range: -1 to 1 (1 = identical direction)
- From scikit-learn: `sklearn.metrics.pairwise.cosine_similarity`

**Nearest Neighbor Search:**
- For efficient retrieval of top-N similar items
- From scikit-learn: `sklearn.neighbors.NearestNeighbors`

### 4.4 Feature Engineering Pipeline

1. **Text Fields:**
   - Clean and normalize (lowercase, remove special characters)
   - Generate SBERT embeddings
   - Result: Dense vector per text field

2. **Categorical Fields:**
   - One-hot encoding or label encoding
   - Result: Sparse/dense vector per categorical field

3. **Numeric Fields:**
   - Normalize/standardize values
   - Result: Scalar or small vector

4. **Combined Vector:**
   - Concatenate all feature vectors
   - Optional: Weight different components
   - Result: Single combined vector per dataset

---

## 5. Functional Requirements

### 5.1 Core Features

#### F1: Dataset Search/Selection
- User can search for datasets by keyword
- User can select a dataset from results
- System displays dataset metadata

#### F2: Similarity Recommendations
- Given a selected dataset, return top N similar datasets
- Display similarity scores
- Allow user to configure N (number of results)

#### F3: Filtering
- Filter by organism (human, mouse, etc.)
- Filter by assay type (ChIP-seq, RNA-seq, etc.)
- Filter by sample count range
- Optional: Weighted filtering (advanced)

#### F4: Visualization
- Dimensionality reduction plot (UMAP or PCA)
- Scatter plot colored by organism/assay
- Hover tooltips showing dataset metadata
- Interactive plot elements

#### F5: Dataset Details
- View full metadata for any dataset
- Link to original ENCODE page

### 5.2 User Flow

1. User enters search term or browses datasets
2. User selects a "seed" dataset of interest
3. System computes similarity to all other datasets
4. System displays ranked list of similar datasets
5. User applies filters to narrow results
6. User explores visualization of dataset relationships
7. User clicks through to ENCODE for full details

### 5.3 Session State Requirements

Streamlit session state must persist:
- Current selected dataset
- Filter settings
- Search history
- Computed embeddings (cached)

---

## 6. Non-Functional Requirements

### 6.1 Performance
- Precompute embeddings for large dataset sets
- Cache similarity computations where possible
- Target: Results displayed within 2-3 seconds after selection

### 6.2 Reproducibility
- Consistent results across sessions
- Document random seeds if used
- Version-lock dependencies

### 6.3 Scalability
- Handle metadata for thousands of ENCODE experiments
- Efficient data structures for similarity lookup

### 6.4 Data Freshness
- Option to refresh data from ENCODE API
- Clear indication of data retrieval date

---

## 7. Sprint Schedule & Milestones

### Week 1: Team Setup & API Onboarding
- [ ] Form teams, establish communication
- [ ] Biology/bioinformatics crash course
- [ ] Get familiar with ENCODE API and Streamlit

### Week 2: API Integration & Dataset Fetching
- [ ] Explore ENCODE REST API, understand endpoints and JSON structure
- [ ] Set up development environment (Python, pandas, Streamlit, scikit-learn)
- [ ] Fetch small subset of metadata from ENCODE API
- [ ] Build functions to fetch experiment metadata dynamically
- [ ] Parse JSON and convert to pandas DataFrame

### Week 3: Metadata Preprocessing
- [ ] Identify key attributes for recommendation
- [ ] Clean and normalize text fields (title, description)
- [ ] Encode categorical fields (organism, assay)
- [ ] Encode numeric fields (sample count)
- [ ] Combine all metadata into single vector per dataset

### Week 4: Text Embeddings
- [ ] Integrate pre-trained SBERT for textual metadata
- [ ] Test embeddings on subset of datasets
- [ ] Validate similarity ranking on examples

### Week 5: Similarity Computation
- [ ] Implement cosine similarity / nearest neighbor search
- [ ] Return top N similar datasets for selected dataset
- [ ] Validate results on known examples

### Week 6: Streamlit App - Basic Functionality
- [ ] Build UI for selecting/searching datasets
- [ ] Display recommended datasets in interactive table
- [ ] Implement filters (organism, assay, sample size)

### Week 7: Visualization
- [ ] Add dimensionality reduction (UMAP/PCA)
- [ ] Create interactive scatter plots colored by organism/assay
- [ ] Implement hover tooltips with dataset metadata

### Week 8: Optimization & Caching
- [ ] Precompute embeddings for large dataset sets
- [ ] Implement optional refresh from ENCODE API
- [ ] Ensure reproducibility and consistent results

### Week 9: Testing & Documentation
- [ ] Test app with multiple datasets and users
- [ ] Document preprocessing, embedding, API calls, app usage
- [ ] Prepare GitHub repo with README, code structure, screenshots

### Week 10: Final Deployment & Presentation (Dino Cage)
- [ ] Deploy app publicly (Streamlit Cloud or similar)
- [ ] Present project (API integration, preprocessing, embeddings, similarity, visualization)
- [ ] Highlight learning outcomes

---

## 8. Deliverables & Grading Criteria

### 8.1 Design Document (5 points)
| Criterion | Points | Requirements |
|-----------|--------|--------------|
| Depth & Thoroughness | 2 | Clear explanation of system architecture, workflow, major components |
| Design Rationale | 1 | Justification of design decisions and tradeoffs |
| Technical Diagrams | 1 | Data flow diagrams, database schema, architecture diagrams |
| Clarity & Professionalism | 1 | Well-written, logically structured, readable |

### 8.2 GitHub Repository (15 points)
| Criterion | Points | Requirements |
|-----------|--------|--------------|
| Code Quality & Organization | 6 | Clean, readable, consistent formatting, modularization, folders, style |
| Documentation in Code | 2 | Docstrings & comments with parameters, return values, intent |
| Testing | 2 | Unit tests/integration tests covering functionality, running successfully |
| Functionality & Performance | 3 | Runs without errors, meets requirements, efficient algorithms |
| Version Control Hygiene | 2 | Meaningful commits, proper branches/PRs, .gitignore configured |

### 8.3 Code Documentation (10 points)
| Criterion | Points | Requirements |
|-----------|--------|--------------|
| Depth & Clarity | 4 | Comprehensive explanation of components, architecture, algorithms |
| Setup & Deployment Instructions | 3 | Local run instructions, environment setup (.env), replication steps |
| Coverage | 2 | Spans full codebase, includes limitations and bugs |
| README.md | 1 | Complete, polished, useful |

### 8.4 Meeting Logs (5 points)
| Criterion | Points | Requirements |
|-----------|--------|--------------|
| Consistency | 1 | Weekly entries with dates and agendas |
| Detail & Thoroughness | 2 | Progress, decisions, next steps recorded |
| Industry/Faculty Meetings | 1 | Mentor meetings labeled and detailed |
| Team Engagement | 1 | Attendance tracked, active participation shown |

### 8.5 Website/Final Report (15 points)
| Criterion | Points | Requirements |
|-----------|--------|--------------|
| Functionality/Completeness | 5 | Website loads properly, all major features work |
| Structure/Content | 4 | Functions as described, usage is self-explanatory |
| Technical Depth | 3 | Explanation of modeling approaches, engineering choices |
| Professionalism & Visual Quality | 3 | Clean formatting, consistent layout, functional links |

**Total Deliverables: 50 points**

### 8.6 DinoCage Presentation (50 points)
| Category | Points | Key Criteria |
|----------|--------|--------------|
| Storytelling, Problem & Motivation | 10 | Problem clarity, narrative flow, background |
| Technical Depth & Credibility | 15 | Methods explanation, data quality, results interpretation |
| Demo Quality | 10 | Functioning demo, clear walkthrough |
| Engagement & Slide Delivery | 10 | Pacing, confidence, balanced speaking, clean slides |
| Q&A Mastery & Professionalism | 5 | Deep understanding, thoughtful handling, professional presence |

---

## 9. Reference: Existing Implementation Patterns

### 9.1 Record Filtering (from Java implementation)

The existing `RegexFilter` class shows how multi-criteria filtering was implemented:

```java
// Filter string parsing: supports "column=value" syntax
// Multiple terms are ANDed together
// Wildcard (*) matches any column
String[] tokens = text.split("\\s+");
for (String t : tokens) {
    String column = "*";
    String value = t.trim();
    if (t.contains("=")) {
        String[] kv = t.split("=");
        column = kv[0].trim();
        value = kv[1].trim();
    }
    // Create regex matcher for value
}
```

### 9.2 Data Loading Pattern

From `EncodeFileBrowser.getEncodeFileRecords()`:
```java
// Fetch from S3 (or could be adapted for direct API)
is = ParsingUtils.openInputStream(
    "https://s3.amazonaws.com/igv.org.app/encode/" + genomeId + ".txt.gz"
);

// Parse header row
String[] headers = reader.readLine().split("\t");

// Parse data rows into records
while ((nextLine = reader.readLine()) != null) {
    String[] tokens = nextLine.split("\t");
    String path = tokens[pathLocation];

    Map<String, String> attributes = new HashMap<>();
    for (int i = 0; i < headers.length; i++) {
        if (tokens[i].length() > 0) {
            attributes.put(headers[i], tokens[i]);
        }
    }

    EncodeFileRecord record = new EncodeFileRecord(path, attributes);
    if (record.hasMetaData()) {
        records.add(record);
    }
}
```

### 9.3 Known File Types (for reference)
```java
private static final HashSet<String> knownFileTypes = new HashSet<>(
    Arrays.asList("bam", "bigBed", "bed", "bb", "bw", "bigWig",
                  "gtf", "broadPeak", "narrowPeak", "gff")
);
```

---

## 10. Learning Objectives

By project completion, team members will have gained experience in:

1. **API Integration**
   - Working with RESTful APIs
   - Parsing JSON responses
   - Handling pagination and rate limits

2. **Streamlit Development**
   - Building interactive web applications
   - Managing session states
   - Creating responsive UI components

3. **Bioinformatics/Genomics**
   - Understanding genome datasets and metadata
   - Working with ENCODE data structures
   - Domain-specific data preprocessing

4. **Machine Learning/NLP**
   - Working with sentence transformers (SBERT)
   - Generating and using text embeddings
   - Implementing similarity scoring

5. **Data Science Techniques**
   - Feature engineering (text, categorical, numeric)
   - Dimensionality reduction (UMAP/PCA)
   - Visualization of high-dimensional data

---

## 11. Key Resources

### Official Documentation
- ENCODE REST API: https://www.encodeproject.org/help/rest-api/
- SBERT / Sentence Transformers: https://www.sbert.net/
- Streamlit: https://docs.streamlit.io/

### Python Libraries
- `requests` - HTTP library for API calls
- `pandas` - Data manipulation
- `sentence-transformers` - SBERT embeddings
- `scikit-learn` - Similarity computation, ML utilities
- `umap-learn` - UMAP dimensionality reduction
- `plotly` or `altair` - Interactive visualizations
- `streamlit` - Web application framework

---

## 12. Glossary

| Term | Definition |
|------|------------|
| **ENCODE** | Encyclopedia of DNA Elements - comprehensive catalog of functional elements in human and mouse genomes |
| **Accession** | Unique identifier for ENCODE objects (e.g., ENCSR000AKS) |
| **Assay** | Experimental technique (ChIP-seq, RNA-seq, ATAC-seq, etc.) |
| **Biosample** | Biological sample used in experiment (cell line, tissue, etc.) |
| **SBERT** | Sentence-BERT - transformer model for generating sentence embeddings |
| **Cosine Similarity** | Measure of similarity between two vectors based on angle |
| **Embedding** | Dense vector representation of text or categorical data |
| **UMAP** | Uniform Manifold Approximation and Projection - dimensionality reduction technique |
| **PCA** | Principal Component Analysis - linear dimensionality reduction |

---

## Appendix A: Claude Code Prompt

The following prompt should be used when working with Claude Code on this project:

```
You are building MetaENCODE, a Streamlit web application for discovering related ENCODE datasets through metadata-driven similarity scoring.

## Project Context
- This is a DS3 x UBIC collaborative bioinformatics project
- Target users are researchers searching ENCODE for related genomic datasets
- Core value: ML-based recommendations to reduce manual filtering

## Technical Stack
- Frontend: Streamlit (with session state management)
- Text Embeddings: sentence-transformers (SBERT)
- Similarity: scikit-learn (cosine similarity, NearestNeighbors)
- Visualization: UMAP/PCA with plotly or altair
- Data: pandas DataFrames
- API: requests library for ENCODE REST API

## ENCODE API Key Details
- Base URL: https://www.encodeproject.org/
- Rate limit: 10 requests/second maximum
- No authentication needed for public data
- Always use headers = {'accept': 'application/json'}
- Key parameters: frame=object, limit=all, format=json, type=Experiment

## Core Functionality Requirements
1. Dataset Search/Selection - search ENCODE, select seed dataset
2. Similarity Recommendations - return top N similar datasets with scores
3. Filtering - by organism, assay type, sample count
4. Visualization - UMAP/PCA scatter plots with tooltips
5. Session State - persist selections, filters, cached embeddings

## Metadata Fields for Similarity
- Text (SBERT): title, description
- Categorical (encoding): organism, assay_term_name, biosample_ontology.term_name, lab
- Numeric (normalize): replicate count, sample count

## Code Quality Standards
- Modular, DRY, well-documented code
- Docstrings for all functions
- Error handling for API calls
- Type hints where appropriate
- Unit tests for core logic

## Reference Implementation Pattern
The Java implementation shows useful patterns:
- Store records as objects with path + attributes map
- Track name: "{cell} {antibody} {dataType} {view} rep {replicate}"
- Validate records have minimum metadata before including
- Support column=value filter syntax

When implementing, always reference the PRD for requirements and API patterns. Ask clarifying questions before making assumptions about functionality or architecture.
```

---

## Appendix B: Quick Reference - Common API Patterns

### Fetch All Experiments
```python
import requests

headers = {'accept': 'application/json'}
url = 'https://www.encodeproject.org/search/?type=Experiment&frame=object&format=json&limit=all'
response = requests.get(url, headers=headers)
data = response.json()
experiments = data['@graph']
```

### Fetch Specific Experiment by Accession
```python
url = f'https://www.encodeproject.org/experiments/{accession}/?frame=object&format=json'
response = requests.get(url, headers=headers)
experiment = response.json()
```

### Search with Multiple Filters
```python
params = {
    'type': 'Experiment',
    'assay_term_name': 'ChIP-seq',
    'biosample_ontology.term_name': 'K562',
    'frame': 'object',
    'format': 'json',
    'limit': 'all'
}
response = requests.get('https://www.encodeproject.org/search/',
                       params=params, headers=headers)
```

### Extract Key Fields from Experiment
```python
def extract_metadata(experiment):
    return {
        'accession': experiment.get('accession'),
        'description': experiment.get('description', ''),
        'assay': experiment.get('assay_term_name'),
        'organism': experiment.get('biosample_ontology', {}).get('term_name'),
        'lab': experiment.get('lab'),
        'status': experiment.get('status'),
        'files': experiment.get('files', []),
        'replicates': experiment.get('replicates', [])
    }
```
