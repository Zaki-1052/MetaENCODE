# MetaENCODE

**ENCODE Dataset Similarity Search using ML-based Recommendations**

MetaENCODE is an interactive web application that helps researchers discover related biological datasets from the Encyclopedia of DNA Elements (ENCODE) through metadata-driven similarity scoring.

## Features

- **Dataset Search**: Search ENCODE for experiments by keyword, assay type, organism, or biosample
- **Similarity Recommendations**: Find the top N most similar datasets based on metadata embeddings
- **Interactive Filtering**: Filter results by organism, assay type, and sample characteristics
- **Visualization**: Explore dataset relationships through UMAP/PCA scatter plots
- **Detailed Metadata**: View full experiment details with links to the ENCODE portal

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/meta-encode.git
   cd meta-encode
   ```

2. Run the setup script (recommended):
   ```bash
   ./scripts/setup.sh
   source .venv/bin/activate
   ```

   Or manually:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements-dev.txt
   pre-commit install
   ```

3. Run the application:
   ```bash
   make run
   ```

4. Open your browser to `http://localhost:8501`

## Development

### Project Structure

```
meta-encode/
├── app.py                      # Main Streamlit application
├── src/
│   ├── api/                    # ENCODE REST API client
│   │   └── encode_client.py
│   ├── processing/             # Metadata processing
│   │   ├── metadata.py
│   │   └── encoders.py
│   ├── ml/                     # Machine learning components
│   │   ├── embeddings.py       # SBERT text embeddings
│   │   └── similarity.py       # Cosine similarity search
│   ├── visualization/          # Plotting utilities
│   │   └── plots.py
│   └── utils/                  # Utilities
│       └── cache.py
├── tests/                      # Test suite
├── data/cache/                 # Cached embeddings and API responses
├── requirements.txt            # Production dependencies
├── requirements-dev.txt        # Development dependencies
└── Makefile                    # Development commands
```

### Available Commands

```bash
make help     # Show all available commands
make install  # Install production dependencies
make dev      # Install all dependencies (prod + dev)
make run      # Run Streamlit application
make test     # Run pytest with coverage
make lint     # Run all linters
make format   # Auto-format code with black and isort
make hooks    # Install pre-commit hooks
make clean    # Remove cache and build artifacts
```

### Code Quality

This project uses:
- **black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing
- **pre-commit** for automated checks

Pre-commit hooks are configured to run automatically on every commit.

### Running Tests

```bash
make test
# Or directly:
pytest -v --cov=src --cov-report=term-missing
```

## Technical Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Text Embeddings | Sentence Transformers (SBERT) |
| Similarity | scikit-learn (cosine similarity, NearestNeighbors) |
| Visualization | Plotly, UMAP |
| Data Processing | pandas, NumPy |
| API Interaction | requests |

## ENCODE API

This application uses the [ENCODE REST API](https://www.encodeproject.org/help/rest-api/) to fetch experiment metadata. No authentication is required for public data access.

**Base URL**: `https://www.encodeproject.org/`
**Rate Limit**: 10 requests/second

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make test && make lint`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [ENCODE Project](https://www.encodeproject.org/) for providing the data and API
- [Sentence Transformers](https://www.sbert.net/) for pre-trained embedding models
- DS3 x UBIC collaborative project team

---

*Built with Streamlit and SBERT*
