# meta-encode/Makefile
# Development workflow commands

.PHONY: install dev setup run test lint format clean help

# Default target
help:
	@echo "MetaENCODE Development Commands"
	@echo "================================"
	@echo "make setup      - Full setup (venv + deps + hooks)"
	@echo "make install    - Install production dependencies"
	@echo "make dev        - Install all dependencies (prod + dev)"
	@echo "make run        - Run Streamlit application"
	@echo "make test       - Run pytest with coverage"
	@echo "make lint       - Run all linters (black, isort, flake8, mypy)"
	@echo "make format     - Auto-format code with black and isort"
	@echo "make clean      - Remove cache and build artifacts"
	@echo "make hooks      - Install pre-commit hooks"

# Full setup for new developers
setup:
	./scripts/setup.sh

# Install production dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Install all dependencies including dev
dev:
	pip install --upgrade pip
	pip install -r requirements-dev.txt

# Run Streamlit app
run:
	streamlit run app.py

# Run tests with coverage
test:
	pytest -v --cov=src --cov-report=term-missing

# Run all linters
lint:
	black --check .
	isort --check .
	flake8 src tests app.py
	mypy src

# Auto-format code
format:
	black .
	isort .

# Install pre-commit hooks
hooks:
	pre-commit install
	pre-commit run --all-files

# Clean cache and build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
