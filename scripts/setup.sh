#!/usr/bin/env bash
# scripts/setup.sh
# One-command setup for new developers

set -e  # Exit on error

echo "Setting up MetaENCODE development environment..."

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION+ required, found $PYTHON_VERSION"
    exit 1
fi

echo "Python $PYTHON_VERSION detected"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install dependencies
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements-dev.txt -q

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Verify installation
echo "Verifying installation..."
python -c "import streamlit; import sentence_transformers; print('All imports OK')"

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the app, run:"
echo "  make run"
