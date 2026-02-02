#!/bin/bash
# HSTTB Local Development Setup
#
# Usage:
#   ./scripts/dev.sh           # Start development server
#   ./scripts/dev.sh --refresh # Force refresh lexicon
#   ./scripts/dev.sh --check   # Check lexicon status only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================"
echo "HSTTB Development Setup"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies if needed
if [ ! -f ".venv/.deps_installed" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install -e ".[dev,api]"
    pip install httpx  # For API fetching
    touch .venv/.deps_installed
fi

# Parse arguments
REFRESH=""
CHECK_ONLY=""
RELOAD="--reload"

for arg in "$@"; do
    case $arg in
        --refresh)
            REFRESH="--refresh"
            ;;
        --check)
            CHECK_ONLY="--check-only"
            ;;
        --no-reload)
            RELOAD=""
            ;;
    esac
done

# Run startup script
echo ""
echo "Starting HSTTB..."
echo ""

python scripts/startup.py $REFRESH $CHECK_ONLY $RELOAD --port 8000
