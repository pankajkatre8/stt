# HSTTB - Healthcare Streaming STT Benchmarking
# Makefile for common development commands

.PHONY: help install install-dev install-all test test-cov lint format typecheck clean webapp webapp-dev docs build publish

# Default target
help:
	@echo "HSTTB - Healthcare Streaming STT Benchmarking"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Installation:"
	@echo "  install        Install package in production mode"
	@echo "  install-dev    Install package with development dependencies"
	@echo "  install-all    Install package with all optional dependencies"
	@echo "  install-api    Install package with API/webapp dependencies"
	@echo ""
	@echo "Development:"
	@echo "  test           Run tests"
	@echo "  test-cov       Run tests with coverage report"
	@echo "  test-fast      Run tests without slow tests"
	@echo "  lint           Run linter (ruff)"
	@echo "  lint-fix       Run linter and auto-fix issues"
	@echo "  format         Format code (ruff format)"
	@echo "  typecheck      Run type checker (mypy)"
	@echo "  check          Run all checks (lint + typecheck + test)"
	@echo ""
	@echo "Web Application:"
	@echo "  webapp         Run web application (production)"
	@echo "  webapp-dev     Run web application with auto-reload"
	@echo "  webapp-docker  Build and run webapp in Docker"
	@echo ""
	@echo "Documentation:"
	@echo "  docs           Open documentation"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          Remove build artifacts and cache"
	@echo "  clean-all      Remove all generated files including venv"
	@echo "  build          Build distribution packages"
	@echo "  publish        Publish to PyPI (requires credentials)"

# =============================================================================
# Installation
# =============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

install-api:
	pip install -e ".[api]"

# =============================================================================
# Testing
# =============================================================================

test:
	PYTHONPATH=src pytest tests/ -v

test-cov:
	PYTHONPATH=src pytest tests/ -v --cov=hsttb --cov-report=html --cov-report=term-missing

test-fast:
	PYTHONPATH=src pytest tests/ -v -m "not slow"

test-unit:
	PYTHONPATH=src pytest tests/unit/ -v

test-integration:
	PYTHONPATH=src pytest tests/ -v -m "integration"

# Quick test with uvx (no install required)
test-quick:
	PYTHONPATH=src uvx --with pytest --with pytest-asyncio --with pytest-cov \
		--with pydantic --with rapidfuzz --with numpy --with pyyaml --with soundfile \
		pytest tests/ -v

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

format:
	ruff format src/ tests/

format-check:
	ruff format src/ tests/ --check

typecheck:
	mypy src/hsttb

# Run all checks
check: lint typecheck test
	@echo "All checks passed!"

# =============================================================================
# Web Application
# =============================================================================

# Production mode
webapp:
	PYTHONPATH=src uvicorn hsttb.webapp.app:app --host 0.0.0.0 --port 8000

# Development mode with auto-reload
webapp-dev:
	PYTHONPATH=src uvicorn hsttb.webapp.app:app --reload --host 127.0.0.1 --port 8000

# Quick start with uvx (no install required)
webapp-quick:
	PYTHONPATH=src uvx --with fastapi --with uvicorn --with jinja2 \
		--with pydantic --with rapidfuzz --with numpy --with pyyaml \
		uvicorn hsttb.webapp.app:app --reload --port 8000

# Build Docker image
webapp-docker-build:
	docker build -t hsttb:latest .

# Run Docker container
webapp-docker-run:
	docker run -p 8000:8000 hsttb:latest

# Build and run Docker
webapp-docker: webapp-docker-build webapp-docker-run

# =============================================================================
# CLI Commands
# =============================================================================

# List available profiles
profiles:
	PYTHONPATH=src python -m hsttb.cli profiles

# List available adapters
adapters:
	PYTHONPATH=src python -m hsttb.cli adapters

# =============================================================================
# Documentation
# =============================================================================

docs:
	@echo "Opening documentation..."
	@open docs/user_guide.md 2>/dev/null || xdg-open docs/user_guide.md 2>/dev/null || echo "See docs/user_guide.md"

docs-serve:
	@echo "Documentation available at:"
	@echo "  - docs/user_guide.md"
	@echo "  - docs/workflow_explainer.md"
	@echo "  - docs/requirements_assessment.md"

# =============================================================================
# Build & Publish
# =============================================================================

build: clean
	python -m build

publish: build
	python -m twine upload dist/*

publish-test: build
	python -m twine upload --repository testpypi dist/*

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean
	rm -rf .venv/
	rm -rf venv/

# =============================================================================
# Development Helpers
# =============================================================================

# Create virtual environment
venv:
	python -m venv .venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source .venv/bin/activate"

# Update dependencies
update:
	pip install --upgrade pip
	pip install -e ".[dev]" --upgrade

# Show project info
info:
	@echo "HSTTB - Healthcare Streaming STT Benchmarking"
	@echo ""
	@echo "Python: $$(python --version)"
	@echo "Pip: $$(pip --version)"
	@echo ""
	@echo "Project Structure:"
	@echo "  src/hsttb/       - Main source code"
	@echo "  tests/           - Test suite"
	@echo "  docs/            - Documentation"
	@echo ""
	@echo "Key Metrics:"
	@echo "  TER - Term Error Rate"
	@echo "  NER - Named Entity Recognition Accuracy"
	@echo "  CRS - Context Retention Score"
	@echo "  SRS - Streaming Robustness Score"

# Count lines of code
loc:
	@echo "Lines of code:"
	@find src -name "*.py" | xargs wc -l | tail -1
	@echo ""
	@echo "Lines of tests:"
	@find tests -name "*.py" | xargs wc -l | tail -1

# Show test count
test-count:
	@PYTHONPATH=src pytest tests/ --collect-only -q 2>/dev/null | tail -1

# =============================================================================
# CI/CD Helpers
# =============================================================================

ci: lint typecheck test-cov
	@echo "CI checks passed!"

ci-quick: lint-fix format test
	@echo "Quick CI checks passed!"

# =============================================================================
# Examples
# =============================================================================

# Run example evaluation
example:
	@echo "Running example evaluation..."
	@PYTHONPATH=src python -c "\
from hsttb.metrics.ter import TEREngine; \
from hsttb.metrics.crs import CRSEngine; \
from hsttb.lexicons.mock_lexicon import MockMedicalLexicon; \
lexicon = MockMedicalLexicon(); \
ter = TEREngine(lexicon); \
crs = CRSEngine(); \
gt = 'Patient takes metformin 500mg for diabetes'; \
pred = 'Patient takes methotrexate 500mg for diabetes'; \
ter_result = ter.compute(gt, pred); \
crs_result = crs.compute([gt], [pred]); \
print(f'Ground Truth: {gt}'); \
print(f'Predicted:    {pred}'); \
print(f'TER: {ter_result.overall_ter:.2%}'); \
print(f'CRS: {crs_result.composite_score:.2%}'); \
"

# Run example with uvx (no install)
example-quick:
	@PYTHONPATH=src uvx --with pydantic --with rapidfuzz --with numpy --with pyyaml \
		python -c "\
from hsttb.metrics.ter import TEREngine; \
from hsttb.metrics.crs import CRSEngine; \
from hsttb.lexicons.mock_lexicon import MockMedicalLexicon; \
lexicon = MockMedicalLexicon(); \
ter = TEREngine(lexicon); \
crs = CRSEngine(); \
gt = 'Patient takes metformin 500mg for diabetes'; \
pred = 'Patient takes methotrexate 500mg for diabetes'; \
ter_result = ter.compute(gt, pred); \
crs_result = crs.compute([gt], [pred]); \
print(f'Ground Truth: {gt}'); \
print(f'Predicted:    {pred}'); \
print(f'TER: {ter_result.overall_ter:.2%}'); \
print(f'CRS: {crs_result.composite_score:.2%}'); \
"
