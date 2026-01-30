# TASK-1C01: Project Setup

## Metadata
- **Status**: pending
- **Complexity**: Small (< 1 hour)
- **Blocked By**: None
- **Blocks**: All other tasks

## Objective

Initialize the Python project structure with proper packaging, dependencies, and development tools.

## Context

This is the first task of the project. It establishes the foundation that all other code builds on. Getting this right ensures smooth development workflow.

## Requirements

- [ ] Create `pyproject.toml` with project metadata
- [ ] Create `requirements.txt` with pinned dependencies
- [ ] Create source directory structure `src/hsttb/`
- [ ] Create test directory structure `tests/`
- [ ] Set up development tools (ruff, mypy, pytest)
- [ ] Create initial `__init__.py` files
- [ ] Create `.gitignore`
- [ ] Verify `pip install -e .` works

## Acceptance Criteria

- [ ] AC1: Running `pip install -e ".[dev]"` succeeds
- [ ] AC2: Running `pytest` completes (even with no tests)
- [ ] AC3: Running `ruff check src/` completes
- [ ] AC4: Running `mypy src/` completes
- [ ] AC5: `from hsttb import __version__` works

## Files to Create

```
hsttb/
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── src/
│   └── hsttb/
│       ├── __init__.py
│       ├── core/
│       │   └── __init__.py
│       ├── audio/
│       │   └── __init__.py
│       ├── adapters/
│       │   └── __init__.py
│       ├── lexicons/
│       │   └── __init__.py
│       ├── nlp/
│       │   └── __init__.py
│       ├── metrics/
│       │   └── __init__.py
│       ├── evaluation/
│       │   └── __init__.py
│       └── reporting/
│           └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   └── __init__.py
│   └── integration/
│       └── __init__.py
└── configs/
    └── streaming_profiles/
```

## Implementation Details

### pyproject.toml
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "hsttb"
version = "0.1.0"
description = "Healthcare Streaming STT Benchmarking Framework"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "pydantic>=2.0.0",
    "soundfile>=0.12.0",
    "pyyaml>=6.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.8.0",
]

[project.scripts]
hsttb = "hsttb.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### src/hsttb/__init__.py
```python
"""
HSTTB - Healthcare Streaming STT Benchmarking Framework.

A model-agnostic evaluation framework for healthcare speech-to-text.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
```

## Testing Requirements

- Unit tests: No (setup only)
- Integration tests: No
- Manual verification:
  - [ ] `pip install -e ".[dev]"` succeeds
  - [ ] `python -c "from hsttb import __version__; print(__version__)"`

## Notes

- Use `src/` layout for proper package isolation
- Pin major versions only to allow patch updates
- Keep initial dependencies minimal
- Add more dependencies as needed in later tasks
