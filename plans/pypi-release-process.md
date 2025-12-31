# ToDACoMM PyPI Release Process

## Overview

This document outlines the process for publishing todacomm to PyPI as version 0.1.0 (alpha release).

## Pre-Release Checklist

### 1. Code Quality Verification

- [x] All tests pass (589 passed, 89 skipped for model downloads)
- [x] Test coverage at 68% overall (core modules 85-99%)
- [x] Package builds successfully (`python -m build`)
- [x] `twine check dist/*` passes
- [x] CLI entry point works (`todacomm --help`)
- [x] `__version__` exported correctly

### 2. Metadata Updates Required

Update `pyproject.toml` classifiers to include Python 3.12 and 3.13:

```toml
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]
```

### 3. Documentation Verification

- [x] README.md is comprehensive
- [x] LICENSE file present (MIT)
- [x] Installation instructions work
- [x] CLI examples documented

## Release Steps

### Step 1: Create Release Branch

```bash
git checkout -b release/v0.1.0
```

### Step 2: Update Version (if needed)

Version is already set to `0.1.0` in:
- `pyproject.toml`
- `todacomm/__init__.py`

### Step 3: Update Classifiers

Add Python 3.12 and 3.13 to pyproject.toml classifiers.

### Step 4: Clean Build

```bash
rm -rf dist/ build/ *.egg-info
python -m build --sdist --wheel
```

### Step 5: Verify Package

```bash
twine check dist/*
```

### Step 6: Test on TestPyPI (Recommended)

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ todacomm
```

### Step 7: Upload to PyPI

```bash
twine upload dist/*
```

### Step 8: Verify Installation

```bash
pip install todacomm
todacomm --help
todacomm list-models
```

### Step 9: Create GitHub Release

```bash
git tag -a v0.1.0 -m "Initial PyPI release"
git push origin v0.1.0
```

Create release on GitHub with:
- Tag: v0.1.0
- Title: ToDACoMM v0.1.0 - Initial Release
- Release notes summarizing features

## Post-Release

### Version Bump

After release, bump version to `0.1.1.dev0` for continued development:

```python
# todacomm/__init__.py
__version__ = "0.1.1.dev0"
```

```toml
# pyproject.toml
version = "0.1.1.dev0"
```

### Merge Back to Main

```bash
git checkout main
git merge release/v0.1.0
git push origin main
```

## Package Name Availability

- Package name `todacomm` is available on PyPI (verified 2024-12-31)

## Dependencies Note

Current dependencies include heavy packages (torch, transformers). Future versions may consider:
- Making torch/transformers optional dependencies
- Creating a `todacomm[full]` extra for complete functionality
- Lightweight `todacomm[core]` with just TDA utilities

## Authentication

PyPI upload requires:
1. PyPI account at https://pypi.org/account/register/
2. API token from https://pypi.org/manage/account/token/
3. Configure `~/.pypirc` or use `TWINE_USERNAME` and `TWINE_PASSWORD` environment variables

```ini
# ~/.pypirc
[pypi]
username = __token__
password = pypi-<your-token>

[testpypi]
username = __token__
password = pypi-<your-test-token>
```
