# Contributing to ToDACoMM

Thank you for your interest in contributing to ToDACoMM!

## Project Scope

ToDACoMM is a **descriptive framework** for characterizing transformer representations using **persistent homology**. Before contributing, please understand:

### What This Project IS

- A **measurement tool** for topological properties of neural network activations
- A **comparative framework** revealing architecture-specific signatures
- A **reproducible methodology** using standard persistent homology (Ripser)

### What This Project Is NOT

- A predictive model for perplexity or performance
- A benchmark for ranking models
- A causal theory explaining model behavior

### Contribution Guidelines by Type

| Contribution Type | Aligned With Project? |
|-------------------|----------------------|
| New TDA metrics (H0, H1, higher homology) | Yes |
| New model support | Yes |
| New dataset support | Yes |
| Visualization improvements | Yes |
| Performance optimizations | Yes |
| Bug fixes | Yes |
| Adding non-topological metrics (e.g., intrinsic dimension, anisotropy) | No - would dilute focus |
| Predictive modeling on TDA features | No - outside scope |

---

## Development Setup

```bash
# Clone repository
git clone https://github.com/aiexplorations/todacomm.git
cd todacomm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v -m "not slow"
```

---

## Code Style

- Follow **PEP 8** for Python formatting
- Use **type hints** for all functions
- Add **docstrings** to all public functions
- Keep functions focused and modular

---

## Testing Requirements

- Add tests for new features
- Ensure all tests pass before submitting PR
- Use `pytest` for testing
- Target **80%+ code coverage** for new code

```bash
# Run tests with coverage
pytest --cov=todacomm --cov-report=html

# Run full suite including slow tests
pytest tests/ -v --run-slow
```

---

## Pull Request Process

1. **Fork** the repository
2. **Create a feature branch** from `main`
3. **Make focused changes** (one feature per PR)
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Submit pull request** with clear description

### PR Description Template

```markdown
## Summary
Brief description of what this PR does.

## Type
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation

## Changes
- List of specific changes

## Testing
- How was this tested?

## Related Issues
Fixes #XX
```

---

## Questions?

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Research questions and methodology discussions
- **Email**: rexplorations@gmail.com
