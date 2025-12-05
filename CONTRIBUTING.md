# TDA Perturbation Analysis - Contributing Guidelines

Thank you for your interest in contributing to the TDA Perturbation Analysis pipeline!

## Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/tda_perturbation_analysis.git
cd tda_perturbation_analysis

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Run tests
pytest tests/ -v
```

## Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all public functions
- Keep functions focused and modular

## Testing

- Add tests for new features
- Ensure all tests pass before submitting PR
- Use `pytest` for testing

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit pull request

## Questions?

Open an issue or discussion on GitHub.
