# Contributing to Flash3D

Thank you for your interest in contributing to Flash3D! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/FlashVision.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Install dev dependencies: `pip install -e ".[dev,full]"`
5. Install pre-commit hooks: `pre-commit install`

## Development Workflow

### Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Line length: 100 characters
- Type hints are required for all public APIs
- Docstrings follow Google style

### Running Tests

```bash
pytest tests/ -v
pytest tests/test_models.py -k "test_gaussian"  # Run specific test
pytest tests/ --cov=flash3d --cov-report=html    # Coverage report
```

### Pre-commit Checks

```bash
pre-commit run --all-files
```

## Contribution Guidelines

### What We Accept

- Bug fixes with test coverage
- New model architectures (with paper reference)
- Dataset loader implementations
- Performance optimizations
- Documentation improvements
- New examples and tutorials

### Pull Request Process

1. Ensure all tests pass: `pytest tests/ -v`
2. Ensure linting passes: `ruff check flash3d/ tests/`
3. Update documentation if adding new features
4. Add tests for new functionality
5. Update CHANGELOG.md
6. Submit PR with clear description

### Commit Messages

Use conventional commits:
- `feat: add support for 4D Gaussian Splatting`
- `fix: correct SH evaluation for degree 3`
- `docs: update depth estimation guide`
- `test: add tests for point cloud filtering`
- `refactor: simplify camera ray generation`

## Architecture Decisions

- All models implement a common `forward()` interface
- Use the Registry pattern for extensibility
- Prefer pure PyTorch over custom CUDA when possible
- Config-driven training (YAML files)
- High-level Solutions API wraps low-level components

## Questions?

Open an issue with the "question" label or reach out to the maintainers.
