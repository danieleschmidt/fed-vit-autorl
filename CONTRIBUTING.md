# Contributing to Fed-ViT-AutoRL

Thank you for your interest in contributing to Fed-ViT-AutoRL! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/terragon-labs/fed-vit-autorl.git
   cd fed-vit-autorl
   ```

2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Run tests**:
   ```bash
   pytest
   ```

4. **Run linting and formatting**:
   ```bash
   ruff check .
   black .
   mypy fed_vit_autorl/
   ```

5. **Create a pull request** with a clear description

## Coding Standards

- **Code Style**: We use Black for formatting and Ruff for linting
- **Type Hints**: All new code must include type hints
- **Documentation**: Include docstrings for all public functions and classes
- **Testing**: Maintain minimum 80% test coverage

## Testing

- Write unit tests for all new functionality
- Use pytest for testing framework
- Test files should be in `tests/` directory
- Run specific tests: `pytest tests/test_specific.py`

## Security

- Follow secure coding practices
- Use Bandit for security scanning
- Never commit secrets or credentials
- Report security issues privately to security@terragon.ai

## Questions?

- Open an issue for bug reports or feature requests
- Use discussions for questions and general topics
- Contact maintainers for sensitive issues

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).