# Contributing to maya-encoding

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/DanielRegaladoUMiami/maya-encoding.git
cd maya-encoding
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

We use `ruff` for linting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Ensure all tests pass
5. Submit a pull request

## Reporting Issues

Please use GitHub Issues and include:
- Python version
- maya-encoding version
- Minimal reproducible example
