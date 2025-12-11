# Contributing to Advanced RAG

First off, thank you for considering contributing to Advanced RAG! It's people like you that make this project better for everyone.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Advanced-RAG.git
   cd Advanced-RAG
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/Advanced-RAG.git
   ```

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, error messages)
- **Describe the behavior you observed and what you expected**
- **Include your environment** (OS, Docker version, GPU, etc.)

### 💡 Suggesting Features

Feature suggestions are welcome! Please:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested feature
- **Explain why this feature would be useful**
- **List any alternatives you've considered**

### 🔧 Pull Requests

1. **Create a branch** from `main`:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit them:

   ```bash
   git commit -m "feat: add amazing feature"
   ```

3. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** against the `main` branch

## Development Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Poetry (recommended) or pip
- NVIDIA GPU (optional, for local vLLM)

### Local Development

```bash
# Install dependencies
pip install poetry
poetry install

# Or with pip
pip install -r requirements.txt

# Run the backend locally
poetry run uvicorn src.main:app --reload --port 8000

# Or start all services with Docker
docker compose up -d
```

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src
```

## Pull Request Process

1. **Ensure your code follows the style guidelines**
2. **Update the README.md** if needed
3. **Add tests** for new functionality
4. **Ensure all tests pass**
5. **Update documentation** as needed
6. **Request review** from maintainers

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(retrieval): add hybrid search with RRF fusion
fix(cache): resolve race condition in semantic cache
docs(readme): update installation instructions
```

## Style Guidelines

### Python

- Follow [PEP 8](https://pep8.org/)
- Use type hints where possible
- Maximum line length: 100 characters
- Use docstrings for functions and classes

```python
def process_document(file_path: str) -> List[TextNode]:
    """
    Process a document and return chunked nodes.

    Args:
        file_path: Path to the document file

    Returns:
        List of TextNode objects with embeddings
    """
    ...
```

### Formatting

We use the following tools:

```bash
# Format code
poetry run black src/
poetry run isort src/

# Lint
poetry run flake8 src/
```

## Questions?

Feel free to open an issue with the `question` label if you have any questions about contributing.

Thank you for contributing! 🎉
