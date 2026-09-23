# Contributing to tyff

Thank you for your interest in contributing to tyff! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/tyff.git
   cd tyff
   ```
3. Install the development dependencies:

   Using conda/mamba (recommended):
   ```bash
   mamba env create -f devtools/conda-envs/dev.yaml
   mamba activate tyff-dev
   pip install -e .
   ```

4. (Optional but recommended) Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

   This will automatically run code formatting and checks before each commit.

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature-name
   ```

2. Make your changes and ensure:
   - Code follows the project's style guidelines
   - All tests pass
   - New features include tests
   - Documentation is updated if needed

3. Run tests:
   ```bash
   pytest
   ```

4. Check code style (easiest to use pre-commit hooks):
   ```bash
   pre-commit run --all-files
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

6. Push to your fork:
   ```bash
   git push origin feature-name
   ```

7. Submit a pull request through GitHub

## Code Style

This project uses:
- **ruff** for code formatting and linting
- Type hints are encouraged but not required

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage
- Use pytest fixtures for common test setup

## Documentation

- Update documentation for new features
- Use NumPy-style docstrings
- Build documentation locally to verify:
  ```bash
  cd docs
  make html
  ```

## Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Include a clear description of the changes
- Reference any related issues
- Ensure CI passes before requesting review
- Be responsive to feedback

## Questions?

Feel free to open an issue for questions or discussion!
