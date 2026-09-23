# tyff

[![CI](https://github.com/openforcefield/tyff/actions/workflows/gh-ci.yaml/badge.svg)](https://github.com/openforcefield/tyff/actions/workflows/gh-ci.yaml)
[![Documentation Status](https://readthedocs.org/projects/tyff/badge/?version=latest)](https://tyff.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/openforcefield/tyff/branch/main/graph/badge.svg)](https://codecov.io/gh/openforcefield/tyff)

**t**rain **y**our **f**orce **f**ield

## Overview

`tyff` is a Python package for fitting force fields via tensor representations.

## Installation

### From source

```bash
git clone https://github.com/openforcefield/tyff.git
cd tyff
pip install -e .
```

### Development installation

For development, use conda/mamba to create an environment from the provided file:

```bash
mamba env create -f devtools/conda-envs/dev.yaml
mamba activate tyff-dev
pip install -e .
```

Alternatively, install with pip using the dev extras:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import tyff

print(f"tyff version: {tyff.__version__}")
```

## Documentation

Full documentation is available at [tyff.readthedocs.io](https://tyff.readthedocs.io).

## Testing

Run tests with pytest:

```bash
pytest
```

Or with coverage:

```bash
pytest --cov=tyff --cov-report=html
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project derives from other projects. See [LICENSE-3RD-PARTY](LICENSE-3RD-PARTY) for details.

## Authors

- Lily Wang

## Acknowledgments

This package structure follows best practices from the [MolSSI Cookiecutter](https://github.com/molssi/cookiecutter-cms).
