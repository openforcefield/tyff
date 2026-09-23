"""
Unit tests for tyff package.
"""

import tyff


def test_import():
    """Test that the package can be imported."""
    assert tyff is not None


def test_version():
    """Test that version is defined."""
    assert hasattr(tyff, "__version__")
    assert isinstance(tyff.__version__, str)


def test_author():
    """Test that author is defined."""
    assert hasattr(tyff, "__author__")
    assert isinstance(tyff.__author__, str)
