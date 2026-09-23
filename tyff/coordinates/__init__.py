"""
Coordinate storage and management for molecular simulations.
"""

from .box import BoxCoordinates, MoleculeSpecies
from .store import CoordinateStore

__all__ = [
    "BoxCoordinates",
    "CoordinateStore",
    "MoleculeSpecies",
]
