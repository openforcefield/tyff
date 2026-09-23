import numpy as np
import pytest

from tyff.coordinates.box import BoxCoordinates, MoleculeSpecies, Substance

rng = np.random.default_rng(42)
# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db_path(tmp_path):
    """
    Provide a temporary database path for testing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory fixture

    Returns
    -------
    pathlib.Path
        Path to temporary database file
    """
    return tmp_path / "test_coordinates.sqlite"


@pytest.fixture
def sample_water_molecule_species():
    return MoleculeSpecies(mapped_smiles="[H:1][O:2][H:3]", count=100)


@pytest.fixture
def sample_molecule_species():
    """
    Create sample MoleculeSpecies objects for testing.

    Returns
    -------
    list[MoleculeSpecies]
        List of sample molecule species
    """
    return [
        MoleculeSpecies(mapped_smiles="[H:1][O:2][H:3]", count=100),
        MoleculeSpecies(mapped_smiles="[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])[H:8]", count=50),
    ]


@pytest.fixture
def sample_substance(sample_molecule_species):
    """
    Create a sample Substance object for testing.

    Parameters
    ----------
    sample_molecule_species : list[MoleculeSpecies]
        List of sample molecule species

    Returns
    -------
    Substance
        Sample Substance object
    """
    return Substance(molecule_species=sample_molecule_species)


@pytest.fixture
def sample_coordinates():
    """
    Create sample coordinate array for testing.

    Returns
    -------
    np.ndarray
        Sample coordinates array (n_atoms, 3)
    """
    return rng.normal(size=(700, 3)) * 10.0  # 100*3 + 50*8 atoms


@pytest.fixture
def sample_box_vectors():
    """
    Create sample box vectors for testing.

    Returns
    -------
    np.ndarray
        Sample box vectors (3, 3)
    """
    return np.array([[30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]])


@pytest.fixture
def sample_binary_box_coordinates(sample_substance, sample_coordinates, sample_box_vectors):
    """
    Create a sample BoxCoordinates object for testing.

    Parameters
    ----------
    sample_substance : Substance
        Sample substance
    sample_coordinates : np.ndarray
        Sample coordinates
    sample_box_vectors : np.ndarray
        Sample box vectors

    Returns
    -------
    BoxCoordinates
        Sample BoxCoordinates object
    """
    return BoxCoordinates(
        substance=sample_substance,
        temperature=298.15,
        pressure=1.0,
        force_field_id="openff-2.1.0",
        potential_energy=-1234.56,
        coordinates=sample_coordinates,
        box_vectors=sample_box_vectors,
        box_metadata={"equilibration_steps": 10000, "production_steps": 100000},
    )
