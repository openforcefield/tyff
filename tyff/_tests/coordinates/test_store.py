"""
Comprehensive tests for tyff.coordinates.store module.

Tests cover all classes and methods with aim for 100% code coverage.
"""

import json

import numpy as np
import openmm
import pytest
from openff.toolkit import ForceField
from sqlmodel import Session, select

from tyff._tests.utils import get_test_data_path
from tyff.coordinates.box import (
    BoxCoordinates,
    CoordinatesDB,
    MoleculeSpecies,
    Substance,
)
from tyff.coordinates.store import CoordinateStore, DBMetadata


@pytest.fixture
def empty_coordinate_store(temp_db_path):
    """
    Create a CoordinateStore instance for testing.

    Parameters
    ----------
    temp_db_path : pathlib.Path
        Temporary database path

    Returns
    -------
    CoordinateStore
        CoordinateStore instance
    """
    return CoordinateStore(temp_db_path)


@pytest.fixture
def temp_coordinate_store_path(tmp_path):
    """
    Create a CoordinateStore instance for testing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory path

    Returns
    -------
    CoordinateStore
        CoordinateStore instance
    """
    existing_store = get_test_data_path("thermoml/test.sqlite")

    # copy to temp location to avoid modifying original test data
    path = tmp_path / "test_copy.sqlite"
    with open(existing_store, "rb") as src, open(path, "wb") as dst:
        dst.write(src.read())
    return path


@pytest.fixture
def temp_coordinate_store(temp_coordinate_store_path):
    store = CoordinateStore(temp_coordinate_store_path)
    return store


@pytest.fixture
def substance_water_1000():
    water = Substance(
        molecule_species=[MoleculeSpecies(mapped_smiles="[H:2][O:1][H:3]", count=1000)],
    )
    return water


class TestCoordinateStoreInit:
    """Tests for CoordinateStore initialization."""

    def test_create_store_creates_database(self, temp_db_path):
        """Test that creating a store creates the database file."""
        assert not temp_db_path.exists()
        store = CoordinateStore(temp_db_path)
        assert temp_db_path.exists()
        assert store.count() == 0

    def test_create_store_creates_parent_directory(self, tmp_path):
        """Test that creating a store creates parent directories."""
        db_path = tmp_path / "subdir" / "another" / "test.db"
        assert not db_path.parent.exists()
        CoordinateStore(db_path)
        assert db_path.exists()
        assert db_path.parent.exists()

    def test_store_initializes_schema_version(self, empty_coordinate_store):
        """Test that store initializes schema version metadata."""
        from sqlmodel import Session, select

        with Session(empty_coordinate_store.engine) as session:
            metadata = session.exec(select(DBMetadata).where(DBMetadata.key == "schema_version")).first()
            assert metadata is not None
            assert metadata.value == "1"

    def test_open_existing_database(self, temp_coordinate_store_path):
        """Test opening an existing database."""
        # Create initial store
        store1 = CoordinateStore(temp_coordinate_store_path)
        assert store1.db_path == temp_coordinate_store_path

        assert store1.count() > 0  # Should have data from test.sqlite

    def test_schema_version_mismatch_raises_error(self, tmp_path):
        """Test that schema version mismatch raises ValueError."""
        # Create store with version 1
        path = tmp_path / "test_mismatch.sqlite"
        store1 = CoordinateStore(path)

        # Manually change schema version in database
        with Session(store1.engine) as session:
            metadata = session.exec(select(DBMetadata).where(DBMetadata.key == "schema_version")).first()
            metadata.value = "999"
            session.add(metadata)
            session.commit()

        # Try to open with current version (should fail)
        with pytest.raises(ValueError, match="Database schema version mismatch"):
            CoordinateStore(path)


class TestCoordinateStoreAddGet:
    """Tests for adding and retrieving boxes."""

    def test_add_box(self, empty_coordinate_store, sample_binary_box_coordinates):
        """Test adding a box to the database."""
        n_boxes = empty_coordinate_store.count()
        assert n_boxes == 0
        box_id = empty_coordinate_store.add(sample_binary_box_coordinates)
        assert isinstance(box_id, int)
        assert box_id == 1
        assert empty_coordinate_store.count() == 1

    @pytest.mark.skip(reason="Not implemented yet")
    def test_add_box_duplicate(self, empty_coordinate_store, sample_binary_box_coordinates):
        """Test adding a duplicate box."""
        assert empty_coordinate_store.count() == 0
        box_id1 = empty_coordinate_store.add(sample_binary_box_coordinates)
        assert empty_coordinate_store.count() == 1
        box_id2 = empty_coordinate_store.add(sample_binary_box_coordinates)
        assert box_id1 == box_id2  # Should deduplicate and return same ID
        assert empty_coordinate_store.count() == 1

    def test_get_box(self, empty_coordinate_store, sample_binary_box_coordinates):
        """Test retrieving a box by ID."""
        box_id = empty_coordinate_store.add(sample_binary_box_coordinates)
        retrieved = empty_coordinate_store.get(box_id)

        assert retrieved is not None
        assert retrieved.id == box_id == 1
        assert retrieved.temperature == sample_binary_box_coordinates.temperature
        assert retrieved.pressure == sample_binary_box_coordinates.pressure
        assert retrieved.force_field_id == sample_binary_box_coordinates.force_field_id
        assert retrieved.n_molecules == sample_binary_box_coordinates.n_molecules

    def test_get_nonexistent_box(self, empty_coordinate_store):
        """Test getting a box that doesn't exist returns None."""
        result = empty_coordinate_store.get(99999)
        assert result is None

    def test_add_multiple_boxes(self, empty_coordinate_store, sample_binary_box_coordinates):
        """Test adding multiple boxes."""
        ids = []
        for i in range(3):
            box = sample_binary_box_coordinates.model_copy(deep=True)
            box.temperature = 300.0 + i * 10.0
            ids.append(empty_coordinate_store.add(box))

        assert len(ids) == 3
        assert len(set(ids)) == 3  # All unique

        # Verify all can be retrieved
        for box_id in ids:
            assert empty_coordinate_store.get(box_id) is not None

    def test_delete(self, empty_coordinate_store, sample_binary_box_coordinates):
        """Test deleting a box."""
        box_id = empty_coordinate_store.add(sample_binary_box_coordinates)
        assert empty_coordinate_store.get(box_id) is not None

        empty_coordinate_store.delete(box_id)
        assert empty_coordinate_store.get(box_id) is None

    def test_delete_nonexistent(self, empty_coordinate_store):
        """Test deleting a box that doesn't exist (should not raise error)."""
        empty_coordinate_store.delete(99999)  # Should not raise error

    def test_get_box_matches_by_substance_no_state(self, temp_coordinate_store):
        """
        Test getting box matches by substance when no state information is provided.
        This tests multiple counts of substance for water.
        """
        water = Substance(
            molecule_species=[MoleculeSpecies(mapped_smiles="[H:2][O:1][H:3]", count=100)],
        )
        matches = temp_coordinate_store.get_box_matches_by_substance(water)
        assert len(matches) == 0

        water.molecule_species[0].count = 1000
        matches = temp_coordinate_store.get_box_matches_by_substance(water)
        assert len(matches) == 4
        for box in matches:
            assert box.substance.to_composition_key() == "InChI=1/H2O/h1H2:1000"

        water.molecule_species[0].count = 2000
        matches = temp_coordinate_store.get_box_matches_by_substance(water)
        assert len(matches) == 3
        for box in matches:
            assert box.substance.to_composition_key() == "InChI=1/H2O/h1H2:2000"

    def test_get_box_matches_by_substance_temperature(self, temp_coordinate_store, substance_water_1000):
        """
        Test getting box matches by substance with temperature
        """

        matches = temp_coordinate_store.get_box_matches_by_substance(
            substance_water_1000, temperature=298.15, temperature_tolerance=0.0
        )
        matches = sorted(matches, key=lambda x: x.id)
        assert len(matches) == 2
        assert matches[0].id == 2
        assert matches[0].temperature == 298.15
        assert matches[0].pressure == 1.0
        assert matches[1].id == 3
        assert matches[1].temperature == 298.15
        assert matches[1].pressure is None

    def test_get_box_matches_by_substance_pressure_wide(self, temp_coordinate_store, substance_water_1000):
        """
        Test getting box matches by substance when pressure is specified
        """
        matches = temp_coordinate_store.get_box_matches_by_substance(
            substance_water_1000, pressure=1.0, pressure_tolerance=0.1
        )
        matches = sorted(matches, key=lambda x: x.id)
        assert len(matches) == 3
        assert matches[0].id == 1
        assert matches[0].temperature == 313.15
        assert np.allclose(matches[0].pressure, 0.9969, atol=0.001)
        assert matches[1].id == 2
        assert matches[1].temperature == 298.15
        assert matches[1].pressure == 1.0
        assert matches[2].id == 4
        assert matches[2].temperature == 313.15
        assert np.allclose(matches[2].pressure, 0.9969, atol=0.001)

    def test_get_box_matches_by_substance_pressure_narrow(self, temp_coordinate_store, substance_water_1000):
        # test filter narrow
        matches = temp_coordinate_store.get_box_matches_by_substance(
            substance_water_1000, pressure=1.0, pressure_tolerance=0.0001
        )
        matches = sorted(matches, key=lambda x: x.id)
        assert len(matches) == 1
        assert matches[0].id == 2
        assert matches[0].temperature == 298.15
        assert matches[0].pressure == 1.0

    def test_get_box_matches_by_substance_pressure_none(self, temp_coordinate_store, substance_water_1000):
        matches = temp_coordinate_store.get_box_matches_by_substance(
            substance_water_1000,
            pressure=None,
            pressure_tolerance=0.01,
        )
        matches = sorted(matches, key=lambda x: x.id)
        assert len(matches) == 1
        assert matches[0].id == 3
        assert matches[0].temperature == 298.15
        assert matches[0].pressure is None

    def test_get_box_matches_by_substance_both(self, temp_coordinate_store, substance_water_1000):
        matches = temp_coordinate_store.get_box_matches_by_substance(
            substance_water_1000,
            temperature=298.15,
            temperature_tolerance=30,
            pressure=1.0,
            pressure_tolerance=0.0001,
        )
        matches = sorted(matches, key=lambda x: x.id)
        assert len(matches) == 1
        assert matches[0].id == 2
        assert matches[0].temperature == 298.15
        assert matches[0].pressure == 1.0


class TestCoordinateStoreGetLowestEnergy:
    """Tests for getting lowest energy box."""

    def test_get_lowest_energy_no_matches(self, temp_coordinate_store, sample_binary_box_coordinates):
        """Test getting lowest energy when no matches exist."""
        result = temp_coordinate_store.get_lowest_energy_box_by_system(sample_binary_box_coordinates, openmm.System())
        assert result is None

    @pytest.mark.skip(reason="Not implemented yet")
    def test_get_lowest_energy_tip3p(self, temp_coordinate_store, substance_water_1000):
        """Check we get the expected box back"""
        ff = ForceField("tip3p.offxml")
        box = BoxCoordinates(
            substance=substance_water_1000,
        )
        lowest = temp_coordinate_store.get_lowest_energy_box_by_force_field(box, ff)
        assert lowest.id == 4

    @pytest.mark.skip(reason="Not implemented yet")
    def test_get_lowest_energy_tip3p_mod(self, temp_coordinate_store, substance_water_1000):
        """Ensure force field parameters affect results"""
        ff = ForceField("tip3p.offxml")
        handler = ff.get_parameter_handler("LibraryCharges")
        handler.parameters[0].charge[0] *= 0.5
        handler.parameters[1].charge[0] *= 0.5

        box = BoxCoordinates(
            substance=substance_water_1000,
        )
        lowest = temp_coordinate_store.get_lowest_energy_box_by_force_field(box, ff)
        assert lowest.id == 3

    @pytest.mark.skip(reason="Not implemented yet")
    def test_get_lowest_energy_tip3p_mod_state(self, temp_coordinate_store, substance_water_1000):
        """Ensure thermophysical state affects results"""
        ff = ForceField("tip3p.offxml")
        box = BoxCoordinates(
            substance=substance_water_1000,
        )
        lowest = temp_coordinate_store.get_lowest_energy_box_by_force_field(
            box, ff, temperature=298, temperature_tolerance=1
        )
        assert lowest.id == 2


class TestCoordinateStoreQuery:
    """Tests for querying database metadata."""

    def test_get_force_fields(self, empty_coordinate_store, sample_binary_box_coordinates):
        """Test getting list of force fields."""
        # Add boxes with different force fields
        for ff_id in ["openff-2.1.0", "openff-2.0.0", "amber14"]:
            box = sample_binary_box_coordinates.model_copy(deep=True)
            box.force_field_id = ff_id
            empty_coordinate_store.add(box)

        force_fields = empty_coordinate_store.get_force_fields()
        assert len(force_fields) == 3
        assert "openff-2.1.0" in force_fields
        assert "amber14" in force_fields

    def test_get_force_fields_empty(self, empty_coordinate_store):
        """Test getting force fields from empty database."""
        force_fields = empty_coordinate_store.get_force_fields()
        assert force_fields == []

    def test_get_compositions_existing(self, temp_coordinate_store):
        """Test getting compositions from existing database."""
        compositions = temp_coordinate_store.get_compositions()
        assert len(compositions) == 6  # Should have at least one composition


class TestCoordinateStoreExportMerge:
    """Tests for exporting and merging databases."""

    def test_export_to_db_all(self, temp_coordinate_store, tmp_path):
        """Test exporting all boxes to another database."""

        # Export to new database
        target_path = tmp_path / "export.sqlite"
        temp_coordinate_store.export_to_db(target_path)

        # Verify export
        target_store = CoordinateStore(target_path)
        assert target_store.count() == temp_coordinate_store.count() == 21

    def test_export_to_db_specific_ids(self, temp_coordinate_store, tmp_path):
        """Test exporting specific boxes by ID."""
        # Add boxes
        ids = [1, 4, 5]
        target_path = tmp_path / "export_specific.sqlite"
        temp_coordinate_store.export_to_db(target_path, ids=ids)

        # Verify export
        target_store = CoordinateStore(target_path)
        assert target_store.count() == 3

    @pytest.mark.skip(reason="Not implemented yet")
    def test_merge_from_db(self, temp_coordinate_store, sample_binary_box_coordinates, tmp_path):
        """Test merging boxes from another database."""

        new_store = CoordinateStore(tmp_path / "new.sqlite")
        new_store.add(sample_binary_box_coordinates)

        # Merge into existing store
        assert temp_coordinate_store.count() == 21

        temp_coordinate_store.merge_from_db(new_store, deduplicate=True)
        assert temp_coordinate_store.count() == 22

        temp_coordinate_store.merge_from_db(new_store, deduplicate=False)
        assert temp_coordinate_store.count() == 23  # Should add duplicate box

        temp_coordinate_store.merge_from_db(new_store, deduplicate=True)
        assert temp_coordinate_store.count() == 23  # Should not add duplicate box again


class TestDatabaseModels:
    """Tests for SQLModel database models."""

    def test_coordinates_db_model(self):
        """Test CoordinatesDB model creation."""
        db_coord = CoordinatesDB(
            composition_key="test_key",
            molecule_species="[H:1][O:2][H:3]<100>",
            n_molecules=100,
            temperature=298.15,
            pressure=1.0,
            force_field_id="openff-2.1.0",
            potential_energy=-1234.56,
            coordinates=b"compressed_data",
            box_vectors=b"compressed_vectors",
            metadata=json.dumps({"test": "data"}),
        )

        assert db_coord.composition_key == "test_key"
        assert db_coord.temperature == 298.15
        assert db_coord.n_molecules == 100

    def test_db_metadata_model(self):
        """Test DBMetadata model creation."""
        metadata = DBMetadata(key="schema_version", value="1")
        assert metadata.key == "schema_version"
        assert metadata.value == "1"
