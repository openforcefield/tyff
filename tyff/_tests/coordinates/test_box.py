import datetime
import json

import numpy as np
import pytest
from openff.toolkit import ForceField, Molecule, Topology

from tyff._tests.utils import get_test_data_path
from tyff.coordinates.box import (
    BoxCoordinates,
    CoordinatesDB,
    MoleculeSpecies,
    Substance,
)

rng = np.random.default_rng(42)


class TestMoleculeSpecies:
    """Tests for MoleculeSpecies class."""

    def test_create_molecule_species(self):
        """Test creating a MoleculeSpecies instance."""
        mol = MoleculeSpecies(mapped_smiles="[H:1][O:2][H:3]", count=100)
        assert mol.mapped_smiles == "[H:1][O:2][H:3]"
        assert mol.count == 100

    def test_to_inchi_composition(self, sample_molecule_species):
        key1 = sample_molecule_species[0].to_inchi_composition()
        assert key1 == ("InChI=1/H2O/h1H2", 100)
        key2 = sample_molecule_species[1].to_inchi_composition()
        assert key2 == ("InChI=1/C2H6/c1-2/h1-2H3", 50)

    def test_to_string(self, sample_molecule_species):
        """Test converting MoleculeSpecies to string representation."""
        result1 = sample_molecule_species[0].to_string()
        assert result1 == "[H:1][O:2][H:3]<100>"

        result2 = sample_molecule_species[1].to_string()
        assert result2 == "[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])[H:8]<50>"

    def test_from_string_single_species(self):
        """Test creating MoleculeSpecies from string (single species)."""
        s = "[H:1][O:2][H:3]<100>"
        result = MoleculeSpecies.from_string(s, flatten=True)
        assert isinstance(result, MoleculeSpecies)
        assert result.mapped_smiles == "[H:1][O:2][H:3]"
        assert result.count == 100

    def test_from_string_multiple_species(self):
        """Test creating MoleculeSpecies from string (multiple species)."""
        s = "[H:1][O:2][H:3]<100>|[C:1][H:2][H:3][H:4]<50>"
        result = MoleculeSpecies.from_string(s, flatten=True)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].mapped_smiles == "[H:1][O:2][H:3]"
        assert result[0].count == 100
        assert result[1].mapped_smiles == "[C:1][H:2][H:3][H:4]"
        assert result[1].count == 50

    def test_from_string_single_species_no_flatten(self):
        """Test creating MoleculeSpecies from string without flattening."""
        s = "[H:1][O:2][H:3]<100>"
        result = MoleculeSpecies.from_string(s, flatten=False)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].mapped_smiles == "[H:1][O:2][H:3]"

    def test_from_string_invalid_format(self):
        """Test that invalid string format raises assertion error."""
        s = "[H:1][O:2][H:3]<100"  # Missing closing >
        with pytest.raises(AssertionError, match="Invalid molecule species string"):
            MoleculeSpecies.from_string(s)

    def test_roundtrip_to_string_from_string(self):
        """Test roundtrip conversion to/from string."""
        mol = MoleculeSpecies(mapped_smiles="[H:1][O:2][H:3]", count=100)
        s = mol.to_string()
        result = MoleculeSpecies.from_string(s, flatten=True)
        assert result.mapped_smiles == mol.mapped_smiles
        assert result.count == mol.count


class TestSubstance:
    """Tests for Substance class."""

    def test_create_substance(self, sample_molecule_species):
        """Test creating a Substance instance."""
        substance = Substance(molecule_species=sample_molecule_species)
        assert len(substance.molecule_species) == 2
        assert substance.molecule_species[0].count == 100
        assert substance.molecule_species[1].count == 50

    def test_to_string(self, sample_substance):
        """Test converting Substance to string representation."""
        result = sample_substance.to_string()
        expected = "[H:1][O:2][H:3]<100>|[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])[H:8]<50>"
        assert result == expected

    def test_from_string(self):
        """Test creating Substance from string representation."""
        s = "[H:1][O:2][H:3]<100>|[C:1][H:2][H:3][H:4]<50>"
        result = Substance.from_string(s)
        assert isinstance(result, Substance)
        assert len(result.molecule_species) == 2
        assert result.molecule_species[0].count == 100
        assert result.molecule_species[1].count == 50

    def test_to_composition_key(self, sample_substance):
        """Test getting canonical composition key."""
        result = sample_substance.to_composition_key()
        # Should be sorted by InChI
        assert result == "InChI=1/C2H6/c1-2/h1-2H3:50|InChI=1/H2O/h1H2:100"

    def test_n_molecules_property(self, sample_substance):
        """Test n_molecules property."""
        assert sample_substance.n_molecules == 150

    @pytest.mark.skip(reason="Not implemented yet")
    def test_is_equivalent_to_remapped(self, sample_substance):
        """Test is_equivalent_to with identical substance."""
        other = Substance(
            molecule_species=[
                MoleculeSpecies(
                    mapped_smiles="[C:7]([H:2])([H:3])([H:4])[C:5]([H:6])([H:1])[H:8]",
                    count=50,
                ),
                MoleculeSpecies(mapped_smiles="[H:2][O:1][H:3]", count=100),
            ]
        )
        assert sample_substance.is_equivalent_to(other)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_is_equivalent_to_different(self, sample_substance):
        """Test is_equivalent_to with different substance."""
        other = Substance(
            molecule_species=[
                MoleculeSpecies(mapped_smiles="[H:1][O:2][H:3]", count=100),
                MoleculeSpecies(mapped_smiles="[C:1][H:2][H:3][H:4][H:5]", count=25),  # Different molecule
            ]
        )
        assert not sample_substance.is_equivalent_to(other)

    def test_roundtrip_string_conversion(self, sample_substance):
        """Test roundtrip conversion to/from string."""
        s = sample_substance.to_string()
        result = Substance.from_string(s)
        assert result.to_string() == sample_substance.to_string()

    def test_to_openff_topology(self, sample_substance):
        """Test converting BoxCoordinates to OpenFF Topology."""
        topology = sample_substance.to_openff_topology()
        assert topology.n_molecules == 150
        assert topology.n_unique_molecules == 2

    def test_from_openff_topology(self):
        jsonfile = get_test_data_path("thermoml/isopropanol-water-topology.json")
        with open(jsonfile) as f:
            contents = f.read()
        topology = Topology.from_json(contents)

        substance = Substance.from_openff_topology(topology)
        assert substance.n_molecules == 1000
        assert len(substance.molecule_species) == 2


class TestBoxCoordinates:
    """Tests for BoxCoordinates class."""

    def test_create_box_coordinates(self, sample_binary_box_coordinates):
        """Test creating a BoxCoordinates instance."""
        assert sample_binary_box_coordinates.temperature == 298.15
        assert sample_binary_box_coordinates.pressure == 1.0
        assert sample_binary_box_coordinates.force_field_id == "openff-2.1.0"
        assert sample_binary_box_coordinates.potential_energy == -1234.56
        assert len(sample_binary_box_coordinates.substance.molecule_species) == 2
        assert sample_binary_box_coordinates.coordinates.shape == (700, 3)
        assert sample_binary_box_coordinates.box_vectors.shape == (3, 3)

    def test_get_molecule_string(self, sample_binary_box_coordinates):
        """Test getting string representation of molecules."""
        result = sample_binary_box_coordinates.get_molecule_string()
        expected = "[H:1][O:2][H:3]<100>|[C:1]([H:2])([H:3])([H:4])[C:5]([H:6])([H:7])[H:8]<50>"
        assert result == expected

    def test_get_composition_key(self, sample_binary_box_coordinates):
        """Test getting canonical composition key."""
        result = sample_binary_box_coordinates.get_composition_key()
        # should be sorted alphabetically by InChI
        assert result == "InChI=1/C2H6/c1-2/h1-2H3:50|InChI=1/H2O/h1H2:100"

    def test_n_molecules_property(self, sample_binary_box_coordinates):
        """Test n_molecules property calculates total correctly."""
        assert sample_binary_box_coordinates.n_molecules == 150

    def test_box_coordinates_without_coordinates(self, sample_substance):
        """Test creating BoxCoordinates without coordinates (None)."""
        box = BoxCoordinates(
            substance=sample_substance,
            temperature=300.0,
            pressure=1.0,
            force_field_id="test",
            potential_energy=0.0,
            coordinates=None,
            box_vectors=None,
        )
        assert box.coordinates is None
        assert box.box_vectors is None

    def test_to_openff_topology(self, sample_binary_box_coordinates):
        """Test converting BoxCoordinates to OpenFF Topology."""
        topology = sample_binary_box_coordinates.to_openff_topology()
        assert topology.n_molecules == 150
        assert topology.n_unique_molecules == 2

    def test_from_openff_topology(self):
        jsonfile = get_test_data_path("thermoml/isopropanol-water-topology.json")
        with open(jsonfile) as f:
            contents = f.read()
        topology = Topology.from_json(contents)

        box = BoxCoordinates.from_openff_topology(topology)
        assert box.n_molecules == 1000
        assert len(box.substance.molecule_species) == 2
        expected_composition_key = "InChI=1/C3H8O/c1-3(2)4/h3-4H,1-2H3:505|InChI=1/H2O/h1H2:495"
        assert box.get_composition_key() == expected_composition_key
        first_coordinates = [1.4742e01, 1.7520e00, 6.1683e01]
        assert np.allclose(box.coordinates[0], first_coordinates)

    @pytest.mark.skip(reason="Not implemented yet")
    def test_get_energy_for_system_openmm(self, sample_binary_box_coordinates):
        topology = sample_binary_box_coordinates.to_openff_topology()
        forcefield = ForceField("openff-2.1.0.offxml")
        system = forcefield.create_openmm_system(topology)
        energy = sample_binary_box_coordinates.get_energy_for_system(system)
        assert isinstance(energy, float)

    def test_has_equivalent_molecular_species_tautomer_fails(self):
        enol = Molecule.from_smiles("Oc1ccncc1").to_smiles(mapped=True)
        keto = Molecule.from_smiles("O=c1cc[nH]cc1").to_smiles(mapped=True)

        species_1 = [
            MoleculeSpecies(mapped_smiles="[H:1][O:2][H:3]", count=3),
            MoleculeSpecies(mapped_smiles=keto, count=2),
        ]
        box1 = BoxCoordinates(
            substance=Substance(molecule_species=species_1),
        )

        species_2 = [
            MoleculeSpecies(mapped_smiles=enol, count=2),
            MoleculeSpecies(mapped_smiles="[H:1][O:2][H:3]", count=3),
        ]

        box2 = BoxCoordinates(
            substance=Substance(molecule_species=species_2),
        )

        assert box1.get_composition_key() != box2.get_composition_key()
        assert not box1.has_equivalent_molecular_species(box2)

    def test_has_equivalent_molecular_species_resonance_fails(self):
        res1 = Molecule.from_smiles("O=CN").to_smiles(mapped=True)
        res2 = Molecule.from_smiles("[O-]C=[NH2+]").to_smiles(mapped=True)

        species_1 = [
            MoleculeSpecies(mapped_smiles=res1, count=5),
        ]
        box1 = BoxCoordinates(
            substance=Substance(molecule_species=species_1),
        )

        species_2 = [
            MoleculeSpecies(mapped_smiles=res2, count=5),
        ]
        box2 = BoxCoordinates(
            substance=Substance(molecule_species=species_2),
        )

        assert box1.get_composition_key() == box2.get_composition_key()
        assert not box1.has_equivalent_molecular_species(box2)

    @pytest.mark.skip(reason="Not implemented yet")
    @pytest.mark.parametrize(
        "smiles",
        [
            "c1ccccc1",
            "CCO",
        ],
    )
    def test_has_equivalent_molecular_species_robust_to_remapping(self, smiles):
        mol = Molecule.from_smiles(smiles)
        explicit_h_smiles = mol.to_smiles(mapped=True)
        randomized_order = rng.permutation(np.arange(len(mol.atoms)))
        new_atom_map = dict(zip(range(len(mol.atoms)), randomized_order))
        new_molecule = mol.remap(new_atom_map)
        randomized_smiles = new_molecule.to_smiles(mapped=True)

        assert explicit_h_smiles != randomized_smiles

        box1 = BoxCoordinates(
            substance=Substance(molecule_species=[MoleculeSpecies(mapped_smiles=explicit_h_smiles, count=10)]),
        )
        box2 = BoxCoordinates(
            substance=Substance(molecule_species=[MoleculeSpecies(mapped_smiles=randomized_smiles, count=10)]),
        )
        assert box1.get_composition_key() == box2.get_composition_key()
        assert box1.has_equivalent_molecular_species(box2)

    def test_load_coordinates_from_other_without_coords_fails(self, sample_binary_box_coordinates):
        """Test that loading from box without coordinates fails."""
        box_no_coords = BoxCoordinates(
            substance=sample_binary_box_coordinates.substance,
            temperature=300.0,
            pressure=1.0,
            force_field_id="test",
            potential_energy=0.0,
            coordinates=None,
        )

        with pytest.raises(AssertionError, match="Other box must have coordinates"):
            sample_binary_box_coordinates.load_coordinates_from_other(box_no_coords)

    @pytest.mark.skip(reason="Remapping not implemented yet")
    def test_load_coordinates_from_other_remapped(self):
        """Test loading coordinates from another box with remapping."""
        # Create another box with same molecules but different coordinates
        # shuffle both order of atoms and molecules
        original_molecule_species = [
            MoleculeSpecies(mapped_smiles="[H:1][O:2][H:3]", count=3),
            MoleculeSpecies(mapped_smiles="[C:1]([H:2])([H:3])([H:4])[H:5]", count=2),
        ]
        original_coords = np.vstack(
            [
                np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]] * 3),
                np.array([[4.0, 5.0, 6.0], [4.1, 5.1, 6.1], [4.2, 5.2, 6.2], [4.3, 5.3, 6.3], [4.4, 5.4, 6.4]] * 2),
            ]
        )
        assert original_coords.shape == (19, 3)
        original_box = BoxCoordinates(
            substance=Substance(molecule_species=original_molecule_species),
            temperature=300.0,
            pressure=1.0,
            force_field_id="test",
            potential_energy=0.0,
            coordinates=original_coords,
            box_vectors=np.eye(3) * 10.0,
        )
        new_molecule_species = [
            MoleculeSpecies(mapped_smiles="[C:5]([H:1])([H:2])([H:3])[H:4]", count=2),
            MoleculeSpecies(mapped_smiles="[H:2][O:1][H:3]", count=3),
        ]
        # just scale for testing
        new_coords = np.vstack(
            [
                np.array([[4.0, 5.0, 6.0], [4.1, 5.1, 6.1], [4.2, 5.2, 6.2], [4.3, 5.3, 6.3], [4.4, 5.4, 6.4]] * 2)
                * 10,
                np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]] * 3) * 10,
            ]
        )
        new_box_vectors = np.eye(3) * 20.0
        new_box = BoxCoordinates(
            substance=Substance(molecule_species=new_molecule_species),
            temperature=310.0,
            pressure=1.0,
            force_field_id="test",
            potential_energy=-1000.0,
            coordinates=new_coords,
            box_vectors=new_box_vectors,
        )

        # Load coordinates from new box into original box
        original_box.load_coordinates_from_other(new_box)

        expected_new_coordinates = np.vstack(
            [
                np.array(
                    [
                        [11.0, 21.0, 31.0],
                        [10.0, 20.0, 30.0],
                        [12.0, 22.0, 32.0],
                    ]
                    * 3
                ),
                np.array(
                    [
                        [41.0, 51.0, 61.0],
                        [42.0, 52.0, 62.0],
                        [43.0, 53.0, 63.0],
                        [44.0, 54.0, 64.0],
                        [40.0, 50.0, 60.0],
                    ]
                    * 2
                ),
            ]
        )
        assert np.allclose(original_box.coordinates, expected_new_coordinates)
        assert np.allclose(original_box.box_vectors, new_box_vectors)

    def test_with_coordinates_from_other(self, sample_binary_box_coordinates):
        """Test creating new box with coordinates from another."""
        result = sample_binary_box_coordinates.with_coordinates_from_other(sample_binary_box_coordinates)
        assert result is not None
        assert isinstance(result, BoxCoordinates)
        assert result is not sample_binary_box_coordinates
        assert np.allclose(result.coordinates, sample_binary_box_coordinates.coordinates)

    def test_to_db_model(self, sample_binary_box_coordinates):
        """Test converting BoxCoordinates to database model."""
        db_model = sample_binary_box_coordinates._to_db_model()

        assert isinstance(db_model, CoordinatesDB)
        assert db_model.temperature == sample_binary_box_coordinates.temperature
        assert db_model.pressure == sample_binary_box_coordinates.pressure
        assert db_model.force_field_id == sample_binary_box_coordinates.force_field_id
        assert db_model.potential_energy == sample_binary_box_coordinates.potential_energy
        assert db_model.n_molecules == 150
        assert isinstance(db_model.coordinates, bytes)
        assert isinstance(db_model.box_vectors, bytes)

        # Test metadata serialization
        metadata = json.loads(db_model.box_metadata)
        assert metadata["equilibration_steps"] == 10000

    def test_to_db_model_without_coordinates_fails(self, sample_substance):
        """Test that converting to DB model without coordinates fails."""
        box = BoxCoordinates(
            substance=sample_substance,
            temperature=300.0,
            pressure=1.0,
            force_field_id="test",
            potential_energy=0.0,
            coordinates=None,
        )

        with pytest.raises(AssertionError, match="Coordinates must be set"):
            box._to_db_model()

    def test_from_db_model(self, sample_binary_box_coordinates):
        """Test creating BoxCoordinates from database model."""
        db_model = sample_binary_box_coordinates._to_db_model()
        db_model.id = 123
        db_model.created_at = datetime.datetime.now()

        result = BoxCoordinates._from_db_model(db_model)

        assert result.id == 123
        assert result.temperature == sample_binary_box_coordinates.temperature
        assert result.pressure == sample_binary_box_coordinates.pressure
        assert result.force_field_id == sample_binary_box_coordinates.force_field_id
        assert result.potential_energy == sample_binary_box_coordinates.potential_energy
        assert len(result.substance.molecule_species) == 2
        assert result.coordinates.shape == sample_binary_box_coordinates.coordinates.shape
        assert result.box_vectors.shape == sample_binary_box_coordinates.box_vectors.shape
        assert result.box_metadata["equilibration_steps"] == 10000

    def test_from_db_model_without_box_vectors_fails(self, sample_binary_box_coordinates):
        """Test creating BoxCoordinates from DB model without box vectors."""
        sample_binary_box_coordinates.box_vectors = None

        with pytest.raises(AssertionError, match="Box vectors must be set"):
            sample_binary_box_coordinates._to_db_model()

    def test_roundtrip_db_conversion(self, sample_binary_box_coordinates):
        """Test roundtrip conversion to and from database model."""
        db_model = sample_binary_box_coordinates._to_db_model()
        db_model.id = 1
        db_model.created_at = datetime.datetime.now()

        result = BoxCoordinates._from_db_model(db_model)

        # Verify all data is preserved
        assert result.temperature == sample_binary_box_coordinates.temperature
        assert result.pressure == sample_binary_box_coordinates.pressure
        assert result.force_field_id == sample_binary_box_coordinates.force_field_id
        assert result.n_molecules == sample_binary_box_coordinates.n_molecules
        assert np.allclose(result.coordinates, sample_binary_box_coordinates.coordinates)
        assert np.allclose(result.box_vectors, sample_binary_box_coordinates.box_vectors)
