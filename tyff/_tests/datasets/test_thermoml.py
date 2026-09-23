import random

import numpy
import pytest
from openff.toolkit import Molecule, Quantity
from openff.toolkit.utils.toolkits import OPENEYE_AVAILABLE
from openff.units import Unit

from tyff._tests.utils import get_test_data_path
from tyff.datasets.provenance import MeasurementSource
from tyff.datasets.thermoml import (
    DuplicateThermoMLEntryWarning,
    ThermoMLDataSet,
    ThermoMLProperty,
    _property_unit_map,
    _tag_property_map,
)
from tyff.substances import Component, Substance


def data_entry_to_property_and_source(
    entry: dict,
) -> ThermoMLProperty:

    property_ = ThermoMLProperty(type_string=_tag_property_map[entry["tag"]])
    property_.default_unit = _property_unit_map[property_.type_string]

    substance = Substance()

    for smiles, x in zip(entry["smiles"], entry["x"]):
        substance.add_component(
            component=Component(smiles=smiles),
            amount=x,
        )

    property_.substance = substance

    property_.temperature = Quantity(entry["temperature"], "kelvin")
    property_.pressure = Quantity(entry["pressure"], "kilopascal") if entry["pressure"] else None

    unit_to_use = _property_unit_map[property_.type_string]

    property_.value = Quantity(entry["value"], unit_to_use)
    property_.uncertainty = Quantity(entry["std"], unit_to_use)

    # if this was in production, need better way to determine if source is a DOI or not
    property_.source = MeasurementSource(doi=entry["source"]) if entry["source"].startswith("10.") else None

    return property_


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            "single_density.xml",
            {
                "x": [1.0],
                "temperature": 293.15,
                "pressure": 101.3,
                "value": 0.96488,
                "std": 0.00005,
                "units": "g/mL",
                "source": "",
            },
        ),
        (
            "single_dhmix.xml",
            {
                "x": [0.219, 0.781],
                "temperature": 298.15,
                "pressure": 101.0,
                "value": 0.03021,
                "std": 0.000151,
                "units": "kcal/mol",
                "source": "10.1016/j.jct.2008.12.004",
            },
        ),
        (
            "single_dhvap.xml",
            {
                "x": [1.0],
                "temperature": 298.15,
                "pressure": None,
                "value": 10.51625,
                "std": 0.1434,
                "units": "kcal/mol",
                "source": "10.1016/j.fluid.2014.12.023",
            },
        ),
        (
            "single_dielectric.xml",
            {
                "x": [1.0],
                "temperature": 293.15,
                "pressure": 101.0,
                "value": 11.76,
                "std": 0.02,
                "units": "dimensionless",
                "source": "",
            },
        ),
    ],
)
class TestThermoMLDataset:
    """Class set up only to make convenient re-use of parametrized test cases."""

    def test_load_property_types(self, filename: str, expected: dict):
        """Test loading a single data type from a ThermoML XML file"""
        dataset = ThermoMLDataSet.from_xml(open(get_test_data_path(f"thermoml/{filename}")).read())
        assert len(dataset) == 1

        entry = next(iter(dataset))
        assert entry["x"] == expected["x"]

        assert len(entry["x"]) == len(entry["smiles"])
        assert len(entry["x"]) == len(expected["x"])

        for found_x, expected_x, found_smiles in zip(
            entry["x"],
            expected["x"],
            entry["smiles"],
        ):
            assert found_x == expected_x

            # just make sure it's valid SMILES
            Molecule.from_smiles(found_smiles)

            # Evaluator uses non-mapped SMILES, pseudocode here used mapped
            # Molecule.from_mapped_smiles(found_smiles)

        assert entry["temperature"] == expected["temperature"]
        if expected["pressure"] is not None:
            assert numpy.isclose(entry["pressure"], expected["pressure"], rtol=1e-8)
        else:
            assert entry["pressure"] is None

        assert numpy.isclose(entry["value"], expected["value"], atol=1e-5)
        assert numpy.isclose(entry["std"], expected["std"], atol=1e-5)
        assert Unit(entry["units"]) == Unit(expected["units"])

        assert entry["source"] == expected["source"]

    def test_same_property_same_hash(self, filename, expected):
        dataset = ThermoMLDataSet.from_xml(open(get_test_data_path(f"thermoml/{filename}")).read())
        this_property = next(iter(dataset))

        reloaded = ThermoMLDataSet.from_xml(open(get_test_data_path(f"thermoml/{filename}")).read())
        reloaded_property = next(iter(reloaded))

        assert reloaded_property["id"] == this_property["id"]

    def test_different_property_different_hash(self, filename, expected):
        dataset = ThermoMLDataSet.from_xml(open(get_test_data_path(f"thermoml/{filename}")).read())
        this_property = next(iter(dataset))

        # Grab a different property from a different file, doesn't really matter which one it is
        other_filename = {
            "single_density.xml": "single_dhmix.xml",
            "single_dhmix.xml": "single_dhvap.xml",
            "single_dhvap.xml": "single_dielectric.xml",
            "single_dielectric.xml": "single_density.xml",
        }[filename]

        different_dataset = ThermoMLDataSet.from_xml(open(get_test_data_path(f"thermoml/{other_filename}")).read())
        different_property = next(iter(different_dataset))

        assert different_property["id"] != this_property["id"]

    def test_pandas_roundtrip(self, filename, expected):
        dataset = ThermoMLDataSet.from_xml(open(get_test_data_path(f"thermoml/{filename}")).read())

        roundtripped = ThermoMLDataSet.from_pandas(dataset.to_pandas())

        assert len(dataset) == len(roundtripped)

        for property1, property2 in zip(dataset, roundtripped):
            assert property1.keys() == property2.keys()
            for key in property1.keys():
                assert property1[key] == property2[key]

            # likely redundant
            assert property1["pressure"] == property2["pressure"]


@pytest.mark.skip(reason="Implement next")
def test_load_single_osmotic():
    """
    Test loading a single osmotic coefficient data point from a ThermoML XML file.

    This is analogous to the test above,
    but is included here to ensure that ions are dealt with correctly.

    """
    dataset = ThermoMLDataSet.from_xml(open(get_test_data_path("thermoml/single_osmotic.xml")).read())
    assert len(dataset) == 1

    entry = next(iter(dataset))

    assert "." in entry["smiles"]
    Molecule.from_mapped_smiles(entry["smiles"])
    assert entry["x"] == 0.00086

    Molecule.from_mapped_smiles(entry["smiles"])
    assert entry["x"] == 0.99914

    assert numpy.isclose(entry["temperature"], 298.15, atol=1e-3)
    assert entry["pressure"] is None
    assert numpy.isclose(entry["value"], 0.7389, atol=1e-5)
    assert numpy.isclose(entry["std"], 0.00655, atol=1e-5)
    assert entry["units"] == "dimensionless"
    assert entry["source"] == "10.1016/j.fluid.2006.09.025"


def test_load_from_doi():
    """Test loading a ThermoML dataset from a DOI"""
    dataset = ThermoMLDataSet.from_doi("10.1016/j.fluid.2014.12.023")
    assert len(dataset) == 186
    for entry in dataset:
        assert entry["source"] == "10.1016/j.fluid.2014.12.023"


def test_load_from_url():
    """A test to ensure that ThermoML archive files can be loaded from a url."""

    data_set = ThermoMLDataSet.from_url("https://trc.nist.gov/ThermoML/10.1021/acs.jced.6b00916.xml")
    assert data_set is not None

    assert len(data_set) > 0

    data_set = ThermoMLDataSet.from_url("https://trc.nist.gov/ThermoML/10.1021/acs.jced.6b00916.xmld")
    assert data_set is None


def test_to_pandas():
    """A test to ensure that data sets are convertable to pandas objects."""

    thermoml_dataset = ThermoMLDataSet()

    density_entry = {
        "id": random.randint(10**15, 10**16 - 1),  # random 16-digit integer
        "tag": "density",
        "x": [1.0],
        "smiles": ["[C:1]([O:5][C:3]([C:2]([O:4][H:13])([H:9])[H:10])([H:11])[H:12])([H:6])([H:7])[H:8]"],
        "temperature": 293.15,
        "pressure": 1.0,
        "value": 0.96488,
        "std": 0.00005,
        "units": "g/mL",
        "source": "",
    }

    thermoml_dataset.add_properties(density_entry)

    dataframe = thermoml_dataset.to_pandas()

    required_columns = [
        "Id",
        "tag",
        "Temperature (K)",
        "Pressure (kPa)",
        "N Components",
        "Component 1",
        "Mole Fraction 1",
        "Value",
        "Uncertainty",
        "Source",
    ]

    assert all(x in dataframe for x in required_columns)

    assert dataframe is not None
    assert dataframe.shape == (1, 10)

    # Source may be an empty string but is not NaN - is this behavior okay?
    data_set_without_na = dataframe.dropna(axis=1, how="all")
    assert data_set_without_na.shape == (1, 10)


@pytest.mark.skipif(not OPENEYE_AVAILABLE, reason="Requires OpenEye toolkit for tautomer resolution.")
def test_thermoml_pyrrolidinone_tautomer_resolution_with_openeye():
    """2-pyrrolidinone (InChI-only entry with mobile-H layer)
    must parse as the lactam O=C1CCCN1, not the lactim OC1=NCCC1.

    The InChI ``InChI=1S/C4H7NO/c6-4-2-1-3-5-4/h1-3H2,(H,5,6)`` round-trips
    to the same string for both tautomers; the common name disambiguates.
    Requires an OpenEye licence for ``Molecule.from_iupac``.
    """
    data_set = ThermoMLDataSet.from_xml(open(get_test_data_path("thermoml/pyrrolidinone.xml")).read())
    assert data_set is not None
    assert len(data_set) > 0

    lactam_smiles = "O=C1CCCN1"
    lactim_smiles = "OC1=NCCC1"

    found_lactam = False
    for property in data_set:
        for smiles in property["smiles"]:
            if smiles == lactam_smiles:
                found_lactam = True
            assert smiles != lactim_smiles, f"Got lactim {lactim_smiles!r} for 2-pyrrolidinone; expected lactam"

    assert found_lactam, "Lactam SMILES not found in any parsed substance"


@pytest.mark.skipif(
    OPENEYE_AVAILABLE,
    reason="Requires OpenEye toolkit for tautomer resolution, but this test checks behavior without it.",
)
def test_thermoml_pyrrolidinone_tautomer_resolution_without_openeye():
    """Without OpenEye, 2-pyrrolidinone (InChI-only entry with mobile-H layer)
    parses as the lactim OC1=NCCC1, not lactam O=C1CCCN1
    """
    with pytest.warns(
        UserWarning,
        match="Multiple tautomers were generated from the InChI string",
    ):
        data_set = ThermoMLDataSet.from_xml(open(get_test_data_path("thermoml/pyrrolidinone.xml")).read())

    assert data_set is not None
    assert len(data_set) > 0

    lactam_smiles = "O=C1CCCN1"
    lactim_smiles = "OC1=NCCC1"

    found_lactam = False
    for property in data_set:
        for smiles in property["smiles"]:
            if smiles == lactam_smiles:
                found_lactam = True
            if smiles == lactim_smiles:
                found_lactim = True
    assert not found_lactam, "Lactam SMILES should not be found in any parsed substance"
    assert found_lactim, "Lactim SMILES not found in any parsed substance"


def test_thermoml_warn_duplicate_properties():
    dataset = ThermoMLDataSet()

    property_ = {
        "id": 35598034897404032129990573617925713559702236029053845170364984208600110431820,
        "tag": "density",
        "smiles": ["COCCO"],
        "x": [1.0],
        "temperature": 293.15,
        "pressure": 101.3,
        "value": 0.9648800000000002,
        "std": 5.0000000000000016e-05,
        "units": "gram / milliliter",
        "source": "",
    }

    with pytest.warns(
        DuplicateThermoMLEntryWarning,
        match="35598034897404032129990573617925713559702236029053845170364984208600110431820",
    ):
        dataset.add_properties(*[property_, property_])

    assert len(dataset) == 1


@pytest.mark.filterwarnings("error")
def test_thermoml_no_warning_for_slightly_different_properties():

    original = {
        "id": 1,
        "tag": "density",
        "smiles": ["COCCO"],
        "x": [1.0],
        "temperature": 293.15,
        "pressure": 101.3,
        "value": 0.9648800000000002,
        "std": 5.0000000000000016e-05,
        "units": "gram / milliliter",
        "source": "10.61092/iaea.ght7-f9qq",  # this DOI is meaningless
    }

    lower_uncertainty = {
        **original,
        "id": 2,
        "std": original["std"] * 0.99,
    }

    different_doi = {
        **original,
        "id": 3,
        "source": "10.61092/iaea.ght7-f9qr",  # this DOI is also meaningless
    }

    # ~roundtrip each through ThermoMLDataSet._property_to_entry to ensure the generated id
    # is based on values of the property, not just set arbitrarily
    entry1 = ThermoMLDataSet._property_to_entry(data_entry_to_property_and_source(original))

    entry2 = ThermoMLDataSet._property_to_entry(data_entry_to_property_and_source(lower_uncertainty))

    entry3 = ThermoMLDataSet._property_to_entry(data_entry_to_property_and_source(different_doi))

    assert entry1["id"] != 1
    assert entry2["id"] != 2
    assert entry3["id"] != 3

    dataset1 = ThermoMLDataSet()

    # different uncertainties should each be added as separate entries
    dataset1.add_properties(*[entry1, entry2])

    assert len(dataset1) == 2

    dataset2 = ThermoMLDataSet()

    # different DOIs should also end up as separate entries, even if otherwise identical
    dataset2.add_properties(*[entry1, entry3])

    assert len(dataset2) == 2
