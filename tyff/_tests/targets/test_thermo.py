import pytest

import tyff.mm
import tyff.utils.dataset
from tyff.molecule import map_smiles
from tyff.targets.thermo import (
    DataEntry,
    create_dataset,
    create_from_evaluator,
    extract_smiles,
)


@pytest.fixture
def mock_density_pure() -> DataEntry:
    return {
        "id": None,
        "tag": "density",
        "smiles": [map_smiles(smiles) for smiles in ["CO"]],
        "x": [1.0],
        "temperature": 298.15,
        "pressure": 1.0,
        "value": 0.785,
        "std": 0.001,
        "units": "g/mL",
        "source": "",
    }


@pytest.fixture
def mock_density_binary() -> DataEntry:
    return {
        "id": None,
        "tag": "density",
        "smiles": [map_smiles(smiles) for smiles in ["CCO", "CO"]],
        "x": [0.5, 0.5],
        "temperature": 298.15,
        "pressure": 1.0,
        "value": 0.9,
        "std": 0.002,
        "units": "g/mL",
        "source": "",
    }


@pytest.fixture
def mock_hvap() -> DataEntry:
    return {
        "tag": "hvap",
        "smiles": [map_smiles(smiles) for smiles in ["CCCC"]],
        "x": [1.0],
        "temperature": 298.15,
        "pressure": 1.0,
        "value": 1.234,
        "std": 0.004,
        "units": "kcal/mol",
        "source": "",
    }


@pytest.fixture
def mock_hmix() -> DataEntry:
    return {
        "tag": "hmix",
        "smiles": [map_smiles(smiles) for smiles in ["CCO", "CO"]],
        "x": [0.5, 0.5],
        "temperature": 298.15,
        "pressure": 1.0,
        "value": 0.4321,
        "std": 0.0025,
        "units": "kcal/mol",
        "source": "",
    }


def test_create_dataset(mock_density_pure, mock_density_binary):
    expected_entries = [mock_density_pure, mock_density_binary]

    dataset = create_dataset(*expected_entries)
    assert len(dataset) == 2

    entries = list(tyff.utils.dataset.iter_dataset(dataset))

    for actual, expected in zip(entries, expected_entries):
        assert actual == expected


def test_extract_smiles(mock_density_pure, mock_density_binary):
    dataset = create_dataset(mock_density_pure, mock_density_binary)
    smiles = extract_smiles(dataset)

    expected_smiles = [
        "[C:1]([C:2]([O:3][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]",
        "[C:1]([O:2][H:6])([H:3])([H:4])[H:5]",
    ]
    assert smiles == expected_smiles


def test_create_from_evaluator(data_dir):
    dataset = create_from_evaluator(dataset_file=data_dir / "evaluator_mock.json")

    entries = list(tyff.utils.dataset.iter_dataset(dataset))
    expected = {
        "id": None,
        "smiles": [
            "[C:1]([C:2]([O:3][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]",
            "[O:1]([H:2])[H:3]",
        ],
        "x": [
            0.48268,
            0.51732,
        ],
        "temperature": 298.15,
        "pressure": 101.3,
        "value": 0.99,
        "std": 0.000505,
        "units": "g/mL",
        "source": "mock",
        "tag": "density",
    }
    assert entries[0] == expected


def test_unsupported_property(data_dir):
    with pytest.raises(KeyError):
        _ = create_from_evaluator(dataset_file=data_dir / "missing_property_evaluator.json")
