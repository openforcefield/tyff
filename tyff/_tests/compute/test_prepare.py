import json
import shutil
from importlib.resources import files

import openmm
import pytest
from openff.interchange import Interchange
from openff.toolkit import Molecule, Topology
from parsl import File

from tyff.compute._pack import PackingFiles
from tyff.compute._prepare import _prepare_openmm_system
from tyff.configs.liquid import BulkLiquid


@pytest.fixture
def bulk_liquid():
    return BulkLiquid(
        tag="liquid",
        # TODO: might make tests quicker to have a slimmed-down Sage with fewer parameters
        # (but still use AshGC charges)
        force_field="openff-2.3.0.offxml",
        n_molecules=100,
        smiles=["CC", "O"],
        x=[0.6, 0.4],
        temperature=301.15,
        pressure=101.325,
        density=0.65,  # very low, just for quicker packing
    )


@pytest.fixture
def packing_future() -> dict[str, BulkLiquid | Topology]:
    with open(
        files("tyff") / "_tests/data/app_files/sample_density/compute_config.json",
    ) as f:
        compute_config = BulkLiquid(
            json.load(f),
        )

    return {
        "compute_config": compute_config,  # do we really need to return this?
        "packed_files": PackingFiles(
            packed_topology=File(files("tyff") / "_tests/data/app_files/sample_density/packed_topology.pdb")
        ),
    }


def test_prepare_openmm_system(packing_future, tmp_path):
    with open(f"{tmp_path}/compute_config.json", "w") as f:
        json.dump(packing_future["compute_config"], f)

    prepare_result = _prepare_openmm_system(
        packing_future=packing_future,
        job_dir=str(tmp_path),
    )

    assert isinstance(prepare_result["prepared_files"], dict)
    assert isinstance(prepare_result["prepared_files"]["openmm_system"], File)

    topology = Topology.from_pdb(
        file_path=packing_future["packed_files"]["packed_topology"].filepath,
        unique_molecules=[Molecule.from_smiles(smiles) for smiles in packing_future["compute_config"]["smiles"]],
    )

    with open(prepare_result["prepared_files"]["openmm_system"].filepath) as f:
        openmm_system = openmm.XmlSerializer.deserialize(f.read())

    assert openmm_system.getNumParticles() == topology.n_atoms


def test_short_circuit(packing_future, tmp_path):
    """Test that the function short-circuits if the serialized OpenMM system already exists."""
    for file in ["openmm_system.xml", "interchange.json"]:
        shutil.copy(
            str(files("tyff") / f"_tests/data/app_files/sample_density/{file}"),
            str(tmp_path / file),
        )

    prepare_result = _prepare_openmm_system(
        packing_future=packing_future,
        job_dir=str(tmp_path),
    )

    assert isinstance(prepare_result["prepared_files"], dict)
    assert isinstance(prepare_result["prepared_files"]["openmm_system"], File)
    assert isinstance(prepare_result["prepared_files"]["interchange"], File)

    # make sure the serialized Interchange and OpenMM system can be loaded
    with open(prepare_result["prepared_files"]["interchange"].filepath) as f:
        interchange = Interchange.model_validate_json(f.read())

    with open(prepare_result["prepared_files"]["openmm_system"].filepath) as f:
        openmm_system = openmm.XmlSerializer.deserialize(f.read())

    # will fail on virtual sites
    assert interchange.topology.n_atoms == openmm_system.getNumParticles()

    with open(tmp_path / "prepare.log") as f:
        for line in f.readlines():
            if "already exists, skipping system prep." in line:
                return

    raise AssertionError("Did not find expected log message about skipping system prep.")
