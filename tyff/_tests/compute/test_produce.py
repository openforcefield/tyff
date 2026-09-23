import shutil
from importlib.resources import files

import pytest
from parsl import File

from tyff.compute._files import EquilibrationFiles
from tyff.compute._produce import _run_production


@pytest.fixture
def equilibration_future() -> dict[str, EquilibrationFiles]:
    return {
        "simulation_files": EquilibrationFiles(
            topology=File(files("tyff") / "_tests/data/app_files/sample_density/equilibrated_topology.pdb"),
            dcd_trajectory=File("foo.dcd"),
            msgpack_trajectory=File("foo.msgpack"),
            log=File(files("tyff") / "_tests/data/app_files/sample_density/equilibrate.log"),
            state_data=File(files("tyff") / "_tests/data/app_files/sample_density/equilibration.csv"),
            system=File(files("tyff") / "_tests/data/app_files/sample_density/equilibration_system.xml"),
            integrator=File(files("tyff") / "_tests/data/app_files/sample_density/equilibration_integrator.xml"),
            checkpoint=File(files("tyff") / "_tests/data/app_files/sample_density/equilibration_checkpoint.chk"),
        )
    }


@pytest.mark.skip(reason="only run on GPUs")
def test_basic(tmp_path, equilibration_future):
    for file in ["compute_config.json", "packed_topology.pdb", "openmm_system.xml"]:
        shutil.copy(
            str(files("tyff") / f"_tests/data/app_files/sample_density/{file}"),
            str(tmp_path / file),
        )

    _run_production(
        production_config=None,  # Replace with an actual ProductionConfig object
        equilibration_future=equilibration_future,
        job_dir=str(tmp_path),
    )
