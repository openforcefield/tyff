import shutil
from importlib.resources import files

import pytest
from parsl import File

from tyff.compute._equilibrate import _run_equilibration
from tyff.compute._files import MinimizationFiles


@pytest.fixture
def minimization_future() -> dict[str, MinimizationFiles]:
    return {
        "simulation_files": MinimizationFiles(
            topology=File(files("tyff") / "_tests/data/app_files/sample_density/minimized_topology.pdb"),
            system=File(files("tyff") / "_tests/data/app_files/sample_density/minimized_system.xml"),
            integrator=File(files("tyff") / "_tests/data/app_files/sample_density/minimized_integrator.xml"),
            checkpoint=File(files("tyff") / "_tests/data/app_files/sample_density/minimized_checkpoint.chk"),
        )
    }


@pytest.mark.skip(reason="only run on GPUs")
def test_basic(tmp_path, minimization_future):
    for file in ["compute_config.json", "packed_topology.pdb", "openmm_system.xml"]:
        shutil.copy(
            str(files("tyff") / f"_tests/data/app_files/sample_density/{file}"),
            str(tmp_path / file),
        )

    _run_equilibration(
        equilibration_config=None,  # Replace with an actual EquilibrationConfig object
        minimization_future=minimization_future,
        job_dir=str(tmp_path),
    )
