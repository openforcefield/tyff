import shutil
from importlib.resources import files

import pytest
from parsl import File

from tyff.compute._files import ProductionFiles
from tyff.compute._jacobian import _get_ensemble_average_and_jacobian


@pytest.fixture
def production_future() -> dict[str, ProductionFiles]:
    return {
        "simulation_files": ProductionFiles(
            topology=File(files("tyff") / "_tests/data/app_files/sample_density/production_topology.pdb"),
            dcd_trajectory=File(files("tyff") / "_tests/data/app_files/sample_density/production_trajectory.dcd"),
            msgpack_trajectory=File(
                files("tyff") / "_tests/data/app_files/sample_density/production_trajectory.msgpack"
            ),
            log=File(files("tyff") / "_tests/data/app_files/sample_density/produce.log"),
            state_data=File(files("tyff") / "_tests/data/app_files/sample_density/production.csv"),
            system=File(files("tyff") / "_tests/data/app_files/sample_density/production_system.xml"),
            integrator=File(files("tyff") / "_tests/data/app_files/sample_density/production_integrator.xml"),
            checkpoint=File(files("tyff") / "_tests/data/app_files/sample_density/production_checkpoint.chk"),
        )
    }


def test_jacobian(production_future, tmp_path):
    for file in [
        "compute_config.json",
        "production_trajectory.msgpack",
        "single_molecule_interchange_0.json",
        "single_molecule_interchange_1.json",
    ]:
        shutil.copy(
            str(files("tyff") / f"_tests/data/app_files/sample_density/{file}"),
            str(tmp_path / file),
        )

    averages, jacobians = _get_ensemble_average_and_jacobian(
        production_future,
        tmp_path,
        reference_force_field_path=File(files("tyff") / "_tests/data/app_files/sample_density/ref.ff.json"),
    )

    assert len(averages) == jacobians.shape[0]
