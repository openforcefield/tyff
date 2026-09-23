import shutil
from importlib.resources import files

import pytest
from parsl import File

from tyff.compute._files import PreparingFiles
from tyff.compute._minimize import _minimize_energy


@pytest.fixture
def prepare_future() -> dict[str, PreparingFiles]:
    return {
        "prepared_files": PreparingFiles(
            openmm_system=File(files("tyff") / "_tests/data/app_files/sample_density/openmm_system.xml"),
            packed_topology=File(files("tyff") / "_tests/data/app_files/sample_density/packed_topology.pdb"),
        ),
    }


def test_minimize_basic(prepare_future, tmp_path):
    # shim - see comment in source code
    for file in ["compute_config.json", "packed_topology.pdb", "openmm_system.xml"]:
        shutil.copy(
            str(files("tyff") / f"_tests/data/app_files/sample_density/{file}"),
            str(tmp_path / file),
        )

    _minimize_energy(
        system_future=prepare_future,
        job_dir=str(tmp_path),
    )

    with open(str(tmp_path / "minimize.log")) as log_file:
        for line in log_file.readlines():
            if "Minimized energy" in line:
                split = line.split()

                index = split.index("to")

                initial = float(split[index - 1])
                final = float(split[index + 1])

                assert final < initial

                return

    raise Exception("'Minimized energy' not found in log file")
