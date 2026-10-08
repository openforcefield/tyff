import json
import shutil
from importlib.resources import files

import pytest
import torch
from parsl import File

from tyff._serialization import load_tensor_force_field
from tyff.compute._files import ProductionFiles
from tyff.compute._jacobian import _get_ensemble_average_and_jacobian
from tyff.mm._ops import _pack_force_field


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


def test_jacobian_shape(production_future, tmp_path):
    # TODO: Can probably create a common base class (or setup method) to avoid copying
    #       code which handles copying the same files over
    for file in [
        "compute_config.json",
        "production_trajectory.msgpack",
        "single_molecule_interchange_0.json",
        "single_molecule_interchange_1.json",
        "ref.ff.json",
    ]:
        shutil.copy(
            str(files("tyff") / f"_tests/data/app_files/sample_density/{file}"),
            str(tmp_path / file),
        )

    averages, jacobians = _get_ensemble_average_and_jacobian(
        production_future,
        tmp_path,
        reference_force_field_path=tmp_path / "ref.ff.json",
    )

    # number of rows in jacobian should be equal to the number of things
    # we're taking the ensemble average of
    assert len(averages) == jacobians.shape[0]

    with open(tmp_path / "ref.ff.json") as ref_f:
        reference_force_field = load_tensor_force_field(json.loads(ref_f.read()))

    tensors, *_ = _pack_force_field(reference_force_field)
    n_values = sum(t.numel() for t in tensors if t is not None)

    # number of columns in jacobian should be equal to the number of values in the
    # (tensor) force field, defined by summing up the number of values in each tensor

    assert jacobians.shape[1] == n_values


@pytest.mark.parametrize("swap_molecules", [False, True])
def test_swapping_molecule_order_same_jacobian_values(production_future, tmp_path, swap_molecules):
    for file in [
        "compute_config.json",
        "production_trajectory.msgpack",
        "single_molecule_interchange_0.json",
        "single_molecule_interchange_1.json",
        "ref.ff.json",
    ]:
        shutil.copy(
            str(files("tyff") / f"_tests/data/app_files/sample_density/{file}"),
            str(tmp_path / file),
        )

    reference_averages, reference_jacobians = _get_ensemble_average_and_jacobian(
        production_future,
        tmp_path,
        reference_force_field_path=tmp_path / "ref.ff.json",
    )

    # check that, using everything else the same including the same global force field, swapping
    # molecules 0 and 1 in the compute batch gets you the same jacobian values as if not
    if swap_molecules:
        shutil.copy(
            str(files("tyff") / "_tests/data/app_files/sample_density/single_molecule_interchange_0.json"),
            str(tmp_path / "single_molecule_interchange_1.json"),
        )
        shutil.copy(
            str(files("tyff") / "_tests/data/app_files/sample_density/single_molecule_interchange_1.json"),
            str(tmp_path / "single_molecule_interchange_0.json"),
        )
    else:
        # this case is a dummy test for the reproducibility of the Jacobian computation;
        # same inputs should produce the same results
        pass

    new_averages, new_jacobians = _get_ensemble_average_and_jacobian(
        production_future,
        tmp_path,
        reference_force_field_path=tmp_path / "ref.ff.json",
    )

    # these don't come out to be equal when swapping molecules - I don't think they should be?
    assert (reference_averages != new_averages) == swap_molecules

    assert reference_jacobians.shape == new_jacobians.shape

    # the Jacobians should have the same value, but they don't quite
    torch.testing.assert_close(reference_jacobians, new_jacobians)
    # >       torch.testing.assert_close(reference_jacobians, new_jacobians)
    # E       AssertionError: Tensor-likes are not close!
    # E
    # E       Mismatched elements: 544 / 592 (91.9%)
    # E       Greatest absolute difference: 1.9144745927413596e+18 at index (6, 51) (up to 1e-07 allowed)
    # E       Greatest relative difference: inf at index (2, 69) (up to 1e-07 allowed)

    # tyff/_tests/compute/test_jacobians.py:118: AssertionError
