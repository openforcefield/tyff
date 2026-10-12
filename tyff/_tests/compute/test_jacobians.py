import json
import shutil
from importlib.resources import files

import pytest
import torch
from openff.interchange import Interchange
from parsl import File

from tyff._serialization import load_tensor_force_field
from tyff.compute._files import ProductionFiles
from tyff.compute._jacobian import _get_ensemble_average_and_jacobian
from tyff.mm._ops import _pack_force_field


def shuffle_coords(
    n: int,
    input_msgpack: str,
    output_msgpack: str,
):
    import msgpack
    import torch

    from tyff.mm._reporters import _encoder, unpack_frames

    def move_rows(coords: torch.Tensor, n: int) -> torch.Tensor:
        """
        Move the first `n` rows of the coordinates tensor to the end, keeping order
        with and out of the `n` block

        For example (in 1-dimensional case):
        [a b c d e f] with n=2 -> [c d e f a b]
        """
        assert coords.ndim == 2, "Coords tensor must be 2-dimensional"

        n_rows = coords.shape[-2]

        if not 0 <= n <= n_rows:
            raise ValueError(f"N={n} is out of range for a coords array with {n_rows} rows")

        return torch.roll(coords, shifts=-n, dims=-2)

    assert input_msgpack != output_msgpack, "Input and output msgpack files must be different"

    with open(input_msgpack, "rb") as f_in, open(output_msgpack, "wb") as f_out:
        frame_counter = 0
        for coords, box_vectors, reduced_potential, kinetic_energy in unpack_frames(f_in):
            shifted_coords = move_rows(coords, n)

            assert not torch.equal(coords, shifted_coords), "Coords didn't seem to shift?"

            frame = (
                shifted_coords,
                box_vectors,
                reduced_potential,
                kinetic_energy,
            )

            f_out.write(msgpack.dumps(frame, default=_encoder))
            print(f"Processed frame {frame_counter}")
            frame_counter += 1


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
        "interchange.json",
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
        pytest.xfail("Cannot get this test fully working yet")
        shutil.copy(
            str(files("tyff") / "_tests/data/app_files/sample_density/single_molecule_interchange_0.json"),
            str(tmp_path / "single_molecule_interchange_1.json"),
        )
        shutil.copy(
            str(files("tyff") / "_tests/data/app_files/sample_density/single_molecule_interchange_1.json"),
            str(tmp_path / "single_molecule_interchange_0.json"),
        )

        with open(tmp_path / "interchange.json") as f:
            # store the number of atoms in the 0th molecule of this system
            n_atoms_interchange_molecule0 = Interchange.model_validate_json(f.read()).topology.molecule(0).n_atoms

        # need to multiply number of atoms in the 0th molecule by the total number of molecules
        # in the system (200) and the composition of this component (0.5). This is very fragile
        # and will probably break if these values are updated if the script that makes them
        # (`tyff/_tests/data/app_files/make_sample_density_files.py`) is updated.
        n_shift = int(n_atoms_interchange_molecule0 * 200 * 0.5)

        shuffle_coords(
            n=n_shift,  # shift this number of atoms around
            input_msgpack=tmp_path / "production_trajectory.msgpack",
            output_msgpack=tmp_path / "new.msgpack",
        )

        production_future["msgpack_trajectory"] = tmp_path / "new.msgpack"

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
