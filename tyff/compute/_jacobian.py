"""Compute the ensemble average(s) and Jacobian matrix associated with a job."""

import glob
import json
import pathlib

import torch
from openff.interchange import Interchange

from tyff.compute._files import ProductionFiles
from tyff.configs.liquid import BulkLiquid


def _get_ensemble_average_and_jacobian(
    production_future: dict[str, ProductionFiles],
    interchanges_path: str | pathlib.Path,
    job_dir: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    import openmm.unit
    import torch

    import tyff.mm
    from tyff.mm._ops import _pack_force_field, _unpack_force_field

    # try to safeguard against the previous step failing
    assert pathlib.Path(production_future["simulation_files"]["msgpack_trajectory"].filepath).exists()

    # the arguments we really care about are:
    #     system: tyff.TensorSystem,
    #     frames_path: pathlib.Path,
    #     temperature: float,
    #     pressure: float | None,
    #     force_field: tyff.TensorForceField,
    #
    # so grab them from scattered files we expect to be in this job directory.

    # TODO: Handle case of cas simulations
    compute_config = BulkLiquid(**json.load(open(f"{job_dir}/compute_config.json")))  # type: ignore[typeddict-item]

    temperature = compute_config["temperature"]  # kelvin, float
    pressure = compute_config.get("pressure")  # atmosphere, float | None

    interchanges = []
    for path_index, interchange_path in enumerate(sorted(glob.glob(f"{interchanges_path}/interchange_*.json"))):
        unique_molecule_index = int(pathlib.Path(interchange_path).stem.split("interchange_")[-1])

        # hope we're loading up the single-molecule interchanges in the same order as we have unique molecules
        assert unique_molecule_index == path_index, (unique_molecule_index, path_index)

        with open(interchange_path) as f:
            interchanges.append(Interchange.model_validate_json(f.read()))

    assert len(interchanges) > 0, "Did not find single-molecule `Interchange`s as expected"

    tensor_force_field, tensor_topologies = tyff.converters.convert_interchange(interchanges)

    # must sync this up with tyff/compute/_pack.py if ever either change
    n_molecules = compute_config["n_molecules"]
    # also hope ordering lines up
    n_copies = [int(n_molecules * x) for x in compute_config["x"]]

    system = tyff.TensorSystem(
        topologies=tensor_topologies,
        n_copies=n_copies,
        is_periodic=True,  # need to handle the case of gas simulations
    )

    frames_path = pathlib.Path(f"{job_dir}/production_trajectory.msgpack")

    # Use existing tyff packing order, including attributes and optional v-sites.
    tensors, parameter_lookup, attribute_lookup, has_v_sites = _pack_force_field(tensor_force_field)

    assert tensors is not None and len(tensors) > 0

    parameters = torch.cat([t.detach().reshape(-1) for t in tensors if t is not None]).requires_grad_(True)
    pieces = iter(parameters.split([t.numel() for t in tensors if t is not None]))
    tensors = tuple(None if t is None else next(pieces).reshape(t.shape) for t in tensors)
    worker_ff = _unpack_force_field(
        tensors,
        parameter_lookup,
        attribute_lookup,
        has_v_sites,
        tensor_force_field,
    )
    means, _ = tyff.mm.compute_ensemble_averages(
        system,
        worker_ff,
        frames_path,
        temperature * openmm.unit.kelvin,
        None if pressure is None else pressure * openmm.unit.atmosphere,
    )

    # originally this was sorted(means),
    # https://github.com/openforcefield/tyff/pull/173#discussion_r4126720329
    #
    # if that's changed back then we need to make BOTH
    # the jacobian and ensemble averages sorted in serialization AND return values
    # https://github.com/openforcefield/tyff/pull/173#discussion_r4126720329
    jacobian = torch.stack([torch.autograd.grad(means[name], parameters, retain_graph=True)[0] for name in means])

    # save ensemble averages and jacobian in each job dir, consider revisiting this decision in the future
    with open(f"{job_dir}/ensemble_averages.json", "w") as f:
        json.dump({name: value.detach().tolist() for name, value in means.items()}, f)

    with open(f"{job_dir}/jacobian.pt", "wb") as f:
        torch.save(jacobian.detach(), f)

    return {name: value.detach() for name, value in means.items()}, jacobian.detach()
