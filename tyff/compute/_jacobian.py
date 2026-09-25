"""Compute the ensemble average(s) and Jacobian matrix associated with a job."""

import glob
import json
import pathlib

import torch
from openff.toolkit import Interchange

from tyff.configs.liquid import BulkLiquid


def _get_ensemble_average_and_jacobian(
    job_dir: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    import openmm.unit
    import torch

    import tyff.mm
    from tyff.mm._ops import _pack_force_field, _unpack_force_field
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
    for path_index, interchange_path in enumerate(glob.glob(f"{job_dir}/single_molecule_interchange_*.json")):
        unique_molecule_index = int(pathlib.Path(interchange_path).stem.split("single_molecule_interchange_")[-1])

        # hope we're loading up the single-molecule interchanges in the same order as we have unique molecules
        assert unique_molecule_index == path_index

        with open(interchange_path) as f:
            interchanges.append(Interchange.model_validate_json(f.read()))

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

    # Jacobian rows follow sorted observable names; columns follow packed parameters.
    jacobian = torch.stack(
        [
            torch.autograd.grad(means[name], parameters, retain_graph=True)[0]
            for name in sorted(means)  # Claude doesn't think this should be sorted
        ]
    )
    return {name: value.detach() for name, value in means.items()}, jacobian.detach()
