"""Compute the ensemble average(s) and Jacobian matrix associated with a job."""

import dataclasses
import glob
import json
import pathlib

import torch
from openff.interchange import Interchange

import tyff
from tyff._serialization import load_tensor_force_field
from tyff.compute._files import ProductionFiles
from tyff.configs.liquid import BulkLiquid


def _gather_from_reference(local: tyff.TensorForceField, reference: tyff.TensorForceField) -> tyff.TensorForceField:
    """Rebuild ``local`` with values indexed out of ``reference``, keeping local row order so the
    topologies' parameter maps stay valid."""
    reference_by_type = reference.potentials_by_type
    potentials = []

    for potential in local.potentials:
        ref = reference_by_type[potential.type]  # KeyError: potential type missing from reference
        assert potential.parameter_cols == ref.parameter_cols, potential.type

        ref_idx = {key: i for i, key in enumerate(ref.parameter_keys)}
        idx = torch.tensor([ref_idx[key] for key in potential.parameter_keys])  # KeyError: unseen parameter

        attributes = potential.attributes
        if potential.attribute_cols is not None:
            attr_idx = torch.tensor([ref.attribute_cols.index(col) for col in potential.attribute_cols])
            attributes = ref.attributes[attr_idx]

        potentials.append(dataclasses.replace(potential, parameters=ref.parameters[idx], attributes=attributes))

    v_sites = local.v_sites
    if v_sites is not None:
        ref_idx = {key: i for i, key in enumerate(reference.v_sites.keys)}
        idx = torch.tensor([ref_idx[key] for key in v_sites.keys])
        v_sites = dataclasses.replace(v_sites, parameters=reference.v_sites.parameters[idx])

    return tyff.TensorForceField(potentials, v_sites)


def _get_ensemble_average_and_jacobian(
    production_future: dict[str, ProductionFiles],
    job_dir: str,
    reference_force_field_path: str,
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
    for path_index, interchange_path in enumerate(sorted(glob.glob(f"{job_dir}/single_molecule_interchange_*.json"))):
        unique_molecule_index = int(pathlib.Path(interchange_path).stem.split("interchange_")[-1])

        # hope we're loading up the single-molecule interchanges in the same order as we have unique molecules
        assert unique_molecule_index == path_index, (unique_molecule_index, path_index)

        with open(interchange_path) as f:
            interchanges.append(Interchange.model_validate_json(f.read()))

    assert len(interchanges) > 0, "Did not find single-molecule `Interchange`s as expected"
    assert len(interchanges) == len(compute_config["smiles"]) > 0, (
        "Did not find same number of single-molecule `Interchange`s as number of unique molecules in compute config"
    )

    local_force_field, tensor_topologies = tyff.converters.convert_interchange(interchanges)

    # before this job (before ANY job ... ) a "global" tensor force field representing the entire data set
    # needs to be created and serialized to somewhere root-like, accessible to all jobs
    reference_force_field = load_tensor_force_field(json.loads(open(reference_force_field_path).read()))

    # must sync this up with tyff/compute/_pack.py if ever either change
    n_molecules = compute_config["n_molecules"]
    # also hope ordering lines up
    n_copies = [int(n_molecules * x) for x in compute_config["x"]]

    # this is on the scale of just this job
    system = tyff.TensorSystem(
        topologies=tensor_topologies,
        n_copies=n_copies,
        is_periodic=True,  # need to handle the case of gas simulations
    )

    frames_path = pathlib.Path(f"{job_dir}/production_trajectory.msgpack")

    # Use existing tyff packing order, including attributes and optional v-sites.
    tensors, parameter_lookup, attribute_lookup, has_v_sites = _pack_force_field(reference_force_field)

    assert tensors is not None and len(tensors) > 0

    parameters = torch.cat([t.detach().reshape(-1) for t in tensors if t is not None]).requires_grad_(True)
    pieces = iter(parameters.split([t.numel() for t in tensors if t is not None]))
    tensors = tuple(None if t is None else next(pieces).reshape(t.shape) for t in tensors)
    reference_worker_force_field = _unpack_force_field(
        tensors,
        parameter_lookup,
        attribute_lookup,
        has_v_sites,
        reference_force_field,
    )

    worker_force_field = _gather_from_reference(local_force_field, reference_worker_force_field)

    means, _ = tyff.mm.compute_ensemble_averages(
        system,
        worker_force_field,
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

    # jacobian should be shaped to "global"/reference force field, not local worker-scale
    return {name: value.detach() for name, value in means.items()}, jacobian.detach()
