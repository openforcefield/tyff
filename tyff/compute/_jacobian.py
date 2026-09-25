"""Compute the ensemble average(s) and Jacobian matrix associated with a job."""

import pathlib

import torch

import tyff


def ensemble_average_jacobian(
    system: tyff.TensorSystem,
    frames_path: pathlib.Path,
    temperature: float,
    pressure: float | None,
    force_field: tyff.TensorForceField,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    import openmm.unit
    import torch

    import tyff.mm
    from tyff.mm._ops import _pack_force_field, _unpack_force_field

    # Use existing tyff packing order, including attributes and optional v-sites.
    tensors, parameter_lookup, attribute_lookup, has_v_sites = _pack_force_field(force_field)
    parameters = torch.cat([t.detach().reshape(-1) for t in tensors if t is not None]).requires_grad_(True)
    pieces = iter(parameters.split([t.numel() for t in tensors if t is not None]))
    tensors = tuple(None if t is None else next(pieces).reshape(t.shape) for t in tensors)
    worker_ff = _unpack_force_field(
        tensors,
        parameter_lookup,
        attribute_lookup,
        has_v_sites,
        force_field,
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
