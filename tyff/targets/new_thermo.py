import json
import pathlib
from collections.abc import Sequence

import torch

import tyff
import tyff.converters
from tyff.configs.targets.thermo import DataEntry


def load_remote_observables(
    force_field_per_epoch: tyff.TensorForceField,
    base_dir: str,
    job_ids: list[str],
) -> dict[str, dict]:
    """Load observables and attach to graph."""
    from tyff.mm._ops import RemoteEnsembleAverage, _pack_force_field

    tensors, *_ = _pack_force_field(
        force_field_per_epoch
    )  # force_field_per_epoch should also be the global reference FF passed to the workers
    parameters: torch.Tensor = torch.cat(
        [t.reshape(-1) for t in tensors if t is not None]
    )  # has to be flat, and not a tuple

    job_id_to_observables = {}

    for job_id in job_ids:
        job_dir = pathlib.Path(base_dir) / job_id

        assert job_dir.exists()

        with open(job_dir / "ensemble_averages.json") as f:
            means_dict: dict[str, float] = json.load(f)  # actually str, torch.float64

        jacobians: torch.Tensor = torch.load(job_dir / "jacobian.pt")

        names = list(
            means_dict
        )  # get the names of the computed properties in the means_dict so we can convert dict -> tensor -> dict again
        tensor_means = parameters.new_tensor([means_dict[x] for x in names])
        attached_means = RemoteEnsembleAverage.apply(
            tensor_means,
            jacobians.to("cpu"),  # to(device) ?, get from somewhere?
            parameters,
        )

        job_id_to_observables[job_id] = dict(zip(names, attached_means))

    return job_id_to_observables


def compute_properties(
    dataset: Sequence[DataEntry],
    observables: dict[str, dict[str, torch.Tensor]],
    simulation_root: str,
) -> list[tuple[dict, dict]]:

    means_and_stds: list[tuple[dict, dict]] = list()

    with open(pathlib.Path(simulation_root) / "target_job_map.json") as f:
        target_compute_mapping = {tuple(k): v for k, v in json.load(f)}

    for entry in dataset:
        if entry["tag"] == "density":
            # this includes replicates, but only jobs for this physical property target
            # structure is a list (length n_replicates) of tuples, each tuple is
            # 1-length and contains a compute config for (a single) liquid job

            compute_configs = list()

            for as_submitted, job_ids in target_compute_mapping.items():
                if as_submitted[0] == entry["id"]:
                    compute_configs.append(job_ids)

            types_of_observables = [*observables[job_ids[0]].keys()]

            # could be simpler to return these as 1-D tensors - make sure to sort
            # keys to avoid ordering issues
            mean = {
                key: torch.stack([observables[job_id][key] for job_id in job_ids]).mean()
                for key in types_of_observables
            }
            std = {
                key: torch.stack([observables[job_id][key] for job_id in job_ids]).std()
                for key in types_of_observables
            }

            # figure out which job id/s (if multiple replicas) in the observable corresponds to
            # and calculate means and stds
        elif entry["tag"] == "dielectric_constant":
            raise NotImplementedError()
            # likely identical to density case
        elif entry["tag"] == "enthalpy_of_mixing":
            raise NotImplementedError()
            # figure out the three job IDs necessary from observables, calculate, etc
        elif entry["tag"] == "excess_molar_volume":
            raise NotImplementedError()
        elif entry["tag"] == "enthalpy_of_vaporization":
            raise NotImplementedError()
        else:
            raise ValueError(f"I do not know what to do with {entry['tag']}")

        means_and_stds.append((mean, std))

    return means_and_stds


def predict(
    dataset: Sequence[DataEntry],
    force_field: tyff.TensorForceField,
    job_ids: list[str],
    base_dir: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    attached_means = load_remote_observables(force_field, base_dir, job_ids)

    means_and_stds: list[dict[str, torch.Tensor]] = compute_properties(
        dataset=dataset,
        observables=attached_means,
        simulation_root=base_dir,
    )

    ref_values = [entry["value"] for entry in dataset]
    ref_stds = [entry["std"] for entry in dataset]
    predicted_values = [mean["density"] for mean, _ in means_and_stds]
    predicted_stds = [std["density"] for _, std in means_and_stds]

    ref_values = torch.tensor(ref_values)
    ref_stds = torch.tensor(ref_stds)
    predicted_values = torch.tensor(predicted_values)
    predicted_stds = torch.tensor(predicted_stds)

    return ref_values, ref_stds, predicted_values, predicted_stds
