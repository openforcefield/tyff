import json
from pathlib import Path

import torch
from datasets import Dataset
from openff.toolkit import ForceField

import tyff
import tyff.converters
import tyff.train
from tyff.compute.configs import local_config
from tyff.compute.workflow import SimulationWorkflow
from tyff.converters.openff._tensors import convert_tensor_force_field
from tyff.mm._ops import RemoteEnsembleAverage
from tyff.targets.thermo import DataEntry


def simulate(
    force_field: tyff.TensorForceField,
    epoch_dir: Path,
    thermo_data: Dataset,
    smirnoff_force_field: ForceField,
    n_molecules: int,
    n_replicates: int,
) -> None:
    """
    Simulate the given force field using the provided thermo data.

    This is meant to be used inside of a training loop.

    Parameters
    ----------
    force_field : tyff.TensorForceField
        The force field to be used in the simulation.
    epoch_dir : Path
        The directory for this epoch where simulation results will be stored.
    thermo_data : Dataset
        The dataset containing thermodynamic data for the simulation.
    smirnoff_force_field : ForceField
        The original SMIRNOFF force field corresponding to the tensor force field.
    n_molecules : int
        The number of molecules to simulate.
    n_replicates : int
        The number of replicates for each simulation.

    Returns
    -------
    None
    """

    # write out this epoch's (tensor) force field as a SMIRNOFF force field
    current_smirnoff_force_field = convert_tensor_force_field(
        original_force_field=smirnoff_force_field,
        tensor_force_field=force_field,
    )

    epoch_dir.mkdir(exist_ok=True)

    smirnoff_force_field_file = str(epoch_dir / "current_force_field.offxml")
    current_smirnoff_force_field.to_file(smirnoff_force_field_file)

    with SimulationWorkflow(
        base_dir=str(epoch_dir / "simulations"),
        parsl_config=local_config(max_workers=6),
    ) as workflow:
        workflow.submit_dataset(
            dataset=thermo_data,
            force_field=smirnoff_force_field_file,
            n_molecules=n_molecules,
            n_replicates=n_replicates,
        )

        workflow.wait()


def load_remote_observables(
    force_field_per_epoch: tyff.TensorForceField,
    base_dir: Path,
    job_ids: list[str],
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Load observables and attach to graph.

    Returns a mapping between job IDs and associated observables.
    """
    from tyff.mm._ops import _pack_force_field

    tensors, *_ = _pack_force_field(
        force_field_per_epoch
    )  # force_field_per_epoch should also be the global reference FF passed to the workers
    parameters: torch.Tensor = torch.cat(
        [t.reshape(-1) for t in tensors if t is not None]
    )  # has to be flat, and not a tuple

    job_id_to_observables = {}

    for job_id in job_ids:
        job_dir = base_dir / "simulations" / job_id

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
    dataset: Dataset,
    observables: dict[str, dict[str, torch.Tensor]],
    epoch_dir: Path,
) -> list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:

    means_and_stds: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = list()

    # think about better standardizing some of these paths
    with open(epoch_dir / "simulations" / "target_job_map.json") as f:
        target_compute_mapping = {tuple(k): v for k, v in json.load(f)}

    for entry in dataset:  # type: ignore
        entry: DataEntry
        if entry["tag"] == "density":
            # this lookup can be improved, but INITIAL_FORCE_FIELD
            # gets mutated by this point so it's not so straightforward
            for as_submitted, job_ids in target_compute_mapping.items():
                if as_submitted[0] == entry["id"]:
                    break

            # this is set with an iteration variable - can we assume that
            # this will be the same for all job_ids within this target?
            # all targets will be the same (i.e. always computing dhvap)
            # but job-to-job maybe not? (i.e. gas vs. liquid jobs)
            types_of_observables = [*observables[job_ids[0][0]].keys()]

            # could be simpler to return these as 1-D tensors - make sure to sort
            # keys to avoid ordering issues
            mean: dict[str, torch.Tensor] = {
                key: torch.stack([observables[job_id[0]][key] for job_id in job_ids]).mean()
                for key in types_of_observables
            }
            std: dict[str, torch.Tensor] = {
                key: torch.stack([observables[job_id[0]][key] for job_id in job_ids]).std()
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
    dataset: Dataset,
    force_field: tyff.TensorForceField,
    epoch_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # in order to get all of the (remote) observables for all simulations run from this
    # dataset, we need the job IDs of these

    with open(epoch_dir / "simulations" / "target_job_map.json") as f:
        target_compute_mapping = {tuple(k): v for k, v in json.load(f)}

    this_datasets_ids: set[str] = set(entry["id"] for entry in dataset)  # type: ignore

    # this needs to be refactored to pass in _targets_, not jobs.
    # synonymous with pure density, but for others (dhmix, dhvap, etc.) need to
    # collect information from multiple jobs (i.e. gas and liquid jobs for dhvap)
    # and condense down to one value (dhvap) to be reported
    job_ids: list[str] = list()

    for (target_id, _, _, _), these_job_ids in target_compute_mapping.items():
        if target_id in this_datasets_ids:
            job_ids.extend(these_job_ids)

    # see comment above
    job_ids = [x[0] for x in job_ids]

    attached_observables = load_remote_observables(force_field, epoch_dir, job_ids)

    means_and_stds: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = compute_properties(
        dataset=dataset,
        observables=attached_observables,
        epoch_dir=epoch_dir,
    )

    ref_values = [entry["value"] for entry in dataset]  # type: ignore
    ref_stds = [entry["std"] for entry in dataset]  # type: ignore
    predicted_values = [mean["density"] for mean, _ in means_and_stds]
    predicted_stds = [std["density"] for _, std in means_and_stds]

    ref_values = torch.tensor(ref_values)
    ref_stds = torch.tensor(ref_stds)
    predicted_values = torch.stack(predicted_values)
    predicted_stds = torch.stack(predicted_stds)

    return ref_values, ref_stds, predicted_values, predicted_stds
