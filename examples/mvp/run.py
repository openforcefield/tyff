import json
import tarfile
from pathlib import Path

import torch
from datasets import Dataset
from openff.toolkit import ForceField, Molecule

import tyff
import tyff.converters
import tyff.train
from tyff.compute._logging import _set_up_logger
from tyff.compute.configs import local_config
from tyff.compute.workflow import SimulationWorkflow
from tyff.converters.openff._tensors import convert_tensor_force_field
from tyff.mm._ops import RemoteEnsembleAverage
from tyff.targets import energy, thermo
from tyff.targets.thermo import DataEntry

N_EPOCHS = 1
N_REPLICATES = 1
N_MOLECULES = 200
INITIAL_FORCE_FIELD = "openff-2.3.0.offxml"


def topologies_for(
    smiles_set: set[str],
    smirnoff_force_field: ForceField,
) -> tuple[dict[str, tyff.TensorTopology], tyff.TensorForceField]:
    # this function is maybe meant to be surely there is a more straightforward way to do this?
    interchanges = [
        smirnoff_force_field.create_interchange(Molecule.from_mapped_smiles(smiles).to_topology())
        for smiles in smiles_set
    ]

    tensor_ff, topologies = tyff.converters.convert_interchange(interchanges)

    return {smiles: topology for smiles, topology in zip(smiles_set, topologies)}, tensor_ff


def load_datasets(root: Path) -> tuple[Dataset, Dataset]:
    """Load datasets stored at data/*.tar.gz"""
    if not (root / "data/energy").exists():
        with tarfile.open(root / "data/energy.tar.gz", "r:gz") as tar:
            tar.extractall(path=root / "data/", filter="data")

    if not (root / "data/thermo").exists():
        with tarfile.open(root / "data/thermo.tar.gz", "r:gz") as tar:
            tar.extractall(path=root / "data/", filter="data")

    thermo_data = Dataset.load_from_disk(root / "data/thermo")
    energy_data = Dataset.load_from_disk(root / "data/energy")

    thermo_data = thermo_data.filter(lambda x: x["tag"] == "density")

    logger.info(f"Loaded {len(thermo_data)} thermo entries and {len(energy_data)} energy entries")

    return thermo_data, energy_data


def simulate(
    force_field: tyff.TensorForceField,
    epoch_dir: Path,
    thermo_data: Dataset,
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
    root_dir : Path
        The root directory where simulation results will be stored.
    thermo_data : Dataset
        The dataset containing thermodynamic data for the simulation.
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
        original_force_field=ForceField(INITIAL_FORCE_FIELD),
        tensor_force_field=force_field,
    )

    logger.info("Making directory for this epoch's simulations")
    epoch_dir.mkdir(exist_ok=True)

    logger.info("Writing out the current SMIRNOFF force field to file")
    smirnoff_force_field_file = str(epoch_dir / "current_force_field.offxml")
    current_smirnoff_force_field.to_file(smirnoff_force_field_file)

    logger.info("Setting up simulation workflow")
    with SimulationWorkflow(
        base_dir=str(epoch_dir / "simulations"),
        parsl_config=local_config(max_workers=6),
    ) as workflow:
        logger.info("Submitting dataset")
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
        job_dir = base_dir / job_id

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
            compute_configs_for_target = list()

            for as_submitted, job_ids in target_compute_mapping.items():
                if as_submitted[0] == entry["id"]:
                    compute_configs_for_target.append(job_ids)

            # this is set with an iteration variable - can we assume that
            # this will be the same for all job_ids within this target?
            # all targets will be the same (i.e. always computing dhvap)
            # but job-to-job maybe not? (i.e. gas vs. liquid jobs)

            types_of_observables = [*observables[job_ids[0]].keys()]

            # could be simpler to return these as 1-D tensors - make sure to sort
            # keys to avoid ordering issues
            mean: dict[str, torch.Tensor] = {
                key: torch.stack([observables[job_id][key] for job_id in job_ids]).mean()
                for key in types_of_observables
            }
            std: dict[str, torch.Tensor] = {
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
    dataset: Dataset,
    force_field: tyff.TensorForceField,
    epoch_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # in order to get all of the (remote) observables for all simulations run from this
    # dataset, we need the job IDs of these

    with open(epoch_dir / "simulations" / "target_job_map.json") as f:
        target_compute_mapping = {tuple(k): v for k, v in json.load(f)}

    this_datasets_ids: set[str] = set(entry["id"] for entry in dataset)  # type: ignore

    job_ids: list[str] = list()

    for (target_id, _, _, _), these_job_ids in target_compute_mapping.items():
        if target_id in this_datasets_ids:
            job_ids.extend(these_job_ids)

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
    predicted_values = torch.tensor(predicted_values)
    predicted_stds = torch.tensor(predicted_stds)

    return ref_values, ref_stds, predicted_values, predicted_stds


def main(root: Path) -> None:
    global logger

    logger = _set_up_logger(f"{root}/mvp.log")

    thermo_data, energy_data = load_datasets(root)

    # should this be convert_interchanges?
    # is `topologies` list[tyff.TensorTopology] or dict[str, tyff.TensorTopology]?
    _topologies, tensor_ff = topologies_for(
        smiles_set={
            *thermo.extract_smiles(thermo_data),
            *energy.extract_smiles(energy_data),
        },
        smirnoff_force_field=ForceField(INITIAL_FORCE_FIELD),
    )

    trainable = tyff.train.Trainable(
        tensor_ff,
        parameters={
            "vdW": tyff.train.ParameterConfig(  # type: ignore
                cols=["epsilon", "sigma"],
                scales={"epsilon": 10.0, "sigma": 1.0},
                limits={"epsilon": (0.0001, None), "sigma": (0.5, None)},
            )
        },
        attributes={},
    )
    theta = trainable.to_values().requires_grad_(True)

    for epoch in range(N_EPOCHS):
        current_ff = trainable.to_force_field(theta)

        simulate(
            force_field=current_ff,
            epoch_dir=root / f"EPOCH_{epoch}",
            thermo_data=thermo_data,
            n_replicates=N_REPLICATES,
            n_molecules=N_MOLECULES,
        )

        predict(
            dataset=thermo_data,
            force_field=current_ff,
            epoch_dir=root / f"EPOCH_{epoch}",
        )
        # -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    return


if __name__ == "__main__":
    main(root=Path(__file__).resolve().parent)
