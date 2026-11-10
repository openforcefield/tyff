import json
from pathlib import Path

import torch
from datasets import Dataset
from openff.toolkit import ForceField
from openff.units import unit

import tyff
import tyff.converters
import tyff.train
from tyff.compute.configs import local_config
from tyff.compute.workflow import SimulationWorkflow
from tyff.converters.openff._tensors import convert_tensor_force_field
from tyff.mm._ops import RemoteEnsembleAverage
from tyff.targets.thermo import DataEntry

KCAL_TO_KJ = 4.184


def simulate(
    tensor_force_field: tyff.TensorForceField,
    epoch_dir: Path,
    thermo_data: Dataset,
    smirnoff_force_field: str,
    n_molecules: int,
    n_replicates: int,
) -> None:
    """
    Simulate the given force field using the provided thermo data.

    This is meant to be used inside of a training loop.

    Parameters
    ----------
    tensor_force_field : tyff.TensorForceField
        The (tensor) force field to be used in the simulation.
    epoch_dir : Path
        The directory for this epoch where simulation results will be stored.
    thermo_data : Dataset
        The dataset containing thermodynamic data for the simulation.
    smirnoff_force_field : str
        The path to the original (SMIRNOFF) force field corresponding to the tensor force field.
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
        original_force_field=ForceField(smirnoff_force_field),
        tensor_force_field=tensor_force_field,
    )

    epoch_dir.mkdir(exist_ok=True)

    smirnoff_force_field_file = str(epoch_dir / "current_force_field.offxml")
    current_smirnoff_force_field.to_file(filename=smirnoff_force_field_file)

    with SimulationWorkflow(
        base_dir=str(epoch_dir / "simulations"),
        parsl_config=local_config(max_workers=6),
    ) as workflow:
        workflow.submit_dataset(
            dataset=thermo_data,
            tensor_force_field=tensor_force_field,
            force_field=smirnoff_force_field_file,  # possibly smoother to pass through openff.toolkit.ForceField
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

    Parameters
    ----------
    force_field_per_epoch : tyff.TensorForceField
        The tensor force field used for the simulations.
    base_dir : Path
        The base directory where simulation results are stored.
    job_ids : list[str]
        A 1-D list of job IDs corresponding to the simulations.
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

    # TODO: Wrap this all up into SimulationWorkflow.estimate_dataset
    for entry in dataset:  # type: ignore
        entry: DataEntry
        if entry["tag"] == "density":
            # this lookup can be improved, but INITIAL_FORCE_FIELD
            # gets mutated by this point so it's not so straightforward
            for (target_id, *_), job_ids in target_compute_mapping.items():
                if target_id == entry["id"]:
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
            for (target_id, *_), job_ids in target_compute_mapping.items():
                if target_id == entry["id"]:
                    break

            values = list()

            for job_ids_in_this_replicate in job_ids:
                assert len(job_ids_in_this_replicate) >= 3, (
                    "Enthalpy of vaporization requires three or more jobs (pure and each component)"
                )
                with (
                    open(
                        epoch_dir / "simulations" / job_ids_in_this_replicate[0] / "compute_config.json",
                    ) as f0,
                    open(
                        epoch_dir / "simulations" / job_ids_in_this_replicate[1] / "compute_config.json",
                    ) as f1,
                    open(
                        epoch_dir / "simulations" / job_ids_in_this_replicate[2] / "compute_config.json",
                    ) as f2,
                ):
                    compute_configs = [
                        json.load(f0),
                        json.load(f1),
                        json.load(f2),
                    ]

                    mixture_config = next(d for d in compute_configs if len(d["smiles"]) == 2)

                    mixture_job_id = job_ids_in_this_replicate[compute_configs.index(mixture_config)]

                    # try to keep this flexible to ternary+ mixtures
                    components = mixture_config["smiles"]
                    x = mixture_config["x"]

                    pure_configs = [
                        next(d for d in compute_configs if d["smiles"] == [component_i]) for component_i in components
                    ]

                    component_job_ids = list()
                    for component_i in components:
                        component_job_id = next(
                            job_ids_in_this_replicate[compute_configs.index(d)]
                            for d in compute_configs
                            if d["smiles"] == [component_i]
                        )
                        component_job_ids.append(component_job_id)

                    # the provided job IDs should be represented by these variables, add several checks to make sure
                    # parsing+bookkeepping is all lined up
                    assert sorted([*component_job_ids, mixture_job_id]) == sorted(job_ids_in_this_replicate)

                    assert len(component_job_ids) == len(pure_configs) == len(mixture_config["smiles"])

                    for component_config in pure_configs:
                        assert len(component_config["smiles"]) == len(component_config["x"]) == 1
                        assert component_config["smiles"][0] in mixture_config["smiles"]

                    mixture_energy = observables[mixture_job_id]["potential_energy"]

                    component_energies = [
                        observables[component_job_id]["potential_energy"] for component_job_id in component_job_ids
                    ]
                    component_n_molecules = [config["n_molecules"] for config in pure_configs]

                    # this is kcal/mol because of what observables is computed as in tyff/mm/_ops.py
                    dhmix = mixture_energy / mixture_config["n_molecules"] - sum(
                        x_i * component_energy_i / n_i
                        for x_i, component_energy_i, n_i in zip(x, component_energies, component_n_molecules)
                    )

                    values.append(dhmix * KCAL_TO_KJ)  # convert to kJ/mol

            types_of_observables = [*observables[job_ids[0][0]].keys()]

            mean: dict[str, torch.Tensor] = {key: torch.stack(values).mean() for key in types_of_observables}
            std: dict[str, torch.Tensor] = {key: torch.stack(values).std() for key in types_of_observables}

        elif entry["tag"] == "excess_molar_volume":
            raise NotImplementedError()
        elif entry["tag"] == "enthalpy_of_vaporization":
            for (target_id, *_), job_ids in target_compute_mapping.items():
                if target_id == entry["id"]:
                    break

            values = list()

            for job_ids_in_this_replicate in job_ids:
                assert len(job_ids_in_this_replicate) == 2, (
                    "Enthalpy of vaporization requires two jobs (gas and liquid)"
                )
                with (
                    open(
                        epoch_dir / "simulations" / job_ids_in_this_replicate[0] / "compute_config.json",
                    ) as f0,
                    open(epoch_dir / "simulations" / job_ids_in_this_replicate[1] / "compute_config.json") as f1,
                ):
                    compute_config0 = json.load(f0)
                    compute_config1 = json.load(f1)

                    if compute_config0["tag"] == "liquid" and compute_config1["tag"] == "gas":
                        liquid_job_id = job_ids_in_this_replicate[0]
                        liquid_config = compute_config0

                        gas_job_id = job_ids_in_this_replicate[1]
                        gas_config = compute_config1

                    elif compute_config0["tag"] == "gas" and compute_config1["tag"] == "liquid":
                        gas_job_id = job_ids_in_this_replicate[0]
                        gas_config = compute_config0

                        liquid_job_id = job_ids_in_this_replicate[1]
                        liquid_config = compute_config1

                    else:
                        raise ValueError(
                            f"Unexpected conmpute configs with tags {compute_config0['tag']} and "
                            f"{compute_config1['tag']}"
                        )

                assert gas_config["temperature"] == liquid_config["temperature"], (
                    "Gas and liquid temperatures must be the same for dHvap calculation"
                )

                # these are stored in kcal/mol, so need to convert
                gas_energy = observables[gas_job_id]["potential_energy"] * KCAL_TO_KJ
                liquid_energy = observables[liquid_job_id]["potential_energy"] * KCAL_TO_KJ

                n_molecules = liquid_config["n_molecules"]

                # 0.00831446261815324
                R = (1 * unit.avogadro_number * unit.boltzmann_constant).m_as("kJ/K")

                print(
                    f"Enthalpy of vaporization calculation: {gas_energy=}, {liquid_energy=}, "
                    f"{n_molecules=}, {R=}, {gas_config['temperature']=}"
                )

                # Enthalpy of vaporization calculation:
                #     gas_energy=tensor(1308.0884, dtype=torch.float64),
                #     liquid_energy=tensor(386691.8175, dtype=torch.float64),
                #     n_molecules=240,
                #     R=0.00831446261815324,
                #     gas_config['temperature']=300.0
                #
                # Enthalpy of vaporization calculation:
                #     gas_energy=tensor(1300.6276, dtype=torch.float64),
                #     liquid_energy=tensor(385886.6263, dtype=torch.float64),
                #     n_molecules=240,
                #     R=0.00831446261815324,
                #     gas_config['temperature']=300.0

                dhvap = gas_energy - liquid_energy / n_molecules + R * gas_config["temperature"]  # kJ/mol

                values.append(dhvap)

            # assume the types of observables are uniform across all these job IDs
            types_of_observables = [*observables[job_ids[0][0]].keys()]

            mean: dict[str, torch.Tensor] = {key: torch.stack(values).mean() for key in types_of_observables}
            std: dict[str, torch.Tensor] = {key: torch.stack(values).std() for key in types_of_observables}

        else:
            raise ValueError(f"I do not know what to do with {entry['tag']}")

        means_and_stds.append((mean, std))

    return means_and_stds


def predict(
    dataset: Dataset,
    tensor_force_field: tyff.TensorForceField,
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

    for (target_id, _target_tag, _, _, _), these_job_ids in target_compute_mapping.items():
        if target_id in this_datasets_ids:
            job_ids.extend(these_job_ids)

    attached_observables = load_remote_observables(
        force_field_per_epoch=tensor_force_field,
        base_dir=epoch_dir,
        job_ids=[x for sublist in job_ids for x in sublist],
    )

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
