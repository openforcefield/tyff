"""Train against thermodynamic properties."""

from __future__ import annotations

import copy
import json
import logging
import pathlib
import typing
from pathlib import Path

import datasets
import datasets.table
import parsl
import pyarrow
import torch
from datasets import Dataset
from openff.toolkit import ForceField
from openff.units import unit

import tyff
import tyff.mm
import tyff.molecule
import tyff.optim
import tyff.utils
import tyff.utils.dataset
import tyff.utils.loss
from tyff.converters.openff._tensors import convert_tensor_force_field
from tyff.mm._ops import RemoteEnsembleAverage

KJ_TO_KCAL = 1 / 4.184

if typing.TYPE_CHECKING:
    import tyff.train


_LOGGER = logging.getLogger(__name__)


DataType = typing.Literal["density", "hvap", "hmix"]

DATA_TYPES = typing.get_args(DataType)

DATA_SCHEMA = pyarrow.schema(
    [
        ("id", pyarrow.string()),
        ("tag", pyarrow.string()),
        ("smiles", pyarrow.list_(pyarrow.string())),
        ("x", pyarrow.list_(pyarrow.float64())),
        ("temperature", pyarrow.float64()),
        ("pressure", pyarrow.float64()),
        ("value", pyarrow.float64()),
        ("std", pyarrow.float64()),
        ("units", pyarrow.string()),
        ("source", pyarrow.string()),
    ]
)

Phase = typing.Literal["bulk", "vacuum"]
PHASES = typing.get_args(Phase)


class DataEntry(typing.TypedDict):
    """Represents a single experimental data point."""

    id: str
    """
    A unique identifier for the data point, hook up to _get_raw_property_hash for physical property data.
    """

    tag: DataType
    """The type of data point."""

    smiles: list[str]
    """The SMILES definitions of each components."""
    x: list[float]
    """The mole fractions of each components. Values must sum to 1.0."""

    temperature: float
    """The temperature at which the data point was measured."""
    pressure: float
    """The pressure at which the data point was measured."""

    value: float
    """The value of the data point."""
    std: float | None
    """The standard deviation of the data point if available."""
    units: str
    """The units of the data point."""

    source: str
    """The source of the data point."""


class _Observables(typing.NamedTuple):
    """Ensemble averages of the observables computed from a simulation."""

    mean: dict[str, torch.Tensor]
    """The mean value of each observable with ``shape=()``."""
    std: dict[str, torch.Tensor]
    """The standard deviation of each observable with ``shape=()``."""


def create_dataset(*entries: DataEntry) -> datasets.Dataset:
    """Create a dataset from a list of existing data points.

    Args:
        entries: The data points to create the dataset from.

    Returns:
        The created dataset.
    """
    copied_entries = [copy.deepcopy(entry) for entry in entries]

    copied_entries = [entry for entry in entries]

    for copied_entry in copied_entries:
        copied_entry["smiles"] = [tyff.molecule.map_smiles(value) for value in copied_entry["smiles"]]

    # could roundtrip through DataEntry here as a form of validation

    table = pyarrow.Table.from_pylist([*copied_entries], schema=DATA_SCHEMA)

    return datasets.Dataset(datasets.table.InMemoryTable(table))


def create_from_evaluator(dataset_file: pathlib.Path) -> datasets.Dataset:
    """
    Create a dataset from an evaluator PhysicalPropertyDataSet

    Args:
        dataset_file: The path to the evaluator dataset

    Returns:
        The created dataset
    """

    _evaluator_to_prop = {
        "openff.evaluator.properties.density.Density": "density",
        "openff.evaluator.properties.enthalpy.EnthalpyOfMixing": "hmix",
        "openff.evaluator.properties.enthalpy.EnthalpyOfVaporization": "hvap",
    }
    _prop_units = {"density": "g/mL", "hmix": "kcal/mol", "hvap": "kcal/mol"}

    properties: list[DataEntry] = []
    property_data = json.load(dataset_file.open())

    for phys_prop in property_data["properties"]:
        try:
            prop_type = _evaluator_to_prop[phys_prop["@type"]]
        except KeyError:
            raise KeyError(f"{phys_prop['@type']} not currently supported.") from None

        smiles_and_role = [
            (comp["smiles"], comp["smiles"] + "{" + comp["role"]["value"] + "}")
            for comp in phys_prop["substance"]["components"]
        ]
        smiles_a, role_a = smiles_and_role[0]
        x_a = phys_prop["substance"]["amounts"][role_a][0]["value"]
        if len(smiles_and_role) == 1:
            smiles_b, x_b = None, None
        elif len(smiles_and_role) == 2:
            smiles_b, role_b = smiles_and_role[1]
            x_b = phys_prop["substance"]["amounts"][role_b][0]["value"]
        else:
            raise NotImplementedError("up to binary mixtures are currently supported")

        temp_unit = getattr(unit, phys_prop["thermodynamic_state"]["temperature"]["unit"])
        temp = phys_prop["thermodynamic_state"]["temperature"]["value"] * temp_unit
        pressure_unit = getattr(unit, phys_prop["thermodynamic_state"]["pressure"]["unit"])
        pressure = phys_prop["thermodynamic_state"]["pressure"]["value"] * pressure_unit
        value = phys_prop["value"]["value"] * getattr(unit, phys_prop["value"]["unit"])
        std = phys_prop["uncertainty"]["value"] * getattr(unit, phys_prop["uncertainty"]["unit"])
        default_units = getattr(unit, _prop_units[prop_type])
        prop = {
            "tag": prop_type,
            "smiles": [smiles_a, smiles_b] if smiles_b is not None else [smiles_a],
            "x": [x_a, x_b] if x_b is not None else [x_a],
            "temperature": temp.to(unit.kelvin).m,
            "pressure": pressure.to(unit.kilopascal).m,
            "value": value.to(default_units).m,
            "units": _prop_units[prop_type],
            "std": std.to(default_units).m,
            "source": phys_prop["source"]["doi"],
        }
        properties.append(prop)

    return create_dataset(*properties)


def extract_smiles(dataset: datasets.Dataset) -> list[str]:
    """Return a list of unique SMILES strings in the dataset.

    Args:
        dataset: The dataset to extract the SMILES strings from.

    Returns:
        The unique SMILES strings with full atom mapping.
    """
    # TODO: This is potentially slow for larger datasets
    return sorted(set([val for row in dataset["smiles"] for val in row]))


def simulate(
    parsl_config: parsl.Config,
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
    parsl_config : parsl.Config
        The Parsl configuration to be used to manage these simulations.
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
    from tyff.compute.workflow import SimulationWorkflow

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
        parsl_config=parsl_config,  # possibly bad to _take in_ an initialized config vs. instantiate it here?
    ) as workflow:
        workflow.submit_dataset(
            dataset=thermo_data,
            tensor_force_field=tensor_force_field,
            force_field=smirnoff_force_field_file,  # possibly smoother to pass through openff.toolkit.ForceField
            n_molecules=n_molecules,
            n_replicates=n_replicates,
        )

        workflow.wait()

    # here may be a place to free up memory, i.e. `del workflow` or something more involved with Parsl
    # but this probably should be handled automatically by SimulationWorkflow?


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
) -> list[tuple[dict[str, str | torch.Tensor], dict[str, str | torch.Tensor]]]:

    # TODO: Make this list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = list()
    means_and_stds: list[tuple[dict[str, str | torch.Tensor], dict[str, str | torch.Tensor]]] = list()

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

            # probably don't need to carry the tag, maybe just return the value?
            mean: dict[str, str | torch.Tensor] = {
                "tag": entry["tag"],
                "value": torch.stack([observables[job_id[0]]["density"] for job_id in job_ids]).mean(),
            }
            std: dict[str, str | torch.Tensor] = {
                "tag": entry["tag"],
                "std": torch.stack([observables[job_id[0]]["density"] for job_id in job_ids]).std(),
            }

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

                    values.append(dhmix)  # leave in kcal/mol

            mean: dict[str, str | torch.Tensor] = {"tag": entry["tag"], "value": torch.stack(values).mean()}
            std: dict[str, str | torch.Tensor] = {"tag": entry["tag"], "std": torch.stack(values).std()}

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

                # these are stored in kcal/mol
                gas_energy = observables[gas_job_id]["potential_energy"]
                liquid_energy = observables[liquid_job_id]["potential_energy"]

                n_molecules = liquid_config["n_molecules"]

                # 0.0019872042586408316, or  0.00831446261815324 / 4.184
                R = (1 * unit.avogadro_number * unit.boltzmann_constant).m_as("kcal/K")

                dhvap = gas_energy - liquid_energy / n_molecules + R * gas_config["temperature"]  # still kcal/mol

                # TODO: Wrap this up into a logger.debug layer
                # print(
                #     f"Enthalpy of vaporization calculation (gas job id={gas_job_id}): {gas_energy=}, "
                #     f"\n\t(liquid job id={liquid_job_id}): {liquid_energy=}, "
                #     f"\n\t{n_molecules=}, {R=}, {gas_config['temperature']=}"
                #     f"\n\t{dhvap=}"
                # )

                # should still be kcal/mol here
                values.append(dhvap)

            mean: dict[str, str | torch.Tensor] = {"tag": entry["tag"], "value": torch.stack(values).mean()}
            std: dict[str, str | torch.Tensor] = {"tag": entry["tag"], "std": torch.stack(values).std()}

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

    means_and_stds: list[tuple[dict[str, str | torch.Tensor], dict[str, str | torch.Tensor]]] = compute_properties(
        dataset=dataset,
        observables=attached_observables,
        epoch_dir=epoch_dir,
    )

    ref_values = torch.tensor([entry["value"] for entry in dataset])  # type: ignore
    ref_stds = torch.tensor([entry["std"] for entry in dataset])  # type: ignore
    predicted_values = torch.stack([mean["value"] for mean, _ in means_and_stds])  # type: ignore
    predicted_stds = torch.stack([std["std"] for _, std in means_and_stds])  # type: ignore

    return ref_values, ref_stds, predicted_values, predicted_stds
