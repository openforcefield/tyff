import tarfile
from pathlib import Path

from datasets import Dataset
from openff.toolkit import ForceField, Molecule

import tyff
import tyff.converters
import tyff.train
from tyff.compute._logging import _set_up_logger
from tyff.compute.configs import local_config
from tyff.compute.workflow import SimulationWorkflow
from tyff.converters.openff._tensors import convert_tensor_force_field
from tyff.targets import energy, thermo

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

    logger.info(f"Loaded {len(thermo_data)} thermo entries and {len(energy_data)} energy entries")

    return thermo_data, energy_data


def simulate(
    force_field: tyff.TensorForceField,
    root_dir: Path,
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
    root_dir.mkdir(exist_ok=True)

    logger.info("Writing out the current SMIRNOFF force field to file")
    smirnoff_force_field_file = str(root_dir / "current_force_field.offxml")
    current_smirnoff_force_field.to_file(smirnoff_force_field_file)

    logger.info("Setting up simulation workflow")
    with SimulationWorkflow(
        base_dir=str(root_dir / "simulations"),
        parsl_config=local_config(max_workers=8),
    ) as workflow:
        logger.info("Submitting dataset")
        workflow.submit_dataset(
            dataset=thermo_data,
            force_field=smirnoff_force_field_file,
            n_molecules=n_molecules,
            n_replicates=n_replicates,
        )

        workflow.wait()


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
            "vdW": tyff.train.ParameterConfig(
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
            root_dir=root / f"EPOCH_{epoch}",
            thermo_data=thermo_data,
            n_replicates=N_REPLICATES,
            n_molecules=N_MOLECULES,
        )

    return


if __name__ == "__main__":
    main(root=Path(__file__).resolve().parent)
