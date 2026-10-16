import tarfile
from pathlib import Path

from datasets import Dataset
from openff.toolkit import ForceField, Molecule

import tyff
import tyff.converters
import tyff.train
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


def main(root: Path) -> None:
    if not (root / "data/energy").exists():
        with tarfile.open(root / "data/energy.tar.gz", "r:gz") as tar:
            tar.extractall(path=root / "data/", filter="data")

    if not (root / "data/thermo").exists():
        with tarfile.open(root / "data/thermo.tar.gz", "r:gz") as tar:
            tar.extractall(path=root / "data/", filter="data")

    thermo_data = Dataset.load_from_disk(root / "data/thermo")
    energy_data = Dataset.load_from_disk(root / "data/energy")

    print(f"Loaded {len(thermo_data)} thermo entries and {len(energy_data)} energy entries")

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
        (root / f"EPOCH_{epoch}").mkdir(exist_ok=True)
        # wrap this up into a single simulate() function?
        # def simulate(
        #   dataset: datasets.Dataset,
        #   tensor_force_field: tyff.TensorForceField,
        #   use_this_dir: str,
        #   n_replicates: int,
        #   n_molecules: int,
        # ) -> None:

        current_ff = trainable.to_force_field(theta)

        # write out this epoch's (tensor) force field as a SMIRNOFF force field
        current_smirnoff_force_field = convert_tensor_force_field(
            original_force_field=ForceField(INITIAL_FORCE_FIELD),
            tensor_force_field=current_ff,
        )

        (root / f"EPOCH_{epoch}").mkdir(exist_ok=True)

        smirnoff_force_field_file = str(root / f"EPOCH_{epoch}" / "current_force_field.offxml")
        current_smirnoff_force_field.to_file(smirnoff_force_field_file)

        with SimulationWorkflow(
            base_dir=str(root / f"EPOCH_{epoch}" / "simulations"),
            parsl_config=local_config(),
        ) as workflow:
            workflow.submit_dataset(
                dataset=thermo_data,
                force_field=smirnoff_force_field_file,
                n_molecules=N_MOLECULES,
                n_replicates=N_REPLICATES,
            )

    return


if __name__ == "__main__":
    main(root=Path(__file__).resolve().parent)
