import tarfile
from json import dump
from pathlib import Path

from datasets import Dataset
from openff.toolkit import ForceField
from rich import print
from torch import no_grad
from torch.optim import Adam

from tyff._serialization import dump_tensor_force_field
from tyff.compute._logging import _set_up_logger
from tyff.converters import topologies_for
from tyff.targets import energy, thermo
from tyff.targets.new_thermo import predict, simulate
from tyff.train import ParameterConfig, Trainable

N_EPOCHS = 10
N_REPLICATES = 3
N_MOLECULES = 200
INITIAL_FORCE_FIELD = "openff-2.3.0.offxml"


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

    # temporary - need to go back and add dhmix, dhvap, etc.
    thermo_data = thermo_data.filter(lambda x: x["tag"] == "density")

    logger.info(f"Loaded {len(thermo_data)} thermo entries and {len(energy_data)} energy entries")

    return thermo_data, energy_data


def main(root: Path) -> None:
    global logger

    logger = _set_up_logger(f"{root}/mvp.log")

    (root / "force_fields").mkdir(exist_ok=True)

    thermo_data, energy_data = load_datasets(root)

    topologies, tensor_ff = topologies_for(
        smiles_set={
            *thermo.extract_smiles(thermo_data),
            *energy.extract_smiles(energy_data),
        },
        smirnoff_force_field=ForceField(INITIAL_FORCE_FIELD),
    )

    trainable = Trainable(
        tensor_ff,
        parameters={
            "vdW": ParameterConfig(  # type: ignore
                cols=["epsilon", "sigma"],
                scales={"epsilon": 10.0, "sigma": 1.0},
                limits={"epsilon": (0.0001, None), "sigma": (0.5, None)},
            )
        },
        attributes={},
    )
    theta = trainable.to_values().requires_grad_(True)
    optimizer = Adam([theta], lr=0.01)

    for epoch in range(N_EPOCHS):
        current_ff = trainable.to_force_field(theta)

        # no return value?
        simulate(
            force_field=current_ff,
            epoch_dir=root / f"EPOCH_{epoch}",
            thermo_data=thermo_data,
            smirnoff_force_field=ForceField(INITIAL_FORCE_FIELD),
            n_replicates=N_REPLICATES,
            n_molecules=N_MOLECULES,
        )

        y_ref, y_ref_std, y_pred, y_pred_std = predict(
            dataset=thermo_data,
            force_field=current_ff,
            epoch_dir=root / f"EPOCH_{epoch}",
        )

        print(f"{y_ref=}\n{y_ref_std=}\n{y_pred=}\n{y_pred_std=}\n")
        # y_ref=tensor([0.8736, 0.8060, 1.1335, 0.8579, 0.8215, 0.8375, 0.8543, 1.0950, 1.1203,
        #         1.1290])
        # y_ref_std=tensor([4.0000e-05, 5.7000e-05, 5.0000e-04, 3.3500e-04, 1.1150e-04, 1.0100e-04,
        #         9.1000e-05, 1.5000e-04, 3.5000e-04, 1.0000e-03])
        # y_pred=tensor([0.8859, 0.7895, 1.1050, 0.8637, 0.8140, 0.8355, 0.8586, 1.0800, 1.0986,
        #         1.1028], dtype=torch.float64, grad_fn=<StackBackward0>)
        # y_pred_std=tensor([0.0037, 0.0043, 0.0068, 0.0021, 0.0024, 0.0025, 0.0065, 0.0065, 0.0092,
        #         0.0034], dtype=torch.float64, grad_fn=<StackBackward0>)

        energy_ref, energy_pred, forces_ref, forces_pred = energy.predict(
            energy_data,
            current_ff,
            topologies,
            normalize=False,
        )

        loss = (
            (energy_pred - energy_ref).square().mean()
            + 0.1 * (forces_pred - forces_ref).square().mean()
            + (y_pred - y_ref).square().mean()
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        with no_grad():
            theta.copy_(trainable.clamp(theta))

        # probably write out FF and end of each epoch?
        with open(root / "force_fields" / f"after_EPOCH_{epoch}.ff.json", "w") as f:
            dump(
                dump_tensor_force_field(
                    trainable.to_force_field(theta),
                ),
                f,
                indent=4,
            )


if __name__ == "__main__":
    main(root=Path(__file__).resolve().parent)
