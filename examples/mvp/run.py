import tarfile
from json import dump
from pathlib import Path

from datasets import Dataset
from openff.toolkit import ForceField
from torch import no_grad
from torch.optim import Adam

from tyff._serialization import dump_tensor_force_field
from tyff.compute._logging import _set_up_logger
from tyff.converters import convert_tensor_force_field, topologies_for
from tyff.targets import energy, thermo
from tyff.targets.new_thermo import predict, simulate
from tyff.train import ParameterConfig, Trainable

N_EPOCHS = 10
N_REPLICATES = 2
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

    # TODO: inner/outer loop, or any way of stepping through QM optimization
    #       without running MD jobs each time
    for epoch in range(N_EPOCHS):
        # maybe do this outside of loop for iteration 0, then
        # at end of loop for [1, ..., N_EPOCHS]?
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

        y_ref, _y_ref_std, y_pred, _y_pred_std = predict(
            dataset=thermo_data,
            force_field=current_ff,
            epoch_dir=root / f"EPOCH_{epoch}",
        )

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

        logger.info(f"EPOCH {epoch}: loss={loss.item():.6f}")

        optimizer.zero_grad()

        loss.backward()
        # ipdb> loss.backward()
        # *** RuntimeError: Function RemoteEnsembleAverageBackward returned an invalid gradient at
        # index 2 - got [208] but expected shape compatible with [287]

        optimizer.step()

        with no_grad():
            theta.copy_(trainable.clamp(theta))

        updated_tensor_force_field = trainable.to_force_field(theta)

        # write out FF as tensor
        with open(root / "force_fields" / f"after_EPOCH_{epoch}.ff.json", "w") as f:
            dump(
                dump_tensor_force_field(updated_tensor_force_field),
                f,
                indent=4,
            )

        # write out FF as SMIRNOFF force field
        convert_tensor_force_field(
            original_force_field=ForceField(INITIAL_FORCE_FIELD),
            tensor_force_field=updated_tensor_force_field,
        ).to_file(str(root / "force_fields" / f"after_EPOCH_{epoch}.offxml"))


if __name__ == "__main__":
    main(root=Path(__file__).resolve().parent)
