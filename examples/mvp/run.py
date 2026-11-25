import socket
import tarfile
from json import dump
from pathlib import Path

from datasets import Dataset
from openff.toolkit import ForceField
from torch import no_grad
from torch.optim import Adam
from tyff.targets.new_thermo import predict, simulate

from tyff._serialization import dump_tensor_force_field
from tyff.compute._logging import _set_up_logger
from tyff.compute.configs import hpc3_config, local_config
from tyff.converters import convert_tensor_force_field, topologies_for
from tyff.targets import energy, thermo
from tyff.train import ParameterConfig, Trainable

N_EPOCHS = 10
N_REPLICATES = 3
N_MOLECULES = 480
INITIAL_FORCE_FIELD = "openff-2.3.0.offxml"

# these weights should be exposed to the user in some sort of training config
# probably a good phase II task
COEFFICIENTS = {
    "energy": 0.1,
    "force": 0.01,
    "density": 1_000.0,
    "enthalpy_of_mixing": 1.0,
    "enthalpy_of_vaporization": 1.0,
}


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


def main(root: Path) -> None:
    global logger

    logger = _set_up_logger(f"{root}/mvp.log")

    (root / "force_fields").mkdir(exist_ok=True)

    thermo_data, energy_data = load_datasets(root)

    # tensor_ff is a subset of all parameters, capturing the parameters
    # used in the datasets
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

    # TODO: Possibly drop in L-M optimizer
    optimizer = Adam([theta], lr=0.01)

    # TODO: inner/outer loop, or any way of stepping through QM optimization
    #       without running MD jobs each time
    for epoch in range(N_EPOCHS):
        # this is linked under the hood to theta - needed to hook gradients together
        current_ff = trainable.to_force_field(theta)

        if "hpc3" in socket.gethostname():
            parsl_config = hpc3_config(partition="gpu", account="dmobley_lab_gpu")
        else:
            parsl_config = local_config(max_workers=6)

        simulate(
            parsl_config=parsl_config,
            tensor_force_field=current_ff,
            epoch_dir=root / f"EPOCH_{epoch}",
            thermo_data=thermo_data,
            smirnoff_force_field=INITIAL_FORCE_FIELD,
            n_replicates=N_REPLICATES,
            n_molecules=N_MOLECULES,
        )

        thermo_results = dict()

        for tag in {*thermo_data["tag"]}:
            thermo_subset = thermo_data.filter(lambda x: x["tag"] == tag)

            y_ref, y_ref_std, y_pred, y_pred_std = predict(
                dataset=thermo_subset,
                tensor_force_field=current_ff,
                epoch_dir=root / f"EPOCH_{epoch}",
            )

            thermo_results[tag] = {
                "y_ref": y_ref,
                "y_ref_std": y_ref_std,
                "y_pred": y_pred,
                "y_pred_std": y_pred_std,
            }

        energy_ref, energy_pred, forces_ref, forces_pred = energy.predict(
            energy_data,
            current_ff,
            topologies,
            normalize=False,
        )

        print(f"EPOCH {epoch}:")

        energy_loss = COEFFICIENTS["energy"] * (energy_pred - energy_ref).square().mean()
        force_loss = COEFFICIENTS["force"] * (forces_pred - forces_ref).square().mean()

        print(
            f"\tenergy component (coefficient {COEFFICIENTS['energy']}) has loss contribution {energy_loss.item():.6f}"
        )

        print(f"\tforce component (coefficient {COEFFICIENTS['force']}) has loss contribution {force_loss.item():.6f}")

        loss = energy_loss + force_loss

        for tag, results in thermo_results.items():
            this_loss = COEFFICIENTS[tag] * (results["y_pred"] - results["y_ref"]).square().mean()
            print(
                f"\tthermo component {tag=} (coefficient {COEFFICIENTS[tag]}) "
                f"has loss contribution {this_loss.item():.6f}"
            )
            loss += this_loss

        logger.info(f"EPOCH {epoch}: loss={loss.item():.6f}")

        print(f"EPOCH {epoch}: total loss={loss.item():.6f}")

        optimizer.zero_grad()

        loss.backward()

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

        # update SMIRNOFF force field with new values from tensor FF
        convert_tensor_force_field(
            original_force_field=ForceField(INITIAL_FORCE_FIELD),
            tensor_force_field=updated_tensor_force_field,
        ).to_file(str(root / "force_fields" / f"after_EPOCH_{epoch}.offxml"))


if __name__ == "__main__":
    main(root=Path(__file__).resolve().parent)
