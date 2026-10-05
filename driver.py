import pathlib
import tyff
from typing import Iterable
import openff.toolkit
from tyff.configs.targets.thermo import DataEntry
import socket
import parsl

def get_thermo_data() -> Iterable[DataEntry]:
    """Stand-in for other ways of procuring physical property data."""
    return [
        DataEntry(**data)
        for data in [
            {
                "id": 74669191270958899039426367776551206167278900382279422138348522811937124722445,
                "tag": "density",
                "smiles": ["COCCO"],
                "x": [1.0],
                "temperature": 293.15,
                "pressure": 101.3,
                "value": 0.9648800000000002,
                "std": 5.0000000000000016e-05,
                "units": "gram / milliliter",
                "source": "",
            },
            {
                "id": "",
                "tag": "density",
                "smiles": ["CCC(=O)OC"],
                "x": [1.0],
                "temperature": 293.15,
                "pressure": 101.325,
                "value": 0.9101800000000002,
                "std": 0.00045,
                "units": "gram / milliliter",
                "source": "",
            },
            {
                "id": "",
                "tag": "density",
                "smiles": ["C=COCCC"],
                "x": [1.0],
                "temperature": 298.15,
                "pressure": 101.325,
                "value": 0.7629800000000002,
                "std": 0.00061,
                "units": "gram / milliliter",
                "source": "",
            },
        ]
    ]


def simulate():
    pass


def create_tensor_force_field(
    unique_smiles: set[str],
    force_field: openff.toolkit.ForceField,
) -> tuple[tyff.TensorForceField, list[tyff.TensorTopology]]:
    import tyff.converters

    interchanges = [
        force_field.create_interchange(openff.toolkit.Molecule.from_smiles(unique_smiles_).to_topology())
        for unique_smiles_ in unique_smiles
    ]

    # this is a "global"/reference force field composed from all unique molecules
    # and a "tensor topology" corresponding to each interchange
    return tyff.converters.convert_interchange(interchanges)


def main(initial_force_field: str, root: str, parsl_config: parsl.Config):
    import openff.toolkit
    import tyff.train
    import tyff.converters
    import torch
    from tyff.compute.workflow import SimulationWorkflow

    smirnoff0 = openff.toolkit.ForceField(initial_force_field)
    thermo_data = get_thermo_data()

    tensor_force_field, tensor_topologies = create_tensor_force_field(
        unique_smiles={smiles for target in thermo_data for smiles in target["smiles"]},
        force_field=smirnoff0
    )

    trainable = tyff.train.Trainable(
        tensor_force_field,
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
    
    optimizer = torch.optim.Adam([theta], lr=0.01)

    with SimulationWorkflow(str(pathlib.Path(root) / "simulations"), parsl_config) as workflow:
        for epoch in range(N_EPOCHS):
            current_ff = trainable.to_force_field(theta)

            current_smirnoff = tyff.converters.convert_tensor_force_field(
                originall_force_field = smirnoff0,
                tesnor_force_field=current_ff,
            )

if __name__ == "__main__":
    from tyff.compute.configs import hpc3_config, local_config

    # inelegant
    config = hpc3_config if "hpc3" in socket.gethostname() else local_config

    main("openff-2.3.0.offxml", "fit", config)
