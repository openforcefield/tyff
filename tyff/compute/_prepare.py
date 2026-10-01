from __future__ import annotations

import json

import openmm
from openff.toolkit import Molecule, Topology
from parsl import File

from tyff.compute._files import PackingFiles, PreparingFiles
from tyff.configs.liquid import BulkLiquid


def _prepare_openmm_system(
    packing_future: dict[str, PackingFiles],
    job_dir: str,
) -> dict[str, PreparingFiles]:
    import pathlib

    from openff.toolkit import ForceField

    from tyff.compute._logging import _set_up_logger

    logger = _set_up_logger(f"{job_dir}/prepare.log")

    logger.info("Starting OpenMM system creation app")

    files = PreparingFiles(
        openmm_system=File(f"{job_dir}/openmm_system.xml"),
        interchange=File(f"{job_dir}/interchange.json"),
        packed_topology=File(f"{job_dir}/packed_topology.pdb"),
    )

    if pathlib.Path(files["openmm_system"].filepath).exists() and pathlib.Path(files["interchange"].filepath).exists():
        logger.info(f"File {files['openmm_system'].filepath} already exists, skipping system prep.")
        return {
            "prepared_files": files,
        }

    compute_config: BulkLiquid = BulkLiquid(**json.load(open(f"{job_dir}/compute_config.json")))  # type: ignore[typeddict-item]
    packing_files: PackingFiles = packing_future["packed_files"]

    packed_topology: Topology = Topology.from_pdb(
        file_path=packing_files["packed_topology"].filepath,
        unique_molecules=[Molecule.from_smiles(smiles) for smiles in compute_config["smiles"]],
    )

    force_field = ForceField(compute_config["force_field"])

    interchange = force_field.create_interchange(packed_topology)

    # later analysis depends on making `TensorSystem`s from `TensoTopology`s which may prefer
    # to be constructed from single-molecule `Interchange`s, so just save them out here
    #
    # ordering is quite fragile, but hope order of smiles list never changes mid-job
    for unique_molecule_index, unique_molecule in enumerate(
        [Molecule.from_smiles(smiles) for smiles in compute_config["smiles"]]
    ):
        single_molecule_interchange = force_field.create_interchange(unique_molecule.to_topology())
        with open(f"{job_dir}/single_molecule_interchange_{unique_molecule_index}.json", "w") as f:
            f.write(single_molecule_interchange.model_dump_json())

    with open(files["interchange"].filepath, "w") as f:
        f.write(interchange.model_dump_json())

    openmm_system = interchange.to_openmm()

    with open(files["openmm_system"].filepath, "w") as f:
        f.write(openmm.XmlSerializer.serialize(openmm_system))

    logger.info("made openmm system!")

    return {
        "prepared_files": files,
    }
