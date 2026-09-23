from __future__ import annotations

import json

from parsl import File

from tyff.compute._files import PackingFiles


def _prepare_packed_topology(
    job_dir: str,
) -> dict[str, PackingFiles]:
    import pathlib
    import time

    from openff.packmol import pack_box
    from openff.toolkit import Molecule, Quantity

    from tyff.compute._logging import _set_up_logger

    logger = _set_up_logger(f"{job_dir}/pack.log")

    logger.info("packing topology")

    compute_config = json.load(open(f"{job_dir}/compute_config.json"))

    files = PackingFiles(
        packed_topology=File(f"{job_dir}/packed_topology.pdb"),
    )

    if pathlib.Path(files["packed_topology"].filepath).exists():
        logger.info(f"File {files['packed_topology'].filepath} already exists, skipping packing.")

        return {
            "packed_files": files,
        }

    n_molecules = compute_config["n_molecules"]

    time.sleep(n_molecules * 0.001)
    packed_topology_file = f"{job_dir}/packed_topology.pdb"

    molecules = [Molecule.from_smiles(smiles) for smiles in compute_config["smiles"]]

    n_copies = [int(n_molecules * x) for x in compute_config["x"]]

    density = compute_config.get("density")

    if density is None:
        if compute_config["tag"] == "liquid":
            density = 1.0
        elif compute_config["tag"] == "gas":
            # probably cleaner to set a box instead of arbitrarily low density
            density = 0.001
        else:
            raise ValueError(f"Unknown compute config tag {compute_config['tag']}")

    result = pack_box(
        molecules,
        n_copies,
        target_density=Quantity(density * 0.7, "g/mL"),
        working_directory=job_dir,
    )

    result.to_file(packed_topology_file, file_format="pdb")

    logger.info(f"packed {result.n_molecules} molecules")

    return {
        "packed_files": files,
    }
