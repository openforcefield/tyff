import hashlib
import json
import os

from tyff.configs._compute import BaseComputeConfig


# TODO: LiquidConfig should have a base class?
#       one for i.e. GasConfig to derive from
def make_job_id(compute_config: BaseComputeConfig) -> str:
    """
    Generate a deterministic job ID from compute parameters.

    Maybe this could be simplified if it only has one argument?
    """
    # TODO: Since this is based on a TypedDict, there is basically no validation at
    #       object creation time. For example, no check that the total mole fraction
    #       is very close to 1. Here might be a good place to do this?
    # TODO: Improve how this handles VacuumGas

    # sort smiles and x according to x, descending
    sorted_x, sorted_smiles = zip(*sorted(zip(compute_config["x"], compute_config["smiles"]), reverse=True))

    params = {
        "tag": compute_config["tag"],  # type: ignore[typeddict-item]
        "force_field": compute_config["force_field"],
        "n_molecules": compute_config["n_molecules"],
        "replicate_index": compute_config["replicate_index"],
        "smiles": sorted_smiles,
        "x": sorted_x,
        "temperature": compute_config["temperature"],
        "pressure": compute_config.get("pressure"),
        "density": compute_config.get("density"),
    }

    # some of attributes - maybe the target value? - could possibly be
    # excluded from hashing criteria here
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def get_job_paths(base_dir, job_id):
    return {
        "root": f"{base_dir}/{job_id}",
        "checkpoint": f"{base_dir}/{job_id}/checkpoint.chk",
        "trajectory": f"{base_dir}/{job_id}/trajectory.dcd",
        "log": f"{base_dir}/{job_id}/simulation.log",
    }


def is_complete(base_dir, job_id):
    """Check if a job already has complete outputs."""
    return os.path.exists(get_job_paths(base_dir, job_id)["trajectory"])
