import pathlib

from tyff.compute.jobs import make_job_id
from tyff.compute.prep import (
    _compute_configs_from_data_entry,
)
from tyff.configs.targets.thermo import DataEntry


def fetch_trajectory_paths_from_target(
    base_dir: str,
    target: DataEntry,
    force_field: str,
    n_molecules: int,
    n_replicates: int,
) -> tuple[str, ...]:
    compute_configs = _compute_configs_from_data_entry(
        target,
        force_field,
        n_molecules,
        n_replicates,
    )

    # flatten list[tuple[BulkLiquid | VacuumGas, ...]] into list[BulkLiquid | VacuumGas]
    job_ids = [make_job_id(compute_config) for tuple_ in compute_configs for compute_config in tuple_]

    # TODO: These might not always be named "production_trajectory.dcd"
    trajectories = [pathlib.Path(base_dir) / job_id / "production_trajectory.dcd" for job_id in job_ids]

    for trajectory in trajectories:
        if not trajectory.exists():
            raise FileNotFoundError(f"Trajectory {trajectory} does not exist")

    return tuple([trajectory.as_posix() for trajectory in trajectories])
