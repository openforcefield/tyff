import platform

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.monitoring.monitoring import MonitoringHub
from parsl.providers import LocalProvider, SlurmProvider


def local_config(max_workers: int = 10):
    """For development and testing - CPU, but force OpenCL on macOS."""
    worker_init = "export OMP_NUM_THREADS=1; "

    if platform.system() == "Darwin":
        worker_init += "export OPENMM_DEFAULT_PLATFORM=OpenCL; "

    return Config(
        executors=[
            HighThroughputExecutor(
                label="local_process_pool",
                provider=LocalProvider(
                    init_blocks=1,
                    min_blocks=1,
                    max_blocks=1,
                    worker_init=worker_init,
                ),
                max_workers_per_node=max_workers,
                cores_per_worker=1,
            )
        ],
        strategy="htex_auto_scale",
        monitoring=MonitoringHub(
            resource_monitoring_interval=5,  # seconds; 0 disables per-task resource sampling
        ),
    )


def hpc3_config(
    partition: str = "free-gpu",
    account: str = "dmobley_omsf_gpu32",
    max_blocks: int = 100,
):
    """
    For use on HPC3 (UCI) only

    Some brittle details include
      - The `hpc3` Pixi environment is used
      - A30 GPUs are forced
      - Walltime is capped at 20 minutes (flexible)
    """
    worker_init = 'eval "$(pixi shell-hook -e hpc3)"'

    return Config(
        executors=[
            HighThroughputExecutor(
                label="gpu_htex",
                max_workers_per_node=1,  # one **worker** per block
                # HPC3 has >1 GPUs/node, but with `exclusive=False` we never request entire nodes at once
                available_accelerators=1,
                cpu_affinity="block",
                worker_debug=True,
                provider=SlurmProvider(
                    partition=partition,
                    account=account,
                    scheduler_options="#SBATCH --gres=gpu:A30:1",
                    exclusive=False,  # do not request entire node per job
                    cmd_timeout=120,  # time to wait for SLURM commands to finish, default is 10 seconds
                    nodes_per_block=1,
                    init_blocks=1,
                    max_blocks=max_blocks,
                    walltime="00:20:00",
                    worker_init=worker_init,
                ),
            )
        ],
        # "htex_auto_scale" may be better
        strategy="simple",
        retries=2,
    )
