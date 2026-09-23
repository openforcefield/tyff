from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider, SlurmProvider
from parsl.utils import get_all_checkpoints


def local_config(max_workers: int = 10):
    """For development and testing."""
    if False:
        # If openff-packmol/temporary_cd was threadsafe, better to use
        # ThreadPoolExecutor and local config
        return Config(
            executors=[ThreadPoolExecutor(label="local")],
            checkpoint_mode="task_exit",
            checkpoint_files=get_all_checkpoints(),
        )

    return Config(
        executors=[
            HighThroughputExecutor(
                label="local_process_pool",
                provider=LocalProvider(
                    init_blocks=1,
                    min_blocks=1,
                    max_blocks=1,
                ),
                max_workers_per_node=max_workers,
                cores_per_worker=1,
            )
        ],
        strategy=None,  # Disable dynamic scaling for simpler local execution
    )


def hpc3_config(
    partition: str = "free-gpu",
    account: str = "dmobley_omsf_gpu32",
    max_blocks=100,
):
    """For production HPC runs."""
    return Config(
        executors=[
            HighThroughputExecutor(
                label="gpu_htex",
                # HPC3 has mostly 4 GPUs/node
                available_accelerators=4,
                cpu_affinity="alternating",
                worker_debug=True,
                provider=SlurmProvider(
                    partition=partition,
                    account=account,
                    scheduler_options="#SBATCH --gres=gpu:1",
                    nodes_per_block=1,
                    init_blocks=1,
                    max_blocks=4,
                    walltime="00:20:00",
                ),
            )
        ],
        # "htex_auto_scale" may be better
        strategy="simple",
        retries=2,
    )
