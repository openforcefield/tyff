import socket

import parsl

from tyff.compute.configs import hpc3_config, local_config
from tyff.compute.workflow import SimulationWorkflow
from tyff.targets.thermo import DataEntry

# 1. Establish target data
targets = [
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
            "id": 4455979010545387927019552539812888795231031600148607549210198983695932067156,
            "tag": "enthalpy_of_vaporization",
            "smiles": ["Fc1ccccc1Cl"],
            "x": [1.0],
            "temperature": 298.15,
            "pressure": None,
            "value": 10.51625239005736,
            "std": 0.14340344168260036,
            "units": "kcal/mol",
            "source": "10.1016/j.fluid.2014.12.023",
        },
    ]
]

job_specs = list()

base_dir = "mvp_jobs"

# 2. Run (compute) jobs
if "hpc3" in socket.gethostname():
    # production runs on HPC3 (SLURM cluster with GPUs)
    with SimulationWorkflow(
        base_dir,
        hpc3_config(
            partition="gpu",
            account="dmobley_lab_gpu",
        ),
    ) as workflow:
        workflow.submit_target_batch(
            targets,
            force_field="openff-2.3.0.offxml",
            n_molecules=200,
            n_replicates=5,
        )

else:
    # local testing
    with SimulationWorkflow(
        base_dir,
        local_config(max_workers=10),
    ) as workflow:
        workflow.submit_target_batch(
            targets,
            force_field="openff-2.3.0.offxml",
            n_molecules=200,
            n_replicates=3,
        )


try:
    parsl.clear()
    parsl.dfk().cleanup()
except Exception:
    pass
