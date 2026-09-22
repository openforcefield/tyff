import pathlib

import mdtraj
import parsl
from rich import print

from tyff.compute.configs import local_config, slurm_config
from tyff.compute.fetch import fetch_trajectory_paths_from_target
from tyff.compute.workflow import SimulationWorkflow
from tyff.datasets.thermoml import ThermoMLDataSet

with open("tyff/_tests/data/thermoml/single_density.xml") as f:
    dataset = ThermoMLDataSet.from_xml(f.read())
    density_target = dataset.properties[0]

job_specs = list()

base_dir = "density_example"


# production on GPU cluster
if False:
    with SimulationWorkflow(base_dir, slurm_config("gpu")) as workflow:
        pass

# local testing
with SimulationWorkflow(base_dir, local_config(max_workers=10)) as workflow:
    for extra_molecules in range(2):
        workflow.submit_target(
            density_target,
            force_field="openff-2.3.0.offxml",
            n_molecules=200 + extra_molecules,
            n_replicates=5,
        )

    for extra_molecules in range(2):
        workflow.estimate_target(
            density_target,
            force_field="openff-2.3.0.offxml",
            n_molecules=200 + extra_molecules,
            n_replicates=5,
        )

# TODO: Show how to check status while running

# get trajectory paths from root job directory and per-target info (without knowing internal compute configs)
trajectory_paths = [
    fetch_trajectory_paths_from_target(
        base_dir=base_dir,
        target=density_target,
        force_field="openff-2.3.0.offxml",
        n_molecules=200 + extra_molecules,
        n_replicates=5,
    )
    for extra_molecules in range(2)
]
print(trajectory_paths)
"""
[
│   (
│   │   'jobs/5090adac8840d612600a2542c0236d47c2de3cdef6d6517961aa68e84f7ab8da/production_trajectory.dcd',
│   │   'jobs/2ed40b18f3773e1a0eb756147e2b56e3da4c47149cdbf97c7bf6b85107df5c1c/production_trajectory.dcd',
│   │   'jobs/86cf94ed4a70093ce9d6d1f480eebaf806c457ca1c7347e5b2f153e59f12d59e/production_trajectory.dcd',
│   │   'jobs/5c6d9889a76d59e8c2cb2b66380b1f11d3fc559ba9d6c53dc843256b17482672/production_trajectory.dcd',
│   │   'jobs/fd86a3fd51a10c6c6310966bb35320ccd34bacc2e756e0ebf7cc7dd41096a8b0/production_trajectory.dcd'
│   ),
│   (
│   │   'jobs/51db1e477d95fd599577da7730d07f65a7e87a571a87c7aaff1b661cd1ba0ae8/production_trajectory.dcd',
│   │   'jobs/0f7cd30b57b03364b78801ae0350b92026725ebd9330f0c9fd88c2427c563a8e/production_trajectory.dcd',
│   │   'jobs/85a9d90d6efb9ac079a58c3f237cf19fd17328d7d1b46b5c586b523da0ce0128/production_trajectory.dcd',
│   │   'jobs/4cf3423f011e107e7c7a03516b673b2a0d6966e6dd9eece39f431482930dbe93/production_trajectory.dcd',
│   │   'jobs/3d836dad5e343ccdfe68fa5efa9928ac7faf5494714b796987fed9bd1fc913b7/production_trajectory.dcd'
│   )
]
"""

for target_paths in trajectory_paths:
    for target_path in target_paths:
        job_id = pathlib.Path(target_path).parts[-2]
        density_all_frames = mdtraj.density(
            mdtraj.load(
                pathlib.Path(base_dir) / job_id / "production_trajectory.dcd",
                top=pathlib.Path(base_dir) / job_id / "production_topology.pdb",
            )
        )

        print(
            f"Density estimate for job {job_id}: "
            f"{density_all_frames.mean() * 0.001:.3f} ± {density_all_frames.std() * 0.001:.3f} g/mL"
        )
"""
'Density estimate for job 5090adac8840d612600a2542c0236d47c2de3cdef6d6517961aa68e84f7ab8da:959.872 ± 11.103 kg/m^3'
'Density estimate for job 2ed40b18f3773e1a0eb756147e2b56e3da4c47149cdbf97c7bf6b85107df5c1c:966.281 ± 10.126 kg/m^3'
'Density estimate for job 86cf94ed4a70093ce9d6d1f480eebaf806c457ca1c7347e5b2f153e59f12d59e:963.841 ± 10.905 kg/m^3'
'Density estimate for job 5c6d9889a76d59e8c2cb2b66380b1f11d3fc559ba9d6c53dc843256b17482672:961.672 ± 8.655 kg/m^3'
'Density estimate for job fd86a3fd51a10c6c6310966bb35320ccd34bacc2e756e0ebf7cc7dd41096a8b0:962.670 ± 10.539 kg/m^3'
'Density estimate for job 51db1e477d95fd599577da7730d07f65a7e87a571a87c7aaff1b661cd1ba0ae8:962.756 ± 10.437 kg/m^3'
'Density estimate for job 0f7cd30b57b03364b78801ae0350b92026725ebd9330f0c9fd88c2427c563a8e:959.012 ± 10.070 kg/m^3'
'Density estimate for job 85a9d90d6efb9ac079a58c3f237cf19fd17328d7d1b46b5c586b523da0ce0128:965.191 ± 9.957 kg/m^3'
'Density estimate for job 4cf3423f011e107e7c7a03516b673b2a0d6966e6dd9eece39f431482930dbe93:962.832 ± 10.417 kg/m^3'
'Density estimate for job 3d836dad5e343ccdfe68fa5efa9928ac7faf5494714b796987fed9bd1fc913b7:963.682 ± 11.553 kg/m^3'
"""

try:
    parsl.clear()
    parsl.dfk().cleanup()
except Exception:
    pass
