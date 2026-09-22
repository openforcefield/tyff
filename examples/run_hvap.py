import parsl

from tyff.compute.configs import local_config, slurm_config
from tyff.compute.workflow import SimulationWorkflow
from tyff.datasets.thermoml import ThermoMLDataSet

with open("tyff/_tests/data/thermoml/single_dhvap.xml") as f:
    dataset = ThermoMLDataSet.from_xml(f.read())
    dhvap_target = dataset.properties[0]

job_specs = list()

base_dir = "dhvap_example"


# production on GPU cluster
if False:
    with SimulationWorkflow(base_dir, slurm_config("gpu")) as workflow:
        pass

# local testing
with SimulationWorkflow(base_dir, local_config(max_workers=10)) as workflow:
    for extra_molecules in range(2):
        workflow.submit_target(
            dhvap_target,
            force_field="openff-2.3.0.offxml",
            n_molecules=500 + extra_molecules,
            n_replicates=5,
        )

    for extra_molecules in range(2):
        workflow.estimate_target(
            dhvap_target,
            force_field="openff-2.3.0.offxml",
            n_molecules=500 + extra_molecules,
            n_replicates=5,
        )

"""
dHvap estimate for target with below ID, force field openff-2.3.0.offxml, 500 molecules, and
5 replicates:
        (target ID:
4455979010545387927019552539812888795231031600148607549210198983695932067156)
        49.108 ± 1.366 kJ/mol
dHvap estimate for target with below ID, force field openff-2.3.0.offxml, 501 molecules, and
5 replicates:
        (target ID:
4455979010545387927019552539812888795231031600148607549210198983695932067156)
        49.096 ± 1.310 kJ/mol
"""

try:
    parsl.clear()
    parsl.dfk().cleanup()
except Exception:
    pass
