import json
import pathlib
import random

from tyff.compute._equilibrate import _run_equilibration
from tyff.compute._minimize import _minimize_energy
from tyff.compute._pack import _prepare_packed_topology
from tyff.compute._prepare import _prepare_openmm_system
from tyff.compute._produce import _run_production
from tyff.compute.prep import _make_liquid_density_compute_configs
from tyff.configs.targets.thermo import DataEntry

pathlib.Path("sample_density/").mkdir(exist_ok=True)

target = DataEntry(
    **{
        "id": random.randint(10**15, 10**16 - 1),
        "tag": "density",
        "x": [0.5, 0.5],
        "smiles": [
            "[O:1]([H:2])[H:3]",
            "[C:1]([C:2]([O:3][H:9])([H:7])[H:8])([H:4])([H:5])[H:6]",
        ],
        "temperature": 298.15,
        "pressure": 1.01325,
        "value": 0.922,
        "std": 0.0001,
        "units": "g/mL",
        "source": "",
    }
)

with open("sample_density/target_config.json", "w") as target_config:
    json.dump(target, target_config, indent=4)

compute = _make_liquid_density_compute_configs(
    data_entry=target,
    force_field="openff-2.3.0.offxml",
    n_molecules=200,
)[0][0]

with open("sample_density/compute_config.json", "w") as compute_config:
    json.dump(compute, compute_config, indent=4)

packing_result = _prepare_packed_topology(
    job_dir="sample_density/",
)

prepare_result = _prepare_openmm_system(
    packing_future=packing_result,
    job_dir="sample_density",
)

minimize_result = _minimize_energy(
    system_future=prepare_result,
    job_dir="sample_density",
)

equilibration_result = _run_equilibration(
    equilibration_config=None,
    minimization_future=minimize_result,
    job_dir="sample_density",
)

production_result = _run_production(
    production_config=None,
    equilibration_future=equilibration_result,
    job_dir="sample_density",
)
