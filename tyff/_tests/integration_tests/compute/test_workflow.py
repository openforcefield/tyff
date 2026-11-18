import json

import openff.toolkit
import pytest

from tyff.compute.configs import local_config
from tyff.compute.workflow import SimulationWorkflow
from tyff.converters import topologies_for
from tyff.targets import thermo
from tyff.targets.thermo import create_dataset


@pytest.mark.gpu
def test_nonstandard_pressure(tmp_path):

    dataset = create_dataset(
        *[
            {  # type: ignore
                "id": "high_pressure_water",
                "tag": "density",
                "x": [1.0],
                "smiles": ["O"],
                "temperature": 298.15,
                "pressure": val,
                "value": 1.0,
                "std": 0.01,
                "units": "g/mL",
                "source": "test_source",
            }
            for val in [101.325, 101325]
        ]
    )

    _topologies, tensor_force_field = topologies_for(
        smiles_set={*thermo.extract_smiles(dataset)},
        smirnoff_force_field=openff.toolkit.ForceField("openff-2.3.0.offxml"),
    )

    with SimulationWorkflow(
        base_dir=tmp_path,
        parsl_config=local_config(),
    ) as workflow:
        workflow.submit_dataset(
            dataset,
            tensor_force_field=tensor_force_field,
            force_field="openff-2.3.0.offxml",
            n_molecules=480,
            n_replicates=1,
        )

        workflow.wait()

        # TODO: Replace this with tyff.targets.new_thermo.predict, which
        #       itself should probably be moved to SimulationWorkflow.estimate_dataset
        job_dirs = [
            path
            for path in (tmp_path).glob("*/")
            if not path.match("thermo_targets")
            and not path.match("reference_force_fields")
            and not path.match("interchanges")
        ]

        results = dict()

        for job_dir in job_dirs:
            with open(job_dir / "compute_config.json") as f:
                compute_config = json.load(f)
                target_pressure = compute_config["pressure"]

                assert compute_config["temperature"] == 298.15

            with open(job_dir / "ensemble_averages.json") as f:
                density = json.load(f)["density"]

            results[target_pressure] = density

        assert results[101325.0] > results[101.325]
