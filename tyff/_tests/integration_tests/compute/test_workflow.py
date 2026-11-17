import json
from pathlib import Path

import openff.toolkit
import pytest
import torch

from tyff.compute.configs import local_config
from tyff.compute.workflow import SimulationWorkflow
from tyff.converters import topologies_for
from tyff.targets import thermo
from tyff.targets.new_thermo import predict, simulate
from tyff.targets.thermo import create_dataset

# this variable is magically detected by pytest, important that it is defined at module level
pytestmark = pytest.mark.gpu

# turn on to store simulation results in local path for debugging/iteration, otherwise (as in CI)
# let pytest start everything fresh in a temporary directory
USE_LOCAL_FILES = False


class TestEnthalpyOfVaporization:
    def test_enthalpy_of_vaporization(self, tmp_path):

        dataset = create_dataset(
            *[
                {  # type: ignore
                    "id": "methanol",
                    "tag": "enthalpy_of_vaporization",
                    "x": [1.0],
                    "smiles": ["CO"],
                    "temperature": 300.0,
                    "pressure": 101.325,
                    "value": 38.3,
                    "std": 0.01,  # arbitrary/unused
                    "units": "kJ/mol",
                    "source": "test_source",
                }
            ]
        )

        # I could be forgetting a cleaner way of going from SMIRNOFF + set[SMILES] -> TensorForceField
        _topologies, tensor_force_field = topologies_for(
            smiles_set={*thermo.extract_smiles(dataset)},
            smirnoff_force_field=openff.toolkit.ForceField("openff-2.3.0.offxml"),
        )

        simulate(
            tensor_force_field=tensor_force_field,
            epoch_dir=Path("vaporization") if USE_LOCAL_FILES else tmp_path,
            thermo_data=dataset,
            smirnoff_force_field="openff-2.3.0.offxml",
            n_replicates=2,
            n_molecules=240,
        )

        y_ref, y_ref_std, y_pred, y_pred_std = predict(
            dataset=dataset,
            tensor_force_field=tensor_force_field,
            epoch_dir=Path("vaporization") if USE_LOCAL_FILES else tmp_path,
        )

        assert not torch.isnan(torch.stack([y_ref, y_ref_std, y_pred, y_pred_std])).any()

        # experimental value is around 38 kJ/mol, could get away with much smaller range
        assert 35 < y_pred < 40, f"{y_pred=} {y_ref=}, {y_pred_std=}, {y_ref_std=}"


class TestEnthalpyOfMixing:
    def test_enthalpy_of_mixing(self, tmp_path):

        dataset = create_dataset(
            *[
                {  # type: ignore
                    "id": "foo",
                    "tag": "enthalpy_of_mixing",
                    "x": [0.8, 0.2],
                    "smiles": ["O", "CCO"],
                    "temperature": 298.15,
                    "pressure": 101.325,
                    "value": -0.765,
                    "std": 0.01,  # arbitrary/unused
                    "units": "kJ/mol",
                    "source": "test_source",
                }
            ]
        )

        _topologies, tensor_force_field = topologies_for(
            smiles_set={smiles for smiles_list in dataset["smiles"] for smiles in smiles_list},
            smirnoff_force_field=openff.toolkit.ForceField("openff-2.3.0.offxml"),
        )

        simulate(
            tensor_force_field=tensor_force_field,
            epoch_dir=Path("mixing") if USE_LOCAL_FILES else tmp_path,
            thermo_data=dataset,
            smirnoff_force_field="openff-2.3.0.offxml",
            n_replicates=3,
            n_molecules=480,
        )

        y_ref, y_ref_std, y_pred, y_pred_std = predict(
            dataset=dataset,
            tensor_force_field=tensor_force_field,
            epoch_dir=Path("mixing") if USE_LOCAL_FILES else tmp_path,
        )

        assert not torch.isnan(torch.stack([y_ref, y_ref_std, y_pred, y_pred_std])).any()

        # experimental value is around -0.7 kJ/mol, but it should be negative (for all concentrations)
        # and numerically somewhere in the few 0.1s of kJ/mol
        assert -1 < y_pred < 0, f"{y_pred=} {y_ref=}, {y_pred_std=}, {y_ref_std=}"


class TestEnthalpyOfVaporization:
    def test_nonstandard_pressure(self, tmp_path):

        dataset = create_dataset(
            *[
                {  # type: ignore
                    "id": "high_pressure_water",
                    "tag": "density",
                    "x": [1.0],
                    "smiles": ["O"],
                    "temperature": 298.15,
                    "pressure": pressure,
                    "value": density,
                    "std": 0.01,
                    "units": "g/mL",
                    "source": "test_source",
                }
                for pressure, density in zip([101.325, 101325], [1.0, 1.05])
            ]
        )

        _topologies, tensor_force_field = topologies_for(
            smiles_set={*thermo.extract_smiles(dataset)},
            smirnoff_force_field=openff.toolkit.ForceField("openff-2.3.0.offxml"),
        )

        with SimulationWorkflow(
            base_dir=str(Path("nonstandard_pressure") if USE_LOCAL_FILES else tmp_path),
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
