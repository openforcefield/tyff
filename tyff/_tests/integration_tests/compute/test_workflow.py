from pathlib import Path

import openff.toolkit
import pytest
import torch

from tyff.converters.openff import topologies_for
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
                    "value": 1.0,
                    "std": 0.01,
                    "units": "kJ/mol",
                    "source": "test_source",
                }
            ]
        )

        # I could be forgetting a cleaner way of going from SMIRNOFF + set[SMILES] -> TensorForceField
        _topologies, tensor_force_field = topologies_for(
            smiles_set={smiles for smiles_list in dataset["smiles"] for smiles in smiles_list},
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

        # experimental value is around 36 kJ/mol, could get away with much smaller range
        assert 30 < y_pred < 50

class TestEnthalpyOfMixing:
    def test_enthalpy_of_mixing(self, tmp_path):

        dataset = create_dataset(
            *[
                {  # type: ignore
                    "id": "foo",
                    "tag": "enthalpy_of_mixing",
                    "x": [0.8, 0.2],
                    "smiles": ["O", "CCO"],
                    "temperature": 300.0,
                    "pressure": 101.325,
                    "value": -5.0,
                    "std": 1.0,
                    "units": "kJ/mol",
                    "source": "test_source",
                }
            ]
        )

        # I could be forgetting a cleaner way of going from SMIRNOFF + set[SMILES] -> TensorForceField
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

        assert -10 < y_pred < 0
