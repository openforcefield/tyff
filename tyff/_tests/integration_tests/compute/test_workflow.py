import openff.toolkit
import pytest

from tyff.converters.openff import topologies_for
from tyff.targets.new_thermo import predict, simulate
from tyff.targets.thermo import create_dataset

# this variable is magically detected by pytest, important that it is defined at module level
pytestmark = pytest.mark.gpu


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

        # I could be forgetting a cleaner way of going from SMIRNOFF + set[SMILES] -> TensorForceField,
        # but this is clean enough
        _topologies, tensor_force_field = topologies_for(
            smiles_set={smiles for smiles_list in dataset["smiles"] for smiles in smiles_list},
            smirnoff_force_field=openff.toolkit.ForceField("openff-2.3.0.offxml"),
        )

        simulate(
            tensor_force_field=tensor_force_field,
            epoch_dir=tmp_path,
            thermo_data=dataset,
            smirnoff_force_field="openff-2.3.0.offxml",
            n_replicates=3,
            n_molecules=360,
        )

        _y_ref, _y_ref_std, _y_pred, _y_pred_std = predict(
            dataset=dataset,
            tensor_force_field=tensor_force_field,
            epoch_dir=tmp_path,
        )

        # ipdb> _y_ref, _y_ref_std, _y_pred, _y_pred_std
        # (tensor([1.]), tensor([0.0100]), tensor([-67.6932],
        #   dtype=torch.float64), tensor([0.8197], dtype=torch.float64)
        # )
