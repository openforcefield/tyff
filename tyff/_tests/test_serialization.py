import torch
import json

from tyff._serialization import dump_tensor_force_field, load_tensor_force_field
from tyff._tests.utils import system_from_smiles
from tyff._models import TensorForceField

def compare_tensor_force_fields(
        force_field1:  TensorForceField,
        force_field2: TensorForceField,
) -> bool:
    if len(force_field1.potentials) != len(force_field2.potentials):
        print("Failed: Number of potentials do not match")
        return False

    for potential1, potential2 in zip(force_field1.potentials, force_field2.potentials):
        if potential1.type != potential2.type:
            print(f"Failed: potential type mismatch ({potential1.type} != {potential2.type})")
            return False
        if potential1.fn != potential2.fn:
            print(f"Failed: potential fn mismatch ({potential1.fn} != {potential2.fn})")
            return False
        if potential1.parameter_units != potential2.parameter_units:
            print(f"Failed: potential parameter_units mismatch ({potential1.parameter_units} != {potential2.parameter_units})")
            return False
        if potential1.parameter_keys != potential2.parameter_keys:
            print(f"Failed: potential parameter_keys mismatch ({potential1.parameter_keys} != {potential2.parameter_keys})")
            return False
        if potential1.parameter_cols != potential2.parameter_cols:
            print(f"Failed: potential parameter_cols mismatch ({potential1.parameter_cols} != {potential2.parameter_cols})")
            return False
        if not torch.equal(potential1.parameters, potential2.parameters):
            print("Failed: potential parameters are not equal")
            return False

        # potential.attributes can be None
        if potential1.attributes is None:
            if potential2.attributes is not None:
                print("Failed: potential1.attributes is None but potential2.attributes is not None")
                return False
        else:
            if not torch.equal(potential1.attributes, potential2.attributes):
                print("Failed: potential1.attributes and potential2.attributes are not equal")
                return False

        if potential1.attribute_units != potential2.attribute_units:
            print(f"Failed: potential attribute_units mismatch ({potential1.attribute_units} != {potential2.attribute_units})")
            return False
        if potential1.attribute_cols != potential2.attribute_cols:
            print(f"Failed: potential attribute_cols mismatch ({potential1.attribute_cols} != {potential2.attribute_cols})")
            return False

    return True

def test_basic_serialization(default_force_field, tmp_path):

    _, tensor_force_field = system_from_smiles(
        ["O"],
        [1],
        default_force_field,
    )

    with open(tmp_path / "tensor_force_field.json", "w") as f:
        json.dump(dump_tensor_force_field(tensor_force_field), f)

    with open(tmp_path / "tensor_force_field.json") as f:
        loaded = load_tensor_force_field(json.load(f))

    assert compare_tensor_force_fields(tensor_force_field, loaded)
