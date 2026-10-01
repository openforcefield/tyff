import json

import pytest
import torch

from tyff._models import TensorForceField
from tyff._serialization import dump_tensor_force_field, load_tensor_force_field
from tyff._tests.utils import system_from_smiles


def compare_tensor_force_fields(
    force_field1: TensorForceField,
    force_field2: TensorForceField,
) -> bool:
    if len(force_field1.potentials) != len(force_field2.potentials):
        pytest.fail("Failed: Number of potentials do not match")

    for potential1, potential2 in zip(force_field1.potentials, force_field2.potentials):
        if potential1.type != potential2.type:
            pytest.fail(f"Failed: potential type mismatch ({potential1.type} != {potential2.type})")
        if potential1.fn != potential2.fn:
            pytest.fail(f"Failed: potential fn mismatch ({potential1.fn} != {potential2.fn})")
        if potential1.parameter_units != potential2.parameter_units:
            pytest.fail(
                f"Failed: potential parameter_units mismatch "
                f"({potential1.parameter_units} != {potential2.parameter_units})"
            )
        if potential1.parameter_keys != potential2.parameter_keys:
            pytest.fail(
                f"Failed: potential parameter_keys mismatch "
                f"({potential1.parameter_keys} != {potential2.parameter_keys})"
            )
        if potential1.parameter_cols != potential2.parameter_cols:
            pytest.fail(
                f"Failed: potential parameter_cols mismatch "
                f"({potential1.parameter_cols} != {potential2.parameter_cols})"
            )
        if not torch.equal(potential1.parameters, potential2.parameters):
            pytest.fail("Failed: potential parameters are not equal")

        # potential.attributes can be None
        if potential1.attributes is None:
            if potential2.attributes is not None:
                pytest.fail("Failed: potential1.attributes is None but potential2.attributes is not None")
        else:
            if not torch.equal(potential1.attributes, potential2.attributes):
                pytest.fail("Failed: potential1.attributes and potential2.attributes are not equal")

        if potential1.attribute_units != potential2.attribute_units:
            pytest.fail(
                f"Failed: potential attribute_units mismatch "
                f"({potential1.attribute_units} != {potential2.attribute_units})"
            )
        if potential1.attribute_cols != potential2.attribute_cols:
            pytest.fail(
                f"Failed: potential attribute_cols mismatch "
                f"({potential1.attribute_cols} != {potential2.attribute_cols})"
            )

    if force_field1.v_sites is None:
        assert force_field2.v_sites is None

        return True
    else:
        v_sites1 = force_field1.v_sites
        v_sites2 = force_field2.v_sites

        if v_sites1.keys != v_sites2.keys:
            pytest.fail("Failed: v_site keys mismatch")

        if not torch.equal(v_sites1.parameters, v_sites2.parameters):
            pytest.fail("Failed: v_site parameters are not equal")

        for w1, w2 in zip(v_sites1.weights, v_sites2.weights):
            if not torch.equal(w1, w2):
                pytest.fail("Failed: v_site weights mismatch")

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


def test_virtual_sites(v_site_force_field, tmp_path):
    _, tensor_force_field = system_from_smiles(
        ["O"],
        [1],
        v_site_force_field,
    )

    with open(tmp_path / "tensor_force_field.json", "w") as f:
        json.dump(dump_tensor_force_field(tensor_force_field), f)

    with open(tmp_path / "tensor_force_field.json") as f:
        loaded = load_tensor_force_field(json.load(f))

    assert compare_tensor_force_fields(tensor_force_field, loaded)
