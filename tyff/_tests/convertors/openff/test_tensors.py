"""Test conversion of tensor force fields back to SMIRNOFF force fields."""

import random

import pytest
from openff.toolkit import ForceField, Molecule
from openff.toolkit.typing.engines.smirnoff.parameters import VirtualSiteType

import tyff.converters


@pytest.fixture
def sage():
    return ForceField("openff-2.3.0.offxml")


@pytest.fixture
def sage_with_bond_charge():
    sage = ForceField("openff-2.3.0.offxml")
    sage.get_parameter_handler("VirtualSites")
    sage["VirtualSites"].add_parameter(
        parameter=VirtualSiteType(
            smirks="[#6:2]-[#17X1:1]",
            type="BondCharge",
            match="all_permutations",
            distance="0.8 * angstrom ** 1",
            charge_increment1="0.123 * elementary_charge ** 1",
            charge_increment2="0.0 * elementary_charge ** 1",
        ),
    )

    return sage


@pytest.fixture
def phenol():
    return Molecule.from_smiles("c1ccccc1O")


@pytest.fixture
def methyl_chloride():
    return Molecule.from_smiles("CCl")


def test_convert_no_modifications(phenol, sage):
    """
    Test basic behavior of convert_tensor_force_field with no modifications of inputs.
    """
    interchange = sage.create_interchange(phenol.to_topology())

    tensor_force_field, _ = tyff.converters.convert_interchange(interchange)

    new_force_field = tyff.converters.convert_tensor_force_field(
        sage,
        tensor_force_field,
    )

    assert hash(new_force_field) == hash(sage)


@pytest.mark.parametrize(
    "handler_to_perturb,column_index",
    [
        ("Bonds", 0),  # k, length
        ("Angles", 0),  # k, angle
        ("ProperTorsions", 0),  # k, etc.
        ("ImproperTorsions", 0),  # k, etc.
    ],
)
def test_convert_after_perturbation(phenol, sage, handler_to_perturb, column_index):
    """
    Test that a tensor force field, randomly perturbed from the original SMIRNOFF source parameters, can be
    converted back into a SMIRNOFF force field.
    """
    factor = random.random()

    interchange = sage.create_interchange(phenol.to_topology())

    tensor_force_field, _ = tyff.converters.convert_interchange(interchange)

    # apply random perturbation to one element in on parameter tensor
    tensor_force_field.potentials_by_type[handler_to_perturb].parameters[:, column_index] *= factor

    new_force_field = tyff.converters.convert_tensor_force_field(
        sage,
        tensor_force_field,
    )

    assert hash(new_force_field) != hash(sage)

    modified_keys = [key.id for key in tensor_force_field.potentials_by_type[handler_to_perturb].parameter_keys]

    if handler_to_perturb in ("ProperTorsions", "ImproperTorsions"):
        # parameter k values are list, so need to gather differently
        found_factors = [
            (a / b).m_as("dimensionless")
            for a, b in zip(
                new_force_field[handler_to_perturb][modified_keys[-1]].k, sage[handler_to_perturb][modified_keys[-1]].k
            )
        ]

    else:
        found_factors = [
            (
                new_force_field[handler_to_perturb][modified_keys[-1]].k
                / sage[handler_to_perturb][modified_keys[-1]].k
            ).m_as("dimensionless")
        ]

    for value in found_factors:
        assert value == pytest.approx(factor), (
            f"Expected k to be scaled by {factor}, but found to be scaled by {value}"
        )


@pytest.mark.skip(reason="not yet implemented")
def test_convert_vsites(methyl_chloride, sage_with_bond_charge):
    """
    Test that a tensor force field, modified from original SMIRNOFF source parameters, can be
    converted back into a SMIRNOFF force field.
    """

    interchange = sage_with_bond_charge.create_interchange(methyl_chloride.to_topology())

    tensor_force_field, _ = tyff.converters.convert_interchange(interchange)

    # modify tensor force field

    new_force_field = tyff.converters.convert_tensor_force_field(
        sage_with_bond_charge,
        tensor_force_field,
    )

    # need to assert this is NOT the case
    assert hash(new_force_field) == hash(sage_with_bond_charge)
