"""Test conversion of tensor force fields back to SMIRNOFF force fields."""

import pytest
import torch
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
def ethanol():
    return Molecule.from_smiles("CCO")


@pytest.fixture
def methyl_chloride():
    return Molecule.from_smiles("CCl")


def test_convert_no_modifications(ethanol, sage):
    """
    Test basic behavior of convert_tensor_force_field with no modifications of inputs.
    """
    interchange = sage.create_interchange(ethanol.to_topology())

    tensor_force_field, _ = tyff.converters.convert_interchange(interchange)

    new_force_field = tyff.converters.convert_tensor_force_field(
        sage,
        tensor_force_field,
    )

    assert hash(new_force_field) == hash(sage)


@pytest.mark.parametrize("handler_to_perturb", ["Bonds", "Angles", "ProperTorsions"])
def test_convert_after_perturbation(ethanol, sage, handler_to_perturb):
    """
    Test that a tensor force field, randomly perturbed from the original SMIRNOFF source parameters, can be
    converted back into a SMIRNOFF force field.
    """

    interchange = sage.create_interchange(ethanol.to_topology())

    tensor_force_field, _ = tyff.converters.convert_interchange(interchange)

    # apply random perturbation to one element in on parameter tensor
    shape = tensor_force_field.potentials_by_type[handler_to_perturb].parameters.shape
    indicies_to_perturb = tuple(torch.randint(0, this_shape, (1,)).item() for this_shape in shape)

    tensor_force_field.potentials_by_type[handler_to_perturb].parameters[indicies_to_perturb] *= 1.432

    new_force_field = tyff.converters.convert_tensor_force_field(
        sage,
        tensor_force_field,
    )

    assert hash(new_force_field) != hash(sage)


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
