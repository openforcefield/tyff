"""Test conversion of tensor force fields back to SMIRNOFF force fields."""

import random

import pytest
from openff.toolkit import ForceField, Molecule
from openff.toolkit.typing.engines.smirnoff.parameters import VirtualSiteType

import tyff.converters


def tidy_force_field(force_field: ForceField) -> ForceField:
    # work around vdWType squishiness when comparing force fields via hash
    # https://github.com/openforcefield/openff-toolkit/issues/2241
    for parameter in force_field["vdW"].parameters:
        parameter.rmin_half = round(parameter.rmin_half, 12)

    return force_field


def force_fields_are_equal(force_field_1: ForceField, force_field_2: ForceField) -> bool:
    return hash(tidy_force_field(force_field_1)) == hash(tidy_force_field(force_field_2))


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


@pytest.fixture
def methyl_phenyl_disulfide():
    # need a molecule with layered torsions and impropers
    return Molecule.from_smiles("c1ccccc1(SSC)")


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

    try:
        force_fields_are_equal(new_force_field, sage)
    except AssertionError as error:
        import copy

        new_force_field.to_file("new.offxml")
        copy.deepcopy(sage).to_file("old.offxml")

        raise error


@pytest.mark.parametrize(
    "handler_to_perturb,column_index",
    [
        ("Bonds", 0),  # k, length
        ("Angles", 0),  # k, angle
        ("ProperTorsions", 0),  # k, etc.
        ("ImproperTorsions", 0),  # k, etc.
    ],
)
def test_convert_after_perturbation(methyl_phenyl_disulfide, sage, handler_to_perturb, column_index):
    """
    Test that a tensor force field, randomly perturbed from the original SMIRNOFF source parameters, can be
    converted back into a SMIRNOFF force field.
    """
    factor = random.random()

    interchange = sage.create_interchange(methyl_phenyl_disulfide.to_topology())

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


def test_convert_vsites_unsupported(methyl_chloride, sage_with_bond_charge):
    """
    Test that an error is raised when attempting to convert back a tensor force field
    which contatins virtual site parameters.

    Remove this test when the feature is implemented.
    """

    interchange = sage_with_bond_charge.create_interchange(methyl_chloride.to_topology())

    tensor_force_field, _ = tyff.converters.convert_interchange(interchange)

    # modify tensor force field

    with pytest.raises(NotImplementedError):
        tyff.converters.convert_tensor_force_field(
            sage_with_bond_charge,
            tensor_force_field,
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
