import openff.interchange
import openff.toolkit

import tyff


def convert_tensor_force_field(
    original_force_field: openff.toolkit.ForceField,
    tensor_force_field: tyff.TensorForceField,
) -> openff.toolkit.ForceField:
    """
    Convert a tensor force field back into a SMIRNOFF force field, using the
    original force field as a template. Electrostatics / applied partial
    charges are not updated.
    """
    import copy

    updated = copy.deepcopy(original_force_field)

    for potential in tensor_force_field.potentials:
        if potential.type == "Electrostatics":
            continue

        if potential.type == "ProperTorsions":
            # TODO: implement this
            continue

        name = str(getattr(potential.type, "value", potential.type))
        handler = updated.get_parameter_handler(name)

        for row, key in enumerate(potential.parameter_keys):
            for index, col in enumerate(potential.parameter_cols):
                value = potential.parameters[row, index].item()

                setattr(handler[key.id], col, value * potential.parameter_units[index])

    return updated
