import openff.interchange
import openff.toolkit
from openff.toolkit.utils.exceptions import ParameterLookupError

import tyff


def convert_tensor_force_field(
    original_force_field: openff.toolkit.ForceField,
    tensor_force_field: tyff.TensorForceField,
) -> openff.toolkit.ForceField:
    """
    Convert a tensor force field back into a SMIRNOFF force field, using the
    original force field as a template.

    Does not look at handler-level attributes (1-4 scaling factors, cutoffs, etc.).

    Electrostatics / applied partial charges are not updated.

    `ImproperTorsionType.idivf` values are ignored.

    Virtual site parameters are not (yet) supported.
    """
    import copy

    updated = copy.deepcopy(original_force_field)

    for potential in tensor_force_field.potentials:
        if potential.type in ("Electrostatics",):
            continue

        if potential.type in ("VirtualSites",):
            raise NotImplementedError("Virtual site parameters are not yet supported.")

        name = str(getattr(potential.type, "value", potential.type))
        handler = updated.get_parameter_handler(name)

        for row, key in enumerate(potential.parameter_keys):
            for index, col in enumerate(potential.parameter_cols):
                if col == "idivf":
                    continue

                value = potential.parameters[row, index].item()
                try:
                    # skip if parameter is already set to this value?
                    # maybe unnecessarily slow ...
                    setattr(handler[key.id], col, value * potential.parameter_units[index])
                except TypeError:
                    # can't set a list of k with just one k value, but can directly set the element
                    # of this parameter's k-list
                    getattr(handler[key.id], col)[key.mult] = value * potential.parameter_units[index]
                except ParameterLookupError as error:
                    # this is probably a failure to look up a virtual site's vdW parameter
                    raise NotImplementedError("Virtual site parameters are not yet supported.") from error

    return updated
