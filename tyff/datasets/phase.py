from enum import IntFlag, unique


@unique
class PropertyPhase(IntFlag):
    """An enum describing the phase that a property was
    collected in.

    Examples
    --------
    Properties measured in multiple phases (e.g. enthalpies of
    vaporization) can be defined be concatenating `PropertyPhase`
    enums:

    >>> gas_liquid_phase = PropertyPhase.Gas | PropertyPhase.Liquid
    """

    Undefined = 0x00
    Solid = 0x01
    Liquid = 0x02
    Gas = 0x04

    @classmethod
    def from_string(cls, enum_string):
        """Parses a phase enum from its string representation.

        Parameters
        ----------
        enum_string: str
            The str representation of a `PropertyPhase`

        Returns
        -------
        PropertyPhase
            The created enum

        Examples
        --------
        To round-trip convert a phase enum:
        >>> phase = PropertyPhase.Liquid | PropertyPhase.Gas
        >>> phase_str = str(phase)
        >>> parsed_phase = PropertyPhase.from_string(phase_str)
        """

        if len(enum_string) == 0:
            return PropertyPhase.Undefined

        components = [cls[x] for x in enum_string.split(" + ")]

        if len(components) == 0:
            return PropertyPhase.Undefined

        enum_value = components[0]

        for component in components[1:]:
            enum_value |= component

        return enum_value

    def __str__(self):
        return " + ".join([phase.name for phase in PropertyPhase if self & phase])

    def __repr__(self):
        return f"<PropertyPhase {self!s}>"
