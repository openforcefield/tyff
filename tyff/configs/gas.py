import typing

from tyff.configs._compute import BaseComputeConfig


class VacuumGas(BaseComputeConfig):
    """
    General model for vacuum simulations

    Note that `pressure` is inherited from the base class as `float | None`,
    but should always be `None` in this model.

    """

    tag: typing.Literal["gas"]

    """
    Maybe add an RNG seed?
    """

    """
    maybe?

    units: str
    """
