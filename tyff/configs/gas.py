import typing

from tyff.configs._compute import BaseComputeConfig


class VacuumGas(BaseComputeConfig):
    tag: typing.Literal["gas"]

    """
    Maybe add an RNG seed?
    """

    """
    maybe?

    units: str
    """
