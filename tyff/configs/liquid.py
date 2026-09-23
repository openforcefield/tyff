import typing

from tyff.configs._compute import BaseComputeConfig


class BulkLiquid(BaseComputeConfig):
    tag: typing.Literal["liquid"]

    """
    Maybe add an RNG seed?
    """

    pressure: float

    density: float | None

    """
    maybe?

    units: str
    """
