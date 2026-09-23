import typing

import typing_extensions  # PEP 728 backport for Python 3.12 - drop when 3.13+


class BaseComputeConfig(typing_extensions.TypedDict, extra_items=typing.Any):
    # mypy does not yet support extra_items
    # https://github.com/python/mypy/issues/18176
    force_field: str

    n_molecules: int

    replicate_index: int

    smiles: list[str]  # sorting these (and x) would avoid false negatives

    x: list[float]

    temperature: float
