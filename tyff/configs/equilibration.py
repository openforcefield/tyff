import typing

from tyff.configs.ensemble import Ensemble


class EquilibrationConfig(typing.TypedDict):
    ensemble: Ensemble

    steps_per_iteration: int

    step_size: float
