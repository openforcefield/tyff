import typing


class MinimizationConfig(typing.TypedDict):
    tolerance: float
    """Energy tolerance, passed directly to LocalEnergyMinimizer.minimize()"""

    max_iterations: int
    """Maximum number of iterations, passed directly to LocalEnergyMinimizer.minimize()"""


default_minimization_config = MinimizationConfig(
    tolerance=10.0,  # kJ/mol
    max_iterations=0,
)
