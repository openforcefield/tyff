from __future__ import annotations

from parsl import python_app

from tyff.compute._equilibrate import EquilibrationConfig
from tyff.compute._files import (
    EquilibrationFiles,
    MinimizationFiles,
    PackingFiles,
    PreparingFiles,
    ProductionFiles,
)
from tyff.compute._produce import ProductionConfig


@python_app
def prepare_packed_topology(
    job_dir: str,
) -> dict[str, PackingFiles]:
    from tyff.compute._pack import _prepare_packed_topology

    return _prepare_packed_topology(job_dir)


@python_app
def prepare_openmm_system(
    packing_future: dict[str, PackingFiles],
    job_dir: str,
) -> dict[str, PreparingFiles]:
    from tyff.compute._prepare import _prepare_openmm_system

    return _prepare_openmm_system(packing_future, job_dir)


@python_app
def minimize_energy(
    system_future: dict[str, PreparingFiles],
    job_dir: str,
) -> dict[str, MinimizationFiles]:
    from tyff.compute._minimize import _minimize_energy

    return _minimize_energy(system_future, job_dir)


@python_app
def run_equilibration(
    equilibration_config: EquilibrationConfig,
    minimization_future: dict[str, MinimizationFiles],
    job_dir: str,
) -> dict[str, EquilibrationFiles]:
    from tyff.compute._equilibrate import _run_equilibration

    return _run_equilibration(equilibration_config, minimization_future, job_dir)


@python_app
def run_production(
    production_config: ProductionConfig,
    equilibration_future: dict[str, EquilibrationFiles],
    job_dir: str,
) -> dict[str, ProductionFiles]:

    from tyff.compute._produce import _run_production

    return _run_production(production_config, equilibration_future, job_dir)


@python_app
def create_jacobian(
    job_dir: str,
) -> dict[str, float]:
    from tyff.compute._jacobian import _get_ensemble_average_and_jacobian

    return _get_ensemble_average_and_jacobian(job_dir)
