from __future__ import annotations

import json

import pandas


def _run_density_analysis(
    job_dir: str,
) -> dict[str, float]:
    """Run a naive density analysis from recorded simulation data."""
    from tyff.compute._logging import _set_up_logger

    logger = _set_up_logger(f"{job_dir}/analyze.log")

    logger.info("Starting density analysis")

    # TODO: Double check for equilibration/stability
    try:
        dataframe = pandas.read_csv(f"{job_dir}/production.csv")
    except TypeError:
        logger.error(f"Production run CSV data file production.csv not found, we are in {job_dir=}")
        raise

    mean = dataframe["Density (g/mL)"].mean()
    std = dataframe["Density (g/mL)"].std()

    logger.info(f"Estimated mean density: {mean:.4f} +/- {std:.4f} g/mL")

    return {
        "mean": mean,
        "std": std,
    }


def _run_dhvap_analysis(
    job_dirs: list[str],
) -> dict[str, float]:
    from openff.units import unit

    configs = {job_dir: json.load(open(f"{job_dir}/compute_config.json")) for job_dir in job_dirs}

    gas_dir = next(dir for dir, config in configs.items() if config["tag"] == "gas")
    liquid_dir = next(dir for dir, config in configs.items() if config["tag"] == "liquid")

    gas_temperature = configs[gas_dir]["temperature"]
    liquid_temperature = configs[liquid_dir]["temperature"]

    n_molecules = configs[liquid_dir]["n_molecules"]

    assert gas_temperature == liquid_temperature, "Gas and liquid temperatures must be the same for dHvap calculation"

    e_gas = pandas.read_csv(f"{gas_dir}/production.csv")["Potential Energy (kJ/mole)"].mean()
    e_liquid = pandas.read_csv(f"{liquid_dir}/production.csv")["Potential Energy (kJ/mole)"].mean()

    R = (1 * unit.avogadro_number * unit.boltzmann_constant).m_as("kJ/K")

    dhvap = e_gas - e_liquid / n_molecules + R * gas_temperature  # kJ/mol

    return {
        "dhvap": dhvap,
    }
