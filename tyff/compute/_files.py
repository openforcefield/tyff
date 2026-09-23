from typing import TypedDict

from parsl import File


class PackingFiles(TypedDict):
    packed_topology: File


class PreparingFiles(TypedDict):
    packed_topology: File
    openmm_system: File


class MinimizationFiles(TypedDict):
    """Files needed in the `minimize_energy` app."""

    topology: File
    system: File
    integrator: File
    checkpoint: File


class EquilibrationFiles(TypedDict):
    """Files needed in the `run_equilibration` app."""

    topology: File
    dcd_trajectory: File
    msgpack_trajectory: File
    log: File
    state_data: File
    system: File
    integrator: File
    checkpoint: File


class ProductionFiles(TypedDict):
    """Files needed in the `run_production` app."""

    topology: File
    dcd_trajectory: File
    msgpack_trajectory: File
    log: File
    state_data: File
    system: File
    integrator: File
    checkpoint: File
