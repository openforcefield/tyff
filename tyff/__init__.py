"""
tyff
Distributed simulation package
"""

from importlib.metadata import version

from tyff._constants import CUTOFF_ATTRIBUTE, SWITCH_ATTRIBUTE, EnergyFn, PotentialType
from tyff._models import (
    NonbondedParameterMap,
    ParameterMap,
    TensorConstraints,
    TensorForceField,
    TensorPotential,
    TensorSystem,
    TensorTopology,
    TensorVSites,
    ValenceParameterMap,
    VSiteMap,
)
from tyff.geometry import add_v_site_coords, compute_v_site_coords
from tyff.potentials import compute_energy, compute_energy_potential

__version__ = version("tyff")

__author__ = "Lily Wang"


__all__ = [
    "CUTOFF_ATTRIBUTE",
    "SWITCH_ATTRIBUTE",
    "EnergyFn",
    "NonbondedParameterMap",
    "ParameterMap",
    "PotentialType",
    "TensorConstraints",
    "TensorForceField",
    "TensorPotential",
    "TensorSystem",
    "TensorTopology",
    "TensorVSites",
    "VSiteMap",
    "ValenceParameterMap",
    "__version__",
    "add_v_site_coords",
    "compute_energy",
    "compute_energy_potential",
    "compute_v_site_coords",
]
