from __future__ import annotations

import logging

import openmm


def _get_platform(logger: logging.Logger) -> openmm.Platform:
    """
    Get the OpenMM platform to use for minimization.

    Returns
    -------
    openmm.Platform
        The OpenMM platform to use for minimization.
    """
    available_platforms = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
    for platform_name in ["CUDA", "OpenCL", "CPU"]:
        if platform_name not in available_platforms:
            logger.info(f"Could not use {platform_name} platform because it is not detected as available")
            continue

        try:
            logger.info(f"Trying to use {platform_name} platform ...")
            platform = openmm.Platform.getPlatformByName(platform_name)
        except openmm.OpenMMException as error:
            logger.info(f"Could not use {platform_name} platform because {error}")
            continue

        logger.info(f"Successfully requested platform {platform.getName()}")

        return platform

    raise ValueError(f"Could not find a usable OpenMM platform. Available platforms: {available_platforms}")
