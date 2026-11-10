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

    for platform_name in ["CUDA", "OpenCL", "CPU"]:
        try:
            logger.info(f"Trying to use {platform_name} platform ...")
            platform = openmm.Platform.getPlatformByName(platform_name)
        except openmm.OpenMMException as error:
            logger.info(f"Could not use {platform_name} platform because {error}")
            continue

        logger.info(f"Successfully requested platform {platform.getName()}")

        return platform
