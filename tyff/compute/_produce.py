from __future__ import annotations

import json
import pathlib

import openmm
import openmm.app
from parsl import File

import tyff.mm
from tyff.compute._files import (
    EquilibrationFiles,
    ProductionFiles,
)
from tyff.configs.liquid import BulkLiquid
from tyff.exceptions import PressureNotDefinedError

ProductionConfig = object


def _run_production(
    production_config: ProductionConfig,
    equilibration_future: dict[str, EquilibrationFiles],
    job_dir: str,
) -> dict[str, ProductionFiles]:
    from tyff.compute._logging import _set_up_logger

    logger = _set_up_logger(f"{job_dir}/produce.log")

    logger.info("Starting production run")

    compute_config = BulkLiquid(**json.load(open(f"{job_dir}/compute_config.json")))  # type: ignore[typeddict-item]

    files = ProductionFiles(
        topology=File(f"{job_dir}/production_topology.pdb"),
        dcd_trajectory=File(f"{job_dir}/production_trajectory.dcd"),
        msgpack_trajectory=File(f"{job_dir}/production_trajectory.msgpack"),
        log=File(f"{job_dir}/production.log"),
        state_data=File(f"{job_dir}/production.csv"),
        system=File(f"{job_dir}/production_system.xml"),
        integrator=File(f"{job_dir}/production_integrator.xml"),
        checkpoint=File(f"{job_dir}/production_checkpoint.chk"),
    )

    equilibrated_files: EquilibrationFiles = equilibration_future["simulation_files"]

    if pathlib.Path(files["topology"].filepath).exists():
        logger.info(f"File {files['topology'].filepath} already exists, skipping production run.")

        return {
            "simulation_files": files,
        }

    with open(equilibrated_files["topology"].filepath) as f:
        pdb_file = openmm.app.PDBFile(f)

    topology = pdb_file.getTopology()

    with open(equilibrated_files["system"].filepath) as f:
        system = openmm.XmlSerializer.deserialize(f.read())

    with open(equilibrated_files["integrator"].filepath) as f:
        integrator = openmm.XmlSerializer.deserialize(f.read())

    simulation = openmm.app.Simulation(
        topology,
        system,
        integrator,
    )

    try:
        simulation.loadCheckpoint(equilibrated_files["checkpoint"].filepath)
    except (openmm.OpenMMException, OSError) as error:
        # loading checkpoint isn't so necessary when starting a new simulation since we are
        # already loading the correct positions. The checkpoint also adds low-level stuff like
        # RNG seeds, platform, hardware-specific stuff. It's nice to have these but I don't think
        # they're **required** for things to run - this is not as true for restarting failed jobs
        logger.warning(
            f"Failed to load checkpoint from {equilibrated_files['checkpoint'].filepath} with below error, "
            "starting from scratch."
        )
        logger.warning(f"{error}")

        # but we need to set positions and box vectors if we fail to load the checkpoint!
        simulation.context.setPositions(pdb_file.getPositions())

        if topology.getPeriodicBoxVectors() is not None:
            simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())

    logger.info("Reinitializing context (in production step)")
    simulation.context.reinitialize(preserveState=True)

    if compute_config["tag"] == "liquid":
        # just double check pressure is loaded back in correctly from state
        barostat = next(
            force for force in simulation.system.getForces() if isinstance(force, openmm.MonteCarloBarostat)
        )

        assert barostat.getDefaultPressure() is not None

    simulation.context.setVelocitiesToTemperature(
        compute_config["temperature"],  # kelvin, but as float
        compute_config["replicate_index"] + 1,
    )

    simulation.reporters.append(
        openmm.app.StateDataReporter(
            file=files["state_data"].filepath,
            reportInterval=1000,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
        )
    )

    dcd_reporter = openmm.app.DCDReporter(
        file=files["dcd_trajectory"].filepath,
        reportInterval=1000,
    )

    simulation.reporters.append(dcd_reporter)

    pressure = compute_config.get("pressure", None)

    if pressure is None:
        raise PressureNotDefinedError("Trying to set up NPT simulation but no pressure defined.")

    tensor_reporter = tyff.mm.TensorReporter(
        output_file=files["msgpack_trajectory"].filepath,
        report_interval=1000,
        beta=1.0 / openmm.unit.kilocalories_per_mole,
        pressure=pressure * openmm.unit.kilopascal,
    )

    simulation.reporters.append(tensor_reporter)

    platform_name = simulation.context.getPlatform().getName()

    logger.info(f"Using OpenMM platform: {platform_name}")

    logger.info("Running 100,000 steps of MD")

    for index in range(10):
        logger.info(f"Running from step {index * 10_000} to step {(index + 1) * 10_000}")
        simulation.step(10_000)

    for index in range(10):
        logger.info(f"Running from step {index * 10_000} to step {(index + 1) * 10_000}")
        simulation.step(10_000)

    with open(files["topology"].filepath, "w") as f:
        openmm.app.PDBFile.writeFile(
            topology=simulation.topology,
            positions=simulation.context.getState(getPositions=True).getPositions(),
            file=f,
        )
    with open(files["system"].filepath, "w") as f:
        f.write(openmm.XmlSerializer.serialize(simulation.system))

    with open(files["integrator"].filepath, "w") as f:
        f.write(openmm.XmlSerializer.serialize(simulation.integrator))

    simulation.saveCheckpoint(files["checkpoint"].filepath)

    return {"simulation_files": files}
