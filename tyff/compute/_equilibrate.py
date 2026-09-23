from __future__ import annotations

import json
import pathlib

from parsl import File

from tyff.compute._files import (
    EquilibrationFiles,
    MinimizationFiles,
)
from tyff.configs.liquid import BulkLiquid
from tyff.exceptions import PressureNotDefinedError

EquilibrationConfig = object


def _run_equilibration(
    equilibration_config: EquilibrationConfig,
    minimization_future: dict[str, MinimizationFiles],
    job_dir: str,
) -> dict[str, EquilibrationFiles]:
    # TODO: Expose barostat (+ thermostat?) to user

    import openmm
    import openmm.app
    import openmm.unit

    from tyff.compute._logging import _set_up_logger

    logger = _set_up_logger(f"{job_dir}/equilibrate.log")

    logger.info("Starting equilibration run")

    compute_config = BulkLiquid(**json.load(open(f"{job_dir}/compute_config.json")))  # type: ignore[typeddict-item]

    minimized_files: MinimizationFiles = minimization_future["simulation_files"]

    files = EquilibrationFiles(
        topology=File(f"{job_dir}/equilibrated_topology.pdb"),
        dcd_trajectory=File(f"{job_dir}/equilibration_trajectory.dcd"),
        msgpack_trajectory=File(f"{job_dir}/equilibration_trajectory.msgpack"),
        log=File(f"{job_dir}/equilibrate.log"),
        state_data=File(f"{job_dir}/equilibration.csv"),
        system=File(f"{job_dir}/equilibration_system.xml"),
        integrator=File(f"{job_dir}/equilibration_integrator.xml"),
        checkpoint=File(f"{job_dir}/equilibration_checkpoint.chk"),
    )

    if pathlib.Path(files["topology"].filepath).exists():
        logger.info(f"File {files['topology'].filepath} already exists, skipping equilibration run.")

        return {
            "simulation_files": files,
        }

    with open(minimized_files["topology"].filepath) as f:
        pdb_file = openmm.app.PDBFile(f)

    topology = pdb_file.getTopology()

    with open(minimized_files["system"].filepath) as f:
        system = openmm.XmlSerializer.deserialize(f.read())

    with open(minimized_files["integrator"].filepath) as f:
        integrator = openmm.XmlSerializer.deserialize(f.read())

    simulation = openmm.app.Simulation(
        topology,
        system,
        integrator,
    )

    try:
        simulation.loadCheckpoint(minimized_files["checkpoint"].filepath)
    except (openmm.OpenMMException, OSError) as error:
        # loading checkpoint isn't so necessary when starting a new simulation since we are
        # already loading the correct positions. The checkpoint also adds low-level stuff like
        # RNG seeds, platform, hardware-specific stuff. It's nice to have these but I don't think
        # they're **required** for things to run - this is not as true for restarting failed jobs
        logger.warning(
            f"Failed to load checkpoint from {minimized_files['checkpoint'].filepath} with below error, "
            "starting from scratch."
        )
        logger.warning(f"{error}")

        # but we need to set positions and box vectors if we fail to load the checkpoint!
        simulation.context.setPositions(pdb_file.getPositions())

        if topology.getPeriodicBoxVectors() is not None:
            simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())

    pressure = compute_config.get("pressure", None)

    if pressure is None:
        raise PressureNotDefinedError("Trying to set up NPT simulation but no pressure defined.")

    # only set pressure in liquid NPT simulations, not in gas-phase NVT simulations
    if compute_config["tag"] == "liquid":
        barostat = openmm.MonteCarloBarostat(
            (pressure * openmm.unit.kilopascal).value_in_unit(openmm.unit.bar),  # pressure in bar
            compute_config["temperature"],  # temperature in kelvin
        )

        simulation.system.addForce(barostat)

    logger.info("Reinitializing context (in equilibration step)")
    simulation.context.reinitialize(preserveState=True)

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

    simulation.context.setVelocitiesToTemperature(
        compute_config["temperature"],  # kelvin, but as float
        compute_config["replicate_index"] + 1,
    )

    logger.info("Running 10,000 steps of MD")

    detected_platform = simulation.context.getPlatform().getName()

    logger.info(f"Using OpenMM platform: {detected_platform}")

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

    # do we no longer need to wire through the compute config ... ?
    return {"simulation_files": files}
