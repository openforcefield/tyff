from __future__ import annotations

import copy

import pytest

pytest.importorskip("openmm")

import numpy
import openmm.app
import openmm.unit

from tyff.mm._reporters import TensorReporter, tensor_reporter, unpack_frames

_BETA = 1.0 / (298.15 * openmm.unit.kelvin * openmm.unit.MOLAR_GAS_CONSTANT_R)


def copy_simulation(simulation: openmm.app.Simulation) -> openmm.app.Simulation:
    positions = simulation.context.getState(getPositions=True).getPositions()

    simulation = openmm.app.Simulation(
        copy.deepcopy(simulation.topology),
        copy.deepcopy(simulation.system),
        copy.deepcopy(simulation.integrator),
    )

    simulation.context.setPositions(positions)

    return simulation


@pytest.fixture(scope="module")
def basic_simulation() -> openmm.app.Simulation:
    import openmm
    from openff.toolkit import ForceField, Molecule, Quantity

    molecule = Molecule.from_smiles("CCO")
    molecule.generate_conformers(n_conformers=1)

    # TensorReporter takes a pressure, can it be used with NVT/gas simulations?
    topology = molecule.to_topology()
    topology.box_vectors = Quantity(numpy.eye(3) * 5.0, "nanometer")

    return (
        ForceField("openff-2.3.0.offxml")
        .create_interchange(topology)
        .to_openmm_simulation(
            integrator=openmm.LangevinIntegrator(
                298.15 * openmm.unit.kelvin,
                1.0 / openmm.unit.picoseconds,
                0.002 * openmm.unit.picoseconds,
            ),
            additional_forces=[
                openmm.MonteCarloBarostat(
                    1.0 * openmm.unit.atmospheres,
                    298.15 * openmm.unit.kelvin,
                )
            ],
        )
    )


class TestTensorReporter:
    def test_describe_next(self, mocker):
        simulation = mocker.MagicMock()
        simulation.currentStep = 5

        reporter = TensorReporter(mocker.MagicMock(), 2, 1.0 / openmm.unit.kilocalories_per_mole, None)
        assert reporter.describeNextReport(simulation) == (1, True, False, False, True)

    def test_report(self, tmp_path, mocker):
        expected_potential = 1.0 * openmm.unit.kilocalories_per_mole
        expected_kinetic = 2.0 * openmm.unit.kilojoules_per_mole

        box_length = 3.0
        expected_volume = box_length**3 * openmm.unit.angstrom**3

        expected_box_vectors = numpy.eye(3) * box_length
        expected_coords = numpy.ones((1, 3))

        mock_state = mocker.MagicMock()
        mock_state.getPotentialEnergy.return_value = expected_potential
        mock_state.getKineticEnergy.return_value = expected_kinetic
        mock_state.getPeriodicBoxVectors.return_value = expected_box_vectors * openmm.unit.angstrom
        mock_state.getPeriodicBoxVolume.return_value = expected_volume
        mock_state.getPositions.return_value = expected_coords * openmm.unit.angstrom

        expected_output_path = tmp_path / "output.msgpack"

        pressure = 1.0 * openmm.unit.atmospheres

        with expected_output_path.open("wb") as file:
            reporter = TensorReporter(file, 1, _BETA, pressure)
            reporter.report(None, mock_state)

        with expected_output_path.open("rb") as file:
            frames = [*unpack_frames(file)]

        assert len(frames) == 1
        coords, box_vectors, reduced_potential, kinetic = frames[0]

        expected_reduced_potential = _BETA * (
            expected_potential + pressure * expected_volume * openmm.unit.AVOGADRO_CONSTANT_NA
        )

        assert coords == pytest.approx(expected_coords)
        assert box_vectors == pytest.approx(expected_box_vectors)

        assert isinstance(reduced_potential, float)
        assert reduced_potential == pytest.approx(expected_reduced_potential)

        assert isinstance(kinetic, float)
        assert kinetic == pytest.approx(expected_kinetic.value_in_unit(openmm.unit.kilocalories_per_mole))

    @pytest.mark.parametrize("potential, contains", [(numpy.nan, "nan"), (numpy.inf, "inf")])
    def test_report_energy_check(self, potential, contains, mocker):
        potential = potential * openmm.unit.kilocalories_per_mole
        kinetic = 2.0 * openmm.unit.kilocalories_per_mole

        mock_state = mocker.MagicMock()
        mock_state.getPotentialEnergy.return_value = potential
        mock_state.getKineticEnergy.return_value = kinetic

        beta = 1.0 / openmm.unit.kilocalories_per_mole

        with pytest.raises(ValueError, match=f"total energy is {contains}"):
            reporter = TensorReporter(mocker.MagicMock(), 1, beta, None)
            reporter.report(None, mock_state)

    @pytest.mark.parametrize("input_type", ["file", "object"])
    def test_report_during_simulation(self, basic_simulation, tmp_path, input_type):

        simulation = copy_simulation(basic_simulation)

        if input_type == "file":
            reporter = TensorReporter(
                output_file=str(tmp_path / "1.msgpack"),
                report_interval=10,
                beta=_BETA,
                pressure=1.0 * openmm.unit.atmospheres,
            )

            simulation.reporters.append(reporter)

            simulation.step(50)

            reporter.close()

        elif input_type == "object":
            with open(tmp_path / "2.msgpack", "wb") as output_file:
                tensor_reporter = TensorReporter(
                    output_file=output_file,
                    report_interval=10,
                    beta=_BETA,
                    pressure=1.0 * openmm.unit.atmospheres,
                )

                simulation.reporters.append(tensor_reporter)

                simulation.step(50)

    @pytest.mark.parametrize("append", [True, False])
    def test_append(self, basic_simulation, tmp_path, append):
        simulation = copy_simulation(basic_simulation)

        reporter = TensorReporter(
            output_file=str(tmp_path / "append.msgpack"),
            report_interval=10,
            beta=_BETA,
            pressure=1.0 * openmm.unit.atmospheres,
            append=True,
        )

        simulation.reporters.append(reporter)

        simulation.step(100)

        reporter.close()

        simulation.reporters.clear()

        second_reporter = TensorReporter(
            output_file=str(tmp_path / "append.msgpack"),
            report_interval=10,
            beta=_BETA,
            pressure=1.0 * openmm.unit.atmospheres,
            append=append,
        )

        simulation.reporters.append(second_reporter)

        simulation.step(100)

        second_reporter.close()

        with (tmp_path / "append.msgpack").open("rb") as file:
            frames = [*unpack_frames(file)]

            assert len(frames) == (20 if append else 10), f"found {len(frames)=}"


def test_tensor_reporter(tmp_path):
    output = tmp_path / "frames.msgpack"

    pressure = 1.0 * openmm.unit.atmospheres

    with tensor_reporter(output, 2, _BETA, pressure) as reporter:
        assert isinstance(reporter, TensorReporter)

    assert output.exists() and output.is_file()
