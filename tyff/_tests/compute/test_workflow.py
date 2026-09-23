import parsl

from tyff.compute.configs import local_config
from tyff.compute.workflow import SimulationWorkflow


class TestSimulationWorkflow:
    def test_init(self, tmp_path):
        # Test that the SimulationWorkflow can be initialized without errors
        with SimulationWorkflow(base_dir=tmp_path, parsl_config=local_config()):
            with open(tmp_path / "workflow.log") as log_file:
                lines = log_file.readlines()

                assert "Initialized" in lines[0]
                assert str(tmp_path) in lines[0]

            parsl.clear()
