import parsl
import pytest
from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor

from tyff.configs.liquid import BulkLiquid


@pytest.fixture(scope="session")
def parsl_test_setup():
    # Use local threads for predictable and fast unit testing
    local_config = Config(
        executors=[ThreadPoolExecutor(label="local_threads", max_threads=2)],
        strategy=None,  # Disable dynamic scaling during unit tests
    )
    parsl.load(local_config)

    yield  # Run tests

    parsl.clear()  # Teardown


@pytest.fixture
def liquid_config() -> BulkLiquid:
    return BulkLiquid(
        tag="liquid",
        force_field="test_force_field",
        n_molecules=10,
        replicate_index=0,
        smiles=["CCO"],
        x=[1.0],
        density=0.8,
        temperature=300.0,
        pressure=101.325,
    )
