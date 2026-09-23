import shutil
from importlib.resources import files

from tyff.compute._analyze import _run_density_analysis


def test_basic_analysis(tmp_path):
    # shim - see comment in source code
    for file in ["production.csv"]:
        shutil.copy(
            str(files("tyff") / f"_tests/data/app_files/sample_density/{file}"),
            str(tmp_path / file),
        )

    analysis_result = _run_density_analysis(
        job_dir=str(tmp_path),
    )

    # just some non-zero number
    assert analysis_result["mean"] > 0.0

    # std should be <10% of the mean
    assert analysis_result["std"] / analysis_result["mean"] < 0.1
