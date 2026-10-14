import json
import pathlib
from collections.abc import Sequence

import torch

import tyff
import tyff.converters
from tyff._serialization import load_tensor_force_field
from tyff.compute.jobs import make_job_id
from tyff.configs.liquid import BulkLiquid
from tyff.configs.targets.thermo import DataEntry

# in leiu of better target-job tracking, just define known argumnets as ~globals
FORCE_FIELD = "openff-2.3.0.offxml"
N_MOLECULES = 200
N_REPLICATES = 3


def load_remote_observables(
    force_field_per_epoch: tyff.TensorForceField,
    base_dir: str,
    job_ids: list[str],
) -> dict[str, dict]:
    """Load observables and attach to graph."""
    from tyff.mm._ops import _pack_force_field

    tensors, *_ = _pack_force_field(
        force_field_per_epoch
    )  # force_field_per_epoch should also be the global reference FF passed to the workers
    parameters: torch.Tensor = torch.cat(
        [t.reshape(-1) for t in tensors if t is not None]
    )  # has to be flat, and not a tuple

    # TODO: Move this class to not be inside of this function
    class RemoteEnsembleAverage(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            values: torch.Tensor,
            jacobian: torch.Tensor,
            parameters: torch.Tensor,
        ) -> torch.Tensor:
            ctx.save_for_backward(jacobian)
            return values.clone()

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_output: torch.Tensor,
        ) -> tuple[None, None, torch.Tensor]:
            (jacobian,) = ctx.saved_tensors
            return None, None, grad_output @ jacobian

    job_id_to_observables = {}

    for job_id in job_ids:
        job_dir = pathlib.Path(base_dir) / job_id

        assert job_dir.exists()

        with open(job_dir / "ensemble_averages.json") as f:
            means_dict: dict[str, float] = json.load(f)  # actually str, torch.float64

        jacobians: torch.Tensor = torch.load(job_dir / "jacobian.pt")

        names = list(
            means_dict
        )  # get the names of the computed properties in the means_dict so we can convert dict -> tensor -> dict again
        tensor_means = parameters.new_tensor([means_dict[x] for x in names])
        attached_means = RemoteEnsembleAverage.apply(
            tensor_means,
            jacobians.to("cpu"),  # to(device) ?, get from somewhere?
            parameters,
        )

        job_id_to_observables[job_id] = dict(zip(names, attached_means))

    return job_id_to_observables


def compute_properties(
    dataset: Sequence[DataEntry],
    observables: dict[str, dict[str, torch.Tensor]],
) -> list[tuple[dict, dict]]:
    from tyff.compute.prep import _compute_configs_from_data_entry

    means_and_stds: list[tuple[dict, dict]] = list()

    for entry in dataset:
        if entry["tag"] == "density":
            # this includes replicates, but only jobs for this physical property target
            # structure is a list (length n_replicates) of tuples, each tuple is
            # 1-length and contains a compute config for (a single) liquid job
            compute_configs: Sequence[tuple[BulkLiquid, ...]] = _compute_configs_from_data_entry(
                data_entry=entry,
                force_field=FORCE_FIELD,
                n_molecules=N_MOLECULES,
                n_replicates=N_REPLICATES,
            )

            # only one job per replicate in density calculatiosn
            job_ids = [make_job_id(replicate[0]) for replicate in compute_configs]

            types_of_observables = [*observables[job_ids[0]].keys()]

            # could be simpler to return these as 1-D tensors - make sure to sort
            # keys to avoid ordering issues
            mean = {
                key: torch.stack([observables[job_id][key] for job_id in job_ids]).mean()
                for key in types_of_observables
            }
            std = {
                key: torch.stack([observables[job_id][key] for job_id in job_ids]).std()
                for key in types_of_observables
            }

            means_and_stds.append((mean, std))

            # figure out which job id/s (if multiple replicas) in the observable corresponds to
            # and calculate means and stds
        elif entry["tag"] == "dielectric_constant":
            raise NotImplementedError()
            # likely identical to density case
        elif entry["tag"] == "enthalpy_of_mixing":
            raise NotImplementedError()
            # figure out the three job IDs necessary from observables, calculate, etc
        elif entry["tag"] == "excess_molar_volume":
            raise NotImplementedError()
        elif entry["tag"] == "enthalpy_of_vaporization":
            raise NotImplementedError()
        else:
            raise ValueError(f"I do not know what to do with {entry['tag']}")

        means_and_stds.append((mean, std))

    return means_and_stds


def predict(
    dataset: Sequence[DataEntry],
    force_field: tyff.TensorForceField,
    job_ids: list[str],
    base_dir: str,
):
    attached_means = load_remote_observables(force_field, base_dir, job_ids)

    means_and_stds = compute_properties(dataset, attached_means)

    return means_and_stds
    # can either just return means_and_stds or can reformat into
    # returning (ref_values, ref_stds, predicted_values, predicted_std)
    # like other targets - reference values can be taken straight from DataEntrys


if __name__ == "__main__":
    with open("density_example/thermo_targets/targets.json") as f:
        targets = json.load(f)

    with open(
        "density_example/reference_force_fields/openff-2.3.0.offxml.json",
    ) as f:
        global_force_field = load_tensor_force_field(json.load(f))

    predict(
        dataset=targets,
        force_field=global_force_field,
        job_ids=[
            "0f6ef5a4fdd601e764c9a70b185a6e27b1b91db6a8b65fb25a8a26922a3fca09",
            "2ed40b18f3773e1a0eb756147e2b56e3da4c47149cdbf97c7bf6b85107df5c1c",
            "5090adac8840d612600a2542c0236d47c2de3cdef6d6517961aa68e84f7ab8da",
            "5a423d141a2e9d2d9ba9d30f988179ca8230bd8b93e7b854cc9c5c61f9ee9093",
            "8418e9155a226c72f4d382d18e190ba023cbe55c9ad90175333d30637267059a",
            "86cf94ed4a70093ce9d6d1f480eebaf806c457ca1c7347e5b2f153e59f12d59e",
            "93e2b533fdec76225cd6af47938c69c40180d90a398a73b99892221471bcebc8",
            "d721ef4f95b38106e81cc51e1d5638d9f09f372f4245cfe96ce2269d17e6a971",
            "f08c269b4006de30978674f5ea1615258f471b5c21e045d57641d67ee86c3bd6",
            "86cf94ed4a70093ce9d6d1f480eebaf806c457ca1c7347e5b2f153e59f12d59e",
            "5090adac8840d612600a2542c0236d47c2de3cdef6d6517961aa68e84f7ab8da",
            "2ed40b18f3773e1a0eb756147e2b56e3da4c47149cdbf97c7bf6b85107df5c1c",
        ],
        base_dir="density_example",
    )

    # TODO: Would be nice to dummy-check .backwards, can do this by constructing
    # a minimal loss function of (predicted - expected) for each density value
