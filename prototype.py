import json
import pathlib
from collections.abc import Sequence

import torch
from rich import print

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

    print(job_id_to_observables)

    return job_id_to_observables


def compute_properties(dataset: Sequence[DataEntry], observables):
    from tyff.compute.prep import _compute_configs_from_data_entry

    means_and_stds = []

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
            """
            ipdb> entry
            {
                'id': 74669191270958899039426367776551206167278900382279422138348522811937124722445,
                'tag': 'density',
                'smiles': ['COCCO'],
                'x': [1.0],
                'temperature': 293.15,
                'pressure': 101.3,
                'value': 0.9648800000000002,
                'std': 5.0000000000000016e-05,
                'units': 'gram / milliliter',
                'source': ''
            }
            ipdb> job_ids
            [
                '5090adac8840d612600a2542c0236d47c2de3cdef6d6517961aa68e84f7ab8da',
                '2ed40b18f3773e1a0eb756147e2b56e3da4c47149cdbf97c7bf6b85107df5c1c',
                '86cf94ed4a70093ce9d6d1f480eebaf806c457ca1c7347e5b2f153e59f12d59e'
            ]
            ipdb> print([observables[job_id] for job_id in job_ids])
            [
                {
                    'potential_energy': tensor(1835.6803, dtype=torch.float64),
                    'potential_energy^2': tensor(3371036.2500, dtype=torch.float64),
                    'volume': tensor(26207.9766, dtype=torch.float64),
                    'volume^2': tensor(6.8692e+08, dtype=torch.float64),
                    'density': tensor(0.9644, dtype=torch.float64),
                    'enthalpy': tensor(3684.1626, dtype=torch.float64),
                    'enthalpy^2': tensor(13574647., dtype=torch.float64),
                    'enthalpy_volume': tensor(96557640., dtype=torch.float64)
                },
                {
                    'potential_energy': tensor(1848.0491, dtype=torch.float64),
                    'potential_energy^2': tensor(3417077., dtype=torch.float64),
                    'volume': tensor(26305.6504, dtype=torch.float64),
                    'volume^2': tensor(6.9209e+08, dtype=torch.float64),
                    'density': tensor(0.9608, dtype=torch.float64),
                    'enthalpy': tensor(3695.4373, dtype=torch.float64),
                    'enthalpy^2': tensor(13658783., dtype=torch.float64),
                    'enthalpy_volume': tensor(97218952., dtype=torch.float64)
                },
                {
                    'potential_energy': tensor(1837.3766, dtype=torch.float64),
                    'potential_energy^2': tensor(3377881.2500, dtype=torch.float64),
                    'volume': tensor(26262.2070, dtype=torch.float64),
                    'volume^2': tensor(6.8976e+08, dtype=torch.float64),
                    'density': tensor(0.9624, dtype=torch.float64),
                    'enthalpy': tensor(3681.6003, dtype=torch.float64),
                    'enthalpy^2': tensor(13557029., dtype=torch.float64),
                    'enthalpy_volume': tensor(96692128., dtype=torch.float64)
                }
            ]
            """
            # do we just return the mean and std for each observable this list?
            mean = {
                key: torch.stack([observables[job_id][key] for job_id in job_ids]).mean()
                for key in types_of_observables
            }
            std = {
                key: torch.stack([observables[job_id][key] for job_id in job_ids]).std()
                for key in types_of_observables
            }

            """
            not sure if these are the right things to return?

            ipdb> print(mean)
            {
                'potential_energy': tensor(1840.3687, dtype=torch.float64),
                'potential_energy^2': tensor(3388664.8333, dtype=torch.float64),
                'volume': tensor(26258.6113, dtype=torch.float64),
                'volume^2': tensor(6.8959e+08, dtype=torch.float64),
                'density': tensor(0.9625, dtype=torch.float64),
                'enthalpy': tensor(3687.0667, dtype=torch.float64),
                'enthalpy^2': tensor(13596819.6667, dtype=torch.float64),
                'enthalpy_volume': tensor(96822906.6667, dtype=torch.float64)
            }
            ipdb> print(std)
            {
                'potential_energy': tensor(6.7053, dtype=torch.float64),
                'potential_energy^2': tensor(24842.5425, dtype=torch.float64),
                'volume': tensor(48.9361, dtype=torch.float64),
                'volume^2': tensor(2586491.9573, dtype=torch.float64),
                'density': tensor(0.0018, dtype=torch.float64),
                'enthalpy': tensor(7.3614, dtype=torch.float64),
                'enthalpy^2': tensor(54380.0468, dtype=torch.float64),
                'enthalpy_volume': tensor(349514.9283, dtype=torch.float64)
            }
            """

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
    # like other targets


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
