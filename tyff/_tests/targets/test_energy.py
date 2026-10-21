import math

import openff.interchange
import openff.toolkit
import pytest
import torch

import tyff.converters
import tyff.utils.dataset
from tyff.targets.energy import (
    Entry,
    create_dataset,
    create_dataset_from_generator,
    extract_smiles,
    predict,
)


@pytest.fixture
def mock_meoh_entry() -> Entry:
    return {
        "smiles": "[C:1]([O:2][H:6])([H:3])([H:4])[H:5]",
        "coords": torch.arange(36, dtype=torch.float32).reshape(2, 6, 3),
        "energy": 3.0 * torch.arange(2, dtype=torch.float32),
        "forces": torch.arange(36, dtype=torch.float32).reshape(2, 6, 3) + 36.0,
    }


@pytest.fixture
def mock_hoh_entry() -> Entry:
    return {
        "smiles": "[H:2][O:1][H:3]",
        "coords": torch.tensor(
            [
                [[0.0, 0.0, 0.0], [-1.0, -0.5, 0.0], [1.0, -0.5, 0.0]],
                [[0.0, 0.0, 0.0], [-0.7, -0.5, 0.0], [0.7, -0.5, 0.0]],
            ]
        ),
        "energy": torch.tensor([2.0, 3.0]),
        "forces": torch.arange(18, dtype=torch.float32).reshape(2, 3, 3),
    }


@pytest.fixture
def mock_hoh_dimer_entry() -> Entry:
    # two waters are needed to test v-sites: a single TIP4P-FB water has zero energy
    # for any geometry (constraints only, all intramolecular pairs excluded)

    # O, H, H in the xy plane, close to TIP4P-FB geometry
    water = torch.tensor([[0.0, 0.0, 0.0], [-0.76, 0.59, 0.0], [0.76, 0.59, 0.0]])

    # geometry 1: second water shifted 3 Å along z
    water_b_1 = water + torch.tensor([0.0, 0.0, 3.0])

    # geometry 2: second water rotated by (x, y, z) -> (z, -y, x), shifted 2.8 Å along x
    water_b_2 = water.flip(-1) * torch.tensor([1.0, -1.0, 1.0]) + torch.tensor([2.8, 0.0, 0.0])

    return {
        "smiles": "[H:2][O:1][H:3].[H:5][O:4][H:6]",
        # (n_confs=2, n_atoms=6, 3), atom-only
        "coords": torch.stack([torch.cat([water, water_b_1]), torch.cat([water, water_b_2])]),
        "energy": torch.tensor([2.0, 3.0]),
        "forces": torch.arange(36, dtype=torch.float32).reshape(2, 6, 3),
    }


@pytest.mark.parametrize(
    "box_vectors",
    [None, torch.eye(3).repeat(2, 1, 1) * 20.0],
    ids=["non-periodic", "periodic"],
)
def test_create_dataset(mock_meoh_entry, box_vectors):
    entry = {**mock_meoh_entry, "box_vectors": box_vectors}

    expected_entries = [
        {
            "id": None,
            "smiles": entry["smiles"],
            "coords": pytest.approx(entry["coords"].flatten()),
            "energy": pytest.approx(entry["energy"]),
            "forces": pytest.approx(entry["forces"].flatten()),
            "box_vectors": None if box_vectors is None else pytest.approx(box_vectors.flatten()),
        },
    ]

    dataset = create_dataset([entry])
    assert len(dataset) == 1

    entries = list(tyff.utils.dataset.iter_dataset(dataset))
    assert entries == expected_entries


@pytest.mark.parametrize(
    "box_vectors",
    [None, torch.eye(3).repeat(2, 1, 1) * 20.0],
    ids=["non-periodic", "periodic"],
)
def test_create_dataset_from_generator(mock_meoh_entry, box_vectors):
    entry = {**mock_meoh_entry, "box_vectors": box_vectors}

    expected_entries = [
        {
            "id": None,
            "smiles": entry["smiles"],
            "coords": pytest.approx(entry["coords"].flatten()),
            "energy": pytest.approx(entry["energy"]),
            "forces": pytest.approx(entry["forces"].flatten()),
            "box_vectors": None if box_vectors is None else pytest.approx(box_vectors.flatten()),
        },
    ]

    dataset = create_dataset_from_generator(lambda: iter([entry]))
    assert len(dataset) == 1

    entries = list(tyff.utils.dataset.iter_dataset(dataset))
    assert entries == expected_entries


def test_extract_smiles(mock_meoh_entry, mock_hoh_entry):
    expected_smiles = ["[C:1]([O:2][H:6])([H:3])([H:4])[H:5]", "[H:2][O:1][H:3]"]

    dataset = create_dataset([mock_meoh_entry, mock_hoh_entry])
    smiles = extract_smiles(dataset)

    assert smiles == expected_smiles


@pytest.mark.parametrize(
    "box_vectors, reference, normalize,"
    "expected_energy_ref, expected_forces_ref, "
    "expected_energy_pred, expected_forces_pred, "
    "entry_name, force_field_name",
    [
        pytest.param(
            None,
            "mean",
            True,
            torch.tensor([-0.5, 0.5]) / math.sqrt(2.0),
            torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0],
                ],
                dtype=torch.float64,
            )
            / math.sqrt(6.0 * 3.0),
            torch.tensor([7.899425506591797, -7.89942741394043]) / math.sqrt(2.0),
            -torch.tensor(
                [
                    [0.0, 83.55978393554688, 0.0],
                    [-161.40325927734375, -41.77988815307617, 0.0],
                    [161.40325927734375, -41.77988815307617, 0.0],
                    [0.0, -137.45770263671875, 0.0],
                    [102.62999725341797, 68.72884368896484, 0.0],
                    [-102.62999725341797, 68.72884368896484, 0.0],
                ],
                dtype=torch.float64,
            )
            / math.sqrt(6.0 * 3.0),
            "mock_hoh_entry",
            "openff_unconstrained-1.3.0.offxml",
            id="non-periodic-mean-normalized",
        ),
        pytest.param(
            None,
            "min",
            False,
            torch.tensor([0.0, 1.0]),
            torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0],
                ],
                dtype=torch.float64,
            ),
            torch.tensor([0.0, -15.798852920532227]),
            -torch.tensor(
                [
                    [0.0, 83.55978393554688, 0.0],
                    [-161.40325927734375, -41.77988815307617, 0.0],
                    [161.40325927734375, -41.77988815307617, 0.0],
                    [0.0, -137.45770263671875, 0.0],
                    [102.62999725341797, 68.72884368896484, 0.0],
                    [-102.62999725341797, 68.72884368896484, 0.0],
                ],
                dtype=torch.float64,
            ),
            "mock_hoh_entry",
            "openff_unconstrained-1.3.0.offxml",
            id="non-periodic-min",
        ),
        pytest.param(
            torch.eye(3).repeat(2, 1, 1) * 30.0,
            "mean",
            True,
            torch.tensor([-0.5, 0.5]) / math.sqrt(2.0),
            torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0],
                ],
                dtype=torch.float64,
            )
            / math.sqrt(6.0 * 3.0),
            torch.tensor([5.585737228393555, -5.585737705230713]),
            torch.tensor(
                [
                    [0.0, -19.695229476617897, 0.0],
                    [38.04311560258793, 9.847614738308948, 0.0],
                    [-38.04311560258793, 9.847614738308948, 0.0],
                    [0.0, 32.3990898002703, 0.0],
                    [-24.190123962094730, -16.19954490013515, 0.0],
                    [24.190123962094730, -16.19954490013515, 0.0],
                ],
                dtype=torch.float64,
            ),
            "mock_hoh_entry",
            "openff_unconstrained-1.3.0.offxml",
            id="periodic-mean-normalized",
        ),
        pytest.param(
            torch.eye(3).repeat(2, 1, 1) * 30.0,
            "min",
            False,
            torch.tensor([0.0, 1.0]),
            torch.tensor(
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0],
                ],
                dtype=torch.float64,
            ),
            torch.tensor([0.0, -15.79885196685791]),
            torch.tensor(
                [
                    [0.0, -83.55977630615234, 0.0],
                    [161.40325927734375, 41.77988815307617, 0.0],
                    [-161.40325927734375, 41.77988815307617, 0.0],
                    [0.0, 137.4576873779297, 0.0],
                    [-102.62999725341797, -68.72884368896484, 0.0],
                    [102.62999725341797, -68.72884368896484, 0.0],
                ],
                dtype=torch.float64,
            ),
            "mock_hoh_entry",
            "openff_unconstrained-1.3.0.offxml",
            id="periodic-min",
        ),
        pytest.param(
            None,
            "mean",
            True,
            torch.tensor([-0.5, 0.5]) / math.sqrt(2.0),
            torch.arange(36, dtype=torch.float64).reshape(12, 3) / math.sqrt(12.0 * 3.0),
            torch.tensor([4.242203712463379, -4.242203712463379]) / math.sqrt(2.0),
            torch.tensor(
                [
                    [0.0, 5.792387, -7.558908],
                    [0.871478, -2.896193, 0.036511],
                    [-0.871478, -2.896193, 0.036511],
                    [0.0, 5.792387, 7.558908],
                    [0.871478, -2.896193, -0.036511],
                    [-0.871478, -2.896193, -0.036511],
                    [-16.529013, -6.123506, 0.0],
                    [1.342896, 1.191738, 0.0],
                    [12.059071, 1.518323, 0.0],
                    [6.345163, 11.937330, 0.0],
                    [-1.609060, -4.261943, -0.585875],
                    [-1.609060, -4.261943, 0.585874],
                ],
                dtype=torch.float64,
            )
            / math.sqrt(12.0 * 3.0),
            "mock_hoh_dimer_entry",
            "tip4p_fb.offxml",
            id="v-sites-mean-normalized",
        ),
    ],
)
def test_predict(
    box_vectors,
    reference,
    normalize,
    expected_energy_ref,
    expected_forces_ref,
    expected_energy_pred,
    expected_forces_pred,
    entry_name,
    force_field_name,
    request,
):
    entry = {**request.getfixturevalue(entry_name)}
    if box_vectors is not None:
        entry["box_vectors"] = box_vectors
    dataset = create_dataset([entry])

    force_field, [topology] = tyff.converters.convert_interchange(
        openff.interchange.Interchange.from_smirnoff(
            openff.toolkit.ForceField(force_field_name),
            openff.toolkit.Molecule.from_mapped_smiles(entry["smiles"]).to_topology(),
        )
    )
    topologies = {entry["smiles"]: topology}

    energy_ref, energy_pred, forces_ref, forces_pred = predict(
        dataset, force_field, topologies, reference=reference, normalize=normalize
    )

    assert energy_pred.shape == expected_energy_pred.shape
    assert torch.allclose(energy_pred, expected_energy_pred)
    assert energy_ref.shape == expected_energy_ref.shape
    assert torch.allclose(energy_ref, expected_energy_ref)

    assert forces_pred.shape == expected_forces_pred.shape
    # atol needed as some expected forces are rounded, or set to 0.0 in place of float32 noise
    assert torch.allclose(forces_pred, expected_forces_pred, atol=1.0e-5)
    assert forces_ref.shape == expected_forces_ref.shape
    assert torch.allclose(forces_ref, expected_forces_ref)


def test_predict_v_sites(mock_hoh_dimer_entry):
    """Test that the v-site parameters receive gradients through ``predict``, and
    that coordinates which already include v-sites are rejected. The predicted values
    themselves are checked by ``test_predict``."""
    smiles = mock_hoh_dimer_entry["smiles"]

    force_field, [topology] = tyff.converters.convert_interchange(
        openff.interchange.Interchange.from_smirnoff(
            openff.toolkit.ForceField("tip4p_fb.offxml"),
            openff.toolkit.Molecule.from_mapped_smiles(smiles).to_topology(),
        )
    )
    assert topology.v_sites is not None
    # needed to check the v-site parameter gradients below (in training,
    # Trainable.to_force_field provides parameters that already require grad)
    force_field.v_sites.parameters.requires_grad_(True)

    _, energy_pred, _, forces_pred = predict(create_dataset([mock_hoh_dimer_entry]), force_field, {smiles: topology})

    # Make sure that the energies and forces are non-zero before summing
    assert all(energy_pred.abs() > 0.0)
    assert all(forces_pred.abs().flatten() > 0.0)
    
    # both the energy and force paths must reach the v-site parameters. The energies
    # are squared as they are mean-subtracted, so their plain sum is always 0.
    (energy_grad,) = torch.autograd.grad(energy_pred.square().sum(), force_field.v_sites.parameters, retain_graph=True)
    (forces_grad,) = torch.autograd.grad(forces_pred.square().sum(), force_field.v_sites.parameters)
    assert energy_grad.abs().sum() > 0.0
    assert forces_grad.abs().sum() > 0.0

    # coords that already include v-sites would get them appended twice
    particle_coords = tyff.geometry.add_v_site_coords(
        topology.v_sites, mock_hoh_dimer_entry["coords"].double(), force_field
    ).detach()
    dataset_v_sites = create_dataset([{**mock_hoh_dimer_entry, "coords": particle_coords}])

    with pytest.raises(ValueError, match="should not include virtual sites"):
        predict(dataset_v_sites, force_field, {smiles: topology})
