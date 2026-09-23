"""Prep (simulation) jobs from (thermophysical) data entries."""

from collections.abc import Sequence

from tyff.configs.gas import VacuumGas
from tyff.configs.liquid import BulkLiquid
from tyff.configs.targets.thermo import DataEntry


def compute_configs_from_data_entries(
    data_entries: list[DataEntry],
    force_field: str,
    n_molecules: int,
) -> Sequence[tuple[BulkLiquid | VacuumGas, ...]]:
    """Convert a list of thermophysical data entries into a list of simulation configs."""
    compute_configs: list[tuple[BulkLiquid | VacuumGas, ...]] = list()

    for data_entry in data_entries:
        these_configs = _compute_configs_from_data_entry(data_entry, force_field, n_molecules)
        compute_configs.extend(these_configs)

    return compute_configs


def _compute_configs_from_data_entry(
    data_entry: DataEntry,
    force_field: str,
    n_molecules: int,
    n_replicates: int = 3,
) -> Sequence[tuple[BulkLiquid | VacuumGas, ...]]:
    """Convert a single thermophysical data entry into a list of simulation configs."""

    match data_entry:
        case {"tag": "density" | "dielectric_constant"}:
            return _make_liquid_density_compute_configs(
                data_entry,
                force_field,
                n_molecules,
                n_replicates,
            )
        case {"tag": "enthalpy_of_mixing" | "excess_molar_volume"}:
            return _make_enthalpy_of_mixing_compute_configs(
                data_entry,
                force_field,
                n_molecules,
                n_replicates,
            )
        case {"tag": "enthalpy_of_vaporization"}:
            return _make_enthalpy_of_vaporization_compute_configs(
                data_entry,
                force_field,
                n_molecules,
                n_replicates,
            )
        case _:
            raise ValueError(f"Unsupported data entry tag: {data_entry['tag']}")


def _make_liquid_density_compute_configs(
    data_entry: DataEntry,
    force_field: str,
    n_molecules: int,
    n_replicates: int = 3,
) -> Sequence[tuple[BulkLiquid, ...]]:
    from tyff.configs.liquid import BulkLiquid

    return [
        tuple(
            (
                BulkLiquid(
                    tag="liquid",
                    force_field=force_field,
                    n_molecules=n_molecules,
                    replicate_index=replicate_index,
                    smiles=data_entry["smiles"],
                    x=data_entry["x"],
                    temperature=data_entry["temperature"],
                    pressure=data_entry["pressure"],
                    density=data_entry["value"] if data_entry["tag"] == "density" else None,
                ),
            )
        )
        for replicate_index in range(n_replicates)
    ]


def _make_enthalpy_of_mixing_compute_configs(
    data_entry: DataEntry,
    force_field: str,
    n_molecules: int,  # error if float? would this value ever be the result of rounding?
    n_replicates: int = 3,
) -> Sequence[tuple[BulkLiquid, ...]]:
    from tyff.configs.liquid import BulkLiquid

    liquid_configs: list[tuple[BulkLiquid, ...]] = list()

    for replicate_index in range(n_replicates):
        this_replicates_configs: list[BulkLiquid] = list()
        this_replicates_configs.append(
            BulkLiquid(
                tag="liquid",
                force_field=force_field,
                n_molecules=n_molecules,
                replicate_index=replicate_index,
                smiles=data_entry["smiles"],
                x=data_entry["x"],
                temperature=data_entry["temperature"],
                pressure=data_entry["pressure"],
                density=data_entry["value"] if data_entry["tag"] == "density" else None,
            )
        )

        for component_smiles, _component_x in zip(data_entry["smiles"], data_entry["x"]):
            this_replicates_configs.append(
                BulkLiquid(
                    tag="liquid",
                    force_field=force_field,
                    n_molecules=n_molecules,
                    replicate_index=replicate_index,
                    smiles=[component_smiles],
                    x=[1.0],
                    temperature=data_entry["temperature"],
                    pressure=data_entry["pressure"],
                    density=data_entry["value"] if data_entry["tag"] == "density" else None,
                )
            )

        liquid_configs.append(tuple(this_replicates_configs))

    return tuple(liquid_configs)


def _make_enthalpy_of_vaporization_compute_configs(
    data_entry: DataEntry,
    force_field: str,
    n_molecules: int,
    n_replicates: int = 3,
) -> list[tuple[BulkLiquid | VacuumGas, ...]]:
    from tyff.configs.gas import VacuumGas
    from tyff.configs.liquid import BulkLiquid

    compute_configs: list[tuple[BulkLiquid | VacuumGas, ...]] = list()

    for replicate_index in range(n_replicates):
        compute_configs.append(
            tuple(
                (
                    BulkLiquid(
                        tag="liquid",
                        force_field=force_field,
                        n_molecules=n_molecules,
                        replicate_index=replicate_index,
                        smiles=data_entry["smiles"],
                        x=data_entry["x"],
                        temperature=data_entry["temperature"],
                        pressure=data_entry["pressure"],
                        density=None,
                    ),
                    VacuumGas(
                        tag="gas",
                        force_field=force_field,
                        n_molecules=1,  # 1 no matter what, n_molecules from the user only goes to the liquid phase
                        replicate_index=replicate_index,
                        smiles=data_entry["smiles"],
                        x=data_entry["x"],
                        temperature=data_entry["temperature"],
                    ),
                )
            )
        )

    return compute_configs


# TODO: Move this into the step that processes multiple configs, maybe SimulationWorkflow.submit_batch
# TODO: When setting up jobs, could have these key a dict that also stores target densities (for when the property is
#       pure liquid but not density, like dielectric constant or enthalpy of mixing). See Issue #103
def get_liquid_deduplication_key(item: BulkLiquid, ignore_keys: list[str] = list()):
    # TODO: Might want to make this item: BaseComputeConfig

    smiles_sorted, x_sorted = [*map(tuple, zip(*sorted(zip(item["smiles"], item["x"]), key=lambda pair: pair[0])))]
    subset = {
        "force_field": item["force_field"],
        "x": x_sorted,
        "smiles": smiles_sorted,
        "temperature": item["temperature"],
        "pressure": item["pressure"],
        "n_molecules": item["n_molecules"],
    }

    return tuple((k, v) for k, v in subset.items() if k not in ignore_keys)
