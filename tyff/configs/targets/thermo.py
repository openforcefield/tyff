import copy
import typing

import datasets
import pyarrow

from tyff.molecule import map_smiles

DATA_SCHEMA = pyarrow.schema(
    [
        ("id", pyarrow.string()),
        ("tag", pyarrow.string()),
        ("smiles", pyarrow.list_(pyarrow.string())),
        ("x", pyarrow.list_(pyarrow.float64())),
        ("temperature", pyarrow.float64()),
        ("pressure", pyarrow.float64()),
        ("value", pyarrow.float64()),
        ("std", pyarrow.float64()),
        ("units", pyarrow.string()),
        ("source", pyarrow.string()),
    ]
)


class DataEntry(typing.TypedDict):
    id: int

    tag: str  # was previously EntryTag

    smiles: list[str]

    x: list[float]

    temperature: float

    pressure: float

    value: float

    std: float | None

    units: str

    source: str


def create_dataset(entries: list[DataEntry]) -> datasets.Dataset:

    copied_entries = list()

    for entry in entries:
        # Copy input data to avoid side effects
        copied_entry = copy.deepcopy(entry)

        copied_entry["smiles"] = [map_smiles(value) for value in copied_entry["smiles"]]

        copied_entries.append(copied_entry)

    table = pyarrow.Table.from_pylist([*copied_entries], schema=DATA_SCHEMA)

    return datasets.Dataset(datasets.table.InMemoryTable(table))
