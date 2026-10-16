import tarfile
from pathlib import Path

from datasets import Dataset


def main(root: Path) -> None:
    if not (root / "data/energy").exists():
        with tarfile.open(root / "data/energy.tar.gz", "r:gz") as tar:
            tar.extractall(path=root / "data/", filter="data")

    if not (root / "data/thermo").exists():
        with tarfile.open(root / "data/thermo.tar.gz", "r:gz") as tar:
            tar.extractall(path=root / "data/", filter="data")

    thermo_data = Dataset.load_from_disk(root / "data/thermo")
    energy_data = Dataset.load_from_disk(root / "data/energy")

    print(f"Loaded {len(thermo_data)} thermo entries and {len(energy_data)} energy entries")

    return


if __name__ == "__main__":
    main(root=Path(__file__).resolve().parent)
