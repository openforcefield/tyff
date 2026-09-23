from openff.toolkit import Molecule


def map_smiles(smiles: str) -> str:
    """Take an (un-mapped) SMILES string and make it a mapped SMILES string."""
    return Molecule.from_smiles(smiles).to_smiles(mapped=True)  # type: ignore[no-any-return]
