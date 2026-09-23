from enum import Enum

import pydantic


class Component(pydantic.BaseModel):
    class Role(Enum):
        Solvent = "solvent"
        Solute = "solute"

        Ligand = "ligand"
        Receptor = "receptor"

    model_config = pydantic.ConfigDict(frozen=True)
    smiles: str
    role: Role = pydantic.Field(default_factory=lambda: Component.Role.Solvent)

    @property
    @pydantic.computed_field
    def identifier(self):
        """str: A unique identifier for this component."""
        return f"{self.smiles}{{{self.role.value}}}"

    @pydantic.field_validator("smiles", mode="before")
    @classmethod
    def standardize_smiles(cls, smiles: str) -> str:
        """Standardizes a SMILES pattern to be canonical (but not necessarily isomeric)
        using the OpenFF Toolkit.

        Parameters
        ----------
        smiles: str
            The SMILES pattern to standardize.

        Returns
        -------
        The standardized SMILES pattern.
        """
        from openff.toolkit.topology import Molecule
        from openff.toolkit.utils.rdkit_wrapper import RDKitToolkitWrapper
        from openff.toolkit.utils.toolkit_registry import ToolkitRegistry

        # This parsing was previously done with `cmiles.utils.load_molecule`, which
        # * did NOT enforce stereochemistry while parsing SMILES and
        # * implicitly used the same toolkit to write the SMILES back from an object
        # This is hard-coded to keep test results consistent across OpenEye status
        # and compared to older versions; if desired this could be relaxed
        rdkit_registry = ToolkitRegistry(toolkit_precedence=[RDKitToolkitWrapper()])

        molecule = Molecule.from_smiles(
            smiles,
            toolkit_registry=rdkit_registry,
            allow_undefined_stereo=True,
        )

        try:
            # Try to make the smiles isomeric.
            smiles = molecule.to_smiles(
                isomeric=True,
                explicit_hydrogens=False,
                mapped=False,
                toolkit_registry=rdkit_registry,
            )
        except ValueError:
            # Fall-back to non-isomeric.
            smiles = molecule.to_smiles(
                isomeric=False,
                explicit_hydrogens=False,
                mapped=False,
                toolkit_registry=rdkit_registry,
            )

        return smiles
