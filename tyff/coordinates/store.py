"""
Coordinate database for storing pre-equilibrated molecular simulation boxes.

This module provides a SQLite-based storage system for molecular simulation
coordinates with searchable metadata including SMILES, thermodynamic states,
force fields, and energies.
"""

import pathlib

import openmm
from openff.toolkit import ForceField
from sqlmodel import Field, Session, SQLModel, create_engine, func, select

from .box import BoxCoordinates, CoordinatesDB, Substance


class DBMetadata(SQLModel, table=True):
    """Database metadata for versioning."""

    __tablename__ = "metadata"

    key: str = Field(primary_key=True)
    value: str


class CoordinateStore:
    """
    SQLite database for storing and querying simulation box coordinates.

    This class provides an API for storing, retrieving, and searching molecular
    simulation coordinates with associated metadata.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: pathlib.Path | str):
        """
        Initialize the coordinate store.

        Parameters
        ----------
        db_path : pathlib.Path | str
            Path to SQLite database file
        """
        self.db_path = pathlib.Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine
        self.engine = create_engine(f"sqlite:///{self.db_path}", connect_args={"check_same_thread": False})

        # Create tables
        SQLModel.metadata.create_all(self.engine)

        # Initialize and validate metadata
        with Session(self.engine) as session:
            existing = session.exec(select(DBMetadata).where(DBMetadata.key == "schema_version")).first()
            if not existing:
                session.add(DBMetadata(key="schema_version", value=str(self.SCHEMA_VERSION)))
                session.commit()
            else:
                existing_version = int(existing.value)
                if existing_version != self.SCHEMA_VERSION:
                    raise ValueError(
                        f"Database schema version mismatch: "
                        f"database has version {existing_version}, "
                        f"but code expects version {self.SCHEMA_VERSION}"
                    )

    def add(self, box: BoxCoordinates) -> int:
        """
        Add a box to the database.

        Parameters
        ----------
        box : BoxCoordinates
            BoxCoordinates object to store

        Returns
        -------
        int
            Database ID of the added box
        """
        with Session(self.engine) as session:
            # Insert main record
            db_coord = box._to_db_model()

            # TODO: deduplicate
            session.add(db_coord)
            session.commit()
            session.refresh(db_coord)

            coord_id = db_coord.id

            session.commit()

        return coord_id  # type: ignore[return-value]

    def get(self, box_id: int) -> BoxCoordinates | None:
        """
        Retrieve a box by its database ID.

        Parameters
        ----------
        box_id : int
            Database ID

        Returns
        -------
        BoxCoordinates or None
            BoxCoordinates object or None if not found
        """
        with Session(self.engine) as session:
            db_coord = session.get(CoordinatesDB, box_id)
            if not db_coord:
                return None

            return BoxCoordinates._from_db_model(db_coord)

    def get_box_matches_by_substance(
        self,
        substance: Substance,
        temperature: float | None = None,
        temperature_tolerance: float | None = None,
        pressure: float | None = None,
        pressure_tolerance: float | None = None,
    ) -> list[BoxCoordinates]:
        """
        Retrieve boxes matching the given substance and optional state filters.

        Parameters
        ----------
        substance : Substance
            Substance object defining the composition to match
        temperature : float | None, optional
            Target temperature in Kelvin.
            If temperature is None and temperature_tolerance is None,
            this filter will be ignored.
            If temperature is None and temperature_tolerance is not None,
            this will be treated as a filter for any temperature.
        temperature_tolerance : float | None, optional
            Temperature tolerance in Kelvin.
        pressure : float | None, optional
            Target pressure in atm.  # NOTE: elsewhere tyff uses kPa
            If pressure is None and pressure_tolerance is None,
            this filter will be ignored.
            If pressure is None and pressure_tolerance is not None,
            this will be treated as a filter for any pressure.
        pressure_tolerance : float | None, optional
            Pressure tolerance in atm

        Returns
        -------
        list[BoxCoordinates]
            List of matching BoxCoordinates objects
        """
        comp_key = substance.to_composition_key()

        with Session(self.engine) as session:
            query = select(CoordinatesDB).where(CoordinatesDB.composition_key == comp_key)

            if temperature is not None and temperature_tolerance is not None:
                query = query.where(
                    CoordinatesDB.temperature.between(
                        temperature - temperature_tolerance,
                        temperature + temperature_tolerance,
                    )
                )
            elif temperature_tolerance is not None:
                query = query.where(CoordinatesDB.temperature == temperature)

            if pressure is not None and pressure_tolerance is not None:
                query = query.where(
                    CoordinatesDB.pressure.between(pressure - pressure_tolerance, pressure + pressure_tolerance)  # type: ignore[union-attr]
                )
            elif pressure_tolerance is not None:
                query = query.where(CoordinatesDB.pressure == pressure)

            results = session.exec(query).all()

        return [BoxCoordinates._from_db_model(db_coord) for db_coord in results]

    def get_box_matches_by_box(
        self,
        box: BoxCoordinates,
        temperature_tolerance: float = 1.0,
        pressure_tolerance: float = 0.001,
    ) -> list[BoxCoordinates]:
        """
        Retrieve boxes matching the given box's composition and optional state filters.

        Parameters
        ----------
        box : BoxCoordinates
            BoxCoordinates object defining the composition to match
        temperature_tolerance : float, default=1.0
            Temperature tolerance in Kelvin
        pressure_tolerance : float, default=0.001
            Pressure tolerance in atm
        pressure_tolerance : float, default=0.001
            Pressure tolerance in atm

        Returns
        -------
        list[BoxCoordinates]
            List of matching BoxCoordinates objects
        """

        return self.find_box_matches_by_substance(  # type: ignore
            substance=box.substance,
            temperature=box.temperature,
            temperature_tolerance=temperature_tolerance,
            pressure=box.pressure,
            pressure_tolerance=pressure_tolerance,
        )

    def get_lowest_energy_box_by_system(
        self,
        box: BoxCoordinates,
        openmm_system: "openmm.System",
        temperature: float | None = None,
        temperature_tolerance: float | None = None,
        pressure: float | None = None,
        pressure_tolerance: float | None = None,
    ) -> BoxCoordinates | None:
        """
        Retrieve the lowest energy box matching the given box's composition
        and optional state filters.
        Energies are calculated using the provided OpenMM System.

        Parameters
        ----------
        box : BoxCoordinates
            BoxCoordinates object defining the composition to match
        openmm_system : openmm.System
            OpenMM System to recalculate energies
        temperature : float | None, optional
            Target temperature in Kelvin (None = ignore)
        temperature_tolerance : float, default=1.0
            Temperature tolerance in Kelvin
        pressure : float | None, optional
            Target pressure in atm (None = ignore)
        pressure_tolerance : float, default=0.001
            Pressure tolerance in atm

        Returns
        -------
        BoxCoordinates or None
            Lowest energy BoxCoordinates object or None if no matches found
        """

        matches = self.get_box_matches_by_substance(
            substance=box.substance,
            temperature=temperature,
            temperature_tolerance=temperature_tolerance,
            pressure=pressure,
            pressure_tolerance=pressure_tolerance,
        )

        if not matches:
            return None

        matches_and_energies = []
        for match in matches:
            box_with_coords = box.with_coordinates_from_other(match)
            matches_and_energies.append((box_with_coords, box_with_coords.get_energy_for_system(openmm_system)))
        lowest_box, _ = min(matches_and_energies, key=lambda be: be[1])
        return lowest_box

    def get_lowest_energy_box_by_force_field(
        self,
        box: BoxCoordinates,
        forcefield: ForceField,
        temperature: float | None = None,
        temperature_tolerance: float = 1.0,
        pressure: float | None = None,
        pressure_tolerance: float = 0.001,
    ) -> BoxCoordinates | None:
        """
        Retrieve the lowest energy box matching the given box's composition,
        force field, and optional state filters.

        Parameters
        ----------
        box : BoxCoordinates
            BoxCoordinates object defining the composition to match
        force_field_id : str
            Force field identifier to filter by
        temperature : float | None, optional
            Target temperature in Kelvin (None = ignore)
        temperature_tolerance : float, default=1.0
            Temperature tolerance in Kelvin
        pressure : float | None, optional
            Target pressure in atm (None = ignore)
        pressure_tolerance : float, default=0.001
            Pressure tolerance in atm

        Returns
        -------
        BoxCoordinates or None
            Lowest energy BoxCoordinates object or None if no matches found
        """
        openmm_system = forcefield.create_openmm_system(box.substance.to_openff_topology())
        return self.get_lowest_energy_box_by_system(
            box=box,
            openmm_system=openmm_system,
            temperature=temperature,
            temperature_tolerance=temperature_tolerance,
            pressure=pressure,
            pressure_tolerance=pressure_tolerance,
        )

    def get_force_fields(self) -> list[str]:
        """
        Get list of all force fields in the database.

        Returns
        -------
        list[str]
            List of force field identifiers
        """
        with Session(self.engine) as session:
            ffs = session.exec(
                select(CoordinatesDB.force_field_id).distinct().order_by(CoordinatesDB.force_field_id)
            ).all()
        return list(ffs)  # type: ignore[arg-type]

    def get_compositions(self) -> list[str]:
        """
        Get list of all unique compositions in the database.

        Returns
        -------
        list[str]
            List of composition keys
        """
        with Session(self.engine) as session:
            comps = session.exec(
                select(CoordinatesDB.composition_key).distinct().order_by(CoordinatesDB.composition_key)
            ).all()
        return list(comps)

    def count(self) -> int:
        """
        Get total number of boxes in the database.

        Returns
        -------
        int
            Total number of boxes
        """
        with Session(self.engine) as session:
            count = session.exec(select(func.count(CoordinatesDB.id))).one()  # type: ignore
        return count

    def delete(self, box_id: int):
        """
        Delete a box from the database.

        Parameters
        ----------
        box_id : int
            Database ID of box to delete
        """
        with Session(self.engine) as session:
            db_coord = session.get(CoordinatesDB, box_id)
            if db_coord:
                session.delete(db_coord)
                session.commit()

    def export_to_db(self, target_db_path: pathlib.Path, ids: list[int] | None = None):
        """
        Export boxes to another database file.

        Parameters
        ----------
        target_db_path : pathlib.Path
            Path to target database
        ids : list[int], optional
            Optional list of IDs to export (None = export all)
        """
        target_store = CoordinateStore(target_db_path)

        if ids is None:
            with Session(self.engine) as session:
                ids = list(session.exec(select(CoordinatesDB.id)).all())  # type: ignore[arg-type]

        for box_id in ids:
            box = self.get(box_id)
            if box:
                box.id = None  # Reset ID for new database
                target_store.add(box)

    def merge_from_db(self, source_store: "CoordinateStore", deduplicate: bool = True):
        """
        Merge boxes from another database into this one.

        Parameters
        ----------
        source_db : CoordinateStore
            Source CoordinateStore instance
        deduplicate : bool, default=True
            If True, skip boxes with identical composition, state, and force field
        """

        with Session(source_store.engine) as session:
            source_ids = list(session.exec(select(CoordinatesDB.id)).all())

        for box_id in source_ids:
            box = source_store.get(box_id)  # type: ignore[arg-type]
            if not box:
                continue

            if deduplicate:
                # Check if identical box already exists
                raise NotImplementedError("Deduplication not implemented yet")

            box.id = None  # Reset ID for new database
            self.add(box)
