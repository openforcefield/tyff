from typing import Any

from pydantic import BaseModel


class Source(BaseModel):
    pass


class MeasurementSource(Source):
    doi: str | None = None

    reference: str | None = None


class CalculationSource(Source):
    fidelity: str | None = None

    provenance: dict[str, Any] | None = None
