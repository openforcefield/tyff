from typing import Any

import openff.interchange
import openff.units
import torch

from tyff._models import TensorForceField, TensorPotential, TensorVSites


def _dump_tensor(t: torch.Tensor) -> dict[str, Any]:
    return {"dtype": str(t.dtype).removeprefix("torch."), "shape": list(t.shape), "data": t.flatten().tolist()}


def _load_tensor(v: Any) -> torch.Tensor:
    return torch.tensor(v["data"], dtype=getattr(torch, v["dtype"])).reshape(v["shape"])


def _dump_exceptions(d: dict[tuple[int, int], int] | None) -> list[tuple[int, int, int]] | None:
    return None if d is None else [(i, j, v) for (i, j), v in d.items()]


def _load_exceptions(v: Any) -> dict[tuple[int, int], int] | None:
    if v is None or isinstance(v, dict):
        return v
    return {(i, j): val for i, j, val in v}


def dump_tensor_potential(p: TensorPotential) -> dict:
    return {
        "handler_type": p.handler_type,
        "fn": p.fn,
        "parameters": _dump_tensor(p.parameters),
        "parameter_keys": [k.model_dump() for k in p.parameter_keys],
        "parameter_cols": list(p.parameter_cols),
        "parameter_units": [str(u) for u in p.parameter_units],
        "attributes": None if p.attributes is None else _dump_tensor(p.attributes),
        "attribute_cols": p.attribute_cols,
        "attribute_units": None if p.attribute_units is None else [str(u) for u in p.attribute_units],
        "exceptions": _dump_exceptions(p.exceptions),
    }


def load_tensor_potential(d: dict) -> TensorPotential:
    return TensorPotential(
        handler_type=d["handler_type"],
        fn=d["fn"],
        parameters=_load_tensor(d["parameters"]),
        parameter_keys=[openff.interchange.models.PotentialKey.model_validate(k) for k in d["parameter_keys"]],
        parameter_cols=tuple(d["parameter_cols"]),
        parameter_units=tuple(openff.units.unit.Unit(u) for u in d["parameter_units"]),
        attributes=None if d["attributes"] is None else _load_tensor(d["attributes"]),
        attribute_cols=tuple(d["attribute_cols"]) if d["attribute_cols"] is not None else None,
        attribute_units=(
            None if d["attribute_units"] is None else tuple(openff.units.unit.Unit(u) for u in d["attribute_units"])
        ),
        exceptions=_load_exceptions(d["exceptions"]),
    )


def dump_tensor_vsites(v: TensorVSites) -> dict:
    return {
        "keys": [k.model_dump() for k in v.keys],
        "weights": [_dump_tensor(w) for w in v.weights],
        "parameters": _dump_tensor(v.parameters),
    }


def load_tensor_vsites(d: dict) -> TensorVSites:
    return TensorVSites(
        keys=[openff.interchange.models.PotentialKey.model_validate(k) for k in d["keys"]],
        weights=[_load_tensor(w) for w in d["weights"]],
        parameters=_load_tensor(d["parameters"]),
    )


def dump_tensor_force_field(tff: TensorForceField) -> dict:
    return {
        "potentials": [dump_tensor_potential(p) for p in tff.potentials],
        "v_sites": None if tff.v_sites is None else dump_tensor_vsites(tff.v_sites),
    }


def load_tensor_force_field(d: dict) -> TensorForceField:
    return TensorForceField(
        potentials=[load_tensor_potential(p) for p in d["potentials"]],
        v_sites=None if d["v_sites"] is None else load_tensor_vsites(d["v_sites"]),
    )
