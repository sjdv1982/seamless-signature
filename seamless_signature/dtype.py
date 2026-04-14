from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypeAlias


IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SCALAR_DTYPE_NAMES = {
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "bool",
    "char",
    "complex64",
    "complex128",
}


@dataclass(frozen=True)
class ScalarDType:
    name: str

    def __post_init__(self) -> None:
        if self.name not in SCALAR_DTYPE_NAMES:
            raise TypeError(f"Unknown scalar dtype: {self.name!r}")


DTypeSpec: TypeAlias = "ScalarDType | StructDType"


@dataclass(frozen=True)
class StructField:
    name: str
    dtype: DTypeSpec
    shape: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not IDENTIFIER_RE.match(self.name):
            raise TypeError(f"Struct field name must be a valid C identifier: {self.name!r}")
        for dim in self.shape:
            if not _is_positive_int(dim):
                raise TypeError(
                    f"Struct field {self.name!r} shape entries must be positive integers"
                )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StructField":
        if not isinstance(d, dict):
            raise TypeError("Struct field entry must be a mapping")
        params = d.copy()
        if "name" not in params:
            raise TypeError("Struct field is missing required key 'name'")
        if "dtype" not in params:
            raise TypeError(f"Struct field {params.get('name')!r} is missing required key 'dtype'")
        params["dtype"] = parse_dtype(params["dtype"])
        if "shape" in params:
            params["shape"] = _parse_struct_shape(params["shape"], params["name"])
        return cls(**params)


@dataclass(frozen=True)
class StructDType:
    fields: list[StructField]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, list) or not self.fields:
            raise TypeError("Structured dtype requires a non-empty 'fields' list")
        for field in self.fields:
            if not isinstance(field, StructField):
                raise TypeError("Structured dtype fields must be StructField instances")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StructDType":
        if not isinstance(d, dict):
            raise TypeError("Structured dtype must be a mapping")
        unknown = set(d) - {"fields"}
        if unknown:
            raise TypeError(f"Unknown structured dtype keys: {sorted(unknown)!r}")
        fields = d.get("fields")
        if not isinstance(fields, list):
            raise TypeError("Structured dtype 'fields' must be a list")
        return cls(fields=[StructField.from_dict(field) for field in fields])


def parse_dtype(node: Any) -> DTypeSpec:
    if isinstance(node, str):
        return ScalarDType(node)
    if isinstance(node, dict):
        if "fields" not in node:
            raise TypeError("Dtype mapping must contain a 'fields' list")
        return StructDType.from_dict(node)
    raise TypeError("Dtype must be a scalar dtype name or a mapping with 'fields'")


def _parse_struct_shape(shape: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(shape, list):
        raise TypeError(f"Struct field {field_name!r} shape must be a list")
    if not shape:
        raise TypeError(f"Struct field {field_name!r} shape must not be empty")
    result = []
    for dim in shape:
        if not _is_positive_int(dim):
            raise TypeError(
                f"Struct field {field_name!r} shape entries must be positive integers"
            )
        result.append(dim)
    return tuple(result)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
