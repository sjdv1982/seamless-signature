from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dtype import DTypeSpec, IDENTIFIER_RE, parse_dtype


ShapeDim = int | str


@dataclass(frozen=True)
class Parameter:
    name: str
    dtype: DTypeSpec
    shape: tuple[ShapeDim, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not IDENTIFIER_RE.match(self.name):
            raise TypeError(f"Parameter name must be a valid C identifier: {self.name!r}")
        if self.shape is None:
            return
        if not isinstance(self.shape, tuple):
            raise TypeError(f"Parameter {self.name!r} shape must be a tuple")
        if not self.shape:
            raise TypeError(f"Parameter {self.name!r} shape must not be empty")

        seen_wildcard_from_right = False
        for dim in reversed(self.shape):
            if _is_positive_int(dim):
                if seen_wildcard_from_right:
                    raise TypeError(
                        f"Parameter {self.name!r} has fixed dimensions before wildcard dimensions"
                    )
            elif isinstance(dim, str) and IDENTIFIER_RE.match(dim):
                seen_wildcard_from_right = True
            else:
                raise TypeError(
                    f"Parameter {self.name!r} shape entries must be positive integers "
                    "or valid wildcard dimension names"
                )

    @property
    def element_shape(self) -> tuple[int, ...]:
        if self.shape is None:
            return ()
        result = []
        for dim in reversed(self.shape):
            if _is_positive_int(dim):
                result.append(dim)
            else:
                break
        return tuple(reversed(result))

    @property
    def wildcard_dims(self) -> tuple[str, ...]:
        if self.shape is None:
            return ()
        wildcard_count = len(self.shape) - len(self.element_shape)
        return tuple(dim for dim in self.shape[:wildcard_count] if isinstance(dim, str))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Parameter":
        if not isinstance(d, dict):
            raise TypeError("Parameter entry must be a mapping")
        params = d.copy()
        if "name" not in params:
            raise TypeError("Parameter is missing required key 'name'")
        if "dtype" not in params:
            raise TypeError(f"Parameter {params.get('name')!r} is missing required key 'dtype'")
        params["dtype"] = parse_dtype(params["dtype"])
        if "shape" in params:
            params["shape"] = _parse_parameter_shape(params["shape"], params["name"])
        return cls(**params)


@dataclass(frozen=True)
class Signature:
    function_name: str
    inputs: list[Parameter]
    outputs: list[Parameter]

    def __post_init__(self) -> None:
        if not isinstance(self.function_name, str) or not IDENTIFIER_RE.match(self.function_name):
            raise TypeError(
                f"Function name must be a valid C identifier: {self.function_name!r}"
            )
        if not isinstance(self.inputs, list):
            raise TypeError("'inputs' must be a list")
        if not isinstance(self.outputs, list):
            raise TypeError("'outputs' must be a list")
        for parameter in self.inputs + self.outputs:
            if not isinstance(parameter, Parameter):
                raise TypeError("Signature inputs and outputs must contain Parameter instances")
        if set(self.input_wildcards) & set(self.output_wildcards):
            raise TypeError("Wildcard dimensions cannot be both input-derived and output-only")

    @property
    def wildcard_names(self) -> tuple[str, ...]:
        names = []
        for parameter in self.inputs + self.outputs:
            for dim in parameter.wildcard_dims:
                if dim not in names:
                    names.append(dim)
        return tuple(names)

    @property
    def input_wildcards(self) -> tuple[str, ...]:
        return _unique_wildcards(self.inputs)

    @property
    def output_wildcards(self) -> tuple[str, ...]:
        input_wildcards = set(self.input_wildcards)
        names = []
        for parameter in self.outputs:
            for dim in parameter.wildcard_dims:
                if dim not in input_wildcards and dim not in names:
                    names.append(dim)
        return tuple(names)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Signature":
        if not isinstance(d, dict):
            raise TypeError("Signature document must be a mapping")
        unknown = set(d) - {"function_name", "inputs", "outputs"}
        if unknown:
            raise TypeError(f"Unknown signature keys: {sorted(unknown)!r}")
        if "function_name" not in d:
            raise TypeError("Signature is missing required key 'function_name'")
        inputs = d.get("inputs", [])
        outputs = d.get("outputs", [])
        if not isinstance(inputs, list):
            raise TypeError("'inputs' must be a list")
        if not isinstance(outputs, list):
            raise TypeError("'outputs' must be a list")
        return cls(
            function_name=d["function_name"],
            inputs=[Parameter.from_dict(parameter) for parameter in inputs],
            outputs=[Parameter.from_dict(parameter) for parameter in outputs],
        )


def _parse_parameter_shape(shape: Any, parameter_name: str) -> tuple[ShapeDim, ...]:
    if not isinstance(shape, list):
        raise TypeError(f"Parameter {parameter_name!r} shape must be a list")
    if not shape:
        raise TypeError(f"Parameter {parameter_name!r} shape must not be empty")
    return tuple(shape)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _unique_wildcards(parameters: list[Parameter]) -> tuple[str, ...]:
    names = []
    for parameter in parameters:
        for dim in parameter.wildcard_dims:
            if dim not in names:
                names.append(dim)
    return tuple(names)
