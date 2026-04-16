from __future__ import annotations

import re

from .dtype import DTypeSpec, ScalarDType, StructDType
from .schema import Parameter, Signature


SCALAR_C_TYPES = {
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "int64": "int64_t",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    "float32": "float",
    "float64": "double",
    "bool": "bool",
    "char": "char",
    "complex64": "_Complex float",
    "complex128": "_Complex double",
}


def generate_header(sig: Signature) -> str:
    typedefs: list[str] = []
    array_typedefs: set[tuple[str, tuple[int, ...]]] = set()
    struct_names: dict[tuple[str, ...], str] = {}

    for parameter in sig.inputs + sig.outputs:
        _collect_dtype_typedefs(parameter.dtype, (parameter.name,), typedefs, struct_names)
        if parameter.element_shape:
            base_name = _parameter_element_base_name(parameter, struct_names)
            key = (base_name, parameter.element_shape)
            if key not in array_typedefs:
                array_typedefs.add(key)
                base_ctype = _parameter_element_base_ctype(parameter, struct_names)
                typedefs.append(
                    _array_typedef(base_ctype, base_name, parameter.element_shape)
                )

    args = []
    args.extend(f"unsigned int {name}" for name in sig.input_wildcards)
    args.extend(f"unsigned int max{name}" for name in sig.output_wildcards)
    args.extend(
        _parameter_arg(parameter, const_array=True, struct_names=struct_names)
        for parameter in sig.inputs
    )
    args.extend(f"unsigned int *{name}" for name in sig.output_wildcards)
    args.extend(
        _parameter_arg(parameter, const_array=False, struct_names=struct_names)
        for parameter in sig.outputs
    )

    lines = [
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "",
    ]
    if typedefs:
        for typedef in typedefs:
            lines.extend(typedef.splitlines())
            lines.append("")
    lines.append("int transform(")
    for index, arg in enumerate(args):
        suffix = "," if index < len(args) - 1 else ""
        lines.append(f"    {arg}{suffix}")
    lines.append(");")
    return "\n".join(lines) + "\n"


def _collect_dtype_typedefs(
    dtype: DTypeSpec,
    path: tuple[str, ...],
    typedefs: list[str],
    struct_names: dict[tuple[str, ...], str],
) -> None:
    if isinstance(dtype, ScalarDType):
        return
    if path in struct_names:
        return

    struct_name = _struct_name(path)
    struct_names[path] = struct_name
    for field in dtype.fields:
        if isinstance(field.dtype, StructDType):
            _collect_dtype_typedefs(field.dtype, path + (field.name,), typedefs, struct_names)

    lines = ["typedef struct {"]
    for field in dtype.fields:
        field_type = _field_ctype(field.dtype, path + (field.name,), struct_names)
        suffix = "".join(f"[{dim}]" for dim in field.shape)
        lines.append(f"    {field_type} {field.name}{suffix};")
    lines.append(f"}} {struct_name};")
    typedefs.append("\n".join(lines))


def _parameter_arg(
    parameter: Parameter,
    *,
    const_array: bool,
    struct_names: dict[tuple[str, ...], str],
) -> str:
    ctype = _parameter_element_ctype(parameter, struct_names)
    if parameter.shape is None:
        if const_array:
            return f"{ctype} {parameter.name}"
        return f"{ctype} *{parameter.name}"
    if const_array:
        return f"const {ctype} *{parameter.name}"
    return f"{ctype} *{parameter.name}"


def _parameter_element_ctype(
    parameter: Parameter,
    struct_names: dict[tuple[str, ...], str],
) -> str:
    base_name = _parameter_element_base_name(parameter, struct_names)
    if parameter.element_shape:
        return _array_name(base_name, parameter.element_shape)
    return _parameter_element_base_ctype(parameter, struct_names)


def _parameter_element_base_name(
    parameter: Parameter,
    struct_names: dict[tuple[str, ...], str],
) -> str:
    if isinstance(parameter.dtype, ScalarDType):
        return parameter.dtype.name
    return struct_names[(parameter.name,)]


def _parameter_element_base_ctype(
    parameter: Parameter,
    struct_names: dict[tuple[str, ...], str],
) -> str:
    if isinstance(parameter.dtype, ScalarDType):
        return SCALAR_C_TYPES[parameter.dtype.name]
    return struct_names[(parameter.name,)]


def _field_ctype(
    dtype: DTypeSpec,
    path: tuple[str, ...],
    struct_names: dict[tuple[str, ...], str],
) -> str:
    if isinstance(dtype, ScalarDType):
        return SCALAR_C_TYPES[dtype.name]
    return struct_names[path]


def _array_typedef(base_ctype: str, base_name: str, element_shape: tuple[int, ...]) -> str:
    suffix = "".join(f"[{dim}]" for dim in element_shape)
    return f"typedef {base_ctype} {_array_name(base_name, element_shape)}{suffix};"


def _array_name(base_name: str, element_shape: tuple[int, ...]) -> str:
    return f"{base_name}_{'x'.join(str(dim) for dim in element_shape)}"


def _struct_name(path: tuple[str, ...]) -> str:
    return "".join(_camel_case(part) for part in path) + "Struct"


def _camel_case(value: str) -> str:
    return "".join(part.capitalize() for part in re.split(r"_+", value) if part)
