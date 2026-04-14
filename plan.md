# Plan: `seamless-signature` — YAML schema language for function signatures

## Context

Seamless needs a precise schema language to describe the signatures of functions whose parameters are numpy scalars and arrays. The schema must be expressive enough that a C header declaring the equivalent `transform()`-style function can be generated deterministically from it. The legacy compiled-transformer pipeline ([legacy-seamless/seamless/workflow/graphs/compiled_transformer/gen_header.py](legacy-seamless/seamless/workflow/graphs/compiled_transformer/gen_header.py)) did something similar by piggy-backing on JSON-schema + Silk "form" metadata, but that design wraps every array in a runtime `{data, shape[], strides[]}` struct, which is heavy and doesn't match how C programmers naturally write compiled kernels.

The new schema introduces two key simplifications:

1. **Named wildcard dimensions** (`"X"`, `"Y"`, …) appear as plain integer parameters in the C signature, shared across the whole signature so the same name always denotes the same length.
2. **Trailing fixed dimensions** collapse into a generated C typedef, so `dtype=int32, shape=("X","Y",3)` yields `typedef int32_t int32_3[3];` and a parameter `const int32_3 *arr` of logical length `X*Y`.

Structured numpy dtypes are supported as recursive nested field definitions, which map directly to C `typedef struct`s.

Design follows the YAML-to-dataclass idiom already used in [seamless-config/seamless_config/cluster.py](seamless-config/seamless_config/cluster.py): plain `@dataclass`, `from_dict` classmethods, `__post_init__` validation, `yaml.safe_load`, no dacite/pydantic.

## Package layout

New sibling package `seamless-signature/` (alongside `seamless-config/`, `seamless-transformer/`):

```
seamless-signature/
  pyproject.toml
  seamless_signature/
    __init__.py          # public API re-exports
    dtype.py             # DType parsing (scalar names + structured)
    schema.py            # Parameter, Signature dataclasses + from_dict
    loader.py            # load_signature(yaml_path) entry point
    c_header.py          # generate_header(Signature) -> str
  tests/
    fixtures/
      simple.yaml
      wildcard.yaml
      structured.yaml
      outputs.yaml
    test_schema.py
    test_c_header.py
```

## YAML schema

Top level has `inputs` and `outputs`, each a list of parameter entries. One file = one function signature (`function_name` at top level).

```yaml
function_name: rmsd
inputs:
  - name: coords1
    dtype: float32
    shape: [X, Y, 3]
  - name: coords2
    dtype: float32
    shape: [X, Y, 3]
  - name: weight            # scalar: no `shape` key
    dtype: float64
  - name: residues          # structured dtype, inline
    dtype:
      fields:
        - {name: resname, dtype: char,    shape: [4]}
        - {name: pos,     dtype: float32, shape: [3]}
        - {name: mass,    dtype: float32}
    shape: [X, Y]
outputs:
  - name: result
    dtype: float64          # scalar output → pointer-out parameter
```

### Rules (enforced in `__post_init__`)

- `shape` absent → scalar; `shape` present and non-empty → array. Empty shape `[]` is an error.
- Each shape entry is either a positive `int` (fixed dim) or a `str` matching `[A-Za-z_][A-Za-z0-9_]*` (wildcard dim name).
- **Trailing-fixed rule**: walking a shape from the right, the maximal run of consecutive fixed ints is the *element shape*; the rest must be *all wildcards*. A fixed dim appearing before any wildcard (e.g. `[X, 3, Y]`) is a `TypeError`.
- Wildcard dim names are collected signature-wide into a single set. Every unique name becomes one `unsigned int` parameter prepended to the C signature, in first-seen order across `inputs` then `outputs`.
- Inside structured-dtype fields, `shape` entries must be fixed ints (structs have fixed layout).
- `dtype` is either a string (recognized numpy scalar name) or a mapping with a `fields` list (each entry itself has `name`, `dtype`, optional `shape`). Recursive.
- Recognized scalar dtypes: `int8/16/32/64`, `uint8/16/32/64`, `float32/64`, `bool`, `char`, `complex64`, `complex128`.
- Arrays are assumed strictly C-contiguous; no stride support. Runtime assertion belongs to the consumer, not this package.

## Dataclasses ([seamless_signature/schema.py](seamless-signature/seamless_signature/schema.py))

```python
DTypeSpec = Union[ScalarDType, StructDType]  # discriminated

@dataclass
class ScalarDType:
    name: str                    # 'int32', 'float64', ...

@dataclass
class StructField:
    name: str
    dtype: DTypeSpec
    shape: tuple[int, ...] = ()  # only fixed dims allowed

@dataclass
class StructDType:
    fields: list[StructField]
    # __post_init__: reject wildcard dims in any nested field

@dataclass
class Parameter:
    name: str
    dtype: DTypeSpec
    shape: tuple[int | str, ...] | None = None  # None = scalar
    # __post_init__: enforce trailing-fixed rule

    @property
    def element_shape(self) -> tuple[int, ...]:
        """The trailing fixed dims (possibly empty)."""
    @property
    def wildcard_dims(self) -> tuple[str, ...]:
        """The leading wildcard dims (possibly empty)."""

    @classmethod
    def from_dict(cls, d: dict) -> "Parameter": ...

@dataclass
class Signature:
    function_name: str
    inputs: list[Parameter]
    outputs: list[Parameter]

    @property
    def wildcard_names(self) -> tuple[str, ...]:
        """Unique wildcard dim names in first-seen order across inputs+outputs."""

    @classmethod
    def from_dict(cls, d: dict) -> "Signature": ...
```

DType parsing lives in [seamless_signature/dtype.py](seamless-signature/seamless_signature/dtype.py) with a single `parse_dtype(node) -> DTypeSpec` that handles the string-or-mapping union, recurses into `fields`, and is reused by `Parameter.from_dict` and `StructField.from_dict`.

## Loader ([seamless_signature/loader.py](seamless-signature/seamless_signature/loader.py))

```python
def load_signature(path: str | Path) -> Signature:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Signature.from_dict(data)
```

Mirrors the entry-point style of [seamless-config/seamless_config/config_files.py](seamless-config/seamless_config/config_files.py) (`load_config_files`).

## C header generation ([seamless_signature/c_header.py](seamless-signature/seamless_signature/c_header.py))

```python
def generate_header(sig: Signature) -> str: ...
```

**Algorithm:**

1. Emit preamble: comment block + `#include <stdint.h>` + `#include <stdbool.h>`. (Reuse the phrasing style from legacy [gen_header.py](legacy-seamless/seamless/workflow/graphs/compiled_transformer/gen_header.py) lines 3-14.)
2. Walk every `Parameter`:
   - For any structured dtype encountered, recursively emit `typedef struct { … } <StructName>;`. Struct name derivation mirrors legacy `gen_struct_name` (path-based CamelCase + `Struct` suffix) in [gen_header.py:35-41](legacy-seamless/seamless/workflow/graphs/compiled_transformer/gen_header.py#L35-L41) — reused verbatim where sensible.
   - For any non-empty `element_shape` on a scalar dtype, emit a typedef named `<dtype>_<d1>[x<d2>...]`, e.g. `typedef int32_t int32_3[3];`, `typedef float float32_4x4[4][4];`. De-duplicate across parameters.
3. Emit `int transform(...)` with args in this order:
   - Every unique wildcard dim name, as `unsigned int <name>` (type choice matches legacy `shape[]` element type in [gen_header.py:114](legacy-seamless/seamless/workflow/graphs/compiled_transformer/gen_header.py#L114)).
   - Each input parameter:
     - Scalar: `<ctype> <name>` by value (`const` not needed — passed by value).
     - Array: `const <element_typedef> *<name>` (where element_typedef is the scalar C name or the generated typedef/struct name).
   - Each output parameter:
     - Scalar: `<ctype> *<name>` (pointer-out).
     - Array: `<element_typedef> *<name>` (non-const, pointer-out; caller pre-allocates length = product of wildcard dims).

**Scalar name → C type mapping** (reuse the legacy table in [gen_header.py:21-32](legacy-seamless/seamless/workflow/graphs/compiled_transformer/gen_header.py#L21-L32), simplified):

```python
SCALAR_C_TYPES = {
    "int8":  "int8_t",  "int16":  "int16_t",  "int32":  "int32_t",  "int64":  "int64_t",
    "uint8": "uint8_t", "uint16": "uint16_t", "uint32": "uint32_t", "uint64": "uint64_t",
    "float32": "float", "float64": "double",
    "bool": "bool", "char": "char",
    "complex64": "_Complex float", "complex128": "_Complex double",
}
```

## Example: end-to-end

**Input** `rmsd.yaml`:

```yaml
function_name: rmsd
inputs:
  - {name: coords1, dtype: float32, shape: [X, Y, 3]}
  - {name: coords2, dtype: float32, shape: [X, Y, 3]}
  - {name: weight,  dtype: float64}
outputs:
  - {name: result,  dtype: float64}
```

**Generated header:**

```c
/* Auto-generated from rmsd.yaml — do not edit. */
#include <stdint.h>
#include <stdbool.h>

typedef float float32_3[3];

int transform(
    unsigned int X, unsigned int Y,
    const float32_3 *coords1,
    const float32_3 *coords2,
    double weight,
    double *result
);
```

## Tests (in `seamless-signature/tests/`)

- `test_schema.py`
  - Round-trip `load_signature` on each fixture YAML.
  - Assert `wildcard_names`, `element_shape`, and structured-field traversal.
  - Negative cases: `[X, 3, Y]` (fixed before wildcard), `shape: []` (empty list), wildcard dim inside struct field, unknown scalar dtype name.
- `test_c_header.py`
  - Golden-file comparison for `simple.yaml`, `wildcard.yaml`, `structured.yaml`, `outputs.yaml`.
  - Assert that identical `(dtype, element_shape)` pairs produce exactly one typedef (de-dup).
  - Sanity check: generated header parses through `gcc -fsyntax-only -xc -` (only if gcc is available; skip otherwise).

## Critical files to read before implementation

- [seamless-config/seamless_config/cluster.py](seamless-config/seamless_config/cluster.py) — canonical `from_dict` + `__post_init__` pattern to copy.
- [seamless-config/seamless_config/config_files.py:187-209](seamless-config/seamless_config/config_files.py#L187-L209) — loader entry-point idiom.
- [seamless-config/pyproject.toml](seamless-config/pyproject.toml) — package metadata template.
- [legacy-seamless/seamless/workflow/graphs/compiled_transformer/gen_header.py](legacy-seamless/seamless/workflow/graphs/compiled_transformer/gen_header.py) — reference for scalar→C-type table, struct-name generation, and preamble phrasing. Do **not** copy the `{data, shape[], strides[]}` wrapping or the Silk `form`/`storage` machinery — the new design deliberately avoids both.

## Verification

1. `cd seamless-signature && pip install -e .` inside the `seamless1` conda env.
2. `pytest seamless-signature/tests/ -v` — all tests pass.
3. Hand-spot-check generated headers for the four fixtures against the in-plan example above.
4. If `gcc` is available in the env, run `gcc -fsyntax-only -xc - < generated.h` on each fixture output to confirm the C is well-formed.

## Explicit non-goals (for this first iteration)

- No stride support (C-contiguous only).
- No integer-constant dim names (e.g. `MAX_N`); only wildcard identifiers or literal ints.
- No cross-file imports / named struct reuse across signatures — structured dtypes are inline-only.
- No CFFI/ctypes runtime binding; only header text generation. Runtime consumption is a separate concern handled by whoever calls this package.
- No replacement of the legacy compiled-transformer pipeline; this package stands alone.
