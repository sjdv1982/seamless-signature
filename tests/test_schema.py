from __future__ import annotations

from pathlib import Path

import pytest

from seamless_signature import ScalarDType, StructDType, load_signature
from seamless_signature.schema import Signature


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    "fixture",
    [
        "simple.yaml",
        "wildcard.yaml",
        "structured.yaml",
        "outputs.yaml",
        "static_return.yaml",
        "dynamic_return.yaml",
    ],
)
def test_load_signature_fixtures(fixture: str) -> None:
    sig = load_signature(FIXTURES / fixture)

    assert all(parameter.name for parameter in sig.inputs + sig.outputs)


def test_wildcard_names_and_element_shape() -> None:
    sig = load_signature(FIXTURES / "simple.yaml")

    assert sig.wildcard_names == ("X", "Y")
    assert sig.inputs[0].wildcard_dims == ("X", "Y")
    assert sig.inputs[0].element_shape == (3,)
    assert sig.inputs[1].element_shape == (3,)
    assert sig.inputs[2].shape is None


def test_wildcards_are_deduplicated_in_first_seen_order() -> None:
    sig = load_signature(FIXTURES / "wildcard.yaml")

    assert sig.wildcard_names == ("Rows", "Cols")
    assert sig.input_wildcards == ("Rows", "Cols")
    assert sig.output_wildcards == ()
    assert sig.inputs[0].wildcard_dims == ("Rows", "Cols")
    assert sig.inputs[1].wildcard_dims == ("Cols",)
    assert sig.outputs[0].wildcard_dims == ("Rows",)


def test_static_return_reuses_input_wildcard() -> None:
    sig = load_signature(FIXTURES / "static_return.yaml")

    assert sig.wildcard_names == ("N",)
    assert sig.input_wildcards == ("N",)
    assert sig.output_wildcards == ()


def test_dynamic_return_classifies_output_only_wildcard() -> None:
    sig = load_signature(FIXTURES / "dynamic_return.yaml")

    assert sig.wildcard_names == ("N", "K")
    assert sig.input_wildcards == ("N",)
    assert sig.output_wildcards == ("K",)


def test_output_wildcards_are_deduplicated_in_first_seen_order() -> None:
    sig = Signature.from_dict(
        {
            "inputs": [{"name": "values", "dtype": "float32", "shape": ["N"]}],
            "outputs": [
                {"name": "first", "dtype": "float32", "shape": ["K", "L"]},
                {"name": "second", "dtype": "float32", "shape": ["L"]},
                {"name": "third", "dtype": "float32", "shape": ["N", "K"]},
            ],
        }
    )

    assert sig.input_wildcards == ("N",)
    assert sig.output_wildcards == ("K", "L")


def test_structured_dtype_fields() -> None:
    sig = load_signature(FIXTURES / "structured.yaml")
    dtype = sig.inputs[0].dtype

    assert isinstance(dtype, StructDType)
    assert [field.name for field in dtype.fields] == ["resname", "pos", "mass"]
    assert dtype.fields[0].shape == (4,)
    assert dtype.fields[1].shape == (3,)
    assert dtype.fields[2].shape == ()
    assert isinstance(dtype.fields[0].dtype, ScalarDType)


@pytest.mark.parametrize(
    "shape",
    [
        ["X", 3, "Y"],
        [3, "X"],
    ],
)
def test_fixed_dimension_before_wildcard_rejected(shape: list[int | str]) -> None:
    with pytest.raises(TypeError, match="fixed dimensions before wildcard"):
        Signature.from_dict(
            {
    
                "inputs": [{"name": "arr", "dtype": "float32", "shape": shape}],
                "outputs": [],
            }
        )


def test_empty_parameter_shape_rejected() -> None:
    with pytest.raises(TypeError, match="must not be empty"):
        Signature.from_dict(
            {
    
                "inputs": [{"name": "arr", "dtype": "float32", "shape": []}],
                "outputs": [],
            }
        )


def test_wildcard_inside_struct_field_rejected() -> None:
    with pytest.raises(TypeError, match="shape entries must be positive integers"):
        Signature.from_dict(
            {
    
                "inputs": [
                    {
                        "name": "arr",
                        "dtype": {
                            "fields": [
                                {"name": "field", "dtype": "float32", "shape": ["X"]}
                            ]
                        },
                        "shape": ["X"],
                    }
                ],
                "outputs": [],
            }
        )


def test_unknown_scalar_dtype_rejected() -> None:
    with pytest.raises(TypeError, match="Unknown scalar dtype"):
        Signature.from_dict(
            {
    
                "inputs": [{"name": "x", "dtype": "float16"}],
                "outputs": [],
            }
        )
