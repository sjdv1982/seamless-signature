from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from seamless_signature import generate_header, load_signature
from seamless_signature.schema import Signature


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("fixture", "golden"),
    [
        ("simple.yaml", "simple.h"),
        ("wildcard.yaml", "wildcard.h"),
        ("structured.yaml", "structured.h"),
        ("outputs.yaml", "outputs.h"),
        ("static_return.yaml", "static_return.h"),
        ("dynamic_return.yaml", "dynamic_return.h"),
    ],
)
def test_generate_header_matches_golden(fixture: str, golden: str) -> None:
    sig = load_signature(FIXTURES / fixture)

    assert generate_header(sig) == (FIXTURES / golden).read_text()


def test_scalar_array_typedefs_are_deduplicated() -> None:
    sig = load_signature(FIXTURES / "simple.yaml")
    header = generate_header(sig)

    assert header.count("typedef float float32_3[3];") == 1


def test_structured_array_fixed_element_shape_gets_typedef() -> None:
    sig = Signature.from_dict(
        {
            "inputs": [
                {
                    "name": "records",
                    "dtype": {
                        "fields": [
                            {"name": "value", "dtype": "float32"},
                        ]
                    },
                    "shape": ["N", 3],
                }
            ],
            "outputs": [],
        }
    )

    header = generate_header(sig)

    assert "typedef struct {\n    float value;\n} RecordsStruct;" in header
    assert "typedef RecordsStruct RecordsStruct_3[3];" in header
    assert "const RecordsStruct_3 *records" in header
    if shutil.which("gcc") is not None:
        subprocess.run(
            ["gcc", "-fsyntax-only", "-xc", "-"],
            input=header,
            text=True,
            check=True,
        )


@pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc is not available")
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
def test_generated_header_is_valid_c(fixture: str) -> None:
    sig = load_signature(FIXTURES / fixture)
    header = generate_header(sig)

    subprocess.run(
        ["gcc", "-fsyntax-only", "-xc", "-"],
        input=header,
        text=True,
        check=True,
    )
