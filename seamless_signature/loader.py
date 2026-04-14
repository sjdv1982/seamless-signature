from __future__ import annotations

from pathlib import Path

import yaml

from .schema import Signature


def load_signature(path: str | Path) -> Signature:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Signature.from_dict(data)
