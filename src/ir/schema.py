"""Phase 2.1 - loading and validating against the DreamScript IR JSON Schemas.

The schemas in `schemas/` are the contract; this module is the only place that knows how to
resolve the `$ref`s between them. Everything else calls `validate_node`, `validate_edge` or
`validate_diagram` and gets back a list of human-readable problems.

    python -m src.ir.schema            # validate the schemas themselves
"""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from typing import Any

import jsonschema
from referencing import Registry, Resource

from src.utils.config import ROOT

SCHEMA_DIR = ROOT / "schemas"
BASE_URI = "https://dreamscript.local/schemas/"


def schema_path(name: str) -> Path:
    """`node` -> schemas/node.schema.json."""
    return SCHEMA_DIR / f"{name}.schema.json"


def load_schema(name: str) -> dict[str, Any]:
    with schema_path(name).open(encoding="utf-8") as fh:
        return json.load(fh)


@functools.cache
def _registry() -> Registry:
    """Every schema in `schemas/`, keyed by its `$id`, so `$ref` resolves offline.

    Without this, jsonschema would try to fetch `https://dreamscript.local/...` over the
    network - which fails, and would make validation depend on being online.
    """
    registry = Registry()
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        with path.open(encoding="utf-8") as fh:
            doc = json.load(fh)
        resource = Resource.from_contents(doc)
        registry = registry.with_resource(doc.get("$id", BASE_URI + path.name), resource)
    return registry


@functools.cache
def validator(name: str) -> jsonschema.protocols.Validator:
    schema = load_schema(name)
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema, registry=_registry())


def problems(name: str, instance: Any) -> list[str]:
    """Every validation failure, deepest path first, as readable one-liners."""
    out = []
    for err in sorted(validator(name).iter_errors(instance), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in err.absolute_path) or "<root>"
        out.append(f"{where}: {err.message}")
    return out


def validate_node(node: Any) -> list[str]:
    return problems("node", node)


def is_valid(name: str, instance: Any) -> bool:
    return not problems(name, instance)


def main(argv: list[str] | None = None) -> int:
    """Check that every schema on disk is itself a legal JSON Schema."""
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    ok = True
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        name = path.name.removesuffix(".schema.json")
        try:
            validator(name)
            print(f"  PASS  {path.relative_to(ROOT)}")
        except Exception as exc:  # noqa: BLE001 - the message is the output
            ok = False
            print(f"  FAIL  {path.relative_to(ROOT)}: {exc}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
