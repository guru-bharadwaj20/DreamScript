"""Phase 0.2.3, completed - the package entry points route to the work in the package.

    python -m src.detect                 # what this package can do
    python -m src.detect train --help    # and doing it
    python -m src.detect --print-config  # the config contract, unchanged

## Why this exists

Eleven packages shipped a `__main__.py` whose `run()` raised `StageNotImplemented`, so
`python -m src.detect` printed a resolved config and exited 2 - while `src/detect/` held sixteen
modules with working `main()` functions, every one of them reachable only by knowing its name.
Fourteen Makefile targets pointed at those eleven entry points and therefore did nothing, and
`make serve` in particular pointed at a stub while the real server was `uvicorn src.serve.api:app`
in both the Dockerfile and CI.

The stage contract itself was not wrong, only empty. `--config` and `--print-config` still work
exactly as `utils.cli.main` defines them - a first argument beginning with `-` goes there
untouched - so `configs/*.yaml` keep the reader they always had. What is new is that a first
argument that *names a module in the package* runs it.

## Discovery is by reading, not by importing

`runnable()` parses each file with `ast` and looks for a top-level `def main`. Importing them to
find out would mean importing torch, ultralytics and OpenCV in order to print a list of names,
which is the difference between a help message and a thirty-second one.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

from src.utils.config import ROOT

#: Modules that are not commands even though they define `main`. Nothing yet; the hook exists so
#: an exclusion is a named decision rather than a silent filter.
HIDDEN: frozenset[str] = frozenset()


def _summary(tree: ast.Module) -> str:
    doc = ast.get_docstring(tree) or ""
    first = doc.strip().splitlines()[0] if doc.strip() else ""
    # Module docstrings here open with "Phase 9.1 - the detector."; the phase is the useful half.
    return first.strip()


def _has_main(tree: ast.Module) -> bool:
    return any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)


def runnable(package: str) -> dict[str, str]:
    """`{command: one-line summary}` for every module in `package` with a top-level `main()`.

    Sub-packages are included under their dotted name (`primitives.arrowheads`), because that is
    what the caller would have to type anyway.
    """
    base = ROOT / Path(*package.split("."))
    if not base.is_dir():
        return {}
    found: dict[str, str] = {}
    for path in sorted(base.rglob("*.py")):
        name = path.relative_to(base).with_suffix("").as_posix().replace("/", ".")
        if name.startswith("__") or name.endswith(".__init__") or name in HIDDEN:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        if _has_main(tree):
            found[name] = _summary(tree)
    return found


def _usage(package: str, commands: dict[str, str]) -> str:
    width = max((len(name) for name in commands), default=0)
    lines = [
        f"usage: python -m {package} <command> [args ...]",
        f"       python -m {package} --print-config [--config PATH] [key=value ...]",
        "",
        f"{len(commands)} commands in {package}:" if commands else f"no commands in {package}",
    ]
    lines += [f"  {name:<{width}}  {summary}" for name, summary in commands.items()]
    lines += ["", f"`python -m {package}.<command> --help` works too; this is the index."]
    return "\n".join(lines)


def dispatch(package: str, argv: list[str] | None = None) -> int:
    """Route `python -m <package> ...` to a module in it, or to the config contract.

    Returns an exit code rather than raising, because that is what `sys.exit` in a `__main__`
    wants and because "no such command" is an answer, not a crash.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    commands = runnable(package)

    if not args:
        print(_usage(package, commands))
        return 0

    if args[0].startswith("-"):
        # The Phase 0.2.3 contract, untouched: --config, overrides, --print-config.
        from src.utils.cli import main as config_main

        return config_main(package, _no_stage, args)

    name, rest = args[0], args[1:]
    if name not in commands:
        print(f"{package}: no command {name!r}", file=sys.stderr)
        print(_usage(package, commands), file=sys.stderr)
        return 2

    module = importlib.import_module(f"{package}.{name}")
    entry = module.main
    # Most take `argv`; a few take nothing. Asking is cheaper than a convention nobody enforces.
    if inspect.signature(entry).parameters:
        return int(entry(rest) or 0)
    if rest:
        print(f"{package}.{name} takes no arguments; got {rest}", file=sys.stderr)
        return 2
    return int(entry() or 0)


def _no_stage(cfg, active):
    """The `run` handed to the config contract when no command was named.

    It resolves and prints the config, which is what `--print-config` asked for, and reports that
    the package has no single default action - it has the list `dispatch` prints.
    """
    from src.utils.cli import StageNotImplemented

    raise StageNotImplemented(
        "this package has no single default stage; name a command "
        "(run it with no arguments to see the list)"
    )


def packages() -> list[str]:
    """Every `src.*` package that ships a `__main__.py`. Used by the tests and by `make help`."""
    import src

    out = []
    for info in pkgutil.iter_modules(src.__path__):
        if info.ispkg and (ROOT / "src" / info.name / "__main__.py").is_file():
            out.append(f"src.{info.name}")
    return sorted(out)
