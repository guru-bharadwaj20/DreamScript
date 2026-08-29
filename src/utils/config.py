"""Phase 0.2.3 — configuration loading.

Every runnable module in DreamScript takes its settings from a YAML file, never from
hardcoded constants:

    python -m src.<module> --config configs/<module>.yaml [key.path=value ...]

Configs compose through a `defaults:` list (the Hydra idea, kept small enough to read):
`configs/base.yaml` holds paths, seed and logging; each module config lists it under
`defaults` and adds its own block. Dotted `key=value` arguments override anything, so a
sweep never needs a new file.

    from src.utils.config import add_config_args, load_config

    parser = argparse.ArgumentParser()
    add_config_args(parser, default="configs/preprocess.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


class ConfigError(RuntimeError):
    """Raised when a config file is missing, malformed, or an override is invalid."""


def _resolve(path: str | Path) -> Path:
    """Accept 'preprocess', 'preprocess.yaml', 'configs/preprocess.yaml' or an absolute path."""
    p = Path(path)
    candidates = [p, p.with_suffix(".yaml"), CONFIG_DIR / p, CONFIG_DIR / p.with_suffix(".yaml")]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise ConfigError(f"config not found: {path} (looked in {CONFIG_DIR})")


def _load_with_defaults(path: Path, seen: set[Path]) -> DictConfig:
    """Load one YAML and merge it on top of everything in its `defaults:` list."""
    if path in seen:
        raise ConfigError(f"circular defaults chain at {path}")
    seen.add(path)

    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ConfigError(f"{path} must contain a mapping at the top level")

    defaults = cfg.pop("defaults", []) or []
    merged = OmegaConf.create({})
    for parent in defaults:
        merged = OmegaConf.merge(merged, _load_with_defaults(_resolve(parent), seen))
    return OmegaConf.merge(merged, cfg)


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> DictConfig:
    """Load a config, apply `key.path=value` overrides, and record where it came from."""
    resolved = _resolve(path)
    cfg = _load_with_defaults(resolved, set())

    if overrides:
        bad = [o for o in overrides if "=" not in o]
        if bad:
            raise ConfigError(f"overrides must be key=value, got: {bad}")
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    # Provenance for logs: repo-relative when possible, absolute for configs outside the tree.
    try:
        provenance = resolved.relative_to(ROOT)
    except ValueError:
        provenance = resolved
    cfg._config_path = str(provenance)  # noqa: SLF001
    return cfg


def add_config_args(parser: argparse.ArgumentParser, default: str | None = None) -> None:
    """Attach the standard --config / override arguments to an entry point."""
    parser.add_argument(
        "--config",
        default=default,
        required=default is None,
        help="path to a YAML config under configs/",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        default=[],
        metavar="key.path=value",
        help="dotted overrides applied on top of the config",
    )


def to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Plain dict, for logging or writing back out as a run record."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def save_config(cfg: DictConfig, path: str | Path) -> Path:
    """Write the fully resolved config next to a run's outputs."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return out
