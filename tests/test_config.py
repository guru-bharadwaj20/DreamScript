"""Phase 0.2.3 acceptance test — the config system and the entry-point contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.utils.config import CONFIG_DIR, ConfigError, load_config, save_config, to_dict

PACKAGES = [
    "ingest",
    "preprocess",
    "features",
    "classify",
    "detect",
    "ocr",
    "parse",
    "rl",
    "synth",
    "eval",
    "serve",
]


def test_base_config_loads():
    cfg = load_config("base")
    assert cfg.project == "dreamscript"
    assert cfg.seed == 42
    assert cfg.paths.raw == "data/raw"


def test_defaults_are_composed():
    """A module config inherits everything from base without repeating it."""
    cfg = load_config("classify")
    assert cfg.seed == 42  # from base
    assert cfg.cv.strategy == "grouped"  # from classify


def test_dotted_overrides_win():
    cfg = load_config("classify", ["model=knn", "cv.n_splits=10", "seed=7"])
    assert cfg.model == "knn"
    assert cfg.cv.n_splits == 10
    assert cfg.seed == 7


def test_override_must_be_key_value():
    with pytest.raises(ConfigError):
        load_config("base", ["not-an-override"])


def test_missing_config_raises():
    with pytest.raises(ConfigError):
        load_config("no_such_config")


def test_config_records_its_provenance():
    cfg = load_config("preprocess")
    assert cfg._config_path.endswith("preprocess.yaml")


def test_save_config_roundtrips(tmp_path: Path):
    cfg = load_config("classify", ["model=tree"])
    out = save_config(cfg, tmp_path / "used.yaml")
    assert load_config(out).model == "tree"


def test_no_hardcoded_paths_in_configs():
    """Configs must stay portable: no absolute paths, no drive letters."""
    for yaml_file in CONFIG_DIR.glob("*.yaml"):
        text = yaml_file.read_text(encoding="utf-8")
        assert "C:\\" not in text and "/home/" not in text, f"{yaml_file} has an absolute path"


@pytest.mark.parametrize("package", PACKAGES)
def test_every_package_has_a_config(package: str):
    assert (CONFIG_DIR / f"{package}.yaml").is_file()


@pytest.mark.parametrize("package", PACKAGES)
def test_entrypoint_contract(package: str):
    """`python -m src.<pkg> --config configs/<pkg>.yaml --print-config` must work for all."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            f"src.{package}",
            "--config",
            f"configs/{package}.yaml",
            "--print-config",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "project: dreamscript" in result.stdout


def test_to_dict_is_plain():
    d = to_dict(load_config("base"))
    assert isinstance(d, dict) and d["paths"]["data"] == "data"
