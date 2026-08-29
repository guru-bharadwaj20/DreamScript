"""Phase 0.3.3 acceptance test — the naming convention is enforceable, not just documented."""

from __future__ import annotations

import pytest

from src.utils.config import CONFIG_DIR, load_config
from src.utils.logging import start_run
from src.utils.naming import (
    NamingError,
    is_valid_run_name,
    make_run_name,
    parse_run_dir,
    parse_run_name,
)

VALID = [
    "p5-knn-k7-s42",
    "p6-mlp-adamw-s42",
    "p6-svmpoly-deg3-s42",
    "p7.3-hmm-viterbi-s42",
    "p9-yolov8-1024px-s42",
    "p11-qlearn-eps0.1-s42",
    "p12-qwen7b-lora32-s1",
    "p14-full-nohmm-s42",
]

INVALID = [
    "knn-k7-s42",  # no phase
    "p5-knn-k7",  # no seed
    "p5-KNN-k7-s42",  # uppercase
    "p5-knn k7-s42",  # space
    "phase5-knn-k7-s42",  # phase not p<number>
    "p5-knn-k7-seed42",  # seed not s<number>
    "",
]


@pytest.mark.parametrize("name", VALID)
def test_valid_names_parse(name: str):
    parsed = parse_run_name(name)
    assert str(parsed) == name
    assert is_valid_run_name(name)


@pytest.mark.parametrize("name", INVALID)
def test_invalid_names_rejected(name: str):
    assert not is_valid_run_name(name)
    with pytest.raises(NamingError):
        parse_run_name(name)


def test_parsed_fields():
    parsed = parse_run_name("p12-qwen7b-lora32-s1")
    assert parsed.phase == "12"
    assert parsed.model == "qwen7b"
    assert parsed.variant == "lora32"
    assert parsed.seed == 1


def test_make_run_name_normalizes_case():
    assert make_run_name(5, "KNN", "K7", 42) == "p5-knn-k7-s42"


def test_make_run_name_rejects_bad_parts():
    with pytest.raises(NamingError):
        make_run_name(5, "k nn", "k7", 42)


def test_run_dir_parses_back(tmp_path):
    run = start_run(name="p5-knn-k7-s42", root=tmp_path)
    run.finish()
    stamp, name = parse_run_dir(run.dir.name)
    assert len(stamp) == 15
    assert parse_run_name(name).model == "knn"


def test_config_run_names_follow_the_convention():
    """Every configured run_name must be parseable, so runs stay findable."""
    offenders = []
    for yaml_file in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = load_config(yaml_file)
        name = cfg.get("logging", {}).get("run_name")
        if name and not is_valid_run_name(name):
            offenders.append(f"{yaml_file.name}: {name}")
    assert not offenders, "config run_names violating docs/conventions.md: " + "; ".join(offenders)
