"""`configs/preprocess.yaml` and the code it describes hold the same numbers.

The config's binarisation block said `method: otsu`, `k: 0.2` and nothing else, while
`src.preprocess.binarize` also carries `PHOTO` - `sauvola`, `k=0.25`, `illumination_mode=flat`,
with a component floor of 2e-4 - which is the setting the photographic corpus is actually
processed with. So the file read as "the binarisation" when it was one of two, and every other
value in it was read by nothing at all: tune `min_component_area_frac` here, commit it, quote it
in a report, and the number it was meant to move does not move.

**The code stays the source of truth.** These are the values every reported preprocessing figure
was measured at, so the config is the published record of them rather than their origin - the
alternative is a YAML read at import time that can silently re-tune a measured pipeline. What
this module adds is that the record cannot drift: every assertion below pairs a config value with
the constant it mirrors, and fails if either side moves alone.
"""

from __future__ import annotations

import pytest
import yaml

from src.utils.config import CONFIG_DIR


@pytest.fixture(scope="module")
def preprocess() -> dict:
    raw = yaml.safe_load((CONFIG_DIR / "preprocess.yaml").read_text(encoding="utf-8"))
    return raw["preprocess"]


def test_the_default_binarisation_is_the_one_binarize_defaults_to(preprocess):
    import inspect

    from src.preprocess import binarize as mod

    block = preprocess["binarization"]
    assert block["method"] == inspect.signature(mod.binarize).parameters["method"].default
    assert block["window_size"] == mod.DEFAULT_WINDOW
    assert block["k"] == pytest.approx(mod.DEFAULT_K)


def test_the_photo_binarisation_is_binarize_photo(preprocess):
    from src.preprocess.binarize import PHOTO, PHOTO_MIN_AREA_FRAC

    block = preprocess["binarization"]["photo"]
    assert block["method"] == PHOTO["method"]
    assert block["window_size"] == PHOTO["window"]
    assert block["k"] == pytest.approx(PHOTO["k"])
    assert block["illumination_mode"] == PHOTO["illumination_mode"]
    assert block["min_component_area_frac"] == pytest.approx(PHOTO_MIN_AREA_FRAC)


def test_the_photo_block_covers_every_key_photo_has(preprocess):
    """A key added to `PHOTO` and not here would be a measured setting the record omits - which
    is exactly how `PHOTO` came to be missing from this file in the first place."""
    from src.preprocess.binarize import PHOTO

    block = preprocess["binarization"]["photo"]
    renamed = {"window": "window_size"}
    assert {renamed.get(key, key) for key in PHOTO} <= set(block)


def test_denoise_matches(preprocess):
    from src.preprocess.denoise import DEFAULT_MEDIAN, MIN_AREA_FRAC

    block = preprocess["denoise"]
    assert block["median_kernel"] == DEFAULT_MEDIAN
    assert block["min_component_area_frac"] == pytest.approx(MIN_AREA_FRAC)


def test_illumination_matches(preprocess):
    from src.preprocess.illumination import DEFAULT_CLIP_LIMIT, DEFAULT_TILE_GRID

    block = preprocess["illumination"]
    assert block["clahe_clip_limit"] == pytest.approx(DEFAULT_CLIP_LIMIT)
    assert block["clahe_tile_grid"] == DEFAULT_TILE_GRID


def test_deskew_matches(preprocess):
    from src.preprocess.deskew import MAX_ANGLE

    assert preprocess["deskew"]["max_angle_deg"] == pytest.approx(MAX_ANGLE)


def test_ruled_line_suppression_matches(preprocess):
    from src.preprocess.rules import MIN_SPAN_FRAC

    assert preprocess["ruled_line_suppression"]["min_line_length_ratio"] == pytest.approx(
        MIN_SPAN_FRAC
    )


# --- a config may not name a corpus that is not there (audit 28) ------------------------------


def test_llm_training_sources_all_exist_in_the_ir_corpus():
    """`didi` was listed as a training source. `data/raw/didi.dvc` is a pointer whose payload is
    "neither locally nor on remote", `data/processed/ir/` holds four sources and didi is not one,
    and `pairs.load` answers a request for it with nothing - so the list was a false account of
    what the model was trained on, costing nothing and claiming something."""
    from src.codegen.pairs import IR_DIR

    raw = yaml.safe_load((CONFIG_DIR / "llm.yaml").read_text(encoding="utf-8"))
    named = {s for s in raw["data"]["sources"] if s != "synthetic"} | set(
        raw["data"]["val_sources"]
    )
    on_disk = {p.name for p in IR_DIR.iterdir() if p.is_dir()} if IR_DIR.is_dir() else set()
    if not on_disk:
        pytest.skip("needs the DVC payload (data/processed/ir)")
    assert named <= on_disk, sorted(named - on_disk)


def test_the_didi_converter_is_kept_and_said_to_be_kept():
    """Removing the name from the config is not a claim that the source is abandoned - the
    converter and the loader entry are what would read the payload if it came back."""
    from src.codegen.pairs import REAL_SOURCES

    assert "didi" in REAL_SOURCES
    text = (CONFIG_DIR / "llm.yaml").read_text(encoding="utf-8")
    assert "src/ir/convert/didi.py" in text
