"""The project's split vocabulary, and the one place its two spellings are related.

There are two, and both are correct where they are:

    validation   the manifest, the feature tables, `codegen.splits`, `codegen.schema`. The
                 project's own word, and what `data/processed/manifest.parquet` holds.
    val          `data/processed/detect/index.json` and everything under `src/detect`. Not a
                 preference - ultralytics reads `val:` out of `data.yaml` and walks a directory
                 called `val`, so the exported corpus is spelled its way or it does not load.

What made this a defect rather than a fact is that **a wrong spelling filtered to nothing
instead of failing**. `corpus.pages(("validation",))` returned an empty tuple, `routing.fit`
raised "no detector output for split 'validation'" three call frames later, and a filter that
silently matches nothing is the most expensive kind of typo in a pipeline that measures things.

    from src.utils.splits import canonical, detect_name, normalise

    normalise("val")          -> "validation"     # the project's word
    detect_name("validation") -> "val"            # what is on disk under data/processed/detect
    normalise("vall")         -> ValueError       # rather than a quiet empty result

`codegen.splits` and `detect.dataset` each had half of this table. They import it now.
"""

from __future__ import annotations

#: The project's vocabulary. `codegen.schema.SPLITS` and `codegen.splits.SPLITS` are this.
CANONICAL: tuple[str, ...] = ("train", "validation", "test")

#: What the same three splits are called on disk under `data/processed/detect`, and in the
#: `data.yaml` ultralytics reads.
DETECT: tuple[str, ...] = ("train", "val", "test")

#: Every spelling seen in this repo or in the datasets it ingests, mapped to the canonical name.
ALIASES: dict[str, str] = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
    "eval": "test",
    "holdout": "test",
}

#: canonical -> the on-disk detect name.
_TO_DETECT = {"train": "train", "validation": "val", "test": "test"}


def normalise(split: object) -> str:
    """Any accepted spelling to the canonical one. Raises on anything else.

    Raising is the point. The alternative - returning None, or the input unchanged - is what
    turns a typo into an empty filter, and an empty filter into a report of nothing.
    """
    text = str(split).strip().lower()
    try:
        return ALIASES[text]
    except KeyError:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(ALIASES)}") from None


def canonical(split: object) -> str:
    """`normalise`, named for the side of the boundary it puts you on."""
    return normalise(split)


def detect_name(split: object) -> str:
    """The spelling `data/processed/detect/index.json` and `data.yaml` use."""
    return _TO_DETECT[normalise(split)]


def same(a: object, b: object) -> bool:
    """Whether two spellings name the same split. `same("val", "validation")` is True."""
    return normalise(a) == normalise(b)
