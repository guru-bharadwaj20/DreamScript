"""Phase 12.1.8 - train/val/test for the code pairs, held out by diagram *and* by scribe.

    python -m src.codegen.splits --self-test
    python -m src.codegen.splits --pairs data/processed/codegen/pairs.jsonl

The rule is the one Phase 1.3.3 already enforces for the whole corpus and that S1 and S3 are
held to: **a person's handwriting appears in exactly one split.** A pair split by diagram alone
would let the model read writer0065's loops in training and be scored on writer0065's loops in
test, and the Phase 14 number would measure memorised penmanship.

So this module does not invent a second split. It **adopts** the two splits this repo already
built, and only decides where they are silent:

    1. manifest       `data/processed/manifest.parquet` (written by `src.ingest.splits`) is the
                      authority for every source it covers. basis `manifest`.
    2. fa_writer_split
                      fa_bresler is not in the manifest, but it is not unsplit either:
                      `src.detect.dataset.fa_splits` is the writer-disjoint split that
                      `src/detect/**` and `src/ocr/textcrops.py` already train and score on.
                      Adopted verbatim. basis `fa_writer_split`.
    3. derived        a scribed source covered by neither would be split by calling
                      `src.ingest.splits.assign` - the corpus's own union-find, at the corpus's
                      own seed. Currently empty. basis `derived_scribe_disjoint`.
    4. no scribe      train only, whether it is machine-generated (12.1.4's synthetic graphs) or
                      merely anonymous (sketch2code). bases `synthetic_train_only` and
                      `no_scribe_train_only`.

## Why every scribe-less pair goes to train and nowhere else

A pair with no writer identity cannot be held out from a writer, so in val or test it can only
dilute the one claim those splits exist to support. Two populations are affected, and both go to
train:

    synthetic     12.1.4 renders ~10K pairs from random graphs. They come from a generator the
                  evaluator could also be handed, so a synthetic test case measures whether the
                  model learned the generator. 10K synthetic against ~150 real held-out pages
                  would also make val loss ~98.5% a number about a program that wrote itself.
    sketch2code   484 wireframe pairs whose target HTML is the dataset's own ground truth. The
                  images are hand-drawn but no writer identity was ever published, so these
                  pairs cannot be attributed and therefore cannot be held out from anyone.

The consequence is the property worth having: **every pair in validation and test has a named
scribe, and that scribe appears in no other split.** Both halves are asserted, not intended -
`summarize()["checks"]["heldout_pairs_all_have_a_scribe"]` and `no_scribe_spans_splits`.

## What it measured (2026-09-10, re-run for this docstring)

No pair file exists yet (12.1.1-12.1.6 open), so the split was built over the 1,477 pairs that
`data/processed/targets/index.json` already describes - the same ids 12.1.x will wrap - plus 600
synthetic stand-ins:

    basis                        pairs   scribes
    manifest                       693       105   hdbpmn, adopted from Phase 1.3.3
    fa_writer_split                300        25   fa_bresler, adopted from src.detect.dataset
    no_scribe_train_only           484         0   sketch2code
    synthetic_train_only           600         0

    split         pairs   fraction   scribes   scribe-less pairs
    train          1739     0.8373        85                1084
    validation      176     0.0847        23                   0
    test            162     0.0780        22                   0

    scribes spanning >1 split      0    (130 scribes over 993 pairs)
    diagrams spanning >1 split     0    (2,077 distinct diagram ids)

On the 993 scribed pairs alone - the ones the held-out claim is about - the split is
655 / 176 / 162 = 0.6596 / 0.1772 / 0.1631. The 0.8373 train fraction of the whole is
scribe-less bulk sitting on top of that, by construction.

## Two things the inherited draft had wrong, and one it had right

- **Its docstring table was not what its own code produced.** It recorded 1497/266/314 over
  "163 scribes"; that same code run here yields 1590/245/242 over 130 scribes. Every number
  above was produced by `python -m src.codegen.splits` on this tree and no other way.
- **It hashed the 484 scribe-less sketch2code pairs across all three splits**, putting 149 pairs
  with no writer into val (69) and test (80) - it reported 1590/245/242 where this module gets
  1739/176/162, and the whole of that difference is anonymous pages propping up the held-out
  sets. They weaken the only claim val and test make.
- It was right that fa_bresler is missing from the manifest. Its fix - re-deriving fa_bresler
  through `src.ingest.splits.assign` - is replaced here by adopting `fa_splits`, but the two
  were checked against each other first and **agree on all 300 diagrams and all 25 writers**
  (`fa_derivation_agreement()`). That agreement is why the swap costs nothing; adoption is still
  preferred, because agreeing today is not a guarantee of agreeing after either side changes.

## The finding: the manifest's sketch2code split is a prefix collision, not a split

`src.ingest.splits` splits writer-less rows by `int.from_bytes(id.encode()[:8], "little") % 100`.
Every sketch2code id begins `sketch2code/`, so **all 731 rows share those first 8 bytes and fall
in one bucket**: the manifest holds 727 validation, 4 test and 0 train for that source. Adopting
it would have put essentially the whole of sketch2code into validation. This module therefore
does not bridge its `s2c_<n>` pair ids onto the manifest's `sketch2code/<n>_<k>` rows, and says
so here rather than letting a silent id mismatch make the decision. The real fix is hashing the
full id in `src/ingest/splits.py`, which is a Phase 1.3 change and not this row's file to make.

`data/processed/label_crops/index.parquet` (read-only here) was used as a third opinion on the
part of the canonical split this module does adopt: 693/693 hdbpmn pages resolve and **0
disagree** once its `val` is read as `validation`. See `crop_index_agreement()`.

## What was rejected

- **Splitting the pairs afresh by scribe.** A second, differently-seeded writer-disjoint split
  would disagree with `reports/splits.md` about which writer is in test, and a diagram could
  then sit in codegen-test while its crops sit in ocr-train. Adopting is the only way the views
  stay one split.
- **Ratio-correcting to 70/15/15.** On scribed pairs the adopted split lands at 0.66/0.18/0.16.
  Nudging pages across the boundary to hit a ratio would break adoption for cosmetics.
- **Hashing scribe-less pairs, or holding out synthetic ones.** See above.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.utils.config import ROOT

SPLITS = ("train", "validation", "test")

#: The canonical corpus manifest written by `python -m src.ingest.splits`.
MANIFEST = ROOT / "data" / "processed" / "manifest.parquet"
IR_DIR = ROOT / "data" / "processed" / "ir"

#: Read-only cross-check on the canonical split. Never written by this module.
LABEL_CROPS = ROOT / "data" / "processed" / "label_crops" / "index.parquet"

#: Sources whose pairs are machine-generated and therefore train-only.
SYNTHETIC_SOURCES = frozenset({"synthetic", "synthetic_graphs", "generated"})


def _normalise(split: Any) -> str | None:
    """`val` and `validation` are the same split under two names in this repo."""
    if split is None:
        return None
    text = str(split).strip().lower()
    if text in ("val", "valid", "validation", "dev"):
        return "validation"
    if text in ("train", "training"):
        return "train"
    if text in ("test", "eval"):
        return "test"
    return None


@lru_cache(maxsize=1)
def manifest_index() -> dict[str, dict]:
    """diagram id -> {split, scribe, source, basis}, from the canonical manifest.

    Keyed both by the full manifest id (`hdbpmn/ex00/ex00_writer0001`) and by its last path
    segment (`ex00_writer0001`), because the IR documents use the bare form. A bare key that is
    ambiguous across sources is dropped rather than guessed at.
    """
    if not MANIFEST.is_file():
        return {}
    import pandas as pd

    frame = pd.read_parquet(MANIFEST)
    index: dict[str, dict] = {}
    tail_owner: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        split = _normalise(getattr(row, "split", None))
        if split is None:
            continue
        entry = {
            "split": split,
            "scribe": None if pd.isna(row.scribe_id) else str(row.scribe_id),
            "source": str(row.source),
            "basis": "manifest",
        }
        index[str(row.id)] = entry
        tail = str(row.id).rsplit("/", 1)[-1]
        if tail in tail_owner and tail_owner[tail] != str(row.id):
            index.pop(tail, None)
            tail_owner[tail] = "<ambiguous>"
        elif tail_owner.get(tail) != "<ambiguous>":
            tail_owner[tail] = str(row.id)
            index[tail] = entry
    return index


@lru_cache(maxsize=1)
def fa_index() -> dict[str, dict]:
    """diagram id -> entry for fa_bresler, adopted from `src.detect.dataset.fa_splits`.

    fa_bresler never entered the manifest, but it is not unsplit: `src/detect/**` and
    `src/ocr/textcrops.py` already train and score on a writer-disjoint split of it. That split
    is the established one for this source, so it is read here, not recomputed.
    """
    directory = IR_DIR / "fa_bresler"
    if not directory.is_dir():
        return {}
    documents = []
    for path in sorted(directory.glob("*.ir.json")):
        try:
            documents.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    if not documents:
        return {}
    from src.detect.dataset import fa_splits

    assigned = fa_splits(documents)
    index: dict[str, dict] = {}
    for document in documents:
        identifier = str(document.get("id"))
        split = _normalise(assigned.get(f"fa_bresler:{identifier}"))
        scribe = (document.get("meta") or {}).get("scribe_id")
        if split is None or not scribe:
            continue
        index[identifier] = {
            "split": split,
            "scribe": f"fa_bresler:{scribe}",
            "source": "fa_bresler",
            "basis": "fa_writer_split",
        }
    return index


def split_index() -> dict[str, dict]:
    """The one authority: the canonical manifest, plus fa_bresler's established split."""
    return {**manifest_index(), **fa_index()}


@lru_cache(maxsize=1)
def ir_scribes() -> dict[str, str]:
    """diagram id -> namespaced scribe, read from the IR corpus metadata.

    Used only to attribute a pair whose split came from neither index; missing IR is not an
    error, the caller falls back.
    """
    out: dict[str, str] = {}
    if not IR_DIR.is_dir():
        return out
    for path in sorted(IR_DIR.glob("*/*.ir.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta = document.get("meta") or {}
        scribe = meta.get("scribe_id")
        if scribe:
            source = str(meta.get("source") or path.parent.name)
            out[str(document.get("id") or path.name)] = f"{source}:{scribe}"
    return out


def _derive(pending: list[tuple[str, str, str, str]], seed: int = 42) -> dict[str, str]:
    """Scribe-disjoint splits for scribed ids that neither index covers.

    `pending` is (diagram_id, source, scribe, diagram_type). The work is delegated to
    `src.ingest.splits.assign` on a manifest-shaped frame, so this module cannot drift from the
    corpus rule. If that import is unavailable, whole *scribes* are dealt round-robin in name
    order - worse, but it still keeps every writer inside exactly one split.
    """
    if not pending:
        return {}
    try:
        import pandas as pd

        from src.ingest.splits import assign as corpus_assign
    except ImportError:
        scribes = sorted({scribe for _, _, scribe, _ in pending})
        wheel = ("train", "train", "train", "train", "train", "validation", "test")
        by_scribe = {s: wheel[i % len(wheel)] for i, s in enumerate(scribes)}
        return {diagram_id: by_scribe[scribe] for diagram_id, _, scribe, _ in pending}

    frame = pd.DataFrame(
        [
            {
                "id": diagram_id,
                "source": source,
                "diagram_type": diagram_type,
                "scribe_id": scribe,
                "native_split": None,
                "split": None,
                "split_basis": None,
            }
            for diagram_id, source, scribe, diagram_type in pending
        ]
    )
    assigned = corpus_assign(frame, seed=seed)
    return {
        str(row.id): _normalise(row.split) or "train" for row in assigned.itertuples(index=False)
    }


def resolve_scribe(record: dict, index: dict[str, dict] | None = None) -> str | None:
    """The writer behind a pair: from the record, then the split indices, then the IR."""
    if record.get("scribe"):
        return str(record["scribe"])
    index = split_index() if index is None else index
    diagram_id = str(record.get("diagram_id", ""))
    tail = diagram_id.rsplit("/", 1)[-1]
    for key in (diagram_id, tail):
        entry = index.get(key)
        if entry and entry.get("scribe"):
            return str(entry["scribe"])
    return ir_scribes().get(diagram_id) or ir_scribes().get(tail)


def is_synthetic(record: dict) -> bool:
    """A pair emitted by a generator rather than drawn by anybody."""
    return str(record.get("source", "")).lower() in SYNTHETIC_SOURCES


def assign(
    records: Iterable[dict], seed: int = 42, index: dict[str, dict] | None = None
) -> list[dict]:
    """Fill `split` on every pair. Returns copies in input order; never mutates the input.

    Also fills `split_basis` and `scribe`, because a split you cannot explain row by row is a
    split nobody can audit. `index` is injectable so a test can poison it with a leaking
    assignment; production passes nothing and gets `split_index()`.
    """
    records = [dict(record) for record in records]
    index = split_index() if index is None else index

    pending: list[tuple[str, str, str, str]] = []
    for record in records:
        diagram_id = str(record.get("diagram_id", ""))
        tail = diagram_id.rsplit("/", 1)[-1]
        scribe = resolve_scribe(record, index)
        record["scribe"] = scribe

        if is_synthetic(record):
            record["split"] = "train"
            record["split_basis"] = "synthetic_train_only"
            continue

        entry = index.get(diagram_id) or index.get(tail)
        if entry:
            record["split"] = entry["split"]
            record["split_basis"] = entry.get("basis", "manifest")
            continue

        if scribe:
            record["split"] = None
            record["split_basis"] = "derived_scribe_disjoint"
            pending.append(
                (
                    diagram_id,
                    str(record.get("source", "unknown")),
                    scribe,
                    str(record.get("diagram_type", "unknown")),
                )
            )
            continue

        # No writer identity anywhere: it cannot be held out from anyone, so it may only train.
        record["split"] = "train"
        record["split_basis"] = "no_scribe_train_only"

    if pending:
        derived = _derive(pending, seed=seed)
        for record in records:
            if record.get("split") is None:
                record["split"] = derived.get(str(record["diagram_id"]), "train")

    return records


def summarize(records: Iterable[dict], index: dict[str, dict] | None = None) -> dict:
    """The split summary, and the leakage checks that make it worth reporting.

    `checks` is the part that matters: every value must be True, and the tests assert exactly
    that on a deliberately leaking input as well as a clean one. A summary without them is a
    table of numbers nobody has to stand behind.
    """
    records = list(records)
    total = len(records)
    sizes = Counter(str(record.get("split")) for record in records)

    scribe_splits: dict[str, set[str]] = defaultdict(set)
    diagram_splits: dict[str, set[str]] = defaultdict(set)
    scribeless_outside_train: list[str] = []
    synthetic_outside_train: list[str] = []
    for record in records:
        split = str(record.get("split"))
        diagram_splits[str(record.get("diagram_id"))].add(split)
        scribe = record.get("scribe") or resolve_scribe(record, index)
        if scribe:
            scribe_splits[str(scribe)].add(split)
        elif split != "train":
            scribeless_outside_train.append(str(record.get("diagram_id")))
        if is_synthetic(record) and split != "train":
            synthetic_outside_train.append(str(record.get("diagram_id")))

    leaking_scribes = sorted(s for s, splits in scribe_splits.items() if len(splits) > 1)
    leaking_diagrams = sorted(d for d, splits in diagram_splits.items() if len(splits) > 1)

    by_basis = Counter(str(record.get("split_basis")) for record in records)
    by_type: dict[str, dict[str, int]] = defaultdict(dict)
    for record in records:
        row = by_type[str(record.get("diagram_type"))]
        key = str(record.get("split"))
        row[key] = row.get(key, 0) + 1

    scribed = [record for record in records if record.get("scribe")]
    scribed_sizes = Counter(str(record.get("split")) for record in scribed)

    return {
        "total": total,
        "sizes": {name: sizes.get(name, 0) for name in SPLITS},
        "fractions": (
            {name: round(sizes.get(name, 0) / total, 4) for name in SPLITS} if total else {}
        ),
        "distinct_diagrams": len(diagram_splits),
        "distinct_scribes": len(scribe_splits),
        "pairs_with_scribe": len(scribed),
        "scribed_sizes": {name: scribed_sizes.get(name, 0) for name in SPLITS},
        "scribed_fractions": (
            {name: round(scribed_sizes.get(name, 0) / len(scribed), 4) for name in SPLITS}
            if scribed
            else {}
        ),
        "scribeless_per_split": {
            name: sizes.get(name, 0) - scribed_sizes.get(name, 0) for name in SPLITS
        },
        "leaking_scribes": leaking_scribes,
        "leaking_diagrams": leaking_diagrams,
        "by_basis": dict(by_basis.most_common()),
        "by_diagram_type": {k: dict(sorted(v.items())) for k, v in sorted(by_type.items())},
        "scribes_per_split": {
            name: len({s for s, sp in scribe_splits.items() if name in sp}) for name in SPLITS
        },
        "checks": {
            "no_scribe_spans_splits": not leaking_scribes,
            "no_diagram_spans_splits": not leaking_diagrams,
            "all_pairs_assigned": all(record.get("split") in SPLITS for record in records),
            "synthetic_only_in_train": not synthetic_outside_train,
            "heldout_pairs_all_have_a_scribe": not scribeless_outside_train,
            "three_splits_present": all(sizes.get(name, 0) > 0 for name in SPLITS),
            "train_is_largest": (sizes.get("train", 0) == max(sizes.values()) if sizes else False),
        },
    }


def fa_derivation_agreement(seed: int = 42) -> dict:
    """Does adopting `fa_splits` agree with deriving fa_bresler through `src.ingest.splits`?

    The docstring's "agree on all 300 diagrams" is this function, kept runnable so the claim can
    be re-tested rather than trusted. A disagreement would not be a bug in either side - it
    would be the reason adoption is preferred over derivation.
    """
    adopted = fa_index()
    if not adopted:
        return {"available": False}
    pending = [
        (identifier, "fa_bresler", entry["scribe"], "state_machine")
        for identifier, entry in sorted(adopted.items())
    ]
    derived = _derive(pending, seed=seed)
    differing = sorted(k for k, v in adopted.items() if derived.get(k) != v["split"])
    return {
        "available": True,
        "diagrams": len(adopted),
        "writers": len({entry["scribe"] for entry in adopted.values()}),
        "agree": len(adopted) - len(differing),
        "disagree": len(differing),
        "examples": [
            {"id": k, "adopted": adopted[k]["split"], "derived": derived.get(k)}
            for k in differing[:5]
        ],
    }


def crop_index_agreement() -> dict:
    """Third opinion: does the label-crop index agree with the manifest? (read-only)

    Not a check this module's correctness depends on - it is how the "0 disagree" number in the
    docstring was obtained, kept runnable so the claim can be re-tested rather than trusted.
    """
    if not (LABEL_CROPS.is_file() and MANIFEST.is_file()):
        return {"available": False}
    import pandas as pd

    crops = pd.read_parquet(LABEL_CROPS).drop_duplicates("page")
    index = manifest_index()
    resolved = agree = disagree = 0
    examples: list[dict] = []
    for row in crops.itertuples(index=False):
        entry = index.get(str(row.page))
        if not entry:
            continue
        resolved += 1
        if _normalise(row.split) == entry["split"]:
            agree += 1
        else:
            disagree += 1
            if len(examples) < 5:
                examples.append(
                    {"page": str(row.page), "crops": str(row.split), "manifest": entry["split"]}
                )
    return {
        "available": True,
        "pages": int(len(crops)),
        "resolved": resolved,
        "agree": agree,
        "disagree": disagree,
        "examples": examples,
    }


def _pairs_from_targets_index() -> list[dict]:
    """Pair records reconstructed from `data/processed/targets/index.json`.

    12.1.1-12.1.6 have not produced `pairs.jsonl` yet, and this row must not wait on them: the
    1,477 pairs Phase 2.2.6 already built carry the same `diagram_id`/`source`/`diagram_type`
    the split cares about, so the split can be built and verified on the real ids today.
    """
    path = ROOT / "data" / "processed" / "targets" / "index.json"
    if not path.is_file():
        return []
    known = {"python", "html", "sql", "spice"}
    records = []
    for entry in json.loads(path.read_text(encoding="utf-8")):
        language = str(entry.get("language", ""))
        records.append(
            {
                "diagram_id": entry["id"],
                "diagram_type": entry.get("diagram_type", "unknown"),
                "ir_text": "",
                "traversal": [],
                "target_code": "",
                "language": language if language in known else "python",
                "source": entry.get("source", "unknown"),
                "split": None,
            }
        )
    return records


def synthetic_stand_ins(count: int = 600) -> list[dict]:
    """Machine-generated pairs standing in for 12.1.4's, to prove they land in train only."""
    return [
        {
            "diagram_id": f"synthetic/graph_{i:05d}",
            "diagram_type": "flowchart",
            "ir_text": "",
            "traversal": [],
            "target_code": "",
            "language": "python",
            "source": "synthetic",
            "split": None,
        }
        for i in range(count)
    ]


def load_pairs(path: Path) -> list[dict]:
    from src.codegen.quality import load_pairs as _load

    return _load(path)


def _print(summary: dict) -> None:
    print(json.dumps({k: v for k, v in summary.items() if k != "by_diagram_type"}, indent=2))
    print()
    for name, ok in summary["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12.1.8 pair splits")
    parser.add_argument("--pairs", type=Path, help="JSONL of training pairs")
    parser.add_argument("--out", type=Path, help="write the assigned pairs back to this JSONL")
    parser.add_argument("--synthetic", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true", help="stand-in pairs; the default")
    args = parser.parse_args(argv)

    if args.pairs:
        records = load_pairs(args.pairs)
    else:
        records = _pairs_from_targets_index() + synthetic_stand_ins(args.synthetic)
        if not records:
            print("no pairs and no targets index; nothing to split", file=sys.stderr)
            return 1
        print(f"no --pairs given; splitting {len(records)} stand-in pairs")

    assigned = assign(records, seed=args.seed)
    summary = summarize(assigned)
    _print(summary)
    print()
    print("fa: adopted vs derived:       " + json.dumps(fa_derivation_agreement(seed=args.seed)))
    print("label-crop index vs manifest: " + json.dumps(crop_index_agreement()))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="\n") as handle:
            for record in assigned:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")

    return 0 if all(summary["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
