"""Phase 1.3.3 — scribe-disjoint train/val/test splits.

The rule this module exists to enforce: **a person's drawings appear in exactly one split.**

Splitting by image is the default mistake in sketch recognition. It lets the model see writer
X's handwriting in training and be tested on the same handwriting, so the reported score
measures memorised style rather than generalization — risk D3/D4 in `docs/risks.md`. Every
score in this project is meant to answer "does it work on someone it has never seen", so the
split has to be built that way from the start.

Three constraints, in priority order:

1. **Scribe-disjoint.** No `scribe_id` spans two splits. Where a source publishes its own
   writer-disjoint split (hdBPMN, FA), that split is adopted rather than re-derived.
2. **Duplicate groups stay whole.** Near-duplicates found in Phase 1.3.2 are pinned to one
   split, so a near-copy cannot straddle train and test.
3. **Stratified by diagram type.** Each split should see every type in roughly corpus
   proportion, subject to constraints 1 and 2.

Rows with no writer identity (flowchartseg, DIDI, sketch2code, IAM) cannot satisfy
constraint 1. They are split by their native split where one exists, and are flagged
`split_basis="no_scribe_id"` so any evaluation that claims cross-writer generalization can
exclude them.

    python -m src.ingest.splits
    python -m src.ingest.splits --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import pandas as pd

from src.utils.config import ROOT
from src.utils.seed import set_seed

RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
REPORT = ROOT / "reports" / "splits.md"


def _duplicate_pins() -> dict[str, int]:
    """id -> duplicate group, from the Phase 1.3.2 hashes (empty if not computed yet)."""
    from src.ingest.dedup import OUT as PHASH_OUT
    from src.ingest.dedup import duplicate_group_map, find_duplicates

    if not PHASH_OUT.is_file():
        return {}
    hashes = pd.read_parquet(PHASH_OUT)
    return duplicate_group_map(find_duplicates(hashes))


def assign(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Return the manifest with `split` and `split_basis` filled in.

    Scribe-disjointness and duplicate-group integrity are *one* constraint, not two: if a
    near-duplicate links writer A to writer B, then A and B must share a split, or one of
    the two rules breaks. So both are resolved together, by building connected components
    over scribes and duplicate groups and assigning whole components.
    """
    set_seed(seed)
    df = df.copy()
    df["split"] = None
    df["split_basis"] = None

    dup_group = _duplicate_pins()
    df["_dup"] = df["id"].map(dup_group)

    # --- union-find over scribes and duplicate groups ------------------------------------
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for s in df.loc[df["scribe_id"].notna(), "scribe_id"].unique():
        find(f"s:{s}")
    for g in df.loc[df["_dup"].notna(), "_dup"].unique():
        find(f"d:{int(g)}")
    # A duplicate group ties together every scribe that owns one of its members.
    for g, members in df[df["_dup"].notna()].groupby("_dup"):
        key = f"d:{int(g)}"
        for s in members.loc[members["scribe_id"].notna(), "scribe_id"].unique():
            union(key, f"s:{s}")

    def unit_of(row) -> str | None:
        if pd.notna(row["scribe_id"]):
            return find(f"s:{row['scribe_id']}")
        if pd.notna(row["_dup"]):
            return find(f"d:{int(row['_dup'])}")
        return None

    df["_unit"] = df.apply(unit_of, axis=1)

    # --- assign whole units, quota-balanced *within each diagram type* --------------------
    # A single global quota starves the rare types: er_diagram has 5 writers and
    # state_machine 25, so a global greedy pass can put every one of them in train and leave
    # the test split with no examples of that class at all. Allocating per type guarantees
    # each type reaches all three splits while units stay whole.
    constrained = df["_unit"].notna()
    if constrained.any():
        sub = df[constrained]
        unit_type = sub.groupby("_unit")["diagram_type"].agg(lambda c: c.value_counts().idxmax())
        sizes = sub.groupby("_unit").size().to_dict()
        native_by_unit = sub.groupby("_unit")["native_split"].agg(
            lambda col: col.dropna().unique().tolist()
        )

        assignment: dict[str, str] = {}
        basis: dict[str, str] = {}

        for dtype in sorted(unit_type.unique()):
            units = [u for u in sizes if unit_type[u] == dtype]
            total = sum(sizes[u] for u in units)
            quota = {k: v * total for k, v in RATIOS.items()}
            filled = dict.fromkeys(RATIOS, 0)
            # Largest first, into whichever split is furthest below its quota: deterministic,
            # and it prevents a big unit from being the only thing in a small split.
            for unit in sorted(units, key=lambda u: -sizes[u]):
                natives = native_by_unit.get(unit, [])
                if len(natives) == 1 and natives[0] in RATIOS:
                    target = natives[0]
                    basis[unit] = "source_writer_split"
                else:
                    target = max(RATIOS, key=lambda k: quota[k] - filled[k])
                    basis[unit] = "scribe_disjoint"
                assignment[unit] = target
                filled[target] += sizes[unit]

        df.loc[constrained, "split"] = df.loc[constrained, "_unit"].map(assignment).to_numpy()
        df.loc[constrained, "split_basis"] = df.loc[constrained, "_unit"].map(basis).to_numpy()

    # --- unconstrained rows: native split, else a deterministic hash ---------------------
    todo = df["split"].isna() & df["native_split"].notna()
    df.loc[todo, "split"] = df.loc[todo, "native_split"]
    df.loc[todo, "split_basis"] = "native_split_no_scribe_id"

    rest = df["split"].isna()
    if rest.any():
        # A digest of the **whole** id, not `int.from_bytes(s.encode()[:8])`. Ids here are
        # `<source>/<page>`, so the first eight bytes are the source name for every row of a
        # source: all 731 sketch2code pages hashed identically and landed in one split, leaving
        # the test set with no wireframes at all. blake2b is stable across runs and processes,
        # which `hash()` is not.
        h = df.loc[rest, "id"].map(
            lambda s: int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")
            % 100
        )
        df.loc[rest, "split"] = pd.cut(
            h, bins=[-1, 69, 84, 99], labels=["train", "validation", "test"]
        ).astype(str)
        df.loc[rest, "split_basis"] = "hashed_no_scribe_id"

    return df.drop(columns=["_dup", "_unit"])


def verify(df: pd.DataFrame) -> dict:
    """The checks that make the split trustworthy. Any failure invalidates every later score."""
    with_scribe = df[df["scribe_id"].notna()]
    per_scribe_splits = with_scribe.groupby("scribe_id")["split"].nunique()
    leaking = per_scribe_splits[per_scribe_splits > 1]

    dup_group = _duplicate_pins()
    dup_df = df[df["id"].isin(dup_group)].copy()
    straddling = 0
    if len(dup_df):
        dup_df["group"] = dup_df["id"].map(dup_group)
        straddling = int((dup_df.groupby("group")["split"].nunique() > 1).sum())

    sizes = df["split"].value_counts().to_dict()
    total = len(df)
    by_type = (
        df.groupby(["split", "diagram_type"]).size().unstack(fill_value=0).to_dict()
        if total
        else {}
    )

    return {
        "total": total,
        "sizes": sizes,
        "fractions": {k: round(v / total, 3) for k, v in sizes.items()} if total else {},
        "rows_with_scribe": int(len(with_scribe)),
        "distinct_scribes": int(with_scribe["scribe_id"].nunique()),
        "leaking_scribes": leaking.index.tolist(),
        "duplicate_groups_straddling": straddling,
        "split_basis_counts": df["split_basis"].value_counts().to_dict(),
        "by_type": {str(k): v for k, v in by_type.items()},
        "checks": {
            "no_scribe_spans_splits": leaking.empty,
            "no_duplicate_group_straddles": straddling == 0,
            "all_rows_assigned": int(df["split"].isna().sum()) == 0,
            "three_splits_present": set(sizes) == {"train", "validation", "test"},
            "train_is_largest": (
                bool(sizes.get("train", 0) == max(sizes.values())) if sizes else False
            ),
        },
    }


def write_report(v: dict) -> object:
    lines: list[str] = []
    add = lines.append
    add("# Split Report")
    add("")
    add("Phase 1.3.3. Generated by `python -m src.ingest.splits`.")
    add("")
    add(f"- Rows: **{v['total']}**")
    add(f"- Distinct scribes: **{v['distinct_scribes']}** over {v['rows_with_scribe']} rows")
    add("")
    add("| split | rows | fraction |")
    add("| :--- | ---: | ---: |")
    for k in ("train", "validation", "test"):
        add(f"| {k} | {v['sizes'].get(k, 0)} | {v['fractions'].get(k, 0):.3f} |")
    add("")
    add("## How each row was assigned")
    add("")
    add("| basis | rows | meaning |")
    add("| :--- | ---: | :--- |")
    meanings = {
        "source_writer_split": "the dataset published a writer-disjoint split; adopted as-is",
        "scribe_disjoint": "assigned here, keeping every scribe (and duplicate group) whole",
        "native_split_no_scribe_id": "no writer identity; the source's own split was used",
        "hashed_no_scribe_id": "no writer identity and no native split; deterministic hash",
    }
    for basis, n in v["split_basis_counts"].items():
        add(f"| `{basis}` | {n} | {meanings.get(basis, '')} |")
    add("")
    add("## Leakage checks")
    add("")
    add("| check | result |")
    add("| :--- | :--- |")
    for name, ok in v["checks"].items():
        add(f"| `{name}` | {'PASS' if ok else 'FAIL'} |")
    add("")
    add("## The caveat that matters")
    add("")
    add("Rows marked `native_split_no_scribe_id` or `hashed_no_scribe_id` come from sources")
    add("that do not publish writer identity, so **they cannot be guaranteed scribe-disjoint**.")
    add("Any result claiming cross-writer generalization must be computed on the")
    add("`source_writer_split` and `scribe_disjoint` rows only. The column exists so that")
    add("filter is one line of code rather than an act of memory.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    while lines and not lines[-1].strip():
        lines.pop()
    with REPORT.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return REPORT


def main(argv: list[str] | None = None) -> int:
    from src.ingest.manifest import OUT, load

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="check an existing assignment only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    df = load()
    if not args.verify:
        df = assign(df, seed=args.seed)
        df.to_parquet(OUT, index=False)
        print(f"wrote {OUT.relative_to(ROOT)}")

    v = verify(df)
    print(json.dumps({k: val for k, val in v.items() if k != "by_type"}, indent=2, default=str))
    path = write_report(v)
    print(f"wrote {path.relative_to(ROOT)}")

    for name, ok in v["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(v["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
