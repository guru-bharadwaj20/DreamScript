"""Phase 14.9 - the gallery: the best and the worst pages, sketch beside the code they produced.

    python -m src.eval.gallery                  # reports/gallery.md + reports/figures/gallery/
    python -m src.eval.gallery --each 4         # four best and four worst

Every other row in Phase 14 is a table. This one exists because a table cannot show *what a
failure looks like*, and the failures on this corpus are not uniform - a page that scores GED 36
is wrong in a visibly different way from one that scores 8, and the difference is legible in
seconds from the photograph.

## How the cases are chosen, and why that is not cherry-picking

The ranking is the criterion's own: pages are ordered by the graph edit distance in
`reports/s5_test_large.json`, which was computed before this module existed and is not touched
here. The best `n` and the worst `n` are taken from the ends of that order. There is no
hand-picking, and the report says which file the order came from so the choice can be checked.
`--each` widens both ends symmetrically; taking more of one end than the other would be exactly
the dishonesty this note exists to rule out.

## What each entry carries

The page thumbnail, the criterion numbers for that page (GED and its decomposition), the
end-to-end verdict from 14.3's `predicted` rung where the page is in that run, and the first
lines of the program 12.1.6 emitted from the *predicted* IR - the code a user would actually
have received, not the reference.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from src.utils.config import ROOT

REPORT_MD = ROOT / "reports" / "gallery.md"
REPORT_JSON = ROOT / "reports" / "gallery.json"
FIGURES = ROOT / "reports" / "figures" / "gallery"

S5_REPORT = ROOT / "reports" / "s5_test_large.json"
PROPAGATION = ROOT / "reports" / "error_propagation.json"

#: How much of the emitted program to show. Enough to see its shape, not a code listing.
CODE_LINES = 18


def ranked_pages() -> list[dict]:
    """S5's own per-page rows, ordered by the criterion - best first."""
    payload = json.loads(S5_REPORT.read_text(encoding="utf-8"))
    return sorted(payload["pages"], key=lambda row: (row["ged"], row["page"]))


def propagation_index() -> dict[str, dict]:
    if not PROPAGATION.is_file():
        return {}
    payload = json.loads(PROPAGATION.read_text(encoding="utf-8"))
    return {row["page"]: row for row in payload["rows"]}


def thumbnail(page, out: Path, width: int = 720) -> Path | None:
    """A width-limited copy of the photograph, so the report is readable in a browser."""
    import cv2

    image = cv2.imread(str(page.image), cv2.IMREAD_COLOR)
    if image is None:
        return None
    height = int(image.shape[0] * width / image.shape[1])
    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), small)
    return out


def emitted_code(page) -> tuple[str, str]:
    """The program 12.1.6 writes from the *predicted* IR - what a user would have received."""
    from src.eval.propagate import _assembled, emit

    try:
        diagram = _assembled(page)
    # An assembly failure is a caption in the gallery, not a stop.
    except Exception as error:  # noqa: BLE001
        return "", f"assembly failed: {type(error).__name__}: {error}"
    code = emit(diagram, page.source)
    if code is None:
        return "", "the emitter refused the predicted IR"
    return code, ""


def build(each: int = 3) -> dict:
    from src.assemble.corpus import pages as corpus_pages

    by_name = {page.name: page for page in corpus_pages(("test",))}
    ranked = [row for row in ranked_pages() if row["page"] in by_name]
    chosen = [("best", row) for row in ranked[:each]] + [("worst", row) for row in ranked[-each:]]
    propagation = propagation_index()

    entries = []
    for band, row in chosen:
        page = by_name[row["page"]]
        image = thumbnail(page, FIGURES / f"{row['page']}.jpg")
        code, note = emitted_code(page)
        rungs = (propagation.get(row["page"]) or {}).get("rungs", {})
        entries.append(
            {
                "band": band,
                "page": row["page"],
                "source": row["source"],
                "ged": row["ged"],
                "node_f1": row.get("node_f1"),
                "edge_f1": row.get("edge_f1"),
                "edits": {
                    key: row[key]
                    for key in (
                        "sub_text",
                        "edge_insert",
                        "edge_delete",
                        "node_insert",
                        "node_delete",
                    )
                    if key in row
                },
                "functional_predicted": (rungs.get("predicted") or {}).get("functional"),
                "functional_reason": (rungs.get("predicted") or {}).get("reason"),
                "functional_gold": (rungs.get("gold") or {}).get("functional"),
                "image": str(image.relative_to(ROOT)).replace("\\", "/") if image else None,
                "code_head": "\n".join(code.splitlines()[:CODE_LINES]),
                "code_lines": len(code.splitlines()),
                "note": note,
            }
        )
    return {
        "each": each,
        "order": "reports/s5_test_large.json, ascending graph edit distance",
        "entries": entries,
    }


def render(result: dict) -> str:
    lines = [
        "# Phase 14.9 - qualitative gallery",
        "",
        f"Generated by `python -m src.eval.gallery --each {result['each']}`. The order is the"
        f" criterion's own (`{result['order']}`), computed before this module existed; the best"
        " and worst entries are the two ends of it, taken symmetrically, so nothing here is"
        " hand-picked. Each entry shows the photograph, the criterion's numbers for that page,"
        " and the head of the program 12.1.6 emitted **from the predicted IR** - the code a user"
        " would have received, not the reference.",
        "",
    ]
    for band in ("best", "worst"):
        lines += [f"## {band.title()} cases", ""]
        for entry in [row for row in result["entries"] if row["band"] == band]:
            lines += [f"### `{entry['page']}` ({entry['source']}) - GED {entry['ged']}", ""]
            if entry["image"]:
                lines += [f"![{entry['page']}]({entry['image'].removeprefix('reports/')})", ""]
            verdict = entry["functional_predicted"]
            verdict_text = (
                "not in the 14.3 run"
                if verdict is None
                else (
                    "passes its functional test"
                    if verdict
                    else f"fails: `{entry['functional_reason']}`"
                )
            )
            lines += [
                f"* node F1 {entry['node_f1']}, edge F1 {entry['edge_f1']}",
                f"* edits: {json.dumps(entry['edits'])}",
                f"* end to end: {verdict_text}"
                + (
                    ""
                    if entry["functional_gold"] is None
                    else f" (from the annotated IR: {'passes' if entry['functional_gold'] else 'fails'})"
                ),
                "",
            ]
            if entry["note"]:
                lines += [f"> {entry['note']}", ""]
            if entry["code_head"]:
                lines += [
                    "```python",
                    entry["code_head"],
                    "```",
                    "",
                    f"({entry['code_lines']} lines in full)",
                    "",
                ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=textwrap.dedent(__doc__ or ""))
    ap.add_argument("--each", type=int, default=3, help="how many from each end")
    ap.add_argument("--out", type=Path, default=REPORT_MD)
    ap.add_argument("--json", dest="out_json", type=Path, default=REPORT_JSON)
    args = ap.parse_args(argv)

    result = build(args.each)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result), encoding="utf-8")
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{len(result['entries'])} entries -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
