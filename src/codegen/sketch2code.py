"""Phase 12.1.5 - Sketch2Code's real webpages converted from HTML into React (JSX) targets.

    python -m src.codegen.sketch2code build        # 484 pages -> JSX, built and rendered in node
    python -m src.codegen.sketch2code stats

    from src.codegen.sketch2code import html_to_jsx, load_pairs
    jsx = html_to_jsx(html)                        # one `export default function Page()`

Output (gitignored): `data/processed/codegen/sketch2code_html/pairs.jsonl` - 12.1.1-schema records
with `source = "sketch2code_html"`, the sketch2code IR as `ir_text` and the converted page as
`target_code` - plus `report.json`. **This shard is deliberately not written into
`data/processed/codegen/pairs/`, so `pairs.merge` does not mix it into the training splits**; see
"Why these are not default training pairs" below. Load it with `load_pairs()` from this module.

## The conversion (`html_to_jsx`)

A streaming `html.parser` walk of `<body>` that emits JSX, element for element:

    dropped whole        script, style, noscript, template, iframe, svg, object, head, comments
    attributes           class -> className, for -> htmlFor, tabindex -> tabIndex (and the other
                         camel-cased DOM names React warns about); `style="a:b; c-d:e"` ->
                         `style={{"a": "b", "cD": "e"}}`; `on*` event handler strings dropped
                         (a string is not a function in React and would throw at render);
                         boolean attributes -> `={true}`; `data-*`/`aria-*` kept verbatim
    void elements        area base br col embed hr img input link meta param source track wbr
                         -> self-closing
    text                 whitespace-collapsed, emitted as `{"..."}` string literals so `{`, `<`
                         and quotes in page text can never become JSX syntax
    unclosed / stray     an end tag with no matching open element is ignored; elements still open
                         at the end are closed, innermost first - the browser's own recovery
    `<p>` auto-close     a block element opening inside an open `<p>` closes the `<p>` first,
                         as the HTML parsing spec does; without it React warns on invalid nesting
    root                 the body's children wrapped in one `<div>`, so `return (...)` has a
                         single root

## How it was checked, rather than eyeballed

Every converted page goes through `src.eval.react`: a real esbuild build and a real
`react-dom/server` render. Then **text fidelity** is measured: the visible text of the rendered
React output against the visible text of the original HTML body (same extractor on both, words
compared as multisets). A converter that dropped a subtree, mangled an entity or double-escaped
a quote shows up as lost or invented words. Numbers are in the 12.1.5 row and `report.json`.

## Why these are not default training pairs

The target is the whole page; the IR is 2.2.2's structure-only, truncated summary of it. So most
of the target's words are not in `ir_text` at all, which is the exact "invent what is not in the
IR" behaviour 12.2.4's contract forbids, and the pages are far beyond a 7B fine-tune's context.
Both are measured (`target_words_in_ir`, token lengths) and reported instead of being mixed
silently into `train.jsonl`; 12.1.1's `sketch2code` shard (IR-faithful React from 12.1.6) remains
the wireframe training source.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from html.parser import HTMLParser

from src.codegen import schema
from src.utils.config import ROOT

RAW = ROOT / "data" / "raw" / "sketch2code" / "webpages"
IR_DIR = ROOT / "data" / "processed" / "ir" / "sketch2code"
OUT = ROOT / "data" / "processed" / "codegen" / "sketch2code_html"

VOID = frozenset("area base br col embed hr img input link meta param source track wbr".split())
SKIP = frozenset("script style noscript template iframe svg object head title".split())
#: Elements whose start tag implicitly closes an open <p> (HTML parsing spec, "in body").
CLOSES_P = frozenset(
    "address article aside blockquote details dialog div dl fieldset figcaption figure footer "
    "form h1 h2 h3 h4 h5 h6 header hgroup hr main menu nav ol p pre section table ul".split()
)
ATTR_RENAME = {
    "class": "className",
    "for": "htmlFor",
    "tabindex": "tabIndex",
    "readonly": "readOnly",
    "maxlength": "maxLength",
    "minlength": "minLength",
    "colspan": "colSpan",
    "rowspan": "rowSpan",
    "cellpadding": "cellPadding",
    "cellspacing": "cellSpacing",
    "contenteditable": "contentEditable",
    "crossorigin": "crossOrigin",
    "srcset": "srcSet",
    "autocomplete": "autoComplete",
    "autofocus": "autoFocus",
    "enctype": "encType",
    "accesskey": "accessKey",
    "frameborder": "frameBorder",
    "allowfullscreen": "allowFullScreen",
    "novalidate": "noValidate",
    "usemap": "useMap",
    "datetime": "dateTime",
    "http-equiv": "httpEquiv",
    "accept-charset": "acceptCharset",
    "spellcheck": "spellCheck",
    "hreflang": "hrefLang",
    "referrerpolicy": "referrerPolicy",
}
BOOLEAN = frozenset(
    "checked disabled selected readonly required multiple autofocus hidden open novalidate "
    "allowfullscreen defer async controls loop muted playsinline".split()
)
#: JSX attribute names must start with a letter and React rejects namespaced (`a:b`) names, so
#: Alpine.js-style `:class` / `x-bind:aria-expanded` are dropped (one page failed to build on it).
_ATTR_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def _style_object(css: str) -> str:
    items = []
    for decl in css.split(";"):
        if ":" not in decl:
            continue
        key, value = decl.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key or not value:
            continue
        if not key.startswith("--"):
            key = re.sub(r"-([a-z])", lambda m: m.group(1).upper(), key.lower())
            if key.startswith("ms"):
                key = "ms" + key[2:]
        items.append(f"{json.dumps(key)}: {json.dumps(value)}")
    return "{{" + ", ".join(items) + "}}"


def _attrs(attrs: list[tuple[str, str | None]], stats: Counter | None = None) -> str:
    out = []
    seen = set()
    for name, value in attrs:
        name = name.lower()
        if name.startswith("on") or not _ATTR_NAME.match(name):
            if stats is not None:
                stats["dropped_attr:" + ("event" if name.startswith("on") else "invalid_name")] += 1
            continue
        jsx = ATTR_RENAME.get(name, name)
        if jsx in seen:
            continue
        seen.add(jsx)
        if name == "style":
            rendered = _style_object(value or "")
            if rendered != "{{}}":
                out.append(f"style={rendered}")
        elif value is None or (name in BOOLEAN and value in ("", name)):
            out.append(f"{jsx}={{true}}")
        else:
            out.append(f"{jsx}={{{json.dumps(value)}}}")
    return (" " + " ".join(out)) if out else ""


class _Converter(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.stack: list[str] = []
        self.skip = 0
        self.in_body = False
        self.stats: Counter = Counter()

    def _emit(self, text: str) -> None:
        self.lines.append("  " * (len(self.stack) + 3) + text)

    def _close_to(self, index: int) -> None:
        while len(self.stack) > index:
            tag = self.stack.pop()
            self._emit(f"</{tag}>")

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in ("html", "body"):
            self.in_body = self.in_body or tag == "body"
            return
        if self.skip or tag in SKIP:
            if tag in SKIP and tag not in VOID:
                self.skip += 1
            self.stats[f"dropped:{tag}"] += 1
            return
        self.in_body = True
        if tag in CLOSES_P and "p" in self.stack:
            self.stats["implicit_p_close"] += 1
            self._close_to(len(self.stack) - self.stack[::-1].index("p") - 1)
        if tag in VOID:
            self._emit(f"<{tag}{_attrs(attrs, self.stats)} />")
        else:
            self._emit(f"<{tag}{_attrs(attrs, self.stats)}>")
            self.stack.append(tag)
        self.stats["elements"] += 1

    def handle_startendtag(self, tag: str, attrs) -> None:
        if self.skip or tag in SKIP or tag in ("html", "body"):
            return
        self._emit(f"<{tag}{_attrs(attrs, self.stats)} />")
        self.stats["elements"] += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP and tag not in VOID:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip or tag in VOID or tag in ("html", "body"):
            return
        if tag not in self.stack:
            self.stats["stray_end_tag"] += 1
            return
        index = len(self.stack) - self.stack[::-1].index(tag) - 1
        self.stats["implicitly_closed"] += len(self.stack) - index - 1
        self._close_to(index)

    def handle_data(self, data: str) -> None:
        if self.skip or not self.in_body:
            return
        # Whitespace is collapsed but never deleted: "a <b>b</b>" rendered as "a<b>b</b>" joins
        # two words on screen, and a word-set metric cannot see it (textContent can).
        text = re.sub(r"\s+", " ", data)
        if text:
            self._emit("{" + json.dumps(text, ensure_ascii=False) + "}")


def html_to_jsx(html: str, name: str = "Page") -> tuple[str, dict]:
    """(JSX module source, conversion stats) for one HTML document."""
    parser = _Converter()
    parser.feed(html)
    parser.close()
    parser._close_to(0)
    body = "\n".join(parser.lines) or '      {""}'
    code = (
        f"export default function {name}() {{\n"
        "  return (\n"
        "    <div>\n"
        f"{body}\n"
        "    </div>\n"
        "  );\n"
        "}\n"
    )
    return code, dict(parser.stats)


class _Text(HTMLParser):
    """Visible words of an HTML fragment - used identically on the source and the render."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.words: list[str] = []
        self.chunks: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIP and tag not in VOID:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in SKIP and tag not in VOID:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.words += data.split()
            self.chunks.append(data)


def visible_words(html: str) -> Counter:
    parser = _Text()
    parser.feed(html)
    parser.close()
    return Counter(parser.words)


def text_content(html: str) -> str:
    """Browser-style textContent with whitespace runs collapsed - sensitive to joined words."""
    parser = _Text()
    parser.feed(html)
    parser.close()
    return re.sub(r"\s+", " ", "".join(parser.chunks)).strip()


def text_fidelity(source_html: str, rendered_html: str) -> dict:
    """Word-multiset agreement between the page and its React render."""
    body = re.search(r"<body[^>]*>(.*)", source_html, re.S | re.I)
    want = visible_words(body.group(1) if body else source_html)
    got = visible_words(rendered_html)
    common = sum((want & got).values())
    source_text = text_content(body.group(1) if body else source_html)
    return {
        "text_content_equal": source_text == text_content(rendered_html),
        "words": sum(want.values()),
        "lost": sum((want - got).values()),
        "invented": sum((got - want).values()),
        "recall": common / max(1, sum(want.values())),
        "precision": common / max(1, sum(got.values())),
    }


def build() -> dict:
    """Convert all pages, render them, measure fidelity, write the (non-default) shard."""
    from src.codegen import pairs
    from src.eval import react

    ids = sorted(p.stem for p in RAW.glob("*.html"))
    sources, codes, stats = [], [], Counter()
    for page in ids:
        html = (RAW / f"{page}.html").read_text(encoding="utf-8", errors="replace")
        code, conv = html_to_jsx(html)
        sources.append(html)
        codes.append(code)
        stats.update(conv)
    results = react.check_many(codes, render=True, keep_html=True, timeout_ms=5000, batch=50)

    fidelity = []
    records = []
    in_ir = []
    kinds = Counter(r["kind"] for r in results)
    for page, html, code, result in zip(ids, sources, codes, results, strict=True):
        if not result["ok"]:
            continue
        fidelity.append(text_fidelity(html, result.get("html", "")))
        ir_path = IR_DIR / f"s2c_{page}.ir.json"
        if not ir_path.is_file():
            continue
        diagram = json.loads(ir_path.read_text(encoding="utf-8"))
        record = pairs.make_pair(
            diagram,
            "sketch2code_html",
            target_code=code,
            meta={"webpage": f"data/raw/sketch2code/webpages/{page}.html"},
        )
        record["language"] = "react"
        ir_words = set(re.findall(r"\w+", record["ir_text"].lower()))
        target_words = re.findall(r'"([^"\\]*)"', code)
        words = [w for t in target_words for w in re.findall(r"\w+", t.lower())]
        in_ir.append(sum(w in ir_words for w in words) / max(1, len(words)))
        records.append(record)
    records = pairs.assign_splits(records)
    OUT.mkdir(parents=True, exist_ok=True)
    written = schema.write_jsonl(records, OUT / "pairs.jsonl")

    def mean(key: str) -> float:
        return round(sum(f[key] for f in fidelity) / max(1, len(fidelity)), 4)

    report = {
        "pages": len(ids),
        "render_kinds": dict(kinds),
        "pairs_written": written,
        "text_fidelity_mean": {k: mean(k) for k in ("recall", "precision")},
        "pages_with_equal_text_content": sum(f["text_content_equal"] for f in fidelity),
        "pages_with_exact_word_multiset": sum(
            f["lost"] == 0 and f["invented"] == 0 for f in fidelity
        ),
        "words_total": sum(f["words"] for f in fidelity),
        "words_lost": sum(f["lost"] for f in fidelity),
        "words_invented": sum(f["invented"] for f in fidelity),
        "target_words_in_ir_mean": round(sum(in_ir) / max(1, len(in_ir)), 4),
        "code_bytes_p50_max": [sorted(map(len, codes))[len(codes) // 2], max(map(len, codes))],
        "conversion": dict(stats.most_common(20)),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_pairs() -> list[dict]:
    return list(schema.read_jsonl(OUT / "pairs.jsonl"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12.1.5 Sketch2Code HTML -> React")
    parser.add_argument("command", choices=("build", "stats"))
    args = parser.parse_args(argv)
    if args.command == "build":
        report = build()
    else:
        report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
