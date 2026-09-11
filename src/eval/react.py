"""Phase 12.3.6 - headless build and render of generated React components, in real node.

    python -m src.eval.react --install        # npm install esbuild + react into the tool dir
    python -m src.eval.react --study          # every React target + 8 injected defects

    from src.eval.react import check_many
    results = check_many(codes, render=True)  # [{ok, kind, detail, html_bytes}, ...]

## What "renders without error" means here, exactly

Every component goes through two real stages in **one** node process per batch:

1. **build** - `esbuild.transformSync(code, {loader: "jsx", format: "cjs", jsx: "transform"})`.
   esbuild is a full JS/JSX parser: an unclosed tag, a stray brace or a JSX expression in an
   attribute position that is not valid JavaScript is a build error with a line and column.
2. **render** - the CommonJS output is executed in a fresh `vm` context whose only importable
   module is `react`, the default export (or the single capitalised function) is instantiated
   with `React.createElement`, and `react-dom/server.renderToString` renders it. Both the module
   body and the render run under a `vm` timeout, so an infinite loop is `react.timeout`, not a
   hung batch. An undefined identifier, calling a non-function, a component that returns an
   object, or an import of any module other than `react` all fail here - which is exactly the
   class the 12.1.7 structural parse cannot see.

`kind` is one of `react.ok / build_error / no_component / render_error / timeout /
unavailable`. `unavailable` means node or the npm packages are missing; it is never folded into
a pass.

The tool directory is `data/processed/codegen/node` (gitignored); `ensure_toolchain()` writes a
pinned `package.json` (esbuild 0.25.10, react 18.3.1, react-dom 18.3.1) and runs `npm install`
there, so nothing under `node_modules` is ever committed and the versions are code, not luck.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from src.utils.config import ROOT

TOOL_DIR = ROOT / "data" / "processed" / "codegen" / "node"
NODE_CANDIDATES = (
    os.environ.get("DREAMSCRIPT_NODE", ""),
    r"C:\tools\node-v22.20.0-win-x64\node.exe",
)
PACKAGES = {"esbuild": "0.25.10", "react": "18.3.1", "react-dom": "18.3.1"}

RUNNER = r"""
const fs = require("fs");
const vm = require("vm");
const path = require("path");
const esbuild = require("esbuild");
const React = require("react");
const ReactDOMServer = require("react-dom/server");

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const out = [];
for (const item of input.items) {
  const res = { ok: false, kind: "react.build_error", detail: "", html_bytes: 0 };
  let js;
  try {
    js = esbuild.transformSync(item.code, {
      loader: "jsx", format: "cjs", jsx: "transform", logLevel: "silent",
    }).code;
  } catch (e) {
    const m = (e.errors && e.errors[0]) || {};
    const loc = m.location ? ` (line ${m.location.line}:${m.location.column})` : "";
    res.detail = (m.text || String(e)).slice(0, 300) + loc;
    out.push(res);
    continue;
  }
  if (!input.render) {
    res.ok = true; res.kind = "react.ok"; res.detail = "built";
    out.push(res);
    continue;
  }
  const module = { exports: {} };
  const sandbox = {
    module, exports: module.exports, React, ReactDOMServer,
    require: (name) => {
      if (name === "react") return React;
      throw new Error(`import of ${JSON.stringify(name)} is not available`);
    },
    console: { log() {}, warn() {}, error() {} },
  };
  const ctx = vm.createContext(sandbox);
  try {
    vm.runInContext(js, ctx, { timeout: input.timeout_ms });
  } catch (e) {
    const timeout = String(e && e.code) === "ERR_SCRIPT_EXECUTION_TIMEOUT";
    res.kind = timeout ? "react.timeout" : "react.render_error";
    res.detail = ("module: " + String(e && e.message || e)).slice(0, 300);
    out.push(res);
    continue;
  }
  let comp = module.exports && (module.exports.default || module.exports);
  if (typeof comp !== "function") {
    const names = Object.keys(module.exports || {}).filter((k) => /^[A-Z]/.test(k));
    comp = names.length === 1 ? module.exports[names[0]] : undefined;
  }
  if (typeof comp !== "function") {
    res.kind = "react.no_component";
    res.detail = "no default-exported component function";
    out.push(res);
    continue;
  }
  ctx.__Comp = comp;
  try {
    const html = vm.runInContext(
      "ReactDOMServer.renderToString(React.createElement(__Comp))", ctx,
      { timeout: input.timeout_ms });
    res.ok = true; res.kind = "react.ok"; res.detail = "rendered";
    res.html_bytes = Buffer.byteLength(html, "utf8");
    if (input.keep_html) res.html = html;
  } catch (e) {
    const timeout = String(e && e.code) === "ERR_SCRIPT_EXECUTION_TIMEOUT";
    res.kind = timeout ? "react.timeout" : "react.render_error";
    res.detail = ("render: " + String(e && e.message || e)).slice(0, 300);
  }
  out.push(res);
}
process.stdout.write(JSON.stringify(out));
"""


def node_binary() -> str | None:
    for candidate in NODE_CANDIDATES:
        if candidate and Path(candidate).is_file():
            return candidate
    return shutil.which("node")


def _npm() -> str | None:
    node = node_binary()
    if node:
        sibling = Path(node).with_name("npm.cmd" if os.name == "nt" else "npm")
        if sibling.is_file():
            return str(sibling)
    return shutil.which("npm")


def toolchain_ready() -> bool:
    return bool(node_binary()) and all(
        (TOOL_DIR / "node_modules" / name / "package.json").is_file() for name in PACKAGES
    )


def ensure_toolchain() -> bool:
    """Install the pinned packages into `TOOL_DIR` if missing. Returns readiness."""
    if toolchain_ready():
        _write_runner()
        return True
    npm = _npm()
    if not npm:
        return False
    TOOL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"name": "dreamscript-react-check", "private": True, "dependencies": PACKAGES}
    (TOOL_DIR / "package.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    env = dict(
        os.environ, PATH=str(Path(node_binary() or npm).parent) + os.pathsep + os.environ["PATH"]
    )
    subprocess.run([npm, "install", "--no-audit", "--no-fund"], cwd=TOOL_DIR, env=env, check=False)
    _write_runner()
    return toolchain_ready()


def _write_runner() -> Path:
    path = TOOL_DIR / "check_react.cjs"
    if not path.is_file() or path.read_text(encoding="utf-8") != RUNNER:
        path.write_text(RUNNER, encoding="utf-8")
    return path


def check_many(
    codes: list[str],
    *,
    render: bool = True,
    timeout_ms: int = 2000,
    keep_html: bool = False,
    batch: int = 500,
) -> list[dict]:
    """Build (and render) every component. One node process per `batch` components."""
    if not codes:
        return []
    if not toolchain_ready():
        return [
            {"ok": False, "kind": "react.unavailable", "detail": "node/esbuild/react missing"}
            for _ in codes
        ]
    runner = _write_runner()
    results: list[dict] = []
    for start in range(0, len(codes), batch):
        chunk = codes[start : start + batch]
        payload = json.dumps(
            {
                "items": [{"code": c} for c in chunk],
                "render": render,
                "timeout_ms": timeout_ms,
                "keep_html": keep_html,
            }
        )
        proc = subprocess.run(
            [node_binary(), str(runner)],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=TOOL_DIR,
            timeout=60 + len(chunk) * (timeout_ms / 1000.0) * 2,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout)[-300:]
            results += [{"ok": False, "kind": "react.crash", "detail": detail} for _ in chunk]
            continue
        results += json.loads(proc.stdout)
    return results


def check(code: str, *, render: bool = True) -> dict:
    """`check_many` for one component."""
    return check_many([code], render=render)[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 12.3.6 React build + render check")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--study", action="store_true")
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    if args.install:
        print(json.dumps({"ready": ensure_toolchain(), "dir": str(TOOL_DIR)}))
    if args.study:
        print(json.dumps(study(), indent=2))
    if args.files:
        codes = [p.read_text(encoding="utf-8") for p in args.files]
        for path, result in zip(args.files, check_many(codes), strict=True):
            print(path, json.dumps(result))
    return 0


# -- the 12.3.6 study: real targets must render, broken ones must not ----------------------------

MUTATIONS = (
    "drop_closing_tag",
    "undefined_identifier",
    "missing_import",
    "infinite_loop",
    "no_export",
    "invalid_expression",
    "throws_in_render",
    "returns_object",
)


def mutate(code: str, kind: str) -> str | None:
    """One known defect injected into a working component, or None when there is no site."""
    if kind == "drop_closing_tag":
        index = code.rfind("</")
        if index <= 0:
            return None
        end = code.find(">", index)
        return code[:index] + code[end + 1 :]
    head, sep, rest = code.partition("return (")
    if not sep:
        return None
    if kind == "undefined_identifier":
        return f"{head}const rows = items.map((x) => x);\n  return ({rest}"
    if kind == "missing_import":
        return f'import {{ Card }} from "@/components/card";\n{code}'
    if kind == "infinite_loop":
        return f"{head}while (true) {{}}\n  return ({rest}"
    if kind == "no_export":
        return code.replace("export default function", "function", 1)
    if kind == "invalid_expression":
        return f"{head}const total = 1 +;\n  return ({rest}"
    if kind == "throws_in_render":
        return f'{head}throw new Error("boom");\n  return ({rest}'
    if kind == "returns_object":
        return f"{head}return {{ a: 1 }};\n  return ({rest}"
    raise ValueError(kind)


def study(limit_per_source: int | None = None) -> dict:
    """Render every React target, then every mutation of a sample; compare to the static gate."""
    import random
    import time

    from src.codegen import quality
    from src.codegen.pairs import load_pairs

    sources: dict[str, list[str]] = {}
    for record in load_pairs(diagram_types=["wireframe"]):
        sources.setdefault(record["source"], []).append(record["target_code"])
    try:
        from src.codegen.sketch2code import load_pairs as html_pairs

        sources["sketch2code_html"] = [r["target_code"] for r in html_pairs()]
    except FileNotFoundError:
        pass
    report: dict = {"targets": {}, "mutations": {}}
    for name, codes in sources.items():
        codes = codes[:limit_per_source] if limit_per_source else codes
        started = time.time()
        results = check_many(codes)
        seconds = time.time() - started
        report["targets"][name] = {
            "components": len(codes),
            "rendered": sum(r["ok"] for r in results),
            "kinds": dict(Counter(r["kind"] for r in results)),
            "per_component_ms": round(1000 * seconds / max(1, len(codes)), 2),
        }
    rng = random.Random(0)
    pool = sources.get("synthetic", [])[:150] + sources.get("sketch2code", [])[:100]
    pool += sources.get("sketch2code_html", [])[:50]
    rng.shuffle(pool)
    for kind in MUTATIONS:
        broken = [m for m in (mutate(c, kind) for c in pool) if m is not None]
        results = check_many(broken, timeout_ms=1000)
        report["mutations"][kind] = {
            "applied": len(broken),
            "rejected_by_render_check": sum(not r["ok"] for r in results),
            "rejected_by_structural_check": sum(not quality.check_react(m)[0] for m in broken),
            "kinds": dict(Counter(r["kind"] for r in results)),
        }
    return report


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
