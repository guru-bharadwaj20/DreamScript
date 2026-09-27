"""Phase 16.3.2 — package the built client as a release archive with a hash you can check.

The APK this row used to describe was going to be protected by a signature. There is no APK and
there is no signing key, so integrity moves to **a SHA-256 published beside the artefact**. That is
weaker and it is worth saying how: a signature proves who built it, and a hash proves only that the
bytes you downloaded are the bytes whose hash was published — by whoever published the hash, in the
same place. Anybody who can replace the archive on a release page can replace the checksum next to
it. What the hash does defend is the ordinary failure: a truncated download, a corrupted mirror, a
proxy that rewrote something.

## The archive is deterministic, and that is not decoration

Every entry is written with a fixed timestamp, fixed permissions, sorted order and no system
metadata, so the same `dist/` always produces the same bytes and therefore the same hash. Without
it the checksum would change on every run for no reason a reader could see, which makes it useless
as a thing to compare — and it would be impossible to tell "somebody rebuilt it" from "somebody
changed it".

## What is in it

`dist/` exactly as `npm run build` wrote it, minus the source maps. The maps are 700 kB, they are
for a developer with the repository, and shipping them in a release is shipping the source of a
thing somebody downloaded to run.

Usage::

    .venv/Scripts/python.exe scripts/make_release.py             # package what is in dist/
    .venv/Scripts/python.exe scripts/make_release.py --verify    # rebuild the archive and compare
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "app" / "frontend"
DIST = FRONTEND / "dist"
OUT = ROOT / "release"
DOC = ROOT / "docs" / "install.md"

#: A fixed timestamp for every entry: 1980-01-01, the earliest a zip can represent.
#:
#: Not "now". An archive whose hash changes every time it is built cannot be compared against a
#: published one, which is the only thing a published hash is for.
EPOCH = (1980, 1, 1, 0, 0, 0)

#: Excluded from the archive. Source maps are for a developer with the repository; a release is
#: for somebody who wants to run the thing.
EXCLUDE_SUFFIXES = (".map",)


def version() -> str:
    return json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))["version"]


def contents() -> list[Path]:
    if not (DIST / "index.html").is_file():
        raise SystemExit(f"no build in {DIST.relative_to(ROOT)} - run `make bundle` first")
    return sorted(
        path
        for path in DIST.rglob("*")
        if path.is_file() and not path.name.endswith(EXCLUDE_SUFFIXES)
    )


def pack(target: Path) -> Path:
    """Write the archive. Byte-identical for identical input."""
    target.parent.mkdir(parents=True, exist_ok=True)
    files = contents()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            # The path inside the archive, with a single top-level directory so unzipping in a
            # home directory does not scatter thirteen files across it.
            arcname = f"dreamscript-app/{path.relative_to(DIST).as_posix()}"
            info = zipfile.ZipInfo(arcname, date_time=EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            # 0o644, as a regular file. Whatever the building filesystem thinks, which on Windows
            # is nothing useful and on Linux is whatever the umask was.
            info.external_attr = (0o100644) << 16
            info.create_system = 3  # Unix, so the mode above is read rather than ignored
            archive.writestr(info, path.read_bytes())
    return target


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def manifest_of(archive: Path) -> list[dict[str, object]]:
    with zipfile.ZipFile(archive) as opened:
        return [
            {"path": info.filename, "bytes": info.file_size, "crc": f"{info.CRC:08x}"}
            for info in sorted(opened.infolist(), key=lambda i: i.filename)
        ]


def install_doc(archive: Path, sha: str, files: list[dict[str, object]], tag: str) -> str:
    total = sum(int(entry["bytes"]) for entry in files)
    lines = [
        "# Installing the DreamScript app",
        "",
        "Phase 16.3.2. The client is an installable web app rather than an APK — Phase 16's",
        "preamble in `contributing.md` records why, and the short version is that it has to run on",
        "iOS too and nothing in this project can compile an Android binary.",
        "",
        "## What you are downloading",
        "",
        "| | |",
        "| :--- | :--- |",
        f"| archive | `{archive.name}` |",
        f"| release | `{tag}` |",
        f"| SHA-256 | `{sha}` |",
        f"| files | {len(files)} |",
        f"| unpacked | {total / 1024:.1f} kB |",
        "",
        "It is a static bundle: HTML, one hashed JavaScript file, one hashed stylesheet, the web",
        "app manifest, the service worker, four icons and the five example sketches. There is no",
        "installer and nothing is executed to unpack it.",
        "",
        "## Check the hash before you use it",
        "",
        "```sh",
        f"sha256sum {archive.name}          # Linux, macOS",
        f"certutil -hashfile {archive.name} SHA256   # Windows",
        "```",
        "",
        f"It must print `{sha}`.",
        "",
        "That hash is **this note's build**. A published release carries its own `.sha256` beside",
        "the archive, built by `.github/workflows/release.yml` on a clean runner, and that is the",
        "authoritative one for the file you downloaded - a note committed to git names one",
        "release's hash and is stale the moment there is a second. The archive is deterministic, so",
        "the two agree whenever the builds do, and the workflow says in its log whether they did.",
        "",
        "**What that proves, and what it does not.** It proves the bytes you have are the bytes",
        "whose hash was published. It does not prove who built them: there is no signing key in",
        "this project, and anyone who can replace the archive on a release page can replace the",
        "checksum printed beside it. The hash defends against a truncated download or a corrupted",
        "mirror, which is the failure that actually happens.",
        "",
        "## Serving it",
        "",
        "The app talks to the backend over **same-origin paths** (`/predict`, `/ir/…`), so it has",
        "to be served by something that also answers those. The backend does both:",
        "",
        "```sh",
        f"unzip {archive.name}",
        "DREAMSCRIPT_BUNDLE=./dreamscript-app \\",
        "  python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 3000",
        "```",
        "",
        "`GET /health` then reports the directory it is serving under `bundle`, so a deployment",
        "that answers the API and shows a blank page is diagnosable rather than a guess.",
        "",
        "Serving the bundle from a plain static server works for the offline examples and nothing",
        "else — every photograph needs the API, and a different origin means CORS, which this",
        "project deliberately does not have anywhere.",
        "",
        "## What it needs",
        "",
        "- **A secure origin.** `getUserMedia` refuses anything but HTTPS or `localhost`, and so",
        "  does the service worker. Over a LAN address both are unavailable without a certificate.",
        "- **A reachable backend**, which needs a reachable model server behind it. With neither,",
        "  the app still opens and shows its five stored example readings, labelled as stored.",
        "",
        "## Installing it on a phone",
        "",
        "- **Android / Chrome**: open it, then use the install prompt, or the About screen's",
        "  *Install DreamScript* button.",
        "- **iOS / Safari**: Share → Add to Home Screen. iOS has no install prompt for a web app",
        "  and Chrome on iOS cannot do it — the About screen says so there too.",
        "",
        "Installed, it opens full screen with no browser chrome over the viewfinder, and the shell",
        "loads with no network.",
        "",
        "## Your photographs leave the device",
        "",
        "The models do not run in the browser. This is not a detail of the deployment — it is what",
        "the app is, and the landing screen says so before the shutter is pressed. Every photograph",
        "is uploaded to the backend, which passes it to the model server. What the server keeps is",
        "the *result* — the shapes, the arrows, the code — under a short id, so the app can show it",
        "again without a second upload; it holds the most recent 500 and drops the oldest. Labels",
        "you correct are kept permanently, because corrections are training data.",
        "",
        "The full statement is `docs/privacy.md` (16.3.4). It is stated here as well rather than",
        "only linked, because an install page that defers the one thing a person should know before",
        "installing has not told them.",
        "",
        "## Contents",
        "",
        "| File | Bytes |",
        "| :--- | ---: |",
    ]
    lines += [f"| `{entry['path']}` | {entry['bytes']} |" for entry in files]
    lines += [
        "",
        "Generated by `scripts/make_release.py`. The archive is deterministic: every entry is",
        "written with a fixed timestamp and fixed permissions in sorted order, so the same build",
        "always produces the same hash.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=None, help="release tag (default: v<package version>)")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="build the archive twice and require the two to be byte-identical",
    )
    args = parser.parse_args(argv)

    tag = args.tag or f"v{version()}"
    archive = OUT / f"dreamscript-app-{version()}.zip"
    pack(archive)
    sha = digest(archive)

    if args.verify:
        # Determinism is the property the whole published-hash scheme rests on, so it is checked
        # rather than asserted in a docstring.
        again = OUT / f".verify-{archive.name}"
        pack(again)
        second = digest(again)
        again.unlink()
        if second != sha:
            print(f"the archive is not deterministic: {sha} != {second}", file=sys.stderr)
            return 1
        print("deterministic: two packs of the same build hash identically")

    files = manifest_of(archive)
    (OUT / f"{archive.name}.sha256").write_text(f"{sha}  {archive.name}\n", encoding="utf-8")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(install_doc(archive, sha, files, tag), encoding="utf-8")

    print(f"  {archive.relative_to(ROOT)}  {archive.stat().st_size / 1024:.1f} kB")
    print(f"  sha256  {sha}")
    print(f"  {len(files)} files, {sum(int(f['bytes']) for f in files) / 1024:.1f} kB unpacked")
    print(f"  wrote {DOC.relative_to(ROOT)} and {archive.name}.sha256")
    print(f"  tag    {tag}")
    if shutil.which("gh") is None:
        print("  (gh is not on PATH - publish by hand)")
    return 0


if __name__ == "__main__":  # pragma: no cover - a script
    raise SystemExit(main())
