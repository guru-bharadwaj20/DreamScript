"""Phase 16.3.2 - the release archive, and the backend serving it.

Two claims, and both of them are the kind that are true on the machine that wrote them and false
everywhere else unless something checks.

**The archive is deterministic.** Integrity here is a published SHA-256 rather than a signature -
there is no APK and no signing key - and a checksum is only worth publishing if the same build
always produces it. An archive whose hash moved on every run would make "somebody rebuilt it"
indistinguishable from "somebody changed it".

**The backend serves the client.** A mount at `/` matches every path, so one registered before the
API routes would make the API unreachable - the single way this can go wrong, and it fails silently
in the direction of a working-looking app whose uploads 404.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from app.backend import limits as limits_module  # noqa: E402
from app.backend.main import BUNDLE_PREFIXES, build_app, bundle_dir  # noqa: E402
from app.backend.store import Store  # noqa: E402
from scripts import make_release  # noqa: E402
from src.utils.config import ROOT  # noqa: E402

FRONTEND = ROOT / "app" / "frontend"
DIST = FRONTEND / "dist"
built = pytest.mark.skipif(
    not (DIST / "index.html").is_file(),
    reason="no client build - run `make bundle`",
)


def bundle(tmp_path: Path) -> Path:
    """A minimal directory shaped like a built client, for the mount tests."""
    root = tmp_path / "unpacked"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<!doctype html><title>x</title>", encoding="utf-8")
    (root / "assets" / "index-abc123.js").write_text("export default 1;\n", encoding="utf-8")
    (root / "manifest.webmanifest").write_text('{"name":"x"}', encoding="utf-8")
    return root


# == the archive ================================================================================


@built
def test_packing_the_same_build_twice_gives_the_same_bytes(tmp_path):
    """The property the whole published-hash scheme rests on.

    Fixed timestamps, fixed permissions and sorted order - a zip written the obvious way records
    the mtime of every file and the umask of the machine, so two packs of one build differ.
    """
    first = make_release.pack(tmp_path / "a.zip")
    second = make_release.pack(tmp_path / "b.zip")
    assert first.read_bytes() == second.read_bytes()
    assert make_release.digest(first) == make_release.digest(second)


@built
def test_every_entry_has_the_fixed_timestamp_rather_than_the_machines_clock(tmp_path):
    with zipfile.ZipFile(make_release.pack(tmp_path / "a.zip")) as archive:
        for info in archive.infolist():
            assert info.date_time == make_release.EPOCH, info.filename


@built
def test_the_archive_unpacks_into_one_directory_rather_than_over_the_cwd(tmp_path):
    """`unzip` in a home directory should not scatter fourteen files across it."""
    with zipfile.ZipFile(make_release.pack(tmp_path / "a.zip")) as archive:
        tops = {name.split("/")[0] for name in archive.namelist()}
    assert tops == {"dreamscript-app"}


@built
def test_source_maps_are_not_shipped(tmp_path):
    """700 kB, for a developer with the repository. A release is for somebody who wants to run it."""
    with zipfile.ZipFile(make_release.pack(tmp_path / "a.zip")) as archive:
        assert not [n for n in archive.namelist() if n.endswith(".map")]


@built
def test_the_archive_carries_everything_the_app_needs_to_start(tmp_path):
    with zipfile.ZipFile(make_release.pack(tmp_path / "a.zip")) as archive:
        names = {n.removeprefix("dreamscript-app/") for n in archive.namelist()}
    # The worker is in the archive even though it is deliberately out of its own precache list.
    for needed in ("index.html", "manifest.webmanifest", "sw.js"):
        assert needed in names, needed
    assert any(n.startswith("assets/") and n.endswith(".js") for n in names)
    assert len([n for n in names if n.startswith("examples/")]) == 5
    assert len([n for n in names if n.startswith("icons/")]) == 4


def test_the_install_note_publishes_a_hash_and_says_what_it_does_not_prove():
    """A checksum with no caveat invites the reading it cannot support.

    It proves the bytes match the published hash. It does not prove who built them: anyone who can
    replace the archive on a release page can replace the checksum printed beside it.
    """
    doc = (ROOT / "docs" / "install.md").read_text(encoding="utf-8")
    sha = re.search(r"SHA-256 \| `([0-9a-f]{64})`", doc)
    assert sha, "no SHA-256 in the install note"
    assert doc.count(sha.group(1)) >= 2, "the hash is published once and never repeated to check"
    assert "does not prove who built them" in doc
    assert "DREAMSCRIPT_BUNDLE" in doc, "no instructions for serving it"
    # The one thing a person should know before installing is stated here rather than only linked.
    assert "uploaded to the backend" in doc


@built
def test_the_published_hash_is_the_hash_of_the_archive_that_was_packaged():
    """The doc and the `.sha256` file are two descriptions of one artefact, so they are joined."""
    archive = ROOT / "release" / f"dreamscript-app-{make_release.version()}.zip"
    if not archive.is_file():
        pytest.skip("no packaged archive - run `make release`")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    doc = (ROOT / "docs" / "install.md").read_text(encoding="utf-8")
    assert sha in doc, "the install note names a different build"
    sidecar = archive.with_suffix(".zip.sha256").read_text(encoding="utf-8")
    assert sidecar.split()[0] == sha
    assert sidecar.split()[1] == archive.name


def test_the_release_is_not_committed_but_the_install_note_is():
    """A 154 kB binary regenerated by one command does not belong in git; the note does."""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert re.search(r"(?m)^release/", ignore), "release/ is not ignored"
    assert (ROOT / "docs" / "install.md").is_file()


def test_the_version_in_the_archive_name_is_the_clients_own():
    assert (
        make_release.version()
        == json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))["version"]
    )


# == the backend serving it ======================================================================


def test_the_backend_serves_a_bundle_when_one_is_pointed_at(tmp_path, monkeypatch):
    monkeypatch.setenv("DREAMSCRIPT_BUNDLE", str(bundle(tmp_path)))
    client = TestClient(build_app(store=Store(tmp_path / "state")))
    root = client.get("/")
    assert root.status_code == 200
    assert root.headers["content-type"].startswith("text/html")
    assert client.get("/assets/index-abc123.js").status_code == 200


def test_the_manifest_is_served_as_a_manifest_and_not_as_a_binary(tmp_path, monkeypatch):
    """`.webmanifest` is not in Python's mimetypes table, and the default is octet-stream.

    Browsers are lenient about it and one platform deciding otherwise is the kind of failure that
    shows up only on somebody else's phone.
    """
    monkeypatch.setenv("DREAMSCRIPT_BUNDLE", str(bundle(tmp_path)))
    client = TestClient(build_app(store=Store(tmp_path / "state")))
    response = client.get("/manifest.webmanifest")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/manifest+json")


def test_the_api_still_answers_with_a_bundle_mounted(tmp_path, monkeypatch):
    """The single way this can go wrong: a mount at `/` matches every path.

    Registered before the routes it would make the whole API unreachable, and it would look like a
    working app whose uploads 404 - which points nowhere near the mount.
    """
    monkeypatch.setenv("DREAMSCRIPT_BUNDLE", str(bundle(tmp_path)))
    client = TestClient(build_app(store=Store(tmp_path / "state")))
    assert client.get("/health").json()["status"] == "ok"
    # A route that exists, answering about an id that does not - not the static mount's 404.
    missing = client.get("/predict/deadbeefdeadbeef")
    assert missing.status_code == 404
    assert missing.json()["detail"]
    assert client.get("/openapi.json").status_code == 200


def test_health_reports_which_bundle_is_being_served(tmp_path, monkeypatch):
    """A deployment that answers the API and shows a blank page is otherwise diagnosed by guessing."""
    root = bundle(tmp_path)
    monkeypatch.setenv("DREAMSCRIPT_BUNDLE", str(root))
    client = TestClient(build_app(store=Store(tmp_path / "state")))
    assert client.get("/health").json()["bundle"] == str(root)


def test_a_checkout_with_no_build_mounts_nothing_rather_than_404ing_its_own_root(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DREAMSCRIPT_BUNDLE", str(tmp_path / "nothing-here"))
    client = TestClient(build_app(store=Store(tmp_path / "state")))
    assert client.get("/health").json()["bundle"] is None
    assert client.get("/").status_code == 404


def test_the_bundle_directory_is_read_per_call_and_not_bound_at_import(tmp_path, monkeypatch):
    """A module-level default is evaluated once, and a process that sets the environment afterwards
    gets the old value - the same trap `trusted_proxy_hops` documents."""
    monkeypatch.setenv("DREAMSCRIPT_BUNDLE", str(tmp_path / "absent"))
    assert bundle_dir() is None
    monkeypatch.setenv("DREAMSCRIPT_BUNDLE", str(bundle(tmp_path)))
    assert bundle_dir() is not None


def test_the_clients_own_assets_do_not_draw_on_the_rate_limit(tmp_path, monkeypatch):
    """A cold load of the gallery is thirteen requests off local disk.

    Six reloads would spend most of a 120-per-minute read budget on the app's own stylesheet, and a
    429 for a stylesheet renders as a page with no CSS - which looks like a catastrophe and is a
    rate limit.
    """
    assert set(BUNDLE_PREFIXES) == set(limits_module.EXEMPT_PREFIXES)
    monkeypatch.setenv("DREAMSCRIPT_BUNDLE", str(bundle(tmp_path)))
    client = TestClient(build_app(store=Store(tmp_path / "state")))
    for _ in range(200):
        assert client.get("/assets/index-abc123.js").status_code == 200


def test_the_mount_is_the_last_thing_the_factory_does():
    """Text, because the ordering is the whole risk and a test that exercised it would need every
    route registered after a mount to prove the negative."""
    source = (ROOT / "app" / "backend" / "main.py").read_text(encoding="utf-8")
    body = source[source.index("def build_app(") :]
    mount = body.index("serve_bundle(app)")
    assert body.index("    return app", mount) > mount
    # Nothing registers a route after it.
    assert "@app." not in body[mount:], "a route is declared after the mount and is unreachable"


# == 16.3.3: the release workflow ===============================================================
#
# A workflow is a description of a machine nobody has, and every claim in it is unverifiable until
# it runs. What *is* checkable here is that it says what the rest of the repository says: the same
# node version, the same install command, the same packaging script - because a release built from
# a resolved-today dependency tree, or by a job that quietly skipped the typecheck, is a release
# nobody can reproduce and nobody would know to distrust.


def release_workflow() -> dict:
    return yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text("utf-8"))


def release_steps() -> list[dict]:
    return release_workflow()["jobs"]["bundle"]["steps"]


def test_the_release_workflow_fires_on_a_tag_and_nothing_else():
    """On every push it would build a release archive nobody asked for, on every PR twice."""
    triggers = release_workflow()[True]  # `on:` is parsed as the boolean True by yaml 1.1
    assert list(triggers["push"]["tags"]) == ["v*"]
    assert "pull_request" not in triggers
    # `workflow_dispatch` so the job can be exercised without minting a tag it cannot un-mint.
    assert "workflow_dispatch" in triggers


def test_the_workflow_may_write_contents_and_nothing_more():
    """`gh release create` needs `contents: write`. Nothing here needs packages, pages or id-token,
    and a token with more scope than the job uses is scope somebody else's action inherits."""
    assert release_workflow()["permissions"] == {"contents": "write"}


def test_the_tag_and_the_clients_version_have_to_agree():
    """`v16.3.0` must package `16.3.0`.

    A tag that disagrees produces an archive whose name says one thing and whose release says
    another, and nothing downstream would notice - so it fails before anything is built.
    """
    steps = release_steps()
    check = next((s for s in steps if "version" in (s.get("name") or "")), None)
    assert check, "nothing compares the tag against package.json"
    assert "package.json" in check["run"]
    assert "exit 1" in check["run"]
    # First, so a mismatch costs no build.
    assert steps.index(check) < min(
        index for index, step in enumerate(steps) if "npm ci" in (step.get("run") or "")
    )


def test_the_release_is_built_from_the_committed_lockfile():
    """`npm ci`, not `npm install`. The lockfile is what pins the build, which is the whole reason
    it is in git - and a release is the one artefact where that matters most."""
    runs = " ".join(step.get("run") or "" for step in release_steps())
    assert "npm ci" in runs
    assert "npm install" not in runs


def test_the_release_job_runs_the_typecheck_and_the_client_suite():
    """esbuild strips types without reading them, so `vite build` alone ships type errors.

    `npm run build` is `tsc --noEmit && vite build`, and `npm test` is the suite whose SSE
    chunk-boundary sweep is the reason a release is a bad place to find out the reader broke.
    """
    runs = " ".join(step.get("run") or "" for step in release_steps())
    assert "npm run build" in runs
    assert "npm test" in runs


def test_the_release_job_requires_the_archive_to_be_deterministic():
    """`--verify` packs twice and compares. Without it, a published hash means nothing."""
    runs = " ".join(step.get("run") or "" for step in release_steps())
    assert "scripts/make_release.py --verify" in runs


def test_the_node_version_is_the_one_ci_already_uses():
    """Two workflows building the same client on two runtimes is two builds to explain."""
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8"))
    ci_node = next(
        step["with"]["node-version"]
        for step in ci["jobs"]["client"]["steps"]
        if "setup-node" in (step.get("uses") or "")
    )
    release_node = next(
        step["with"]["node-version"]
        for step in release_steps()
        if "setup-node" in (step.get("uses") or "")
    )
    assert ci_node == release_node


def test_the_assets_attached_are_the_archive_its_hash_and_the_instructions():
    """Three files, and the second is useless without being beside the first."""
    publish = next(s for s in release_steps() if "gh release create" in (s.get("run") or ""))
    for asset in ("release/*.zip", "release/*.sha256", "docs/install.md"):
        assert asset in publish["run"], asset
    # Only on a tag: `workflow_dispatch` has no release to publish to.
    assert "startsWith(github.ref, 'refs/tags/')" in publish["if"]


def test_the_release_body_carries_the_same_caveat_the_install_note_does():
    """The hash is most likely to be read on the release page, so the caveat belongs there too.

    A checksum published with no caveat invites the reading it cannot support - that it proves who
    built the file.
    """
    publish = next(s for s in release_steps() if "gh release create" in (s.get("run") or ""))
    assert "does not" in publish["run"] and "prove who built them" in publish["run"]


def test_publishing_uses_gh_rather_than_a_third_party_action():
    """`gh` is on every runner and needs no pinning to a commit sha to be trustworthy.

    A third-party action in a job holding a `contents: write` token is a supply chain in the one
    workflow that produces the thing people download.
    """
    steps = release_steps()
    allowed = {
        "actions/checkout",
        "actions/setup-node",
        "actions/setup-python",
        "actions/upload-artifact",
    }
    for step in steps:
        uses = step.get("uses")
        if not uses:
            continue
        assert uses.split("@")[0] in allowed, uses
