"""Phase 16.2.1 - the client's skeleton, checked as text.

Nothing here runs a browser or a build. Those happen in `npm run build` and in the screenshot pass,
and a pytest suite is the wrong place to install 67 npm packages. What *is* checked here is the set
of claims a Python test can hold honestly: that the tree exists, that its pins say what the repo says
elsewhere, that the mobile-specific pieces which are easy to delete and hard to notice are present,
and that the client makes no claim the backend cannot keep.

The last one is the reason this file is worth having. `app/frontend/src/lib/api.ts` is a second
description of `app/backend/main.py`'s routes, and two descriptions of one contract drift. So the
paths the client calls are compared against the routes the backend actually registers.
"""

from __future__ import annotations

import json
import re

import pytest

from src.utils.config import ROOT

FRONTEND = ROOT / "app" / "frontend"

pytestmark = pytest.mark.skipif(not FRONTEND.is_dir(), reason="the client is not present")


def read(*parts: str) -> str:
    return (FRONTEND.joinpath(*parts)).read_text(encoding="utf-8")


def package() -> dict:
    return json.loads(read("package.json"))


def without_comments(text: str) -> str:
    """Source with `/* */`, `//` and `<!-- -->` removed.

    Three of these tests failed on their first run by matching the *comment* that explains why a
    thing is absent - `maximum-scale` inside "`maximum-scale` is deliberately absent", `VITE_API`
    inside "no `VITE_API_URL` to be wrong". A test that a well-documented file cannot pass is a test
    that punishes documentation.
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", text)


def viewport() -> str:
    """The viewport meta's `content`, which is the only place these directives mean anything."""
    match = re.search(r'<meta\s+name="viewport"\s+content="([^"]*)"', read("index.html"))
    assert match, "no viewport meta"
    return match.group(1)


# == the tree =================================================================================


@pytest.mark.parametrize(
    "path",
    [
        "package.json",
        "package-lock.json",
        "tsconfig.json",
        "vite.config.ts",
        "index.html",
        "src/main.tsx",
        "src/App.tsx",
        "src/styles/tokens.css",
        "src/styles/base.css",
        "src/lib/api.ts",
        "src/lib/theme.ts",
        "src/lib/route.ts",
        "src/ui/index.tsx",
    ],
)
def test_the_skeleton_is_there(path: str):
    assert (FRONTEND / path).is_file(), path


def test_the_lockfile_is_committed_because_it_is_what_pins_the_build():
    """`node_modules` is ignored and `dist` is ignored; the lockfile is the only thing in git that
    decides what `npm ci` restores. Ignoring it too would make the build unreproducible."""
    tracked = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "app/frontend/node_modules/" in tracked
    assert "app/frontend/dist/" in tracked
    assert "package-lock.json" not in tracked
    assert (FRONTEND / "package-lock.json").is_file()


def test_react_is_pinned_to_the_version_the_generated_wireframes_are_checked_against():
    """12.3.2 builds and renders generated JSX against react 18.3.1 (`src/eval/react.py`'s
    `PACKAGES`). A client on a different major would be showing code it validates differently."""
    from src.eval.react import PACKAGES

    assert package()["dependencies"]["react"] == PACKAGES["react"]
    assert package()["dependencies"]["react-dom"] == PACKAGES["react-dom"]


def test_every_dependency_is_pinned_exactly_like_every_requirements_layer():
    """`^` and `~` in a manifest are how two machines build different bundles from one commit."""
    manifest = package()
    loose = {
        name: spec
        for group in ("dependencies", "devDependencies")
        for name, spec in manifest.get(group, {}).items()
        if not re.fullmatch(r"\d+\.\d+\.\d+", spec)
    }
    assert not loose, f"unpinned: {loose}"


def test_the_build_typechecks_before_it_bundles():
    """`vite build` does not typecheck - esbuild strips types without reading them. A build script
    that is only `vite build` is a build that ships a type error."""
    assert package()["scripts"]["build"].startswith("tsc --noEmit &&")


# == the mobile-specific pieces ===============================================================


def test_the_viewport_opts_into_the_safe_area():
    """Without `viewport-fit=cover` every `env(safe-area-inset-*)` returns 0 on a notched iPhone,
    and the CSS that keeps a control clear of the home bar silently does nothing."""
    assert "viewport-fit=cover" in viewport()
    assert "width=device-width" in viewport()


def test_pinch_zoom_is_not_disabled():
    """`maximum-scale=1` and `user-scalable=no` make an app feel native by taking zoom away from
    people who need it. Neither is here, and this test is why it stays that way.

    Checked against the meta's `content`, not the file: the first version searched the whole HTML
    and matched the comment saying "`maximum-scale` is deliberately absent".
    """
    assert "maximum-scale" not in viewport()
    assert "user-scalable" not in viewport()


@pytest.mark.parametrize(
    ("needle", "why"),
    [
        ("100dvh", "100vh on iOS is taller than the visible viewport"),
        ("env(safe-area-inset-bottom)", "a control under the home indicator is unreachable"),
        ("overscroll-behavior", "pull-to-refresh mid-capture reloads away the photograph"),
        ("-webkit-tap-highlight-color", "Safari's grey flash reads as a rendering fault"),
        ("touch-action", "without it every tap carries a 300 ms double-tap delay"),
        (
            "prefers-reduced-motion",
            "the stage list is exactly the motion that makes some people ill",
        ),
    ],
)
def test_the_reset_keeps_the_eight_lines_that_are_about_phones(needle: str, why: str):
    assert needle in read("src/styles/base.css"), why


def test_the_tap_floor_is_declared_once_and_is_44px():
    tokens = read("src/styles/tokens.css")
    assert re.search(r"--tap:\s*44px", tokens), "44px is the smallest target a thumb hits reliably"


def test_light_and_dark_are_both_defined_and_system_is_a_third_state():
    """A toggle with two states has overridden a choice the person already made for every app on
    their phone."""
    tokens = read("src/styles/tokens.css")
    assert "prefers-color-scheme: dark" in tokens
    assert ':root[data-theme="dark"]' in tokens
    assert ':root:not([data-theme="light"])' in tokens
    theme = read("src/lib/theme.ts")
    assert '"system"' in theme
    assert "removeAttribute" in theme, "system must remove the attribute, not set it to 'system'"


def test_no_web_font_is_fetched():
    """16.3.1 makes this installable and offline-capable; a font over the network is a font missing
    on the one occasion the offline shell exists for."""
    for path in ("index.html", "src/styles/tokens.css", "src/styles/base.css"):
        text = read(path)
        assert "fonts.googleapis" not in text, path
        assert "@import url(" not in text, path
        assert "@font-face" not in text, path


# == the two colour rules the design system makes ============================================


def test_gold_has_a_second_value_for_light_mode():
    """`#e8c36a` on white is about 1.9:1. A palette that inverts cleanly is a palette with no gold
    in it."""
    tokens = read("src/styles/tokens.css")
    assert "--gold-text: #7a5f12" in tokens, "the readable gold for a light ground"
    assert "--gold: #e8c36a" in tokens


def test_degraded_is_not_the_same_hue_as_the_brand():
    """Amber is the obvious caution colour and sits at roughly gold's hue, so a degraded pill in
    amber beside a gold button reads as brand rather than as a warning."""
    tokens = read("src/styles/tokens.css")
    assert "--degraded: #b4510a" in tokens or "--degraded: #fb923c" in tokens
    assert "amber" not in tokens.lower() or "Amber is the obvious" in tokens


def test_the_dot_means_trust_and_only_the_four_trust_tones_have_one():
    """Colour alone is not a signal for roughly one man in twelve, so the four trust pills carry a
    dot as well - which only works as a signal if nothing else does."""
    css = read("src/ui/ui.css")
    block = css.split("/* == pills", 1)[1].split("/* == surfaces", 1)[0]
    dotted = re.findall(r"\.pill-(\w+)::before", block)
    assert set(dotted) == {"ok", "degraded", "stopped", "confirm"}, dotted


# == the client makes no claim the backend cannot keep ========================================


def test_every_route_the_client_calls_is_one_the_backend_registers():
    """Two descriptions of one contract drift. This is the join.

    The client's paths are template literals (`/ir/${id}`), so they are normalised to the backend's
    parameter form before comparison rather than matched loosely - a loose match would accept a
    typo'd route that happens to share a prefix.
    """
    pytest.importorskip("fastapi")
    import tempfile

    from app.backend.main import build_app
    from app.backend.store import Store
    from app.backend.upstream import Upstream

    with tempfile.TemporaryDirectory() as tmp:
        registered = {
            route.path  # type: ignore[attr-defined]
            for route in build_app(store=Store(tmp), upstream=Upstream("http://model:8000")).routes
            if hasattr(route, "path")
        }

    api = without_comments(read("src/lib/api.ts"))
    called = set()
    for raw in re.findall(r'fetch\(\s*[`"]([^`"]+)[`"]', api):
        # Interpolations first, then the query string - in that order, because a `${...}` can itself
        # contain a `?`: `/health${probe ? "?upstream=1" : ""}` split on `?` before substitution and
        # produced a route with a brace in it. One id becomes the backend's parameter name; any
        # remaining interpolation (a query fragment) is dropped.
        path = re.sub(r"\$\{[^{}]*\}", lambda m: "{record_id}" if "id" in m.group(0) else "", raw)
        path = path.split("?")[0].rstrip("/") or "/"
        # `/run/${id}${query}` leaves two substitutions adjacent; collapse the repeat.
        path = re.sub(r"(\{record_id\})+", "{record_id}", path)
        called.add(path)

    assert called, "no fetch calls found - has the client stopped using template literals?"
    missing = {path for path in called if path not in registered}
    assert not missing, f"the client calls routes the backend does not serve: {sorted(missing)}"


def test_the_clients_reason_table_matches_the_only_place_reasons_are_produced():
    """`doubts.ts` turns the tracer's reason codes into sentences, which makes it a second copy of
    a vocabulary defined in `src/ir/model.py`.

    The first version was written from memory and had three names no payload contains
    (`source-open`, `target-open`, `ambiguous-target`), so a real `no-source` fell through to the
    fallback text - correct behaviour concealing an incorrect table, which is the kind of thing only
    a comparison catches.
    """
    model = (ROOT / "src" / "ir" / "model.py").read_text(encoding="utf-8")
    block = model.split("def record_unresolved", 1)[1].split("def sync_unresolved", 1)[0]
    produced = set(re.findall(r'reason = "([a-z-]+)"', block))
    produced |= set(re.findall(r'"reason", "([a-z-]+)"', block))

    client = read("src/lib/doubts.ts")
    table = client.split("const REASONS", 1)[1].split("};", 1)[0]
    explained = set(re.findall(r'"([a-z-]+)":', table))

    assert produced, "no reason codes found in record_unresolved"
    missing = produced - explained
    assert not missing, f"the client has no words for {sorted(missing)}"
    invented = explained - produced
    assert not invented, f"the client explains reasons nothing emits: {sorted(invented)}"


def test_every_route_the_client_calls_is_in_the_dev_proxy():
    """The third description of the same contract, and the one with no compiler behind it.

    `vite.config.ts` lists the paths the dev server forwards to the backend. The list is explicit
    on purpose - a `/api/*` prefix would need a rewrite, and a rewrite is a second description of
    the API - but the cost is that adding a route means editing three places. 16.2.8 edited two:
    the backend registered `POST /correct/{id}`, the client called it, and the correction sheet
    submitted into a **404** that nothing caught, because every other test was satisfied.
    """
    proxied = set(
        re.findall(
            r'"(/[a-z.]+)"', read("vite.config.ts").split("const ROUTES = [", 1)[1].split("]", 1)[0]
        )
    )
    assert proxied, "no ROUTES list found in vite.config.ts"

    api = without_comments(read("src/lib/api.ts"))
    called = {
        "/" + raw.lstrip("/").split("/")[0].split("?")[0].split("$")[0]
        for raw in re.findall(r'fetch\(\s*[`"](/[^`"]+)[`"]', api)
    }
    missing = {path for path in called if path not in proxied}
    assert not missing, (
        f"the client calls {sorted(missing)}, which the dev proxy does not forward - "
        "these would 404 in development while working in production"
    )


def test_the_client_reads_the_four_trust_fields_the_backend_goes_to_trouble_to_send():
    """13.4, 13.7 and 13.8 spent three phases making the pipeline say when it fell back. A client
    that rendered `code` alone would throw that away at the last hop."""
    api = read("src/lib/api.ts")
    for field in ("ok", "degraded", "stopped_at", "needs_confirmation", "stages"):
        assert field in api, field
    ui = read("src/ui/index.tsx")
    assert "TrustPill" in ui
    assert "stoppedAt" in ui and "needsConfirmation" in ui


def test_the_correction_kinds_match_the_backends_closed_set():
    """A log that also contains "typo" cannot be counted against 14's taxonomy, and the client is
    where a free-text kind would get invented."""
    from app.backend.main import FEEDBACK_KINDS

    api = read("src/lib/api.ts")
    declared = set(
        re.findall(r'\|\s*"(\w+)"', api.split("export type CorrectionKind", 1)[1].split(";", 1)[0])
    )
    first = re.search(r'export type CorrectionKind\s*=\s*\|?\s*"(\w+)"', api)
    if first:
        declared.add(first.group(1))
    assert declared == set(FEEDBACK_KINDS), declared ^ set(FEEDBACK_KINDS)


# == the client has its own suite, and CI runs it =============================================


def test_the_client_has_unit_tests_for_the_two_things_text_cannot_check():
    """A grep can assert `camera.ts` mentions `NotAllowedError`. It cannot assert the SSE reader
    survives a chunk boundary, which is the defect that only appears against a real socket."""
    assert (FRONTEND / "src" / "lib" / "stream.test.ts").is_file()
    assert (FRONTEND / "src" / "lib" / "camera.test.ts").is_file()
    assert package()["scripts"]["test"] == "vitest run"


def test_the_sse_reader_is_tested_against_chunk_boundaries():
    """The one test in that file that earns its keep: a parser that assumes each chunk holds whole
    lines passes every hand-written case and loses one frame per read in production."""
    suite = read("src/lib/stream.test.ts")
    assert "however the bytes are split" in suite
    assert "TextEncoder" in suite, "the fixture has to be bytes, not a string"


def test_ci_runs_the_client_suite():
    """A test suite nothing runs is indistinguishable from no test suite."""
    import yaml

    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    client = workflow["jobs"].get("client")
    assert client, "no `client` job in CI"
    runs = [step.get("run") for step in client["steps"] if step.get("run")]
    assert "npm ci" in runs, "`npm install` would ignore the lockfile the repo commits"
    assert "npm test" in runs
    assert "npm run build" in runs


def test_the_camera_asks_for_the_rear_lens_as_ideal_rather_than_exact():
    """`exact: "environment"` fails outright on every laptop, which is where this is developed."""
    camera = without_comments(read("src/lib/camera.ts"))
    assert 'facingMode: { ideal: "environment" }' in camera
    assert 'exact: "environment"' not in camera


def test_the_stream_is_read_from_fetch_rather_than_eventsource():
    """`EventSource` only issues GET. The stream begins by uploading a photograph."""
    stream = without_comments(read("src/lib/stream.ts"))
    assert "EventSource" not in stream
    assert "getReader()" in stream
    assert "AbortSignal" in stream or "signal" in stream


def test_the_clients_ir_types_use_the_field_names_the_schemas_declare():
    """The client's `Node` and `Edge` are a second description of `schemas/*.schema.json`.

    TypeScript cannot catch a misspelling here: both interfaces carry an index signature, so
    `edge.source` on an object that has `src` is a legal `unknown` rather than an error. The first
    version of `api.ts` declared `source`/`target` and `type`, where the IR has `src`/`dst` and
    `shape` - so the overlay found no edges for any node and reported "No arrows touch this shape"
    on every page, with nothing failing anywhere.
    """
    api = without_comments(read("src/lib/api.ts"))

    def declared(interface: str) -> set[str]:
        body = api.split(f"export interface {interface} {{", 1)[1].split("\n}", 1)[0]
        return set(re.findall(r"^\s{2}(\w+)[?]?:", body, re.M))

    for interface, schema in (("Node", "node"), ("Edge", "edge")):
        allowed = set(
            json.loads((ROOT / "schemas" / f"{schema}.schema.json").read_text(encoding="utf-8"))[
                "properties"
            ]
        )
        used = declared(interface) - {"key"}
        unknown = used - allowed
        assert not unknown, f"{interface} declares fields the IR schema does not have: {unknown}"

    # And the ones the overlay cannot work without are actually present.
    assert {"id", "bbox", "text", "confidence", "shape"} <= declared("Node")
    assert {"id", "src", "dst", "polyline", "confidence"} <= declared("Edge")


def test_nothing_in_the_client_reads_an_edge_by_the_wrong_name():
    """The regression for the defect above, at the use sites rather than the declaration."""
    for path in ("src/ui/Overlay.tsx", "src/screens/Result.tsx"):
        source = without_comments(read(path))
        assert ".source ===" not in source, path
        assert ".target ===" not in source, path


def test_the_client_talks_to_the_same_origin_so_there_is_no_cors_to_get_wrong():
    """The backend has no CORS middleware and needs none: Vite proxies in development and the
    backend serves the bundle in production. A `VITE_API_URL` would reintroduce both."""
    api = without_comments(read("src/lib/api.ts"))
    assert "http://localhost:3000" not in api
    assert "VITE_API" not in api
    assert "proxy" in read("vite.config.ts")


# == 16.2.10: the offline bundle ===============================================================
#
# The claim this section defends is provenance. `src/lib/offline.data.json` is the one payload in
# this repository that a person sees on a screen without a pipeline having run to produce it, which
# makes it the easiest place in the project to ship a fabricated answer and the hardest place to
# notice one - a hand-written example looks exactly like the product working perfectly.


def offline_bundle() -> dict:
    return json.loads(read("src/lib/offline.data.json"))


def test_the_offline_bundle_holds_one_answer_per_bundled_example():
    """Three descriptions of one set: the fixtures, the copies the gallery shows, and the answers.

    A gallery card with no stored answer is a card that does nothing when the network is gone, and
    a stored answer for a file the gallery does not offer is dead weight in the bundle.
    """
    files = [entry["file"] for entry in offline_bundle()["examples"]]
    assert len(files) == 5, files
    for file in files:
        assert (ROOT / "tests" / "fixtures" / file).is_file(), f"{file} is not a committed fixture"
        assert (FRONTEND / "public" / "examples" / file).is_file(), f"{file} is not shipped"

    # And the gallery's own list is the same set, in the same order, so the screen cannot offer a
    # sixth example that has no answer behind it.
    gallery = without_comments(read("src/screens/Gallery.tsx"))
    listed = re.findall(r'file:\s*"([^"]+\.png)"', gallery)
    assert listed == files, (listed, files)


def test_every_offline_id_is_one_the_store_and_the_router_would_accept():
    """A non-hex id would make its card land on the capture screen instead of the answer.

    `route.ts` only parses `[0-9a-f]{1,64}` and `store.py`'s `path_for` validates the same shape.
    An id that fails either is an example that is dead in exactly the situation it exists for.
    """
    ids = [entry["id"] for entry in offline_bundle()["examples"]]
    assert len(set(ids)) == len(ids), ids
    for value in ids:
        assert re.fullmatch(r"[0-9a-f]{1,64}", value), value


def test_the_offline_answers_were_captured_with_the_stage_cache_cold():
    """The first capture was taken warm and every stage came back `cached: true` at 0.06 s a page.

    Real, and a lie about the product: an offline screen whose header reads `0.06s` teaches a person
    that this pipeline is instant. Checked on this side as well as in vitest, because a regeneration
    is a Python script and this is the suite that runs in CI without npm.
    """
    for entry in offline_bundle()["examples"]:
        prediction = entry["prediction"]
        stages = prediction["stages"]
        assert [stage["stage"] for stage in stages] == [
            "detect",
            "classify",
            "assemble",
            "traverse",
            "serialise",
            "generate",
            "verify",
        ], entry["file"]
        assert not any(stage["cached"] for stage in stages), f"{entry['file']} was captured warm"
        assert prediction["seconds"] > 0.05, entry["file"]


def test_the_offline_answers_carry_the_degradation_fields_unflattened():
    """13.4's rule at the last hop it can be broken at.

    A summary with `ok` and nothing else would be the one payload in this project that dropped the
    fields every other hop goes to trouble to pass through.
    """
    for entry in offline_bundle()["examples"]:
        prediction = entry["prediction"]
        for field in ("ok", "degraded", "stopped_at", "needs_confirmation", "diagram_type", "ir"):
            assert field in prediction, (entry["file"], field)
        assert isinstance(prediction["ok"], bool)
        assert isinstance(prediction["stages"], list)


def test_the_offline_answers_keep_the_readings_that_are_wrong():
    """Three of five come back as flowcharts, and the bundle keeps it.

    `tests/fixtures/manifest.json` names each fixture after its own diagram type, so the mismatch is
    measurable rather than a matter of opinion. A bundle where all five were right would be the
    signal that someone had tidied the demo - which is the failure this whole section is about.
    """
    wrong = [
        entry["file"]
        for entry in offline_bundle()["examples"]
        if entry["prediction"]["diagram_type"] != entry["file"].removesuffix(".png")
    ]
    assert wrong == ["er_diagram.png", "wireframe.png", "circuit.png"], wrong


def test_the_gallery_does_not_assert_the_misread_count_in_words():
    """The count is computed from the bundle, and it used to be written by hand - wrongly.

    The panel said "two of these are read as the wrong kind" for as long as it existed; capturing
    the answers for the offline cache showed the count is three. A number in prose next to a number
    in data is a number that will be stale, so the prose one is gone and this keeps it gone.
    """
    gallery = without_comments(read("src/screens/Gallery.tsx"))
    for claim in ("Two of these", "two of these", "reads as a flowchart"):
        assert claim not in gallery, claim
    assert "misread()" in gallery, "the count is no longer derived"


def test_the_offline_bundle_stays_small_enough_to_be_worth_shipping():
    """It is in the JS bundle, not in `public/`, so its size is the offline shell's size.

    In the bundle on purpose: a file under `public/` is a separate request, which is exactly the
    thing that is unavailable when this data is needed. The cost of that choice is that it is bytes
    every visitor downloads, so there is a ceiling on it.
    """
    size = (FRONTEND / "src" / "lib" / "offline.data.json").stat().st_size
    assert size < 64 * 1024, f"the offline bundle is {size / 1024:.1f} kB"


def test_the_cache_is_never_consulted_before_the_server():
    """A stored answer that pre-empted a request would show a reading from before a correction.

    The order is the rule, and it is visible in the source: `cachedFor` is called inside the
    `catch`, never in the `then` and never before the request.
    """
    result = without_comments(read("src/screens/Result.tsx"))
    before, _, after = result.partition(".catch(")
    assert "cachedFor(" not in before, "the cache is consulted before the request fails"
    assert "cachedFor(" in after


def test_the_two_controls_that_need_a_server_are_disabled_for_a_stored_answer():
    """Run and correct both reach the backend, and a live-looking button that answers with a network
    error is the failure this row exists to remove."""
    code = without_comments(read("src/ui/Code.tsx"))
    assert "stored" in code and "disabled" in code
    result = without_comments(read("src/screens/Result.tsx"))
    assert "stored={!!stored}" in result


def test_navigator_online_is_only_trusted_in_the_false_direction():
    """`true` means an interface is up, not that a server is reachable - a captive portal reports it.

    So nothing in this client may read `navigator.onLine` as permission to claim reachability. The
    one accessor is `definitelyOffline`, and every screen goes through it.
    """
    offline = read("src/lib/offline.ts")
    assert "navigator.onLine === false" in offline
    for path in (
        "src/App.tsx",
        "src/screens/Capture.tsx",
        "src/screens/Camera.tsx",
        "src/screens/Gallery.tsx",
        "src/screens/Result.tsx",
    ):
        source = without_comments(read(path))
        assert "navigator.onLine" not in source, f"{path} reads onLine directly"


def test_the_offline_bundle_is_regenerable_and_says_so():
    """A committed payload with no generator is a payload nobody can refresh or verify."""
    script = ROOT / "scripts" / "make_offline_examples.py"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "offline.data.json" in text
    bundle = offline_bundle()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", bundle["generated"]), bundle["generated"]
    assert bundle["backend"], "no backend version recorded"
