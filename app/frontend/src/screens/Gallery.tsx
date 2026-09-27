/**
 * Phase 16.2.11 - five bundled sketches, so the app can be shown without paper.
 * Phase 16.2.10 - and the one screen that still works with no network.
 *
 * The demo problem this solves is real. A phone, a projector and no whiteboard is the usual setting
 * for showing this, and "please find me a pen and something to draw on" is a bad first minute. These
 * five are the repository's own committed fixtures - the same images `tests/fixtures/` holds and
 * `scripts/make_fixtures.py` regenerates deterministically - so the gallery cannot drift from what
 * every test in the project runs against.
 *
 * ## They are examples, not marketing
 *
 * Each one is a *synthetic* sketch, and the screen says so. A gallery of pages the reader happens to
 * handle perfectly would be a demo that lies about the accuracy: S5 sits at median GED 14.5 against
 * a target of 3, and Phase 16's own preamble gives that as the reason this is not on an app store.
 * The honest version shows what the pipeline actually does with them, including where it struggles -
 * which is what every other screen in this app is built to show.
 *
 * ## Nothing on the card is hand-written any more, and that is 16.2.10's doing
 *
 * The first version of this file carried a hand-written `reads` note on two cards and claimed, in a
 * panel underneath, that "two of these are read as the wrong kind". Capturing the five answers for
 * the offline cache measured it: the count is **three**. `circuit.png` also comes back as a
 * flowchart, and the panel had been wrong since it was written.
 *
 * So the misreads are no longer asserted. Each card compares the fixture's own name - which *is* its
 * diagram type, by `tests/fixtures/manifest.json`'s convention - against the `diagram_type` in the
 * stored answer for it, and the count under the grid is `filter(...).length`. A claim that is
 * computed from a captured payload cannot be out of date by one.
 *
 * ## Online they go through the real pipeline; offline they are the only thing left
 *
 * With a server, tapping one fetches the PNG and hands it to the same upload path a photograph takes:
 * dewarp, stream, result. Nothing is pre-computed. With no server, the card links straight to
 * 16.2.10's stored answer for the same file, and says **stored** on its face before it is tapped -
 * because a person who taps expecting a live reading and gets a cached one has been misled by the
 * screen rather than by the network.
 */

import { useCallback, useEffect, useState } from "react";

import { Glyph } from "../App";
import { ApiError, api } from "../lib/api";
import { CACHED, CAPTURED, cachedForFile, definitelyOffline, looksOffline } from "../lib/offline";
import { go } from "../lib/route";
import { Card, Pill } from "../ui";
import "./Gallery.css";

interface Example {
  file: string;
  name: string;
  /** What the drawing *is*, so a person can tell a good read from a bad one. */
  expect: string;
}

/**
 * The five diagram types the pipeline handles, one each.
 *
 * `tests/fixtures/manifest.json` is the source of the set; these are copied into `public/examples`
 * at 20 kB for all five, which is small enough for 16.3.1 to precache.
 *
 * `expect` is the only hand-written string left here and it describes the *drawing*, which no
 * payload contains. Everything the reader claims about these pages is read from the captured
 * answers instead - see the note at the top of the file.
 */
export const EXAMPLES: Example[] = [
  { file: "flowchart.png", name: "Flowchart", expect: "a branch and a loop" },
  { file: "state_machine.png", name: "State machine", expect: "states and transitions" },
  { file: "er_diagram.png", name: "ER diagram", expect: "two entities" },
  { file: "wireframe.png", name: "Wireframe", expect: "a screen and its rows" },
  { file: "circuit.png", name: "Circuit", expect: "resistors and a source" },
];

/**
 * The type this fixture *is*, from its filename.
 *
 * `tests/fixtures/manifest.json` gives each fixture an `id` equal to its stem and a `diagram_type`
 * equal to that id, so the stem is the ground truth without shipping the manifest to the browser.
 */
export function expectedType(file: string): string {
  return file.replace(/\.png$/, "");
}

/** What the reader actually answered for this file, or `null` if no answer was captured for it. */
export function readsAs(file: string): string | null {
  return cachedForFile(file)?.prediction.diagram_type ?? null;
}

/** Measured, not asserted: the files whose stored answer disagrees with the fixture's own type. */
export function misread(): string[] {
  return EXAMPLES.map((e) => e.file).filter((file) => {
    const actual = readsAs(file);
    return actual !== null && actual !== expectedType(file);
  });
}

/** Can a tap reach the pipeline, or is the cache all there is? */
type Reach = "checking" | "live" | "cache";

export function Gallery({ onStaged }: { onStaged: (blob: Blob | null) => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reach, setReach] = useState<Reach>(() => (definitelyOffline() ? "cache" : "checking"));
  /**
   * Thumbnails that would not load - which offline is all of them, until 16.3.1 precaches them.
   *
   * Caught by the first offline screenshot of this screen: five of Chrome's **broken-image icons**
   * with the alt text spilling out of the frame, on the one screen in the app that is supposed to
   * still work without a network. The answer behind each card was intact; the card looked broken.
   * A composed placeholder says the picture is missing without making the card look like a fault.
   */
  const [noThumb, setNoThumb] = useState<string[]>([]);

  /**
   * One `/health` before the cards decide what they are.
   *
   * A read, so it costs 1 of 16.1.4's 120-per-minute budget rather than 1 of the 6 uploads - which
   * matters, because the alternative is discovering the server is gone by spending an upload on it.
   * `?upstream=1` is not used: whether the *model* server is cold is the Capture screen's question,
   * and here the only thing that changes the cards is whether anything answers at all.
   */
  useEffect(() => {
    if (definitelyOffline()) return;
    let live = true;
    api
      .health()
      .then(() => live && setReach("live"))
      .catch((failure: unknown) => {
        if (!live) return;
        // An `ApiError` means the backend answered - it is up, and an upload is worth attempting
        // even if this particular request was refused. Only a failure to reach anything at all
        // turns the cards into stored answers.
        setReach(looksOffline(failure) ? "cache" : "live");
      });
    return () => {
      live = false;
    };
  }, []);

  const choose = useCallback(
    async (example: Example) => {
      const cached = cachedForFile(example.file);
      if (reach === "cache" && cached) {
        go({ view: "result", id: cached.id });
        return;
      }
      setBusy(example.file);
      setError(null);
      try {
        const response = await fetch(`examples/${example.file}`);
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        const blob = await response.blob();
        onStaged(blob);
        go({ view: "camera" });
      } catch (failure) {
        // The network went between the probe and the tap, which on a phone is an ordinary event
        // rather than an edge case. There is a correct answer for exactly this file, so it is shown
        // instead of an apology - and the result screen labels it.
        if (looksOffline(failure) && cached) {
          setReach("cache");
          setBusy(null);
          go({ view: "result", id: cached.id });
          return;
        }
        // A bundled asset that will not load for any other reason means the build is broken or the
        // cache is stale - worth saying rather than leaving a card that does nothing when tapped.
        setError(
          `${example.name} could not be loaded: ${
            failure instanceof ApiError || failure instanceof Error
              ? failure.message
              : String(failure)
          }`,
        );
        setBusy(null);
      }
    },
    [onStaged, reach],
  );

  const wrong = misread();
  const cache = reach === "cache";

  return (
    <div className="scroll">
      <div className="page stack">
        <div className="stack-sm">
          <span className="eyebrow">Examples</span>
          <h2 style={{ fontFamily: "var(--serif)", fontSize: 24, letterSpacing: "-0.02em" }}>
            Five sketches, no paper needed
          </h2>
          <p className="muted" style={{ fontSize: 14 }}>
            {cache
              ? "One of each kind the reader handles. With no server to send them to, each one opens the answer this app shipped with — a real run, kept as it came out."
              : "One of each kind the reader handles. They go through the same pipeline a photograph does — nothing here is pre-computed, so what you see is what it actually makes of them."}
          </p>
        </div>

        {cache ? (
          <Card
            className="row"
            style={{
              padding: "var(--sp-4)",
              borderColor: "color-mix(in srgb, var(--degraded) 32%, transparent)",
              background: "var(--degraded-soft)",
            }}
          >
            <Pill tone="degraded">stored</Pill>
            <span className="grow muted" style={{ fontSize: 13 }}>
              Nothing is being read. These five answers were captured on {CAPTURED} and are the whole
              of what this app can show without a server.
            </span>
          </Card>
        ) : null}

        {error ? (
          <Card
            style={{
              padding: "var(--sp-4)",
              borderColor: "color-mix(in srgb, var(--stopped) 30%, transparent)",
            }}
          >
            <p style={{ fontSize: 13 }}>{error}</p>
          </Card>
        ) : null}

        <div className="gallery">
          {EXAMPLES.map((example) => {
            const actual = readsAs(example.file);
            const wrongType = actual !== null && actual !== expectedType(example.file);
            return (
              <button
                key={example.file}
                className="example"
                onClick={() => choose(example)}
                disabled={busy !== null}
                aria-busy={busy === example.file}
              >
                {noThumb.includes(example.file) ? (
                  <span className="example-thumb is-missing" aria-hidden="true">
                    <Glyph d="M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1Zm0 11 4.6-4.6a2 2 0 0 1 2.8 0L16 16m-2-2 1.6-1.6a2 2 0 0 1 2.8 0L20 14" size={22} />
                  </span>
                ) : (
                  <img
                    src={`examples/${example.file}`}
                    alt={`A hand-drawn ${example.name.toLowerCase()}`}
                    width={320}
                    height={240}
                    loading="lazy"
                    onError={() =>
                      setNoThumb((seen) =>
                        seen.includes(example.file) ? seen : [...seen, example.file],
                      )
                    }
                  />
                )}
                <span className="example-meta">
                  <strong>{example.name}</strong>
                  <small>{example.expect}</small>
                  {/* Read from the captured answer, never written here. The hand-written version of
                      this note was wrong about one of the five for as long as it existed. */}
                  {wrongType ? (
                    <Pill tone="degraded">reads as a {actual.replace(/_/g, " ")}</Pill>
                  ) : null}
                  {cache ? <Pill tone="plain">stored answer</Pill> : null}
                </span>
                <span className="example-go" aria-hidden="true">
                  {busy === example.file ? "…" : <Glyph d="m9 6 6 6-6 6" size={18} />}
                </span>
              </button>
            );
          })}
        </div>

        {wrong.length ? (
          <Card style={{ padding: "var(--sp-4)" }} className="stack-sm">
            <span className="eyebrow">
              {wrong.length} of these {wrong.length === 1 ? "is" : "are"} read as the wrong kind
            </span>
            <p className="muted" style={{ fontSize: 13.5 }}>
              {/* The count and the names are computed from the stored answers. The hand-written
                  sentence that used to be here said "two", and capturing the five answers for the
                  offline cache showed it was three. */}
              {wrong.map((file) => file.replace(/\.png$/, "").replace(/_/g, " ")).join(", ")} come
              back as flowcharts. That is a known leak rather than a surprise: the type
              classifier&apos;s strongest single feature is the page&apos;s overall shape, and all
              five of these are the same size — so it has learned something about the image rather
              than about the drawing. Tapping them shows it happening.
            </p>
            <p className="dim" style={{ fontSize: 12.5 }}>
              They are synthetic sketches, not photographs. A real page is messier and the reader is
              less sure of it, which the result screen will tell you either way.
            </p>
          </Card>
        ) : null}

        <p className="dim" style={{ fontSize: 12 }}>
          {CACHED.length} answers are bundled with this app, one per example.
        </p>

        <a href="#/" className="btn btn-block">
          Back
        </a>
      </div>
    </div>
  );
}
