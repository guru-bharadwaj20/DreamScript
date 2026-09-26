/**
 * Phase 16.2.11 - five bundled sketches, so the app can be shown without paper.
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
 * ## They go through the real pipeline
 *
 * Tapping one fetches the PNG and hands it to the same upload path a photograph takes: dewarp,
 * stream, result. Nothing is pre-computed and no answer is cached in the bundle. 16.2.10's offline
 * mode is the one place a stored answer appears, and it is labelled there.
 */

import { useCallback, useState } from "react";

import { Glyph } from "../App";
import { go } from "../lib/route";
import { Card, Pill } from "../ui";
import "./Gallery.css";

interface Example {
  file: string;
  name: string;
  /** What the drawing *is*, so a person can tell a good read from a bad one. */
  expect: string;
  /**
   * What the reader actually answers for this fixture today, measured rather than hoped.
   *
   * There is no "language" field here on purpose. The first version promised one per card, and on
   * these five fixtures the type classifier answers correctly **twice**: `er_diagram` and
   * `wireframe` both come back `flowchart -> python`. That is not a bug in the gallery, it is
   * 4.1.5's documented leak - global aspect ratio is the classifier's strongest single feature and
   * every fixture here is 320x240 - so a card promising `sql` would be the app breaking its own
   * promise three times in five.
   */
  reads?: string;
}

/**
 * The five diagram types the pipeline handles, one each.
 *
 * `tests/fixtures/manifest.json` is the source of the set; these are copied into `public/examples`
 * at 20 kB for all five, which is small enough to precache for 16.2.10.
 */
export const EXAMPLES: Example[] = [
  { file: "flowchart.png", name: "Flowchart", expect: "a branch and a loop" },
  { file: "state_machine.png", name: "State machine", expect: "states and transitions" },
  { file: "er_diagram.png", name: "ER diagram", expect: "two entities", reads: "reads as a flowchart" },
  {
    file: "wireframe.png",
    name: "Wireframe",
    expect: "a screen and its rows",
    reads: "reads as a flowchart",
  },
  { file: "circuit.png", name: "Circuit", expect: "resistors and a source" },
];

export function Gallery({ onStaged }: { onStaged: (blob: Blob | null) => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const choose = useCallback(
    async (example: Example) => {
      setBusy(example.file);
      setError(null);
      try {
        const response = await fetch(`examples/${example.file}`);
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        const blob = await response.blob();
        onStaged(blob);
        go({ view: "camera" });
      } catch (failure) {
        // A bundled asset that will not load means the build is broken or the cache is stale -
        // worth saying rather than leaving a card that does nothing when tapped.
        setError(
          `${example.name} could not be loaded: ${
            failure instanceof Error ? failure.message : String(failure)
          }`,
        );
        setBusy(null);
      }
    },
    [onStaged],
  );

  return (
    <div className="scroll">
      <div className="page stack">
        <div className="stack-sm">
          <span className="eyebrow">Examples</span>
          <h2 style={{ fontFamily: "var(--serif)", fontSize: 24, letterSpacing: "-0.02em" }}>
            Five sketches, no paper needed
          </h2>
          <p className="muted" style={{ fontSize: 14 }}>
            One of each kind the reader handles. They go through the same pipeline a photograph does
            — nothing here is pre-computed, so what you see is what it actually makes of them.
          </p>
        </div>

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
          {EXAMPLES.map((example) => (
            <button
              key={example.file}
              className="example"
              onClick={() => choose(example)}
              disabled={busy !== null}
              aria-busy={busy === example.file}
            >
              <img
                src={`examples/${example.file}`}
                alt={`A hand-drawn ${example.name.toLowerCase()}`}
                width={320}
                height={240}
                loading="lazy"
              />
              <span className="example-meta">
                <strong>{example.name}</strong>
                <small>{example.expect}</small>
                {example.reads ? <Pill tone="degraded">{example.reads}</Pill> : null}
              </span>
              <span className="example-go" aria-hidden="true">
                {busy === example.file ? "…" : <Glyph d="m9 6 6 6-6 6" size={18} />}
              </span>
            </button>
          ))}
        </div>

        <Card style={{ padding: "var(--sp-4)" }} className="stack-sm">
          <span className="eyebrow">Two of these are read as the wrong kind</span>
          <p className="muted" style={{ fontSize: 13.5 }}>
            The ER diagram and the wireframe both come back as flowcharts. That is a known leak
            rather than a surprise: the type classifier's strongest single feature is the page&apos;s
            overall shape, and all five of these are the same size — so it has learned something
            about the image rather than about the drawing. Tapping them shows it happening.
          </p>
          <p className="dim" style={{ fontSize: 12.5 }}>
            They are synthetic sketches, not photographs. A real page is messier and the reader is
            less sure of it, which the result screen will tell you either way.
          </p>
        </Card>

        <a href="#/" className="btn btn-block">
          Back
        </a>
      </div>
    </div>
  );
}
