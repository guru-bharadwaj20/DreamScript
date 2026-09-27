/**
 * Phase 16.3.4 - what happens to your photograph, on the screen rather than in a repository.
 *
 * The row asks for `docs/privacy.md` linked from the app, and a link alone would not have been
 * enough on its own: **the app is installable and works with no network**, so a link to a file on
 * github.com is a link that fails in exactly the situation this app was built to survive. So the
 * statement is here, in the bundle, readable in flight mode - and the canonical document is linked
 * from it for anyone who wants the version with the file paths in it.
 *
 * ## It is placed where the decision is made
 *
 * The landing screen's footer says the photograph is uploaded and links here, *above* the camera
 * button rather than below the fold. A privacy note that can only be reached from an About screen
 * is one that is read after the photograph has gone.
 *
 * ## The two facts this screen exists for
 *
 * Everything else here a person could guess. These two they could not:
 *
 *   the library picker   a photograph *chosen* from the library is uploaded byte for byte, EXIF
 *                        and all - including GPS, if the camera recorded it. A camera capture goes
 *                        through a canvas and comes out with none, which is a side effect of how
 *                        the capture works rather than a feature
 *   corrections          a retyped label is kept **for ever**, and is not evicted with the reading
 *                        it belongs to. That is the point of the feature and it is also the one
 *                        thing here that outlives everything else
 */

import { Glyph } from "../App";
import { Card, Pill } from "../ui";

/** The canonical document, for the version with the file paths in it. */
export const CANONICAL =
  "https://github.com/guru-bharadwaj20/DreamScript/blob/main/docs/privacy.md";

export function Privacy() {
  return (
    <div className="scroll">
      <div className="page stack">
        <div className="stack-sm">
          <span className="eyebrow">Privacy</span>
          <h2 style={{ fontFamily: "var(--serif)", fontSize: 24, letterSpacing: "-0.02em" }}>
            Your photograph is uploaded to a server
          </h2>
          <p className="muted" style={{ fontSize: 14 }}>
            The models do not run in your browser. That is not a detail of how this is deployed — it
            is what the app is, and it is the first thing worth knowing about it.
          </p>
        </div>

        <Card
          style={{
            padding: "var(--sp-4)",
            borderColor: "color-mix(in srgb, var(--degraded) 30%, transparent)",
            background: "var(--degraded-soft)",
          }}
          className="stack-sm"
        >
          <div className="row" style={{ gap: "var(--sp-2)" }}>
            <Pill tone="degraded">worth knowing</Pill>
          </div>
          <p style={{ fontSize: 13.5 }}>
            A photograph you <strong>choose from your library</strong> is uploaded exactly as it is
            on your device, <strong>including its EXIF metadata</strong> — which on most phones
            records the time, the camera, and often the <strong>place</strong> it was taken.
          </p>
          <p className="muted" style={{ fontSize: 13 }}>
            Nothing here reads that data and nothing here removes it. A photograph taken with the
            in-app camera is redrawn and re-encoded on its way out, which drops all of it — so if
            location matters to you, use the camera rather than the picker.
          </p>
        </Card>

        <Section title="What is kept">
          <Row label="The reading">
            The shapes, the arrows, the recognised words, the graph and the code — stored under a
            short id so the app can show it to you again without a second upload.
          </Row>
          <Row label="Not the picture">
            The image is never written down by the app server. That is why a reopened or shared
            result shows you the reading with no photograph behind it: the only copy is the one this
            browser is still holding, until the tab closes or you take another.
          </Row>
          <Row label="The most recent 500">
            Older readings are dropped as new ones arrive. One that has aged out is gone, and the
            app says so in those words rather than showing you an error.
          </Row>
          <Row label="Corrections, for ever">
            A label you retype is kept permanently, and is <em>not</em> dropped with the reading it
            belongs to. That is the whole point of the feature — corrections are what make the
            reader better — and it is the one thing here that outlives everything else. If you would
            rather it was not kept, do not correct it.
          </Row>
        </Section>

        <Section title="What there is none of">
          <Row label="No account">
            No login, no email address, no password. Nothing identifies you.
          </Row>
          <Row label="No analytics">
            No telemetry, no crash reporting, no usage statistics, no advertising, no cookies. This
            app makes requests to the server that reads your page and to nothing else at all — no
            fonts, no scripts, no images from anywhere.
          </Row>
          <Row label="No location">
            Nothing ever asks your browser where you are. See the note above about a photograph you
            choose rather than take.
          </Row>
          <Row label="The camera is released immediately">
            The moment a frame is taken, the stream is stopped — the indicator light goes out before
            the page is read, rather than staying on for the ten seconds it takes.
          </Row>
        </Section>

        <Section title="Two things that are easy to get wrong">
          <Row label="A result link is a key">
            The address of a result is not secret. Anyone who has it and can reach the same server
            can read the page, the words on it and the code. There are no accounts here, so the
            address is the only thing standing between a reading and whoever has the link.
          </Row>
          <Row label="Whose server it is">
            This is not a hosted service. The server is one somebody runs — perhaps you — and that
            person can read everything described on this page.
          </Row>
        </Section>

        <Card style={{ padding: "var(--sp-4)" }} className="stack-sm">
          <span className="eyebrow">The full statement</span>
          <p className="muted" style={{ fontSize: 13 }}>
            Written from the code, naming the file that makes each claim true, so it can be checked
            rather than believed.
          </p>
          <a
            href={CANONICAL}
            target="_blank"
            rel="noreferrer noopener"
            className="btn btn-block"
          >
            docs/privacy.md
            <Glyph d="M14 4h6v6M20 4l-8.5 8.5M18 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h6" size={16} />
          </a>
          <p className="dim" style={{ fontSize: 12 }}>
            That link needs a network. Everything above is part of the app and is readable without
            one.
          </p>
        </Card>

        <a href="#/" className="btn btn-block">
          Back
        </a>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card style={{ padding: "var(--sp-4)" }} className="stack-sm">
      <span className="eyebrow">{title}</span>
      <dl className="facts">{children}</dl>
    </Card>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </>
  );
}
