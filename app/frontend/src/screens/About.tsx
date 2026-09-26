/**
 * Phase 16.2.1 - what this is, what it keeps, and the theme picker.
 *
 * Three sections, and the middle one is not padding. **A person about to photograph their own
 * whiteboard is entitled to know the photograph leaves the device before they take it**, and 16.3.4
 * makes `docs/privacy.md` a row of its own - so this screen links to it and states the short version
 * here, where it is actually read.
 *
 * The palette is on screen rather than only in `tokens.css` because the four trust colours are a
 * *vocabulary* this app uses everywhere, and a legend for it is more useful than a colour swatch
 * grid usually is.
 */

import { type Theme, resolvedTheme } from "../lib/theme";
import { Button, Card, Mark, Pill, TrustPill } from "../ui";

export function About({ theme, onTheme }: { theme: Theme; onTheme: (next: Theme) => void }) {
  return (
    <div className="scroll">
      <div className="page stack">
        <div className="stack-sm">
          <Mark size={30} />
          <p className="muted">
            A vision → structure → code pipeline. A photograph of a hand-drawn diagram becomes an
            intermediate representation, and the representation becomes a program that runs.
          </p>
        </div>

        <Card style={{ padding: "var(--sp-5)" }} className="stack-sm">
          <span className="eyebrow">Where the work happens</span>
          <p className="muted" style={{ fontSize: 14 }}>
            The recogniser is a 334M-parameter handwriting model and the detector runs at 1280px.
            Neither belongs on a phone, so this app captures, straightens and displays — the models
            answer over HTTP. That is why it needs a server, and why it tells you when it cannot
            reach one.
          </p>
        </Card>

        <Card style={{ padding: "var(--sp-5)" }} className="stack-sm">
          <span className="eyebrow">Your photograph</span>
          <p className="muted" style={{ fontSize: 14 }}>
            It is uploaded to the server that runs the models. The server keeps the result — the
            shapes it found, the code it wrote — under a short id so this app can show it to you
            without uploading the picture again, and drops the oldest once 500 are held. Labels you
            correct are kept permanently, because corrections are the training data that makes the
            reader better.
          </p>
          <p className="dim" style={{ fontSize: 13 }}>
            The full statement is in <code>docs/privacy.md</code>.
          </p>
        </Card>

        <Card style={{ padding: "var(--sp-5)" }} className="stack-sm">
          <span className="eyebrow">How it says what it is unsure of</span>
          <p className="muted" style={{ fontSize: 14 }}>
            The pipeline reports when a stage fell back to a simpler method, where it gave up, and
            when it would rather ask than guess. This app shows you all of it — an answer you should
            check looks different from one you should not.
          </p>
          <div className="row wrap" style={{ gap: "var(--sp-2)", marginTop: "var(--sp-2)" }}>
            <TrustPill ok degraded={false} stoppedAt={null} needsConfirmation={false} />
            <TrustPill ok degraded stoppedAt={null} needsConfirmation={false} />
            <TrustPill ok={false} degraded={false} stoppedAt="assemble" needsConfirmation={false} />
            <TrustPill ok={false} degraded={false} stoppedAt={null} needsConfirmation />
          </div>
        </Card>

        <Card style={{ padding: "var(--sp-5)" }} className="stack-sm">
          <span className="eyebrow">Appearance</span>
          <div className="theme-choice">
            {(["system", "light", "dark"] as Theme[]).map((option) => (
              <Button
                key={option}
                aria-pressed={theme === option}
                onClick={() => onTheme(option)}
                style={{ flexDirection: "column" }}
              >
                <span style={{ textTransform: "capitalize" }}>{option}</span>
                {option === "system" ? (
                  <small className="dim" style={{ fontSize: 11 }}>
                    now {resolvedTheme(theme === "system" ? "system" : option)}
                  </small>
                ) : null}
              </Button>
            ))}
          </div>
          <p className="dim" style={{ fontSize: 12.5 }}>
            <strong>System</strong> follows your phone, which follows the time of day. It is the
            default because you already made that choice once.
          </p>
        </Card>

        <Card style={{ padding: "var(--sp-5)" }} className="stack-sm">
          <span className="eyebrow">Palette</span>
          <div className="swatches">
            {[
              ["--gold", "gold"],
              ["--silver-ish", "silver"],
              ["--ink", "ink"],
              ["--ok", "read"],
              ["--degraded", "degraded"],
              ["--stopped", "stopped"],
              ["--confirm", "asking"],
            ].map(([token, name]) => (
              <div className="swatch" key={name}>
                <i
                  style={{
                    background: token === "--silver-ish" ? "var(--ink-2)" : `var(${token})`,
                  }}
                />
                <span>{name}</span>
              </div>
            ))}
          </div>
          <p className="dim" style={{ fontSize: 12.5 }}>
            Gold is the brand and nothing else. The four on the right are the only colours that carry
            meaning, and each one is paired with a dot and a word — colour alone is not a signal for
            about one man in twelve.
          </p>
        </Card>

        <div className="row" style={{ justifyContent: "center", paddingTop: "var(--sp-2)" }}>
          <Pill tone="plain">app 16.2</Pill>
          <a href="#/" className="btn btn-quiet">
            Back
          </a>
        </div>
      </div>
    </div>
  );
}
