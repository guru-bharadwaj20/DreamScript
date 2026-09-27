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

import { useCallback, useEffect, useState } from "react";

import { CACHED, CAPTURED } from "../lib/offline";
import { type Install, install, installability, watchInstall } from "../lib/pwa";
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
          <a href="#/privacy" className="btn btn-block">
            What is kept, and for how long
          </a>
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

        <InstallCard />

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
          <Pill tone="plain">app 16.3</Pill>
          <a href="#/" className="btn btn-quiet">
            Back
          </a>
        </div>
      </div>
    </div>
  );
}

/**
 * Phase 16.3.1 - put it on the home screen, and the three different things that means.
 *
 * The button is only shown when the browser has actually offered a prompt, because `prompt()` can
 * only be called on a saved `beforeinstallprompt` and a button that does nothing is worse than no
 * button. On iOS there is never such an event and never will be, so that platform gets the gesture
 * written out instead - which is the case most people reading this screen on a phone will be in.
 *
 * The card also says what installing *gets* you, because "install" on a web app is a word people
 * have learned to ignore. Here it is concrete: full screen with no browser chrome over the
 * viewfinder, and the five examples readable with no network at all.
 */
function InstallCard() {
  const [state, setState] = useState<Install>(installability);
  const [outcome, setOutcome] = useState<string | null>(null);

  useEffect(() => watchInstall(() => setState(installability())), []);

  const ask = useCallback(async () => {
    const answer = await install();
    setState(installability());
    setOutcome(
      answer === "accepted"
        ? "Installed. Look for the gold D on your home screen."
        : answer === "dismissed"
          ? "Not installed — the browser will offer again later."
          : null,
    );
  }, []);

  if (state.state === "installed") {
    return (
      <Card style={{ padding: "var(--sp-5)" }} className="stack-sm">
        <span className="eyebrow">Installed</span>
        <p className="muted" style={{ fontSize: 14 }}>
          This is running from your home screen, full screen, with no browser chrome over the
          viewfinder. The shell is cached, so it opens whether or not there is a network.
        </p>
        <p className="dim" style={{ fontSize: 12.5 }}>
          {CACHED.length} stored example readings are bundled with it, captured on {CAPTURED}.
        </p>
      </Card>
    );
  }

  return (
    <Card style={{ padding: "var(--sp-5)" }} className="stack-sm">
      <span className="eyebrow">Put it on your home screen</span>
      <p className="muted" style={{ fontSize: 14 }}>
        It opens full screen, so nothing sits over the viewfinder, and the app itself loads with no
        network — the {CACHED.length} bundled examples are readable in flight mode. Photographing a
        page still needs a server; that part cannot be cached.
      </p>

      {state.state === "ready" ? (
        <Button variant="primary" block onClick={ask}>
          Install DreamScript
        </Button>
      ) : state.state === "manual" ? (
        <>
          {/* Safari fires no `beforeinstallprompt` and there is no API to ask. The gesture is the
              feature on this platform, so it is written out rather than hinted at. */}
          <ol className="steps">
            <li>
              Tap <strong>Share</strong> in Safari&apos;s toolbar
            </li>
            <li>
              Choose <strong>Add to Home Screen</strong>
            </li>
            <li>
              Tap <strong>Add</strong>
            </li>
          </ol>
          <p className="dim" style={{ fontSize: 12.5 }}>
            iOS has no install button for a web app — only this gesture. It has to be Safari;
            Chrome on iOS cannot do it.
          </p>
        </>
      ) : (
        <p className="dim" style={{ fontSize: 12.5 }}>
          Your browser has not offered an install prompt for this page. On Android, Chrome offers
          one after a visit or two; on a desktop it is usually an icon at the right-hand end of the
          address bar.
        </p>
      )}

      {outcome ? (
        <p className="dim" style={{ fontSize: 12.5 }}>
          {outcome}
        </p>
      ) : null}
    </Card>
  );
}
