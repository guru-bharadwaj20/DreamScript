/**
 * Phase 16.2.1 - the landing screen: the gesture, and whether the server can answer it.
 *
 * Two jobs. **Make the gesture obvious before a word is read** - hence a frame with four gold
 * brackets around a deliberately crooked sketch, which is the same frame the live camera draws over
 * its preview in 16.2.2, so the landing screen and the working screen are visibly one object. And
 * **say whether the pipeline is reachable**, which is not decoration: this client holds no model, so
 * "nothing happens when I tap" has exactly one likely cause and it is worth naming before the tap
 * rather than after.
 *
 * The probe uses `/health?upstream=1`, the opt-in form. 16.1.1 made that opt-in precisely so an
 * orchestrator's liveness check would not fail because a *different* container was cold - and a
 * client that wants to draw "server offline" is the caller it was made opt-in *for*.
 *
 * All three affordances are live: the camera (16.2.2), the library picker (16.2.4) and the bundled
 * examples (16.2.11).
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { Glyph } from "../App";
import { type Health, ApiError, api } from "../lib/api";
import { definitelyOffline, looksOffline, watchOnline } from "../lib/offline";
import { type PickError, accept, explainPick, isPicked } from "../lib/pick";
import { go } from "../lib/route";
import { Card, Pill } from "../ui";

type Probe =
  | { state: "checking" }
  | { state: "up"; health: Health }
  /**
   * 16.2.10 splits this in two, because the two have different fixes and different sentences.
   *
   * `offline` means nothing reached anything: the answer is a network, and the only thing this app
   * can still do is show the stored examples. Without the flag, a phone in a lift got "is it running
   * on port 3000?" - a sentence that sends someone to check a server that is perfectly fine.
   */
  | { state: "down"; detail: string; offline: boolean };

export function Capture({ onStaged }: { onStaged: (blob: Blob | null) => void }) {
  const [probe, setProbe] = useState<Probe>({ state: "checking" });
  const [rejected, setRejected] = useState<PickError | null>(null);
  const picker = useRef<HTMLInputElement>(null);

  const choose = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      // Reset the input before anything else: without it, choosing the *same* file twice fires no
      // change event the second time and the app looks frozen.
      event.target.value = "";
      if (!file) return;
      setRejected(null);
      const checked = await accept(file);
      if (!isPicked(checked)) {
        setRejected(checked);
        return;
      }
      onStaged(checked.file);
      go({ view: "camera" });
    },
    [onStaged],
  );

  /**
   * The probe, re-run when the browser says the network came back.
   *
   * `nonce` rather than a function called from two places: the effect already owns the `live` flag
   * that stops a late answer overwriting a newer one, and a second caller would need its own copy of
   * that. Bumping a number re-runs the effect and the existing guard covers the race.
   */
  const [nonce, setNonce] = useState(0);
  useEffect(() => watchOnline((online) => online && setNonce((n) => n + 1)), []);

  useEffect(() => {
    // Skipped when the operating system says there is no route: the request cannot succeed, and
    // spending a second on it before saying so is a spinner in place of an answer.
    if (definitelyOffline()) {
      setProbe({
        state: "down",
        offline: true,
        detail: "there is no network connection",
      });
      return;
    }
    let live = true;
    api
      .health(true)
      .then((health) => live && setProbe({ state: "up", health }))
      .catch((error: unknown) => {
        if (!live) return;
        const offline = looksOffline(error);
        const detail =
          error instanceof ApiError
            ? error.message
            : offline
              ? "nothing answered - the request did not leave this device"
              : "the app backend did not answer - is it running on port 3000?";
        setProbe({ state: "down", detail, offline });
      });
    return () => {
      live = false;
    };
  }, [nonce]);

  return (
    <div className="scroll">
      <section className="hero">
        {/* The project's own line, from the README: vision -> structure -> code. It says what the
            thing does in four words, which "Phase 16 - mobile" did not: a plan's row number is
            internal vocabulary and the first thing anyone sees should not be. */}
        <span className="eyebrow">sketch → structure → code</span>
        <h1>
          Photograph a sketch.
          <br />
          Get <em>runnable code</em>.
        </h1>
        <p>
          Hold your phone over a whiteboard or a sheet of paper. DreamScript reads the shapes, the
          arrows and the handwriting, then writes the program the drawing describes.
        </p>

        <div className="frame">
          <div className="frame-brackets" aria-hidden="true">
            <span />
            <span />
            <span />
            <span />
          </div>
          <CrookedFlowchart />
          <span className="frame-caption">your paper goes here</span>
        </div>

        <div className="row wrap" style={{ justifyContent: "center", gap: "var(--sp-2)" }}>
          {["flowchart", "state machine", "ER diagram", "wireframe", "circuit"].map((kind) => (
            <Pill key={kind} tone="plain">
              {kind}
            </Pill>
          ))}
        </div>
      </section>

      <div className="page stack">
        <ServerStatus probe={probe} />

        <div className="actions">
          <a href="#/camera" className="action" aria-describedby="camera-note">
            <span className="action-icon">
              <Glyph d="M14.5 4h-5L8 6H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-3l-1.5-2ZM15 13a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
            </span>
            <span className="grow">
              <strong>Use the camera</strong>
              <small id="camera-note">Live preview, with the page corners found for you</small>
            </span>
            <Glyph d="m9 6 6 6-6 6" />
          </a>

          <button className="action" onClick={() => picker.current?.click()}>
            <span className="action-icon">
              <Glyph d="M4 16l4.6-4.6a2 2 0 0 1 2.8 0L16 16m-2-2 1.6-1.6a2 2 0 0 1 2.8 0L20 14M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1Zm6 4.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0Z" />
            </span>
            <span className="grow">
              <strong>Choose a photo</strong>
              <small>One you have already taken</small>
            </span>
            <Glyph d="m9 6 6 6-6 6" />
          </button>
          {/*
            No `capture` attribute, deliberately. `capture="environment"` replaces the library
            picker with the system camera, so the one control labelled "choose a photo" would not
            offer any photo the person already has. The camera is the other button.
          */}
          <input
            ref={picker}
            type="file"
            accept="image/*"
            className="sr-only"
            onChange={choose}
            aria-label="Choose a photograph"
          />

          <a href="#/gallery" className="action">
            <span className="action-icon">
              <Glyph d="M4 6h16M4 12h16M4 18h10" />
            </span>
            <span className="grow">
              <strong>Try an example</strong>
              <small>Five bundled sketches — no paper needed</small>
            </span>
            <Glyph d="m9 6 6 6-6 6" />
          </a>
        </div>

        {rejected ? <Rejected error={rejected} onDismiss={() => setRejected(null)} /> : null}

        <p className="dim" style={{ fontSize: 12.5, textAlign: "center" }}>
          Your photograph is uploaded to the server that runs the models.{" "}
          <a href="#/about">What is kept, and for how long.</a>
        </p>
      </div>
    </div>
  );
}

/**
 * The server, in one line, with the four states a person can act on.
 *
 * Not a spinner that resolves to nothing. `checking` pulses gold, `up` is emerald, and the two ways
 * of being down are **different sentences**, because they have different fixes: this backend
 * unreachable means the app is not running, and the *model server* unreachable means the backend is
 * fine and the thing behind it is not. Collapsing them into "offline" would send someone to restart
 * the wrong process.
 */
function ServerStatus({ probe }: { probe: Probe }) {
  if (probe.state === "checking") {
    return (
      <div className="status">
        <span className="status-dot" data-state="checking" />
        <span className="grow muted">Checking the server…</span>
      </div>
    );
  }

  if (probe.state === "down") {
    return (
      <Card
        className="stack-sm"
        style={{
          padding: "var(--sp-4)",
          borderColor: probe.offline
            ? "color-mix(in srgb, var(--degraded) 32%, transparent)"
            : "color-mix(in srgb, var(--stopped) 30%, transparent)",
        }}
      >
        <div className="row">
          <span className="status-dot" data-state="down" />
          <span className="grow">
            <strong>
              {probe.offline
                ? "There is no network, so nothing can be read."
                : "The app backend is not answering."}
            </strong>
            <br />
            <span className="dim mono" style={{ fontSize: 12 }}>
              {probe.detail}
            </span>
          </span>
        </div>
        {probe.offline ? (
          <>
            <p className="muted" style={{ fontSize: 13 }}>
              The models run on a server and this app holds none of them, so the camera has nowhere
              to send a photograph. Five answers are bundled and readable as they are.
            </p>
            <a href="#/gallery" className="btn btn-primary btn-block">
              See the stored examples
            </a>
          </>
        ) : null}
      </Card>
    );
  }

  const { health } = probe;
  const model = health.model_server;
  if (model && !model.reachable) {
    return (
      <Card className="status" style={{ borderColor: "color-mix(in srgb, var(--degraded) 32%, transparent)" }}>
        <span className="status-dot" data-state="down" />
        <span className="grow">
          <strong>The backend is up; the model server is not.</strong>
          <br />
          <span className="dim mono" style={{ fontSize: 12 }}>
            nothing answered at {model.url}
          </span>
        </span>
      </Card>
    );
  }

  return (
    <div className="status">
      <span className="status-dot" data-state="up" />
      <span className="grow">
        Ready.{" "}
        <span className="dim">
          backend {health.version} · {health.stored} page{health.stored === 1 ? "" : "s"} held ·{" "}
          {health.limits.budgets.predict}/min
        </span>
      </span>
    </div>
  );
}

/**
 * Why a chosen file was not accepted.
 *
 * One sentence and the thing the person can do about it. The HEIC case is the one that earns this
 * component: every photograph on a recent iPhone is HEIC, no browser decodes it, and "that image
 * could not be opened" sends someone looking for a corrupt file that is not corrupt.
 */
function Rejected({ error, onDismiss }: { error: PickError; onDismiss: () => void }) {
  const { title, detail } = explainPick(error);
  return (
    <Card
      style={{
        padding: "var(--sp-4)",
        borderColor: "color-mix(in srgb, var(--degraded) 32%, transparent)",
      }}
      className="stack-sm"
    >
      <div className="row">
        <span className="eyebrow grow" style={{ color: "var(--degraded)" }}>
          {title}
        </span>
        <button className="btn btn-quiet" onClick={onDismiss} aria-label="Dismiss">
          <Glyph d="M6 6l12 12M18 6 6 18" size={16} />
        </button>
      </div>
      <p className="muted" style={{ fontSize: 13.5 }}>
        {detail}
      </p>
    </Card>
  );
}

/**
 * A flowchart drawn badly, on purpose.
 *
 * A crisp vector diagram here would advertise the wrong input. The corpus behind this pipeline is
 * photographs of real handwriting, the arrows in it are broken, and S5's median graph edit distance
 * is 14.5 against a target of 3. A frame promising a tidy drawing would be promising the case that
 * works; a wobbly one promises the case the project is actually about.
 */
function CrookedFlowchart() {
  return (
    <svg
      className="frame-sketch"
      viewBox="0 0 120 96"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="34" y="6" width="50" height="17" rx="8.5" transform="rotate(-1.2 59 14.5)" />
      <path d="M59.5 23.5 60 35" />
      <path d="m56.5 31.5 3.4 4.2 3.2-4.4" />
      <path d="M59 36 44 51l15.5 14.5L75 50.5 59 36Z" transform="rotate(1.6 59.5 50.7)" />
      <path d="M44.5 51 26 51.6 25.6 72" />
      <path d="M75.5 50.5 94 50l.6 21.5" />
      <rect x="10" y="72" width="32" height="15" rx="2.5" transform="rotate(0.9 26 79.5)" />
      <rect x="79" y="71.5" width="32" height="15" rx="2.5" transform="rotate(-1.4 95 79)" />
      <path d="M25.6 68.5 25.9 72m-2.4-3 2.2 3.4 2.6-3.1" />
      <path d="M94.6 68 94.8 71.5m-2.5-3 2.3 3.4 2.6-3.1" />
    </svg>
  );
}
