/**
 * Phase 16.2.7 - the code panel: read it, copy it, share it, run it.
 *
 * The payoff screen. A person photographed a whiteboard forty seconds ago and this is the program
 * it described - so the three things they will actually do with it are first-class: **copy** to a
 * clipboard, **share** to wherever they write code, and **run** it, which reaches 16.1.3's sandbox
 * and comes back with what it printed.
 *
 * ## Copy and share are two controls, not one
 *
 * `navigator.share` exists on a phone and mostly does not on a desktop; `navigator.clipboard` needs
 * a secure origin and can be refused outright. Offering one button that silently does whichever
 * happens to work leaves a person unsure what just happened. So both are shown, each is hidden when
 * the browser does not have it, and each says what it did.
 *
 * ## The run button is honest about four different outcomes
 *
 * 16.1.3 answers with a `kind`, and they are not interchangeable. `exec.ok` ran. `exec.timeout` and
 * `exec.memory` are the sandbox doing its job on a program that would not stop - which is
 * information about the *drawing*, usually a loop the tracer closed wrongly. `exec.refused` is the
 * import pre-scan. `*.unavailable` means the toolchain is not on the host and nothing was run at
 * all, which must never be shown as a failure: a wireframe whose code was never run has to look
 * different from one whose code ran and failed.
 */

import { useCallback, useMemo, useState } from "react";

import { Glyph } from "../App";
import { type Prediction, type RunVerdict, ApiError, api } from "../lib/api";
import { type Token, highlight, languageOf } from "../lib/highlight";
import { Button, Card, Pill } from "../ui";
import "./Code.css";

export function Code({ prediction }: { prediction: Prediction }) {
  const language = languageOf(prediction.language);
  const lines = useMemo(
    () => (prediction.code && language ? highlight(prediction.code, language) : null),
    [prediction.code, language],
  );

  const [copied, setCopied] = useState(false);
  const [shared, setShared] = useState(false);
  const [running, setRunning] = useState(false);
  const [verdict, setVerdict] = useState<RunVerdict | null>(null);
  const [runError, setRunError] = useState<string | null>(null);

  const copy = useCallback(async () => {
    if (!prediction.code) return;
    try {
      await navigator.clipboard.writeText(prediction.code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      // Refused: an insecure origin, or a gesture the call has lost. The text is on screen and
      // selectable, so this is a missing convenience rather than a dead end - and saying "copied"
      // when nothing was copied would be worse than saying nothing.
      setCopied(false);
    }
  }, [prediction.code]);

  const share = useCallback(async () => {
    if (!prediction.code) return;
    try {
      await navigator.share({
        title: `DreamScript — ${prediction.diagram_type}`,
        text: prediction.code,
      });
      setShared(true);
      window.setTimeout(() => setShared(false), 1600);
    } catch {
      // `AbortError` is a person tapping Cancel on the share sheet, which is not a failure and
      // must not be reported as one. Nothing else here is worth a message either.
    }
  }, [prediction.code, prediction.diagram_type]);

  const run = useCallback(async () => {
    setRunning(true);
    setRunError(null);
    setVerdict(null);
    try {
      setVerdict(await api.run(prediction.id));
    } catch (error) {
      setRunError(
        error instanceof ApiError
          ? error.status === 503 && error.retryAfter
            ? `${error.message} Try again in ${error.retryAfter} seconds.`
            : error.message
          : "The server did not answer.",
      );
    } finally {
      setRunning(false);
    }
  }, [prediction.id]);

  if (!prediction.code) {
    return (
      <Card style={{ padding: "var(--sp-5)" }} className="stack-sm">
        <span className="eyebrow">No code</span>
        <p className="muted" style={{ fontSize: 14 }}>
          This page produced no program
          {prediction.stopped_at ? `; the pipeline stopped at ${prediction.stopped_at}.` : "."}
        </p>
      </Card>
    );
  }

  const canShare = typeof navigator !== "undefined" && "share" in navigator;
  const canCopy = typeof navigator !== "undefined" && !!navigator.clipboard;

  return (
    <div className="stack-sm">
      <Card className="code" style={{ padding: 0, overflow: "hidden" }}>
        <div className="code-bar">
          <span className="eyebrow grow">{prediction.language}</span>
          {canCopy ? (
            <Button variant="quiet" onClick={copy} aria-label="Copy the program">
              {copied ? "Copied" : "Copy"}
            </Button>
          ) : null}
          {canShare ? (
            <Button variant="quiet" onClick={share} aria-label="Share the program">
              {shared ? "Shared" : "Share"}
            </Button>
          ) : null}
        </div>

        {/* `pre` scrolls sideways rather than wrapping: indentation is structure in three of these
            four languages, and re-wrapping a long line destroys the only cue that says which
            branch a statement is in. */}
        <pre className="code-body mono selectable">
          {lines ? (
            lines.map((tokens, index) => (
              <span className="code-line" key={index}>
                <span className="code-gutter" aria-hidden="true">
                  {index + 1}
                </span>
                <span className="code-text">
                  {tokens.map((token, t) => (
                    <Span key={t} token={token} />
                  ))}
                  {/*
                    A newline *between* lines, never after the last one. One per line adds a
                    character the program does not have, and the browser check caught it: the panel
                    selected as 636 bytes against the 635 the server serves. `tokenise`
                    round-tripping is not enough on its own — the renderer has to as well, because
                    this text is read and copied verbatim.
                  */}
                  {index < lines.length - 1 ? "\n" : ""}
                </span>
              </span>
            ))
          ) : (
            <span className="code-text">{prediction.code}</span>
          )}
        </pre>
      </Card>

      <Button variant="primary" block onClick={run} disabled={running}>
        {running ? "Running…" : "Run it"}
        {running ? null : <Glyph d="M6 4.5v15l13-7.5-13-7.5Z" size={16} />}
      </Button>

      {runError ? (
        <Card
          style={{
            padding: "var(--sp-4)",
            borderColor: "color-mix(in srgb, var(--stopped) 30%, transparent)",
          }}
        >
          <p style={{ fontSize: 13.5 }}>{runError}</p>
        </Card>
      ) : null}

      {verdict ? <Verdict verdict={verdict} /> : null}
    </div>
  );
}

function Span({ token }: { token: Token }) {
  if (token.kind === "plain") return <>{token.value}</>;
  return <span className={`t-${token.kind}`}>{token.value}</span>;
}

/**
 * What the sandbox did, in the four shapes it can come back in.
 *
 * `available === false` is the one that must not look like a failure: the toolchain is absent and
 * **nothing ran**, so a wireframe on a host without node reads differently from a wireframe whose
 * JSX threw.
 */
function Verdict({ verdict }: { verdict: RunVerdict }) {
  const unavailable = verdict.available === false || verdict.kind.endsWith(".unavailable");
  const tone = unavailable ? "plain" : verdict.ok ? "ok" : "degraded";
  const output = (verdict.stdout ?? "").trimEnd();
  const errors = (verdict.stderr ?? "").trimEnd();

  return (
    <Card style={{ padding: "var(--sp-4)" }} className="stack-sm">
      <div className="row wrap" style={{ gap: "var(--sp-2)" }}>
        <Pill tone={tone}>{unavailable ? "not run here" : verdict.ok ? "it ran" : verdict.kind}</Pill>
        <span className="grow" />
        <span className="dim num" style={{ fontSize: 12 }}>
          {verdict.seconds.toFixed(2)}s
          {verdict.peak_memory_bytes
            ? ` · ${(verdict.peak_memory_bytes / 1024 / 1024).toFixed(0)} MB`
            : ""}
        </span>
      </div>

      <p className="muted" style={{ fontSize: 13 }}>
        {explainRun(verdict, unavailable)}
      </p>

      {output ? (
        <pre className="run-output mono selectable">{output}</pre>
      ) : verdict.ok ? (
        <p className="dim" style={{ fontSize: 12.5 }}>
          It ran and printed nothing, which for a generated program is the usual outcome.
        </p>
      ) : null}

      {errors ? <pre className="run-output run-stderr mono selectable">{errors}</pre> : null}

      {verdict.truncated ? (
        <p className="dim" style={{ fontSize: 12 }}>
          Output was cut at 64 kB.
        </p>
      ) : null}

      {verdict.sandbox ? (
        <p className="dim" style={{ fontSize: 11.5 }}>
          Run in a fresh interpreter: {verdict.sandbox.memory_limit_mb} MB cap,{" "}
          {verdict.sandbox.active_process_limit} processes, {verdict.sandbox.denied_imports} modules
          refused, {verdict.timeout_s}s budget.
        </p>
      ) : null}
    </Card>
  );
}

/** One sentence per outcome, in the language of the drawing rather than of the sandbox. */
function explainRun(verdict: RunVerdict, unavailable: boolean): string {
  if (unavailable) {
    return `Nothing was run: this server has no toolchain for ${verdict.language}. That is not a failure of the code — it was never executed.`;
  }
  switch (verdict.kind) {
    case "exec.ok":
    case "sql.ok":
      return "The program ran to completion inside the sandbox.";
    case "exec.timeout":
      return `It was still running after ${verdict.timeout_s}s and was stopped. A generated program that does not finish usually means a loop closed on an arrow that should not have been there.`;
    case "exec.memory":
      return "It ran out of the memory the sandbox allows. Usually an unbounded list built by a loop that never ends.";
    case "exec.refused":
      return `It was refused before it started: ${verdict.detail}. A program derived from a drawing has no business importing that.`;
    case "run.no_entry":
      return "There was no function or class to call, so there was nothing to run.";
    case "run.no_code":
      return verdict.detail;
    default:
      return verdict.detail || "It did not finish cleanly.";
  }
}
