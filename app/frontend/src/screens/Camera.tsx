/**
 * Phase 16.2.2 - the live camera, and 16.1.2's stream arriving on a screen at last.
 *
 * One screen, two halves. The **viewfinder** is full-bleed video with the same four gold brackets the
 * landing frame draws, so the two are visibly one object; the **progress** half replaces it the
 * moment a frame is captured, and fills in the seven stages as the server reports them.
 *
 * ## The viewfinder is not a `<img>` of a camera
 *
 *   playsInline   without it iOS Safari takes the video **fullscreen** on play, which replaces the
 *                 whole app with a system player and the shutter is gone
 *   muted         autoplay is refused for an unmuted stream. There is no audio track anyway; the
 *                 attribute is what the policy checks
 *   object-fit    the sensor is 4:3 and the screen is 19.5:9. `cover` crops, which is right: the
 *                 brackets mark the region the person is framing, and letterboxing would show them
 *                 black bars instead of paper
 *
 * ## What is captured is the sensor frame, not what is on screen
 *
 * `capture()` reads `videoWidth`/`videoHeight`. The preview is cropped by `object-fit` and the
 * element is whatever CSS made it; uploading the visible rectangle would silently send a
 * centre-crop of the page, which is the most expensive possible way to lose the corner of a diagram.
 *
 * ## The progress list is the whole point of 16.1.2
 *
 * A cold run of a fixture page is 12.08 s and `assemble` alone is 9.15 s. Seven rows that light up
 * in turn, each with its own time, are the difference between a nine-second wait that is working and
 * one that is stuck - and none of those rows is invented: the stage names arrive in `run_started`
 * and each timing arrives when the stage finished.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { Glyph } from "../App";
import { ApiError, type Prediction, type Stage } from "../lib/api";
import {
  type CameraState,
  capture,
  captureName,
  close,
  explain,
  open,
  ready,
} from "../lib/camera";
import { replace } from "../lib/route";
import { predictStream } from "../lib/stream";
import { Button, Card, Pill } from "../ui";
import "./Camera.css";

type Phase =
  | { kind: "viewfinder" }
  | { kind: "running"; preview: string }
  | { kind: "failed"; preview: string; status: number; detail: string; retryAfter: number | null };

interface Progress {
  stages: string[];
  done: Record<string, Stage>;
  active: string | null;
  bytes: number;
}

const EMPTY: Progress = { stages: [], done: {}, active: null, bytes: 0 };

export function Camera({ initial }: { initial?: Blob | null }) {
  const video = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const abort = useRef<AbortController | null>(null);

  const [camera, setCamera] = useState<CameraState>({ kind: "idle" });
  const [phase, setPhase] = useState<Phase>({ kind: "viewfinder" });
  const [progress, setProgress] = useState<Progress>(EMPTY);
  // The shutter waits for a frame worth sending. See `MIN_USABLE_WIDTH`.
  const [framing, setFraming] = useState(false);

  // -- the device ---------------------------------------------------------------------------

  useEffect(() => {
    // Not opened when an image was handed in (16.2.4's gallery import): asking for a camera the
    // person is not about to use is a permission prompt for nothing.
    if (initial) return;
    let cancelled = false;
    setCamera({ kind: "opening" });
    open().then((state) => {
      if (cancelled) {
        // The effect was torn down while `getUserMedia` was in flight - which is exactly what
        // StrictMode's double invocation does - and the stream that arrived belongs to nobody.
        if (state.kind === "live") close(state.stream);
        return;
      }
      if (state.kind === "live") streamRef.current = state.stream;
      setCamera(state);
    });
    return () => {
      cancelled = true;
      close(streamRef.current);
      streamRef.current = null;
    };
  }, [initial]);

  useEffect(() => {
    if (camera.kind !== "live" || !video.current) return;
    const element = video.current;
    element.srcObject = camera.stream;
    // `play()` rejects if the element is torn down first; there is nothing to do about it and
    // nothing to report, so the rejection is swallowed rather than logged as an error.
    void element.play().catch(() => {});

    // `loadedmetadata` fires once, and on some devices it fires with a placeholder size. Polling
    // until the frame is usable is unglamorous and it is what actually holds: the event tells you
    // the track exists, not that it is ready to be photographed.
    setFraming(false);
    const timer = window.setInterval(() => {
      if (ready(element)) {
        setFraming(true);
        window.clearInterval(timer);
      }
    }, 120);
    return () => window.clearInterval(timer);
  }, [camera]);

  // -- the run ------------------------------------------------------------------------------

  const send = useCallback(async (image: Blob, filename: string) => {
    const preview = URL.createObjectURL(image);
    setPhase({ kind: "running", preview });
    setProgress({ ...EMPTY, bytes: image.size });
    // The camera is released the moment a frame is taken. Holding it through a ten-second pipeline
    // run keeps the indicator light on and the sensor warm for no reason.
    close(streamRef.current);
    streamRef.current = null;

    const controller = new AbortController();
    abort.current = controller;

    try {
      await predictStream(
        image,
        filename,
        {
          onUpload: (bytes) => setProgress((p) => ({ ...p, bytes })),
          onStages: (stages) => setProgress((p) => ({ ...p, stages })),
          onStageStarted: (stage) => setProgress((p) => ({ ...p, active: stage })),
          onStageFinished: (row) =>
            setProgress((p) => ({
              ...p,
              active: null,
              done: { ...p.done, [row.stage]: row },
            })),
          onResult: (prediction: Prediction) => {
            URL.revokeObjectURL(preview);
            // `replace`, not a push: going "back" to a finished progress screen would show a dead
            // stage list with nothing running.
            replace({ view: "result", id: prediction.id });
          },
          onError: (status, detail) =>
            setPhase({ kind: "failed", preview, status, detail, retryAfter: null }),
        },
        controller.signal,
      );
    } catch (error) {
      if (controller.signal.aborted) return;
      const status = error instanceof ApiError ? error.status : 0;
      const detail =
        error instanceof ApiError ? error.message : "The server could not be reached.";
      const retryAfter = error instanceof ApiError ? error.retryAfter : null;
      setPhase({ kind: "failed", preview, status, detail, retryAfter });
    } finally {
      abort.current = null;
    }
  }, []);

  // A photograph handed in by 16.2.4 goes straight to the pipeline; there is nothing to frame.
  useEffect(() => {
    if (initial) void send(initial, captureName());
  }, [initial, send]);

  useEffect(() => () => abort.current?.abort(), []);

  const shutter = useCallback(async () => {
    if (!video.current) return;
    try {
      const frame = await capture(video.current);
      void send(frame, captureName());
    } catch (error) {
      setPhase({
        kind: "failed",
        preview: "",
        status: 0,
        detail: error instanceof Error ? error.message : "The frame could not be read.",
        retryAfter: null,
      });
    }
  }, [send]);

  // -- render -------------------------------------------------------------------------------

  if (phase.kind === "running") {
    return <Running preview={phase.preview} progress={progress} onCancel={() => abort.current?.abort()} />;
  }

  if (phase.kind === "failed") {
    return <Failed phase={phase} />;
  }

  if (camera.kind !== "live") {
    return <Unavailable state={camera} onRetry={() => open().then(setCamera)} />;
  }

  return (
    <div className="viewfinder">
      <video ref={video} playsInline muted autoPlay aria-label="Camera preview" />
      <div className="viewfinder-brackets" aria-hidden="true">
        <span />
        <span />
        <span />
        <span />
      </div>
      <p className="viewfinder-hint">
        {framing ? "Fill the frame with the paper. Hold steady." : "Focusing…"}
      </p>
      <div className="viewfinder-controls safe-bottom">
        <a href="#/" className="btn btn-quiet" aria-label="Back">
          <Glyph d="m15 6-6 6 6 6" />
        </a>
        <button
          className="shutter"
          onClick={shutter}
          disabled={!framing}
          aria-label="Take the photograph"
        >
          <span />
        </button>
        {/* Balances the back button so the shutter is centred. A flex spacer, not a control. */}
        <span style={{ width: 44 }} aria-hidden="true" />
      </div>
    </div>
  );
}

/**
 * The seven stages, filling in.
 *
 * Every row is drawn from `run_started`'s list, so the client never hardcodes the pipeline's shape.
 * Before the list arrives there is one row saying what is actually happening - the upload - because
 * on a mobile uplink that is a real and attributable part of the wait, and a blank screen during it
 * is the failure this row exists to fix.
 */
function Running({
  preview,
  progress,
  onCancel,
}: {
  preview: string;
  progress: Progress;
  onCancel: () => void;
}) {
  const finished = Object.keys(progress.done).length;
  const total = progress.stages.length || 7;

  return (
    <div className="scroll">
      <div className="page stack">
        {preview ? (
          <div className="capture-preview">
            <img src={preview} alt="The photograph being read" />
            <span className="capture-scan" aria-hidden="true" />
          </div>
        ) : null}

        <div className="row">
          <span className="eyebrow grow">Reading the page</span>
          <span className="dim num" style={{ fontSize: 12.5 }}>
            {finished}/{total}
          </span>
        </div>

        <Card style={{ padding: "var(--sp-4)" }}>
          {progress.stages.length === 0 ? (
            <StageRow
              name="upload"
              state="active"
              note={progress.bytes ? `${(progress.bytes / 1024).toFixed(0)} kB` : ""}
            />
          ) : (
            progress.stages.map((stage) => {
              const done = progress.done[stage];
              const state = done ? (done.ok ? "done" : "failed") : progress.active === stage ? "active" : "waiting";
              return (
                <StageRow
                  key={stage}
                  name={stage}
                  state={state}
                  degraded={done?.degraded}
                  note={done ? (done.cached ? "cached" : `${done.seconds.toFixed(2)}s`) : ""}
                  reason={done && !done.ok ? done.reason : ""}
                />
              );
            })
          )}
        </Card>

        <Button variant="quiet" block onClick={onCancel}>
          Cancel
        </Button>
      </div>
    </div>
  );
}

function StageRow({
  name,
  state,
  note,
  degraded,
  reason,
}: {
  name: string;
  state: "waiting" | "active" | "done" | "failed";
  note?: string;
  degraded?: boolean;
  reason?: string;
}) {
  return (
    <div className="stage-row" data-state={state}>
      <span className="stage-mark" aria-hidden="true" />
      <span className="stage-name">{name}</span>
      <span className="grow" />
      {degraded ? <Pill tone="degraded">fell back</Pill> : null}
      <span className="dim num stage-note">{note}</span>
      {reason ? <p className="stage-reason">{reason}</p> : null}
    </div>
  );
}

function Failed({ phase }: { phase: Extract<Phase, { kind: "failed" }> }) {
  const retryable = phase.status === 0 || phase.status >= 500 || phase.status === 429;
  return (
    <div className="page stack">
      {phase.preview ? (
        <div className="capture-preview">
          <img src={phase.preview} alt="The photograph that was sent" />
        </div>
      ) : null}
      <Card
        style={{
          padding: "var(--sp-5)",
          borderColor: "color-mix(in srgb, var(--stopped) 30%, transparent)",
        }}
        className="stack-sm"
      >
        <span className="eyebrow" style={{ color: "var(--stopped)" }}>
          {phase.status ? `error ${phase.status}` : "no answer"}
        </span>
        <p>{phase.detail}</p>
        {phase.retryAfter ? (
          <p className="dim" style={{ fontSize: 13 }}>
            The server asked for {phase.retryAfter} seconds before the next try.
          </p>
        ) : null}
      </Card>
      <div className="row" style={{ gap: "var(--sp-2)" }}>
        {retryable ? (
          <a href="#/camera" className="btn btn-primary grow" onClick={() => window.location.reload()}>
            Try again
          </a>
        ) : null}
        <a href="#/" className="btn grow">
          Back
        </a>
      </div>
    </div>
  );
}

/** One of the five refusals, with the sentence that matches it. */
function Unavailable({ state, onRetry }: { state: CameraState; onRetry: () => void }) {
  if (state.kind === "opening" || state.kind === "idle") {
    return (
      <div className="page stack">
        <div className="status">
          <span className="status-dot" data-state="checking" />
          <span className="grow muted">Opening the camera…</span>
        </div>
      </div>
    );
  }
  const { title, detail, retry } = explain(state);
  return (
    <div className="page stack">
      <Card style={{ padding: "var(--sp-5)" }} className="stack-sm">
        <span className="eyebrow">{title}</span>
        <p className="muted" style={{ fontSize: 14 }}>
          {detail}
        </p>
      </Card>
      <div className="row" style={{ gap: "var(--sp-2)" }}>
        {retry ? (
          <Button variant="primary" className="grow" onClick={onRetry}>
            Try again
          </Button>
        ) : null}
        <a href="#/" className="btn grow">
          Back
        </a>
      </div>
    </div>
  );
}
