/**
 * Phase 16.2.2 - reading 16.1.2's progress stream, and why not `EventSource`.
 *
 * `EventSource` is the obvious tool and it cannot be used here: **it only issues GET requests**. The
 * stream starts by uploading a photograph, which is a multipart POST, and there is no form of
 * `EventSource` that sends a body. So the stream is read from `fetch` with a `ReadableStream`, which
 * also buys two things `EventSource` would not have given:
 *
 *   an abort    a person who backs out mid-upload cancels the request rather than leaving a
 *               megabyte in flight and a pipeline running for an answer nobody will read
 *   the status  `EventSource` surfaces a failed handshake as an opaque `onerror`. 16.1.2 probes the
 *               model server *before* the stream opens precisely so an unreachable one is a real
 *               503, and that status is readable here
 *
 * What is lost is automatic reconnection, which would be wrong anyway: re-POSTing a photograph
 * because a socket blinked would run the pipeline twice and mint two ids.
 *
 * ## Framing
 *
 * The same four rules as the server's: `event:` names it, `data:` carries it, a blank line ends it,
 * a leading `:` is a keep-alive. The buffer is split on `\n` and the **remainder is kept**, because a
 * chunk boundary lands in the middle of a line far more often than not - and a parser that assumed
 * whole lines would lose exactly one frame per read, silently.
 */

import type { Prediction, Stage } from "./api";
import { ApiError } from "./api";

export interface StreamEvent {
  event: string;
  [key: string]: unknown;
}

export interface RunStarted extends StreamEvent {
  event: "run_started";
  stages: string[];
}

export interface StageFinished extends StreamEvent, Stage {
  event: "stage_finished";
  index: number;
}

export interface Handlers {
  onUpload?: (bytes: number, filename: string) => void;
  /** The stage list, before any stage runs, so seven rows can be drawn rather than guessed. */
  onStages?: (stages: string[]) => void;
  onStageStarted?: (stage: string, index: number) => void;
  onStageFinished?: (row: Stage & { index: number }) => void;
  onResult?: (prediction: Prediction) => void;
  /** A failure after the headers left, carrying the status it would have been. */
  onError?: (status: number, detail: string) => void;
}

/**
 * POST one image to `/predict/stream` and call the handlers as frames arrive.
 *
 * Returns when the stream closes. Throws `ApiError` only for a failure *before* the first byte -
 * a 415, a 413, a 429, or the 503 the backend answers when the model server is unreachable. Once the
 * stream is open a failure is an `onError` call, because the status line has already gone out and
 * pretending otherwise would mean holding the whole stream to decide what to throw.
 */
export async function predictStream(
  image: Blob,
  filename: string,
  handlers: Handlers,
  signal?: AbortSignal,
): Promise<void> {
  const body = new FormData();
  body.append("image", image, filename);

  const response = await fetch("/predict/stream", { method: "POST", body, signal });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const failure = await response.json();
      if (typeof failure?.detail === "string") detail = failure.detail;
    } catch {
      // A non-JSON error body is not worth a second failure.
    }
    const after = response.headers.get("retry-after");
    throw new ApiError(response.status, detail, after ? Number(after) : null);
  }
  if (!response.body) throw new ApiError(502, "the response carried no stream");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let name = "";
  let data = "";

  const flush = () => {
    if (!name && !data) return;
    name = "";
    const payload = data;
    data = "";
    let parsed: unknown;
    try {
      parsed = payload ? JSON.parse(payload) : {};
    } catch {
      // One malformed frame is not worth losing the run for. The producer is our own server and
      // always sends JSON, so this is a path that should never execute - but the alternative is a
      // reader that dies on it and takes the finished page with it.
      return;
    }
    if (parsed && typeof parsed === "object") dispatch(parsed as StreamEvent, handlers);
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      // The last element is whatever came after the final newline - a partial line most of the
      // time. Kept, not parsed: dropping it loses one frame per chunk, silently.
      buffer = lines.pop() ?? "";
      for (const raw of lines) {
        const line = raw.replace(/\r$/, "");
        if (line.startsWith(":")) continue;
        if (!line) {
          flush();
          continue;
        }
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) data = line.slice(5).trim();
      }
    }
    // A stream that ended without its final blank line still delivered its last frame, and that
    // frame is the `result` - the one thing the whole request was for.
    buffer += decoder.decode();
    for (const raw of buffer.split("\n")) {
      const line = raw.replace(/\r$/, "");
      if (line.startsWith("event:")) name = line.slice(6).trim();
      else if (line.startsWith("data:")) data = line.slice(5).trim();
    }
    flush();
  } finally {
    // Releasing the lock matters on an abort: without it the body stays locked and the connection
    // is not returned to the pool.
    reader.releaseLock();
  }
}

function dispatch(event: StreamEvent, handlers: Handlers): void {
  switch (event.event) {
    case "upload":
      handlers.onUpload?.(Number(event.bytes_in ?? 0), String(event.filename ?? ""));
      return;
    case "run_started":
      handlers.onStages?.((event as RunStarted).stages ?? []);
      return;
    case "stage_started":
      handlers.onStageStarted?.(String(event.stage ?? ""), Number(event.index ?? 0));
      return;
    case "stage_finished":
      handlers.onStageFinished?.(event as unknown as Stage & { index: number });
      return;
    case "run_finished":
      // Deliberately not surfaced. `result` follows it and carries everything this one hints at;
      // a client that acted on `run_finished` would show "done" a frame before it had the answer.
      return;
    case "result":
      handlers.onResult?.(event as unknown as Prediction);
      return;
    case "error":
      handlers.onError?.(Number(event.status ?? 502), String(event.detail ?? "the run failed"));
      return;
    default:
      return;
  }
}
