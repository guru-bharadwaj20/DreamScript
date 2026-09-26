/**
 * Phase 16.2.2 - the SSE reader, against the thing that actually breaks it.
 *
 * A parser that assumes each network chunk contains whole lines works perfectly in every test
 * written by hand and loses exactly one frame per read against a real socket, silently. So the
 * fixture here is a **byte-level stream** that can be sliced anywhere, and the central test slices
 * the same transcript at every single offset and requires the same frames out of all of them.
 */

import { describe, expect, it } from "vitest";

import { ApiError } from "./api";
import { predictStream } from "./stream";

const TRANSCRIPT = [
  'event: upload\ndata: {"event":"upload","filename":"p.jpg","bytes_in":2901}\n\n',
  ': ping\n\n',
  'event: run_started\ndata: {"event":"run_started","stages":["detect","classify","assemble"]}\n\n',
  'event: stage_started\ndata: {"event":"stage_started","stage":"detect","index":1}\n\n',
  'event: stage_finished\ndata: {"event":"stage_finished","stage":"detect","ok":true,"seconds":2.9,"cached":false,"degraded":false,"confidence":1,"reason":"","index":1}\n\n',
  'event: run_finished\ndata: {"event":"run_finished","stopped_at":null}\n\n',
  'event: result\ndata: {"event":"result","id":"abc123","ok":true,"code":"print(1)"}\n\n',
].join("");

/** A `Response` whose body delivers `text` in chunks of exactly `size` bytes. */
function streaming(text: string, size: number, status = 200): Response {
  const bytes = new TextEncoder().encode(text);
  let offset = 0;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (offset >= bytes.length) {
        controller.close();
        return;
      }
      controller.enqueue(bytes.slice(offset, offset + size));
      offset += size;
    },
  });
  return new Response(body, {
    status,
    headers: { "Content-Type": "text/event-stream" },
  });
}

function collect() {
  const seen: string[] = [];
  const stages: string[][] = [];
  const finished: unknown[] = [];
  let result: { id?: string } | null = null;
  let failure: { status: number; detail: string } | null = null;
  let uploaded = 0;
  return {
    seen,
    stages,
    finished,
    get result() {
      return result;
    },
    get failure() {
      return failure;
    },
    get uploaded() {
      return uploaded;
    },
    handlers: {
      onUpload: (bytes: number) => {
        seen.push("upload");
        uploaded = bytes;
      },
      onStages: (list: string[]) => {
        seen.push("run_started");
        stages.push(list);
      },
      onStageStarted: () => seen.push("stage_started"),
      onStageFinished: (row: unknown) => {
        seen.push("stage_finished");
        finished.push(row);
      },
      onResult: (prediction: { id?: string }) => {
        seen.push("result");
        result = prediction;
      },
      onError: (status: number, detail: string) => {
        seen.push("error");
        failure = { status, detail };
      },
    },
  };
}

function withFetch(response: Response | (() => Response), run: () => Promise<void>) {
  const original = globalThis.fetch;
  globalThis.fetch = (async () =>
    typeof response === "function" ? response() : response) as typeof fetch;
  return run().finally(() => {
    globalThis.fetch = original;
  });
}

const IMAGE = new Blob([new Uint8Array([0x89, 0x50, 0x4e, 0x47])], { type: "image/jpeg" });

describe("predictStream", () => {
  it("reads every frame in order", async () => {
    const sink = collect();
    await withFetch(streaming(TRANSCRIPT, 4096), () =>
      predictStream(IMAGE, "p.jpg", sink.handlers),
    );
    // `run_finished` is deliberately not surfaced: `result` follows it and carries everything.
    expect(sink.seen).toEqual([
      "upload",
      "run_started",
      "stage_started",
      "stage_finished",
      "result",
    ]);
    expect(sink.uploaded).toBe(2901);
    expect(sink.stages[0]).toEqual(["detect", "classify", "assemble"]);
    expect(sink.result?.id).toBe("abc123");
  });

  it("gives the same frames however the bytes are split", async () => {
    // The test this file exists for. One byte at a time is the worst case and the one that catches
    // a parser assuming whole lines; the sweep covers every boundary in between.
    const reference = collect();
    await withFetch(streaming(TRANSCRIPT, 1 << 20), () =>
      predictStream(IMAGE, "p.jpg", reference.handlers),
    );

    for (const size of [1, 2, 3, 7, 13, 64, 97, 256, 999]) {
      const sink = collect();
      await withFetch(streaming(TRANSCRIPT, size), () =>
        predictStream(IMAGE, "p.jpg", sink.handlers),
      );
      expect(sink.seen, `chunk size ${size}`).toEqual(reference.seen);
      expect(sink.result?.id, `chunk size ${size}`).toBe("abc123");
      expect(sink.finished, `chunk size ${size}`).toEqual(reference.finished);
    }
  });

  it("keeps the last frame when the stream ends without its blank line", async () => {
    // The `result` frame is the one thing the whole request was for, and a server that closes a
    // byte early must not cost it.
    const clipped = TRANSCRIPT.replace(/\n\n$/, "\n");
    const sink = collect();
    await withFetch(streaming(clipped, 17), () => predictStream(IMAGE, "p.jpg", sink.handlers));
    expect(sink.result?.id).toBe("abc123");
  });

  it("skips keep-alive comments without breaking the frame around them", async () => {
    const noisy = TRANSCRIPT.replace(
      "event: stage_started",
      ": ping\n\n: ping\n\nevent: stage_started",
    );
    const sink = collect();
    await withFetch(streaming(noisy, 11), () => predictStream(IMAGE, "p.jpg", sink.handlers));
    expect(sink.seen.filter((e) => e === "stage_started")).toHaveLength(1);
    expect(sink.result?.id).toBe("abc123");
  });

  it("drops one malformed frame rather than losing the run", async () => {
    const broken = TRANSCRIPT.replace(
      'data: {"event":"stage_started","stage":"detect","index":1}',
      "data: {not json",
    );
    const sink = collect();
    await withFetch(streaming(broken, 32), () => predictStream(IMAGE, "p.jpg", sink.handlers));
    expect(sink.seen).not.toContain("stage_started");
    expect(sink.result?.id).toBe("abc123");
  });

  it("surfaces a mid-stream error with the status it would have been", async () => {
    const failing =
      TRANSCRIPT.split("event: run_finished")[0] +
      'event: error\ndata: {"event":"error","status":502,"detail":"pipeline failed"}\n\n';
    const sink = collect();
    await withFetch(streaming(failing, 23), () => predictStream(IMAGE, "p.jpg", sink.handlers));
    expect(sink.failure).toEqual({ status: 502, detail: "pipeline failed" });
    expect(sink.result).toBeNull();
  });

  it("throws ApiError for a failure before the first byte", async () => {
    // A 503 from the pre-stream probe is a real status line, and the client shows it differently
    // from a mid-stream failure because retrying it can actually work.
    const refused = new Response(JSON.stringify({ detail: "the model server is not reachable" }), {
      status: 503,
      headers: { "Content-Type": "application/json", "Retry-After": "5" },
    });
    const sink = collect();
    await expect(
      withFetch(refused, () => predictStream(IMAGE, "p.jpg", sink.handlers)),
    ).rejects.toMatchObject({ status: 503, retryAfter: 5 });
    expect(sink.seen).toEqual([]);
  });

  it("reports a 429 as retryable with its Retry-After", async () => {
    const limited = new Response(JSON.stringify({ detail: "6 predict requests per 60s" }), {
      status: 429,
      headers: { "Content-Type": "application/json", "Retry-After": "48" },
    });
    const sink = collect();
    const error = await withFetch(limited, () =>
      predictStream(IMAGE, "p.jpg", sink.handlers),
    ).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).retryable).toBe(true);
    expect((error as ApiError).retryAfter).toBe(48);
  });

  it("reports a 415 as not retryable", async () => {
    // This photograph will never work, so offering "try again" would be a lie.
    const wrongType = new Response(JSON.stringify({ detail: "unsupported file type .pdf" }), {
      status: 415,
      headers: { "Content-Type": "application/json" },
    });
    const error = await withFetch(wrongType, () =>
      predictStream(IMAGE, "p.jpg", collect().handlers),
    ).catch((e: unknown) => e);
    expect((error as ApiError).retryable).toBe(false);
  });

  it("splits a multi-byte character across a chunk boundary without corrupting it", async () => {
    // `TextDecoder` with `{stream: true}` is what makes this work; without it a label containing an
    // em dash or a non-Latin script comes back with a replacement character in it.
    const unicode = TRANSCRIPT.replace('"print(1)"', '"# café — 図\\nprint(1)"');
    for (const size of [1, 2, 3, 5]) {
      const sink = collect();
      await withFetch(streaming(unicode, size), () =>
        predictStream(IMAGE, "p.jpg", sink.handlers),
      );
      expect((sink.result as { code?: string } | null)?.code, `chunk ${size}`).toBe(
        "# café — 図\nprint(1)",
      );
    }
  });
});
