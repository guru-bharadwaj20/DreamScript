/**
 * Phase 16.2.2 - the camera's refusals, and the readiness guard.
 *
 * `getUserMedia` itself cannot be tested without a device, and there is nothing worth testing in one
 * call. What is worth testing is everything around it: that each of the five refusals is a distinct
 * state with its own sentence, that the guard refuses a frame too small to be worth uploading, and
 * that closing a stream stops *every* track rather than the first one.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  MIN_USABLE_WIDTH,
  type CameraState,
  captureName,
  close,
  explain,
  open,
  ready,
  secureEnough,
} from "./camera";

function named(name: string): Error {
  const error = new Error(name);
  error.name = name;
  return error;
}

/** A `navigator.mediaDevices` that fails the way a browser does. */
function withMedia(behaviour: (() => Promise<MediaStream>) | null, secure = true) {
  const g = globalThis as unknown as {
    navigator: { mediaDevices?: { getUserMedia: () => Promise<MediaStream> } };
    window: { isSecureContext: boolean };
    isSecureContext: boolean;
  };
  const previous = g.navigator?.mediaDevices;
  Object.defineProperty(globalThis, "isSecureContext", { value: secure, configurable: true });
  Object.defineProperty(globalThis, "window", {
    value: { isSecureContext: secure },
    configurable: true,
    writable: true,
  });
  Object.defineProperty(globalThis, "navigator", {
    value: behaviour ? { mediaDevices: { getUserMedia: behaviour } } : {},
    configurable: true,
    writable: true,
  });
  return () => {
    if (previous) {
      Object.defineProperty(globalThis, "navigator", {
        value: { mediaDevices: previous },
        configurable: true,
        writable: true,
      });
    }
  };
}

afterEach(() => vi.restoreAllMocks());

describe("open", () => {
  it("is insecure before it is anything else", async () => {
    // `navigator.mediaDevices` is **undefined** on an insecure origin, not an error - so the check
    // has to come before the call. This is the state every developer testing on a phone over the
    // LAN hits first.
    const restore = withMedia(null, false);
    expect(await open()).toEqual({ kind: "insecure" });
    restore();
  });

  it.each([
    ["NotAllowedError", "denied"],
    ["SecurityError", "denied"],
    ["NotFoundError", "none"],
    ["OverconstrainedError", "none"],
    ["NotReadableError", "busy"],
    ["AbortError", "busy"],
  ])("maps %s to %s", async (name, kind) => {
    const restore = withMedia(() => Promise.reject(named(name)));
    expect((await open()).kind).toBe(kind);
    restore();
  });

  it("keeps the name of an unrecognised failure rather than swallowing it", async () => {
    const restore = withMedia(() => Promise.reject(named("SomethingNewError")));
    const state = await open();
    expect(state.kind).toBe("failed");
    expect((state as Extract<CameraState, { kind: "failed" }>).detail).toContain("SomethingNewError");
    restore();
  });

  it("asks for the rear camera as ideal, not exact", async () => {
    // `exact: "environment"` fails outright on every laptop. `ideal` prefers the rear camera and
    // falls back to whatever exists, so the app works where it is used and where it is developed.
    // Typed as taking the constraints, so `mock.calls[0][0]` is a real element rather than an
    // index into an empty tuple - which is what `vi.fn(() => ...)` infers from a zero-arg lambda.
    const getUserMedia = vi.fn((_constraints?: MediaStreamConstraints) =>
      Promise.resolve({} as MediaStream),
    );
    const restore = withMedia(getUserMedia as unknown as () => Promise<MediaStream>);
    await open();
    const constraints = getUserMedia.mock.calls[0]?.[0] as MediaStreamConstraints;
    const video = constraints.video as MediaTrackConstraints;
    expect(video.facingMode).toEqual({ ideal: "environment" });
    expect(constraints.audio).toBe(false);
    restore();
  });
});

describe("explain", () => {
  it.each(["insecure", "denied", "none", "busy"] as const)("gives %s its own sentence", (kind) => {
    const { title, detail } = explain({ kind } as CameraState);
    expect(title).toBeTruthy();
    expect(detail.length).toBeGreaterThan(30);
  });

  it("offers a retry only where one can work", () => {
    // An insecure origin and a missing camera do not become fixable by tapping again.
    expect(explain({ kind: "insecure" }).retry).toBe(false);
    expect(explain({ kind: "none" }).retry).toBe(false);
    expect(explain({ kind: "denied" }).retry).toBe(true);
    expect(explain({ kind: "busy" }).retry).toBe(true);
  });

  it("gives the four refusals four different sentences", () => {
    const said = (["insecure", "denied", "none", "busy"] as const).map(
      (kind) => explain({ kind } as CameraState).detail,
    );
    expect(new Set(said).size).toBe(4);
  });
});

describe("ready", () => {
  const frame = (w: number, h: number) => ({ videoWidth: w, videoHeight: h }) as HTMLVideoElement;

  it("refuses a frame the track has not sized yet", () => {
    expect(ready(null)).toBe(false);
    expect(ready(frame(0, 0))).toBe(false);
  });

  it("refuses the placeholder Chrome's fake device reports", () => {
    // 2x2, observed. A real phone does the same thing for a shorter moment, and a 2x2 upload is a
    // wasted round trip plus a confusing "the detector found nothing".
    expect(ready(frame(2, 2))).toBe(false);
  });

  it("accepts anything a real camera produces", () => {
    expect(ready(frame(MIN_USABLE_WIDTH, 240))).toBe(true);
    expect(ready(frame(1920, 1440))).toBe(true);
  });

  it("sets the floor far below any camera and far above any placeholder", () => {
    expect(MIN_USABLE_WIDTH).toBeGreaterThan(64);
    expect(MIN_USABLE_WIDTH).toBeLessThan(640);
  });
});

describe("close", () => {
  it("stops every track, not the first", () => {
    // `MediaStream` has no `stop()`, and stopping one track leaves the camera light on.
    const stops = [vi.fn(), vi.fn(), vi.fn()];
    const stream = { getTracks: () => stops.map((stop) => ({ stop })) } as unknown as MediaStream;
    close(stream);
    for (const stop of stops) expect(stop).toHaveBeenCalledOnce();
  });

  it("does nothing, loudly or otherwise, for a stream that never opened", () => {
    expect(() => close(null)).not.toThrow();
    expect(() => close(undefined)).not.toThrow();
  });
});

describe("captureName", () => {
  it("is a jpg the backend accepts, with the time in it", () => {
    const name = captureName(new Date("2026-09-26T11:27:03.500Z"));
    expect(name).toBe("dreamscript-2026-09-26T11-27-03.jpg");
    // `.jpg` is in `ALLOWED_SUFFIXES`, and the backend rejects on the suffix before reading a byte.
    expect(name.endsWith(".jpg")).toBe(true);
    expect(name).not.toMatch(/[:]/);
  });
});

describe("secureEnough", () => {
  it("asks the browser rather than parsing the protocol", () => {
    // `isSecureContext` is true for localhost as well as HTTPS; a hand-rolled protocol check gets
    // the localhost exemption wrong and refuses the camera during development.
    const restore = withMedia(() => Promise.resolve({} as MediaStream), true);
    expect(secureEnough()).toBe(true);
    restore();
  });
});
