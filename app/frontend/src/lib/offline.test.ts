/**
 * Phase 16.2.10 - the offline bundle, and the two ways it could quietly be a lie.
 *
 * The first is a **hand-written answer**. A demo payload is the easiest thing in a codebase to
 * fabricate and the hardest to notice, because a fabricated one looks like the product working
 * perfectly. So the assertions below are about provenance as much as shape: seven stages, none of
 * them `cached`, all of 13.4's degradation fields present, and a `diagram_type` that is allowed to
 * be *wrong* - three of the five are, and a bundle where all five were right would be the signal
 * that someone had tidied them.
 *
 * The second is a **stored answer that cannot be reached**. The result route only parses hexadecimal
 * ids (`route.ts`), so an id in this bundle that route rejects would make its card land on the
 * capture screen instead - a dead example, visible only by tapping it offline. It is checked here
 * against the real parser rather than against a copy of its regular expression.
 */

import { describe, expect, it } from "vitest";

import { ApiError } from "./api";
import {
  CACHED,
  CAPTURED,
  CAPTURED_BY,
  cachedFor,
  cachedForFile,
  definitelyOffline,
  looksOffline,
  watchOnline,
} from "./offline";
import { parse } from "./route";

/**
 * Swap `navigator.onLine` for one call. Restored even if the body throws.
 *
 * `defineProperty` rather than assignment, which is what the first version of this helper used and
 * what node refuses: `globalThis.navigator` is an accessor with **only a getter** from node 21
 * onward, so `globalThis.navigator = ...` throws. Worth knowing beyond the test - it is also why
 * `definitelyOffline()` answers `false` under vitest: node's `navigator` has no `onLine` at all,
 * which is the case the "does not guess" test below is about.
 */
function withOnLine<T>(value: boolean | undefined, body: () => T): T {
  const had = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", {
    value: { onLine: value },
    configurable: true,
    writable: true,
  });
  try {
    return body();
  } finally {
    if (had) Object.defineProperty(globalThis, "navigator", had);
    else delete (globalThis as { navigator?: unknown }).navigator;
  }
}

describe("the bundled answers", () => {
  it("has one for every example the gallery offers", () => {
    // Five cards, five stored answers. A card with no answer is a card that does nothing when the
    // network is gone, which is the exact failure this row exists to remove.
    expect(CACHED.map((entry) => entry.file)).toEqual([
      "flowchart.png",
      "state_machine.png",
      "er_diagram.png",
      "wireframe.png",
      "circuit.png",
    ]);
  });

  it("records when and against what they were captured", () => {
    expect(CAPTURED).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(CAPTURED_BY).toBeTruthy();
  });

  it.each(CACHED)("$file carries a whole prediction, not a summary", ({ prediction }) => {
    // 13.4's rule at the last possible hop: the degradation fields travel unflattened, and a
    // hand-written "example" would be the one payload in this project that dropped them.
    expect(typeof prediction.ok).toBe("boolean");
    expect(typeof prediction.degraded).toBe("boolean");
    expect(typeof prediction.needs_confirmation).toBe("boolean");
    expect(prediction).toHaveProperty("stopped_at");
    expect(prediction.diagram_type).toBeTruthy();
    expect(Array.isArray(prediction.traversal)).toBe(true);
    expect(Array.isArray(prediction.corrections)).toBe(true);
  });

  it.each(CACHED)("$file reports all seven stages", ({ prediction }) => {
    expect(prediction.stages.map((stage) => stage.stage)).toEqual([
      "detect",
      "classify",
      "assemble",
      "traverse",
      "serialise",
      "generate",
      "verify",
    ]);
  });

  it.each(CACHED)("$file was captured with the stage cache cold", ({ prediction }) => {
    // The first capture was taken warm and every stage came back `cached: true` at 0.06 s a page.
    // Real, and a lie about the product: an offline screen whose header says `0.06s` teaches a
    // person this pipeline is instant. A warm re-capture would trip this.
    expect(prediction.stages.every((stage) => !stage.cached)).toBe(true);
    expect(prediction.seconds).toBeGreaterThan(0.05);
  });

  it.each(CACHED)("$file is addressable by the app's own router", ({ id }) => {
    // Against `parse`, not against a copy of its pattern. A non-hex id here would send the card to
    // the capture screen and the example would be dead offline and nowhere else.
    expect(parse(`#/r/${id}`)).toEqual({ view: "result", id });
  });

  it("gives every answer a distinct id", () => {
    expect(new Set(CACHED.map((entry) => entry.id)).size).toBe(CACHED.length);
  });

  it("keeps the readings that are wrong rather than tidying them", () => {
    // Three of five: `er_diagram`, `wireframe` and `circuit` all come back as flowcharts, which is
    // 4.1.5's aspect-ratio leak - every fixture is 320x240. The gallery computes this count instead
    // of asserting it; this test is where the measurement is pinned, so a regeneration that changes
    // it is noticed here rather than on a screen.
    const wrong = CACHED.filter(
      (entry) => entry.prediction.diagram_type !== entry.file.replace(/\.png$/, ""),
    );
    expect(wrong.map((entry) => entry.file)).toEqual([
      "er_diagram.png",
      "wireframe.png",
      "circuit.png",
    ]);
  });

  it("holds real IR, with the ambiguity the pipeline recorded", () => {
    const flowchart = cachedForFile("flowchart.png");
    expect(flowchart?.prediction.ir?.nodes.length).toBeGreaterThan(0);
    // 2.1.6's unresolved edges are in the bundle, so 16.2.9's panel has something to show offline.
    // A tidied example would have none of these, on any page.
    const unresolved = CACHED.reduce(
      (total, entry) => total + (entry.prediction.ir?.unresolved_edges?.length ?? 0),
      0,
    );
    expect(unresolved).toBeGreaterThan(0);
  });
});

describe("cachedFor", () => {
  it("finds a bundled answer by the id the backend minted for it", () => {
    const first = CACHED[0];
    expect(cachedFor(first.id)?.file).toBe(first.file);
  });

  it("is null for any other id, so a live reading is never shadowed", () => {
    expect(cachedFor("deadbeef")).toBeNull();
    expect(cachedFor("")).toBeNull();
  });

  it("finds one by file for the gallery", () => {
    expect(cachedForFile("circuit.png")?.name).toBe("Circuit");
    expect(cachedForFile("nothing.png")).toBeNull();
  });
});

describe("definitelyOffline", () => {
  it("is true only when the operating system says there is no route", () => {
    expect(withOnLine(false, definitelyOffline)).toBe(true);
    expect(withOnLine(true, definitelyOffline)).toBe(false);
  });

  it("is false when the browser does not say, rather than guessing", () => {
    // `onLine` missing is not evidence of anything. Defaulting to "offline" here would hide the
    // whole app behind a stored example on any runtime that does not implement the property.
    expect(withOnLine(undefined, definitelyOffline)).toBe(false);
  });
});

describe("looksOffline", () => {
  it("is false for anything the server answered, however badly", () => {
    // The distinction the Capture screen's two sentences rest on: a status came back, so the server
    // is there and the fix is not a network.
    for (const status of [400, 413, 415, 429, 500, 502, 503, 504]) {
      expect(looksOffline(new ApiError(status, "no"))).toBe(false);
    }
  });

  it("is false for a 503 even though a 503 means the model server is down", () => {
    // 16.1.2 answers 503 when the *model* server is unreachable. The app backend answered it, so
    // this is not an offline device and must not be reported as one.
    expect(looksOffline(new ApiError(503, "the model server is unreachable"))).toBe(false);
  });

  it("is true for the TypeError fetch rejects with on a network failure", () => {
    expect(withOnLine(true, () => looksOffline(new TypeError("Failed to fetch")))).toBe(true);
  });

  it("does not match on the message, which three engines word differently", () => {
    // Chrome says "Failed to fetch", Firefox "NetworkError when attempting to fetch resource",
    // Safari "Load failed". The rejection *type* is specified; the text is not.
    for (const message of ["Failed to fetch", "Load failed", "NetworkError", ""]) {
      expect(withOnLine(true, () => looksOffline(new TypeError(message)))).toBe(true);
    }
  });

  it("is true for anything at all once the device says it is offline", () => {
    expect(withOnLine(false, () => looksOffline(new Error("something else")))).toBe(true);
  });

  it("is false for an ordinary Error while the device says it is online", () => {
    // A bug in this client is not a network problem and must not be reported as one - "you are
    // offline" is the most effective way to stop someone reporting a real fault.
    expect(withOnLine(true, () => looksOffline(new Error("boom")))).toBe(false);
  });
});

describe("watchOnline", () => {
  it("reports both transitions and unsubscribes cleanly", () => {
    const listeners = new Map<string, Set<() => void>>();
    const fake = {
      addEventListener(name: string, fn: () => void) {
        (listeners.get(name) ?? listeners.set(name, new Set()).get(name)!).add(fn);
      },
      removeEventListener(name: string, fn: () => void) {
        listeners.get(name)?.delete(fn);
      },
    };
    const target = globalThis as { window?: unknown };
    const previous = target.window;
    target.window = fake;
    try {
      const seen: boolean[] = [];
      const stop = withOnLine(true, () => watchOnline((online) => seen.push(online)));
      withOnLine(false, () => listeners.get("offline")?.forEach((fn) => fn()));
      withOnLine(true, () => listeners.get("online")?.forEach((fn) => fn()));
      expect(seen).toEqual([false, true]);
      stop();
      expect(listeners.get("online")?.size).toBe(0);
      expect(listeners.get("offline")?.size).toBe(0);
    } finally {
      if (previous === undefined) delete target.window;
      else target.window = previous;
    }
  });
});
