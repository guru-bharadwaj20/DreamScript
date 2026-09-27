/**
 * Phase 16.2.10 - what this app can still do with no network, and what it must not pretend.
 *
 * This client holds no model. Every answer comes from a server, so "offline" is not a degraded mode
 * here the way it is in a note-taking app - it is the **absence of the entire product**. The only
 * honest offline feature is therefore a set of *stored* answers, and the only thing that makes them
 * honest is that a person can tell at a glance they are stored.
 *
 * ## Nothing in `offline.data.json` is written by hand
 *
 * The five answers are what `app/backend/main.py` actually replied for the five fixtures this
 * repository commits, captured by `scripts/make_offline_examples.py` against a running backend with
 * the stage cache **cold**, so the timings in them are real work rather than cache hits.
 *
 * The first capture was taken with the cache warm and every stage came back `cached: true` at 0.06 s
 * a page. Technically real, and a lie about the product: an offline screen whose header reads
 * `0.06s` teaches a person that this pipeline is instant. Cold, the same pages are 0.45-0.69 s of
 * genuine work with `assemble` taking 85% of it, which is the shape of the real thing.
 *
 * And they are kept **warts included**. Three of the five come back read as flowcharts - 4.1.5's
 * aspect-ratio leak, since every fixture is 320x240 - and the cache keeps that. A demo mode is the
 * most tempting place in a codebase to store a perfect answer, and the hardest place to notice one,
 * because it looks like the product working.
 *
 * ## Network first. Always.
 *
 * `cachedFor(id)` is consulted only after a request has failed, never instead of one. A stored
 * answer that pre-empted the server would show a person the reading from before the correction they
 * just made - and would do it silently, which is the worst available outcome for a screen whose
 * whole job is telling someone how much to trust what they are looking at.
 *
 * ## `navigator.onLine` is not a boolean about connectivity
 *
 * It is two different claims wearing one type. `false` is trustworthy: the operating system is
 * saying there is no route, and a request will fail. `true` means only that *an* interface is up -
 * a captive portal, a hotel wifi that has not been paid for, or a laptop on a LAN with no uplink all
 * report `true`. So it is used in exactly one direction here: `false` skips a request that cannot
 * succeed, and `true` is never taken as permission to claim the server is reachable. The other
 * direction is answered by a request that actually failed, which is what `looksOffline` classifies.
 *
 * ## The shell still has to load, and that is 16.3.1
 *
 * None of this makes the app open on a cold start with no network - that needs a service worker, and
 * it is the next row. What this row delivers is an app that is **already open** behaving correctly
 * when the network goes: no spinner that never resolves, no "error 0", and the examples still
 * readable. The two halves are verified separately and the plan says which row did which.
 */

import type { Prediction } from "./api";
import { ApiError } from "./api";
// Named `captured` rather than `data` or `document`: `document` is a global in every browser
// and shadowing it inside a module that also touches `navigator` and `window` is a trap.
import captured from "./offline.data.json";

export interface Cached {
  /** The file in `public/examples/`, which is also the gallery's key. */
  file: string;
  name: string;
  /** The id the backend minted for this run. A real address, and the route the gallery links to. */
  id: string;
  bytes: number;
  prediction: Prediction;
}

/** When these answers were captured, as a date a person can read. */
export const CAPTURED: string = captured.generated;

/** The backend version that produced them, so a stale cache is attributable rather than mysterious. */
export const CAPTURED_BY: string = captured.backend;

export const CACHED: Cached[] = captured.examples as unknown as Cached[];

const BY_ID = new Map(CACHED.map((entry) => [entry.id, entry]));
const BY_FILE = new Map(CACHED.map((entry) => [entry.file, entry]));

/** The stored answer for an id, or `null`. Consulted **after** a failed request, never before. */
export function cachedFor(id: string): Cached | null {
  return BY_ID.get(id) ?? null;
}

export function cachedForFile(file: string): Cached | null {
  return BY_FILE.get(file) ?? null;
}

/**
 * Does the operating system say there is no route?
 *
 * Only ever used in the `false` direction - see the note above. A `true` here is not evidence the
 * server is reachable and nothing in this app treats it as such.
 */
export function definitelyOffline(): boolean {
  return typeof navigator !== "undefined" && navigator.onLine === false;
}

/**
 * Was this failure the network rather than the server?
 *
 * The distinction is the whole reason the Capture screen has two different sentences. An `ApiError`
 * means bytes went out and a status came back: the server is there and refused, and retrying the
 * same photograph may be pointless. A `TypeError` from `fetch` - "Failed to fetch", "NetworkError
 * when attempting to fetch resource", "Load failed" on Safari, each browser with its own wording -
 * means nothing reached anything, and the fix is a network rather than a different photograph.
 *
 * The wording is deliberately not matched. `fetch` rejects with `TypeError` for a network failure
 * and that is specified; the message text is not, and three engines disagree about it.
 */
export function looksOffline(error: unknown): boolean {
  if (error instanceof ApiError) return false;
  if (definitelyOffline()) return true;
  return error instanceof TypeError;
}

/**
 * Call back whenever the browser changes its mind about being online.
 *
 * Both events, because they are not symmetric in usefulness: `offline` is the one worth reacting to
 * (it is trustworthy), and `online` only means it is worth *trying* again.
 */
export function watchOnline(onChange: (online: boolean) => void): () => void {
  const fire = () => onChange(!definitelyOffline());
  window.addEventListener("online", fire);
  window.addEventListener("offline", fire);
  return () => {
    window.removeEventListener("online", fire);
    window.removeEventListener("offline", fire);
  };
}
