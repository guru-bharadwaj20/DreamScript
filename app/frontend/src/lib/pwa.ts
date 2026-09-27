/**
 * Phase 16.3.1 - registering the shell, updating it, and offering to install.
 *
 * Three jobs that all belong together because they are the same object's lifecycle, and each one
 * has a way of going wrong that is invisible until somebody else's phone is holding it.
 *
 * ## Only in a production build
 *
 * A service worker in front of Vite's dev server caches a module graph that the dev server is
 * actively rewriting, and the symptom - an edit that does not appear, sometimes, on one machine -
 * costs far more than the worker is worth in development. There is also nothing to test there: the
 * shell precaches *hashed* assets, and in development there are none.
 *
 * ## The update is offered, not taken
 *
 * `registration.waiting` means a new worker has installed and is holding. It holds because `sw.js`
 * does not call `skipWaiting` on its own: a worker that takes over immediately swaps the asset
 * cache under a page that is already running, and the running page then asks for a chunk from a
 * build that has just been deleted - a 404 from its own origin, on a page that was working a second
 * ago.
 *
 * So the page is told, a strip appears, and the person's tap does two things in the only order that
 * works: tell the waiting worker to `skipWaiting`, then reload once `controllerchange` fires. The
 * `refreshing` guard is not paranoia - `controllerchange` can fire more than once, and a reload
 * inside its handler without a guard is a reload loop, which is one of the few bugs a web app can
 * have that a person genuinely cannot get out of.
 *
 * ## `beforeinstallprompt`, and the platform where it does not exist
 *
 * Chrome fires it, and it is the *only* way to show an install button: `prompt()` may only be
 * called on a saved event and only from a user gesture. **Safari fires nothing at all.** On iOS the
 * person has to choose Share -> Add to Home Screen themselves, so `installability()` reports which
 * of the three situations this is - installable now, already installed, or iOS-with-instructions -
 * and the About screen shows the right one instead of a button that would do nothing on the
 * platform where most people will read it.
 */

export type Install =
  /** The browser has offered a prompt and it is saved. */
  | { state: "ready" }
  /** Running from the home screen already. */
  | { state: "installed" }
  /** iOS Safari: installable by hand, and only by hand. */
  | { state: "manual" }
  /** No prompt has arrived and this is not iOS - it may yet, or the criteria are not met. */
  | { state: "unknown" };

/**
 * The saved `beforeinstallprompt` event.
 *
 * Module scope rather than React state, for the same reason `held.ts` is: the event arrives when it
 * arrives, usually before any component that cares has mounted, and there is no way to ask for it
 * again. Missing it means no install button for the whole session.
 */
let prompt: BeforeInstallPromptEvent | null = null;
const listeners = new Set<() => void>();

interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>;
  readonly userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

function announce(): void {
  for (const listener of listeners) listener();
}

/** Subscribe to changes in what `installability()` would answer. */
export function watchInstall(onChange: () => void): () => void {
  listeners.add(onChange);
  return () => listeners.delete(onChange);
}

/** Is this document running as an installed app rather than in a browser tab? */
export function standalone(): boolean {
  if (typeof window === "undefined") return false;
  // Two checks because two platforms answer differently. `display-mode: standalone` is the
  // standard one; `navigator.standalone` is Safari's, which is the platform that does not
  // implement the standard one for this purpose.
  const media = window.matchMedia?.("(display-mode: standalone)")?.matches ?? false;
  const ios = (window.navigator as { standalone?: boolean }).standalone === true;
  return media || ios;
}

/** iOS - where there is no install prompt and never will be, so instructions are the feature. */
export function isIOS(): boolean {
  if (typeof navigator === "undefined") return false;
  const ua = navigator.userAgent;
  // iPadOS 13+ reports itself as a Mac. The touch-point check is the usual way to tell a real Mac
  // from an iPad lying about being one, and a Mac cannot install this anyway, so a false positive
  // here costs a set of instructions nobody needed rather than a broken button.
  return /iPad|iPhone|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
}

export function installability(): Install {
  if (standalone()) return { state: "installed" };
  if (prompt) return { state: "ready" };
  if (isIOS()) return { state: "manual" };
  return { state: "unknown" };
}

/**
 * Show the browser's install prompt. Returns what the person chose.
 *
 * The saved event is single-use: once `prompt()` has been called the browser will not accept it
 * again, so it is cleared whatever the answer. A dismissed prompt is not a failure and is not
 * reported as one - Chrome will offer another one later on its own terms.
 */
export async function install(): Promise<"accepted" | "dismissed" | "unavailable"> {
  const saved = prompt;
  if (!saved) return "unavailable";
  prompt = null;
  announce();
  try {
    await saved.prompt();
    const { outcome } = await saved.userChoice;
    return outcome;
  } catch {
    return "dismissed";
  }
}

export interface Registered {
  /** A new version has installed and is waiting for the page to let it take over. */
  onUpdate(ready: () => void): void;
  /** Activate the waiting worker and reload. Safe to call when nothing is waiting. */
  update(): void;
}

/**
 * Register the shell, if this is a production build in a browser that has service workers.
 *
 * Returns the handle synchronously so the caller does not have to await a registration whose only
 * purpose is a side effect; callbacks fire later if and when there is something to say.
 */
export function registerShell(): Registered {
  let waiting: ServiceWorker | null = null;
  let notify: (() => void) | null = null;
  let refreshing = false;

  const handle: Registered = {
    onUpdate(ready) {
      notify = ready;
      if (waiting) ready();
    },
    update() {
      if (!waiting) return;
      waiting.postMessage({ type: "skip-waiting" });
    },
  };

  if (typeof navigator === "undefined" || !("serviceWorker" in navigator)) return handle;
  // `import.meta.env.PROD` is replaced at build time, so the whole registration is removed from the
  // development bundle rather than skipped at runtime.
  if (!import.meta.env.PROD) return handle;

  const found = (worker: ServiceWorker | null) => {
    if (!worker) return;
    waiting = worker;
    notify?.();
  };

  navigator.serviceWorker.addEventListener("controllerchange", () => {
    // Guarded: `controllerchange` can fire more than once, and an unguarded reload in here is a
    // reload loop with no way out for the person in it.
    if (refreshing) return;
    refreshing = true;
    window.location.reload();
  });

  /**
   * After `load`, or immediately if that has already happened.
   *
   * The `readyState` half is not defensive coding, it is the fix for a bug the browser pass found:
   * a bare `addEventListener("load", ...)` **never fired**, and the worker was never registered in
   * the built app at all while registering it by hand from the console worked perfectly.
   *
   * `createRoot().render()` does not render synchronously - React 18 schedules the initial render
   * through the scheduler, so `registerShell()` is first called from a task that can run *after*
   * the document has finished loading. Subscribing to an event that has already happened is
   * subscribing to nothing, silently and forever.
   *
   * Waiting for `load` at all is still right: registration competes with the page's own resources
   * for the connection, and on a first visit those are the ones a person is looking at.
   */
  const start = () => {
    void navigator.serviceWorker
      .register("sw.js", { scope: "./" })
      .then((registration) => {
        // Already waiting when we registered: a second tab installed it, or this tab was reloaded
        // between the install and the takeover.
        found(registration.waiting);
        registration.addEventListener("updatefound", () => {
          const installing = registration.installing;
          if (!installing) return;
          installing.addEventListener("statechange", () => {
            // `installed` with a controller present means an *update*. Without a controller it is
            // the first install, which needs no announcement - there is no older version running.
            if (installing.state === "installed" && navigator.serviceWorker.controller) {
              found(installing);
            }
          });
        });
      })
      .catch(() => {
        // A refused registration - an insecure origin, or a worker the browser would not parse -
        // leaves an app that works and does not work offline. Nothing a person can act on, so
        // nothing is said.
      });
  };

  if (document.readyState === "complete") start();
  else window.addEventListener("load", start, { once: true });

  return handle;
}

if (typeof window !== "undefined") {
  window.addEventListener("beforeinstallprompt", (event) => {
    // Chrome shows its own mini-infobar unless this is prevented, and then the app has two install
    // affordances that disagree about where they are.
    event.preventDefault();
    prompt = event as BeforeInstallPromptEvent;
    announce();
  });
  window.addEventListener("appinstalled", () => {
    prompt = null;
    announce();
  });
}
