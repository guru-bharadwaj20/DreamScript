/**
 * Phase 16.3.1 - the offline shell, in one file nobody has to take on trust.
 *
 * ## Why this is written out rather than generated
 *
 * Workbox is the obvious answer and it is the wrong size for the question. It is a build-time
 * plugin plus a runtime bundle, and what it would be configured to express here fits in a page: one
 * precache of a shipped build, one rule for documents, one rule for hashed assets, and a list of
 * paths that must **never** be cached. A service worker is also the one piece of a web app that can
 * make a site permanently broken for a person who already visited it - a bad cache outlives the
 * deploy that caused it - so it is the last place to want a dependency whose behaviour has to be
 * inferred from its documentation.
 *
 * ## The API is never cached, and that is the most important line in this file
 *
 * `/predict`, `/ir`, `/code`, `/run`, `/correct`, `/feedback` and `/health` go to the network and
 * nowhere else. A cached `GET /predict/{id}` would serve the reading from before a correction
 * somebody had just made, silently, which is exactly the failure 16.2.10 refused when it put the
 * offline cache *after* the request instead of in front of it. A service worker that cached them
 * would reintroduce it one layer lower, where no screen could label it.
 *
 * So there is no fallback for those either. With no network `/predict/{id}` fails, `Result` catches
 * it, and 16.2.10's stored answers appear **with a banner that says they are stored**. A shell that
 * quietly answered from a cache would take that banner away.
 *
 * ## Two rules for two kinds of thing
 *
 *   the document    network first, cache second. `index.html` names hashed assets, and a stale one
 *                   names assets that a later deploy has deleted - a blank page for anyone whose
 *                   cache is a version behind. Network-first means the stale copy is only ever used
 *                   when there is no network, and then it is correct by construction, because the
 *                   assets it names are in the same cache generation
 *   everything else cache first. They are content-hashed: `index-CWrBmrk3.js` is that file forever,
 *                   so a cache hit is not a stale answer, it is the same answer without a request
 *
 * ## It does not `skipWaiting` on its own
 *
 * A new worker that takes over immediately swaps the asset cache under a page that is already
 * running, and a React app that then lazy-loads a chunk from the previous build gets a 404 from its
 * own origin. This one waits, the page notices it waiting and offers a reload, and the reload is
 * what activates it. `skipWaiting` happens only on a message from a page that is about to reload.
 */

/**
 * Filled in at build time by the `offline-shell` plugin in `vite.config.ts`.
 *
 * The default is the literal below, which is what `public/sw.js` serves in development - where this
 * worker is never registered, because a cached module graph is a whole class of bug that only
 * exists on the machine of the person who wrote the code.
 */
const BUILD = "dev";
const PRECACHE = ["index.html"];

const CACHE = `dreamscript-${BUILD}`;

/**
 * The paths that go to the network and are never stored.
 *
 * The same list `vite.config.ts` proxies, for the same reason it is a list and not a prefix: the
 * backend's routes are not under one, and inventing `/api/*` here would be a second description of
 * the API that can drift from the first. `tests/test_app_frontend.py` compares the three.
 */
const NEVER_CACHE = [
  "/health",
  "/predict",
  "/ir",
  "/code",
  "/run",
  "/correct",
  "/feedback",
  "/openapi.json",
];

function isApi(url) {
  return NEVER_CACHE.some((route) => url.pathname === route || url.pathname.startsWith(`${route}/`));
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      // One at a time rather than `addAll`, which rejects the whole install if any single request
      // fails. A shell that refuses to install because one example image 404'd is a shell that is
      // not there at all, which is worse than one missing a thumbnail.
      await Promise.all(
        PRECACHE.map(async (path) => {
          try {
            const response = await fetch(new Request(path, { cache: "reload" }));
            if (response.ok) await cache.put(path, response);
          } catch {
            // Recorded by its absence: the fetch handler will go to the network for it later.
          }
        }),
      );
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      // Every generation but this one. A cache keyed by build id and never swept grows by the size
      // of the bundle on every deploy, in storage the person did not agree to give up.
      const names = await caches.keys();
      await Promise.all(
        names.filter((name) => name.startsWith("dreamscript-") && name !== CACHE).map((name) => caches.delete(name)),
      );
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("message", (event) => {
  // Sent by `lib/pwa.ts` immediately before it reloads the page. See the note at the top.
  if (event.data && event.data.type === "skip-waiting") self.skipWaiting();
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  // Cross-origin is somebody else's to cache. Nothing here loads any, and a worker that started
  // storing third-party responses would be storing things it cannot reason about.
  if (url.origin !== self.location.origin) return;
  if (isApi(url)) return;

  if (request.mode === "navigate") {
    event.respondWith(
      (async () => {
        try {
          const fresh = await fetch(request);
          const cache = await caches.open(CACHE);
          await cache.put("index.html", fresh.clone());
          return fresh;
        } catch {
          const cached = await caches.match("index.html");
          // A navigation with nothing cached is a first visit with no network. There is no useful
          // page to invent, so the failure is a real one rather than a fabricated error page.
          return cached ?? Response.error();
        }
      })(),
    );
    return;
  }

  event.respondWith(
    (async () => {
      const cached = await caches.match(request, { ignoreSearch: false });
      if (cached) return cached;
      try {
        const fresh = await fetch(request);
        // Only successful, basic (same-origin, non-opaque) responses. An opaque response has a
        // status of 0 and a body this worker cannot read, so caching one caches an unknown.
        if (fresh.ok && fresh.type === "basic") {
          const cache = await caches.open(CACHE);
          await cache.put(request, fresh.clone());
        }
        return fresh;
      } catch (failure) {
        throw failure;
      }
    })(),
  );
});
