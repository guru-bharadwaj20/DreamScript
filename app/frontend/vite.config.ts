import { createHash } from "node:crypto";
import { readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { join, relative, sep } from "node:path";

import react from "@vitejs/plugin-react";
import { type Plugin, defineConfig } from "vite";

/**
 * Phase 16.2.1 - the build.
 *
 * ## Why there is a dev proxy and no CORS middleware anywhere
 *
 * The client calls the backend with **same-origin paths** (`/predict`, `/ir/:id`). In production
 * the backend serves the built bundle, so same-origin is literally true. In development Vite
 * serves the bundle on 5173 and proxies those paths to the backend on 3000, so same-origin is
 * true from the browser's point of view as well.
 *
 * That is the whole reason `app/backend` has no CORS middleware and needs none. The alternative -
 * the client calling `http://localhost:3000` directly - means a preflight on every upload, an
 * `Access-Control-Allow-Origin` that has to be right in three environments, and a class of bug
 * that appears only in a browser. A proxy in the one place that already knows both addresses is
 * cheaper and has no production surface at all.
 *
 * ## The proxy list is explicit
 *
 * Not `/api/*`, because the backend's routes are not under a prefix and inventing one here would
 * mean rewriting paths - a rewrite is a second description of the API that can drift from the
 * first. Each route the client uses is listed, so adding one is a visible edit rather than a
 * silent match.
 *
 * **That visibility is the whole cost of the choice, and it was paid once already.** 16.2.8 added
 * `POST /correct/{id}` to the backend and to the client and not to this list, so the sheet
 * submitted a correction into a **404** that no test caught - the route existed on the server and
 * the client called it correctly; only the development proxy did not know. `tests/
 * test_app_frontend.py` compares this list against the paths `api.ts` fetches now.
 */
const BACKEND = process.env.DREAMSCRIPT_BACKEND ?? "http://127.0.0.1:3000";

const ROUTES = [
  "/health",
  "/predict",
  "/ir",
  "/code",
  "/run",
  "/correct",
  "/feedback",
  "/openapi.json",
];


/**
 * Phase 16.3.1 - fill `sw.js` in with the build it is the shell for.
 *
 * The service worker is a committed, readable file under `public/`. What it cannot know when it is
 * written is the **hashed names** of the assets it has to precache, because those are decided by
 * this build - and an offline shell that precaches `index.js` when the bundle emits
 * `index-CWrBmrk3.js` is an offline shell that caches nothing.
 *
 * So the list is taken from `dist/` after it exists, which is the one description that cannot be
 * wrong: it is what was actually shipped. Two exclusions, both deliberate. Source maps, which are
 * 700 kB and are for a developer with a network. And `sw.js` itself, because a worker that
 * precaches itself serves its own previous version out of its own cache, which is a worker that
 * can never be replaced.
 *
 * `BUILD` is a hash of those files' **names and contents**, so the cache name changes exactly when
 * the shipped bytes do and not on every `npm run build` of unchanged source. A cache keyed on a
 * timestamp would evict and refetch the whole shell on a rebuild that changed nothing.
 *
 * It hashes the contents and not only the names because of a defect the update harness caught: with
 * names alone, a build that changed `index.html` and nothing else produced a **byte-identical**
 * `sw.js` - Vite's asset names are content hashes, but `index.html` keeps its name - and a
 * byte-identical worker is not an update as far as the browser is concerned. The new generation
 * would never have activated. Thirteen files and 340 kB is nothing to hash once per build.
 */
function offlineShell(): Plugin {
  const SKIP = new Set(["sw.js"]);
  return {
    name: "dreamscript-offline-shell",
    apply: "build",
    closeBundle() {
      const out = "dist";
      const files: string[] = [];
      const walk = (dir: string) => {
        for (const entry of readdirSync(dir)) {
          const full = join(dir, entry);
          if (statSync(full).isDirectory()) walk(full);
          else files.push(relative(out, full).split(sep).join("/"));
        }
      };
      walk(out);

      const precache = files
        .filter((file) => !file.endsWith(".map") && !SKIP.has(file))
        .sort();
      const build = createHash("sha256")
        .update(
          precache
            .map((file) => `${file}:${createHash("sha256").update(readFileSync(join(out, file))).digest("hex")}`)
            .join("|"),
        )
        .digest("hex")
        .slice(0, 12);

      const path = join(out, "sw.js");
      const source = readFileSync(path, "utf8");
      const filled = source
        .replace('const BUILD = "dev";', `const BUILD = ${JSON.stringify(build)};`)
        .replace('const PRECACHE = ["index.html"];', `const PRECACHE = ${JSON.stringify(precache)};`);
      if (filled === source) {
        // A silent no-op here ships a worker that caches one file. Better to fail the build.
        throw new Error("sw.js did not contain the BUILD/PRECACHE placeholders");
      }
      writeFileSync(path, filled);
      const bytes = precache.reduce((total, file) => total + statSync(join(out, file)).size, 0);
      this.info(`offline shell: ${precache.length} files, ${(bytes / 1024).toFixed(1)} kB, build ${build}`);
    },
  };
}

function proxy() {
  return Object.fromEntries(
    ROUTES.map((route) => [route, { target: BACKEND, changeOrigin: false }]),
  );
}

export default defineConfig({
  plugins: [react(), offlineShell()],
  server: {
    port: 5173,
    // Bound to every interface on purpose: `getUserMedia` needs a secure origin, and the way to
    // test the camera on a real phone is to reach this dev server from that phone over the LAN.
    // `localhost` is a secure origin and a LAN IP is not, so a phone needs `--host` plus either
    // HTTPS or Chrome's "treat as secure origin" flag. Recorded because it is the single most
    // confusing part of developing a camera app.
    host: true,
    proxy: proxy(),
  },
  // `vite preview` serves `dist/` - the real bundle, the real service worker, the real hashed
  // asset names - and 16.3.1 can only be tested there, because none of those exist in development.
  // It needs the same proxy for the same reason, and gets it from the same list.
  preview: {
    port: 4173,
    host: true,
    proxy: proxy(),
  },
  build: {
    // Hashed filenames, because 16.3.1's service worker caches by URL: an asset that keeps its
    // name across releases is an asset a stale cache serves forever.
    assetsDir: "assets",
    sourcemap: true,
    target: "es2020",
  },
});
