import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

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

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Bound to every interface on purpose: `getUserMedia` needs a secure origin, and the way to
    // test the camera on a real phone is to reach this dev server from that phone over the LAN.
    // `localhost` is a secure origin and a LAN IP is not, so a phone needs `--host` plus either
    // HTTPS or Chrome's "treat as secure origin" flag. Recorded because it is the single most
    // confusing part of developing a camera app.
    host: true,
    proxy: Object.fromEntries(
      ROUTES.map((route) => [route, { target: BACKEND, changeOrigin: false }]),
    ),
  },
  build: {
    // Hashed filenames, because 16.3.1's service worker caches by URL: an asset that keeps its
    // name across releases is an asset a stale cache serves forever.
    assetsDir: "assets",
    sourcemap: true,
    target: "es2020",
  },
});
