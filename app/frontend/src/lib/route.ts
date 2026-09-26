/**
 * Phase 16.2.1 - forty lines of hash routing, and why not a router.
 *
 * Three views in a line - capture, progress, result - plus one address worth being an address:
 * `#/r/<id>`. That id is the whole point of 16.1.1's store, which keeps a prediction on disk
 * precisely so it outlives the process; a result that could not be reopened after a reload would
 * make that store pointless from the client's side.
 *
 * `react-router` would be 12 kB gzipped to express that, in an app whose offline shell (16.3.1) has
 * to be small enough to be worth caching. The hash rather than the History API for the same reason
 * it is usually the wrong choice and here is the right one: a hash needs **no server rewrite**, so
 * the same bundle works served by the backend, by `vite preview`, from a `file://` URL and from a
 * service worker with no configuration in any of them.
 *
 * Unknown routes fall to capture rather than to a 404 screen, because there is nothing a person can
 * do with a 404 in a three-screen app but start again - which is what capture is.
 */

export type Route =
  | { view: "capture" }
  | { view: "camera" }
  | { view: "progress" }
  | { view: "result"; id: string }
  | { view: "gallery" }
  | { view: "about" };

const ID = /^[0-9a-f]{1,64}$/;

export function parse(hash: string): Route {
  const path = hash.replace(/^#\/?/, "").split("?")[0];
  if (path === "camera") return { view: "camera" };
  if (path === "progress") return { view: "progress" };
  if (path === "gallery") return { view: "gallery" };
  if (path === "about") return { view: "about" };
  const result = /^r\/([^/]+)$/.exec(path);
  // The id is shaped-checked here as well as at the store. A malformed one in the address bar
  // should land on capture rather than issue a request that can only 404.
  if (result && ID.test(result[1])) return { view: "result", id: result[1] };
  return { view: "capture" };
}

export function href(route: Route): string {
  switch (route.view) {
    case "camera":
      return "#/camera";
    case "progress":
      return "#/progress";
    case "gallery":
      return "#/gallery";
    case "about":
      return "#/about";
    case "result":
      return `#/r/${route.id}`;
    default:
      return "#/";
  }
}

export function go(route: Route): void {
  const next = href(route);
  if (window.location.hash === next) return;
  window.location.hash = next;
}

/**
 * Replace rather than push. Used when a capture finishes: the progress view is not somewhere a
 * person wants "back" to return to, because going back to a finished run would show a dead stage
 * list with nothing running.
 */
export function replace(route: Route): void {
  const url = new URL(window.location.href);
  url.hash = href(route);
  window.history.replaceState(null, "", url.toString());
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}

export function watch(onChange: (route: Route) => void): () => void {
  const handler = () => onChange(parse(window.location.hash));
  window.addEventListener("hashchange", handler);
  return () => window.removeEventListener("hashchange", handler);
}
