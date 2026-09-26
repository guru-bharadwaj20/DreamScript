/**
 * Phase 16.2.1 - light, dark, and the third state people forget.
 *
 * Three states, not two. `"system"` is the default and it is not the same as either explicit
 * choice: it follows the phone, which follows the time of day on both iOS and Android. A theme
 * toggle that only flips between light and dark has quietly overridden a setting the person already
 * made once, for every app on their device.
 *
 * The mechanism is one attribute. `data-theme="light"` or `"dark"` on `<html>`; **absent** for
 * system, because `tokens.css` reads `prefers-color-scheme` under `:root:not([data-theme="light"])`
 * and an attribute of `"system"` would satisfy that selector and pin the page to dark.
 *
 * `localStorage` can throw - a private window, blocked site data, or a WebView with storage
 * disabled - so every access is guarded and the failure is "you get system", which is the correct
 * default anyway.
 */

export type Theme = "system" | "light" | "dark";

const KEY = "dreamscript.theme";
const THEMES: Theme[] = ["system", "light", "dark"];

export function readTheme(): Theme {
  try {
    const held = localStorage.getItem(KEY);
    if (held && (THEMES as string[]).includes(held)) return held as Theme;
  } catch {
    // Blocked storage is not an error worth surfacing; it means the default.
  }
  return "system";
}

export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  if (theme === "system") {
    // Removed, not set to "system". See the note above: an attribute here defeats the media query.
    root.removeAttribute("data-theme");
  } else {
    root.setAttribute("data-theme", theme);
  }
  // Tells the browser which UA-painted surfaces to use - form controls, the scrollbar, and on iOS
  // the colour behind a rubber-band overscroll. Without it a dark page flashes white at the edges.
  root.style.colorScheme = theme === "system" ? "light dark" : theme;
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    // The theme still applies for this session; only the memory of it is lost.
  }
}

/** What the page is actually showing right now, which for `"system"` only the browser knows. */
export function resolvedTheme(theme: Theme): "light" | "dark" {
  if (theme !== "system") return theme;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function nextTheme(theme: Theme): Theme {
  return THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
}

/**
 * Keep the address bar and the status bar in step with the page.
 *
 * `theme-color` is what iOS Safari paints behind the status bar and what Android Chrome paints in
 * the address bar. A static one in the HTML is wrong half the time; without this the top of the
 * screen is white above a black app, which is the single most obvious way a web app announces
 * itself as a web page.
 */
export function syncBrowserChrome(): void {
  const colour = getComputedStyle(document.documentElement).getPropertyValue("--ground").trim();
  if (!colour) return;
  let tag = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
  if (!tag) {
    tag = document.createElement("meta");
    tag.name = "theme-color";
    document.head.appendChild(tag);
  }
  tag.content = colour;
}

/** Subscribe to the OS changing its mind. Returns an unsubscribe. */
export function watchSystemTheme(onChange: () => void): () => void {
  const query = window.matchMedia?.("(prefers-color-scheme: dark)");
  if (!query) return () => {};
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}
