/**
 * Phase 16.2.1 - the shell: one bar, one route, one theme.
 *
 * The screens arrive in 16.2.2 onward. What is here is the frame they hang in, and the two pieces of
 * state that are genuinely global: which view is showing, and which theme.
 *
 * **No state management library, and no context for the prediction.** The flow is linear - capture,
 * progress, result - and a capture produces exactly one id, which the address bar already holds.
 * `#/r/<id>` is the app's state, so the state survives a reload, a share and a cold start; a store
 * in memory would not, and would have to be rebuilt from the id anyway.
 */

import { useCallback, useEffect, useState } from "react";

import { definitelyOffline, watchOnline } from "./lib/offline";
import { registerShell } from "./lib/pwa";
import { type Route, parse, watch } from "./lib/route";
import {
  type Theme,
  applyTheme,
  nextTheme,
  readTheme,
  resolvedTheme,
  syncBrowserChrome,
  watchSystemTheme,
} from "./lib/theme";
import { About } from "./screens/About";
import { Camera } from "./screens/Camera";
import { Capture } from "./screens/Capture";
import { Gallery } from "./screens/Gallery";
import { Privacy } from "./screens/Privacy";
import { Result } from "./screens/Result";
import { Button, Mark } from "./ui";
import "./styles/tokens.css";
import "./styles/base.css";
import "./App.css";

export default function App() {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));
  const [theme, setTheme] = useState<Theme>(readTheme);
  /**
   * A photograph chosen from the library (16.2.4) or an example (16.2.11), on its way to the
   * pipeline.
   *
   * Held here rather than in the address bar because a `File` cannot be one. That is a real
   * limitation and it is accepted rather than worked around: reloading `#/camera` with a staged
   * file loses the staging and opens the camera, which is the correct thing to do with a reload -
   * the alternative is stashing megabytes in IndexedDB to survive a gesture nobody made.
   */
  const [staged, setStaged] = useState<Blob | null>(null);
  /**
   * 16.2.10: whether the operating system says there is no route.
   *
   * Global because it changes what *every* screen can honestly offer, not because it is
   * convenient - this client holds no model, so with no network there is no product, only the five
   * stored answers. A strip at the top is the one place that fact belongs: the alternative is each
   * screen discovering it separately by watching a request fail, which is a spinner first and an
   * explanation second.
   *
   * Read in the `false` direction only. See `lib/offline.ts` - `navigator.onLine === true` says an
   * interface is up and nothing more.
   */
  const [offline, setOffline] = useState(definitelyOffline);
  /**
   * 16.3.1: a newer shell has installed and is waiting for this page to stand aside.
   *
   * Offered rather than taken. `sw.js` deliberately does not `skipWaiting` on its own, because a
   * worker that takes over immediately swaps the asset cache under a page that is already running -
   * and the running page then requests a chunk from a build that has just been deleted, which is a
   * 404 from its own origin on a screen that was working a second ago.
   */
  const [update, setUpdate] = useState(false);

  useEffect(() => watch(setRoute), []);
  useEffect(() => watchOnline((online) => setOffline(!online)), []);

  // Registered once, for the life of the document. The handle is kept in a ref-like closure rather
  // than in state because nothing renders from it - only from the boolean it sets.
  const [shell] = useState(registerShell);
  useEffect(() => shell.onUpdate(() => setUpdate(true)), [shell]);

  useEffect(() => {
    // Cleared on the way out. Without this, tapping the camera later re-sends the photograph that
    // was chosen ten minutes ago instead of opening the viewfinder.
    if (route.view !== "camera" && route.view !== "gallery") setStaged(null);
  }, [route.view]);

  useEffect(() => {
    applyTheme(theme);
    // After the attribute lands, so the computed `--ground` is the new one. Ordering matters here:
    // reading it first paints the status bar the colour the app just stopped being.
    syncBrowserChrome();
  }, [theme]);

  // The OS changing its mind only matters while the theme is "system", but the listener is cheap and
  // unconditional - a conditional subscription is one more thing to get wrong on a theme change.
  useEffect(() => watchSystemTheme(syncBrowserChrome), []);

  const cycleTheme = useCallback(() => setTheme((current) => nextTheme(current)), []);

  // The viewfinder is full-bleed and draws its own chrome over the preview. A sticky app bar above
  // it would take 52px off the picture and put a wordmark where the paper is.
  const bare = route.view === "camera";

  return (
    <>
      {/* Above the bar, and drawn in the camera view too - a capture that cannot be uploaded is
          exactly where someone needs to be told, and that screen has no bar to hang it under. */}
      {offline ? <OfflineStrip inCamera={bare} /> : null}
      {update ? <UpdateStrip onUpdate={() => shell.update()} /> : null}
      {bare ? null : (
      <header className="bar safe-top safe-x">
        <a href="#/" className="row" style={{ textDecoration: "none", gap: "var(--sp-2)" }}>
          <Mark />
        </a>
        <span className="grow" />
        <Button
          variant="quiet"
          icon
          onClick={cycleTheme}
          aria-label={`Theme: ${theme}. Tap to change.`}
          title={`Theme: ${theme} (showing ${resolvedTheme(theme)})`}
        >
          <ThemeGlyph theme={theme} />
        </Button>
        <a href="#/about" className="btn btn-quiet btn-icon" aria-label="About DreamScript">
          <Glyph d="M12 17v-6m0-4h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
        </a>
      </header>
      )}

      <main className="screen">
        {route.view === "result" ? (
          <Result id={route.id} />
        ) : route.view === "camera" ? (
          // `key` on the staging, so choosing a second photograph remounts rather than reusing a
          // component whose effects have already run for the first one.
          <Camera key={staged ? "staged" : "live"} initial={staged} />
        ) : route.view === "gallery" ? (
          <Gallery onStaged={setStaged} />
        ) : route.view === "privacy" ? (
          <Privacy />
        ) : route.view === "about" ? (
          <About theme={theme} onTheme={setTheme} />
        ) : (
          <Capture onStaged={setStaged} />
        )}
      </main>
    </>
  );
}

/**
 * Phase 16.2.10 - one line, the moment the network goes.
 *
 * It says what is gone and what is left, in that order, and the second half is a link rather than a
 * consolation: with no network the five stored examples are the entire remaining app, so the strip
 * that reports the loss is also the way to the only thing still working.
 *
 * Orange - `--degraded` - and not rose. The app has not stopped and nothing failed; it is running on
 * a lower rung, which is precisely what that colour means everywhere else in this client. Using the
 * `stopped` rose here would make "no wifi" look like "the pipeline died".
 */
function OfflineStrip({ inCamera }: { inCamera: boolean }) {
  return (
    <div className={`offline-strip${inCamera ? " is-over" : ""} safe-top safe-x`} role="status">
      <Glyph d="M2 2l20 20M8.5 16.4a5 5 0 0 1 7 0M5 13a9 9 0 0 1 3.2-2.1m7.6.1A9 9 0 0 1 19 13M2 8.8A15 15 0 0 1 8 5.4m8 .1a15 15 0 0 1 6 3.3M12 20h.01" size={16} />
      <span className="grow">
        <strong>No network.</strong> Nothing can be read while this is showing — this app holds no
        models of its own.
      </span>
      <a href="#/gallery" className="offline-strip-go">
        Stored examples
      </a>
    </div>
  );
}

/**
 * Phase 16.3.1 - a newer version is cached and one tap takes it.
 *
 * Gold rather than orange, because this is not a degradation and nothing is wrong: the four
 * trust colours mean something specific in this app and spending one of them on "there is an
 * update" would blunt all four. Gold is the brand, and an update is the app talking about itself.
 *
 * It does not reload on its own. An app that reloads underneath someone is an app that throws away
 * the correction they were half way through typing.
 */
function UpdateStrip({ onUpdate }: { onUpdate: () => void }) {
  return (
    <div className="update-strip safe-x" role="status">
      <Glyph d="M4 12a8 8 0 0 1 13.7-5.6L20 8m0-4v4h-4m4 4a8 8 0 0 1-13.7 5.6L4 16m0 4v-4h4" size={16} />
      <span className="grow">
        <strong>A newer version is ready.</strong> It will load when you reload.
      </span>
      <button className="update-strip-go" onClick={onUpdate}>
        Reload
      </button>
    </div>
  );
}

/**
 * Three states need three glyphs, not a sun that toggles.
 *
 * A single toggling icon cannot show that the app is following the phone - which is the default and
 * the one most people will be in. Sun, moon, and a half-filled circle for system.
 */
function ThemeGlyph({ theme }: { theme: Theme }) {
  if (theme === "light") {
    return <Glyph d="M12 3v2m0 14v2M3 12h2m14 0h2M5.6 5.6l1.4 1.4m10 10 1.4 1.4m0-12.8-1.4 1.4m-10 10-1.4 1.4M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z" />;
  }
  if (theme === "dark") {
    return <Glyph d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z" />;
  }
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="8.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M12 3.5a8.5 8.5 0 0 0 0 17Z" fill="currentColor" />
    </svg>
  );
}

export function Glyph({ d, size = 20 }: { d: string; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={d} />
    </svg>
  );
}
