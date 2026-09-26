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
import { Result } from "./screens/Result";
import { Button, Mark } from "./ui";
import "./styles/tokens.css";
import "./styles/base.css";
import "./App.css";

export default function App() {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));
  const [theme, setTheme] = useState<Theme>(readTheme);

  useEffect(() => watch(setRoute), []);

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
          <Camera />
        ) : route.view === "about" ? (
          <About theme={theme} onTheme={setTheme} />
        ) : (
          <Capture />
        )}
      </main>
    </>
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
