/**
 * Phase 16.3.1 - the three questions the install card asks, and why each is easy to get wrong.
 *
 * Nothing here registers a worker: that needs a browser, a secure origin and a built bundle, and it
 * is verified in one against `vite preview`. What is testable without any of that is the
 * *classification* - which of four situations a given browser is in - and that is where the
 * platform-specific traps live.
 */

import { afterEach, describe, expect, it } from "vitest";

import { installability, isIOS, standalone } from "./pwa";

interface Fake {
  userAgent?: string;
  maxTouchPoints?: number;
  standalone?: boolean;
}

const saved = {
  navigator: Object.getOwnPropertyDescriptor(globalThis, "navigator"),
  window: Object.getOwnPropertyDescriptor(globalThis, "window"),
};

function as(fake: Fake, displayMode = false): void {
  const nav = { userAgent: "", maxTouchPoints: 0, ...fake };
  Object.defineProperty(globalThis, "navigator", {
    value: nav,
    configurable: true,
    writable: true,
  });
  Object.defineProperty(globalThis, "window", {
    value: {
      navigator: nav,
      matchMedia: (query: string) => ({ matches: displayMode && query.includes("standalone") }),
    },
    configurable: true,
    writable: true,
  });
}

afterEach(() => {
  for (const [key, descriptor] of Object.entries(saved)) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete (globalThis as Record<string, unknown>)[key];
  }
});

const IPHONE =
  "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1";
const IPAD =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15";
const PIXEL =
  "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36";
const MAC =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36";

describe("isIOS", () => {
  it("recognises an iPhone", () => {
    as({ userAgent: IPHONE });
    expect(isIOS()).toBe(true);
  });

  it("recognises an iPad that is claiming to be a Mac", () => {
    // iPadOS 13 and later send a desktop Safari string. Without the touch-point check the one
    // platform that needs the Add to Home Screen instructions is the one platform that would not
    // be shown them.
    as({ userAgent: IPAD, maxTouchPoints: 5 });
    expect(isIOS()).toBe(true);
  });

  it("does not mistake a real Mac for an iPad", () => {
    as({ userAgent: MAC, maxTouchPoints: 0 });
    expect(isIOS()).toBe(false);
  });

  it("is false on Android", () => {
    as({ userAgent: PIXEL });
    expect(isIOS()).toBe(false);
  });
});

describe("standalone", () => {
  it("is true when the display mode says so", () => {
    as({ userAgent: PIXEL }, true);
    expect(standalone()).toBe(true);
  });

  it("is true for Safari's own flag, which is the only one iOS sets", () => {
    // `display-mode: standalone` does not answer this on iOS. Checking only the standard one would
    // show an installed iPhone the "put it on your home screen" card it is already past.
    as({ userAgent: IPHONE, standalone: true }, false);
    expect(standalone()).toBe(true);
  });

  it("is false in an ordinary tab", () => {
    as({ userAgent: PIXEL }, false);
    expect(standalone()).toBe(false);
  });
});

describe("installability", () => {
  it("reports an installed app before anything else", () => {
    as({ userAgent: PIXEL }, true);
    expect(installability()).toEqual({ state: "installed" });
  });

  it("offers the manual gesture on iOS, where no prompt event exists", () => {
    as({ userAgent: IPHONE });
    expect(installability()).toEqual({ state: "manual" });
  });

  it("says nothing rather than showing a button that cannot work", () => {
    // No saved `beforeinstallprompt` means `prompt()` would throw. A button here would be a
    // control that does nothing, which is worse than no control.
    as({ userAgent: PIXEL });
    expect(installability()).toEqual({ state: "unknown" });
  });
});
