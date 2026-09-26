import { defineConfig } from "vitest/config";

/**
 * Phase 16.2.2 / 16.2.3 - the unit suite.
 *
 * `environment: "node"`, deliberately. Nothing under test touches the DOM: the SSE reader is a byte
 * stream and the dewarp is arithmetic over a `Uint8ClampedArray`. The one DOM *type* either needs
 * is `ImageData`, and `vitest.setup.ts` defines it in ten lines rather than installing jsdom to
 * obtain a buffer and two numbers.
 */
export default defineConfig({
  test: {
    environment: "node",
    setupFiles: ["./vitest.setup.ts"],
    include: ["src/**/*.test.ts"],
  },
});
