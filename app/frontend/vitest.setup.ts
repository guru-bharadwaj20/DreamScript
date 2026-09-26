/**
 * Phase 16.2.3 - `ImageData` in node, in ten lines rather than in jsdom.
 *
 * `dewarp.ts` takes and returns `ImageData`, which is a DOM class; vitest's default environment is
 * node, where it does not exist. The two usual answers are `environment: "jsdom"` or
 * `"happy-dom"` - between 3 and 10 MB of dependency to obtain one structure that is, in full, a
 * `Uint8ClampedArray` and two numbers.
 *
 * So it is defined here. This is not a mock of `ImageData`: it is the same shape, the same
 * zero-filled buffer, and the same two constructor forms the specification defines, so a test that
 * passes against it passes against the browser's for the same reason. Nothing under test touches
 * any other DOM API - the detector is arithmetic over a byte array, which is exactly why it could
 * be written without OpenCV in the first place.
 *
 * `globalThis.ImageData` is only defined when it is absent, so running these tests in a browser
 * environment uses the browser's.
 */

class NodeImageData {
  readonly data: Uint8ClampedArray;
  readonly width: number;
  readonly height: number;
  readonly colorSpace = "srgb" as const;

  constructor(dataOrWidth: Uint8ClampedArray | number, widthOrHeight: number, maybeHeight?: number) {
    if (typeof dataOrWidth === "number") {
      this.width = dataOrWidth;
      this.height = widthOrHeight;
      this.data = new Uint8ClampedArray(this.width * this.height * 4);
    } else {
      this.data = dataOrWidth;
      this.width = widthOrHeight;
      this.height = maybeHeight ?? dataOrWidth.length / 4 / widthOrHeight;
    }
    if (!Number.isFinite(this.width) || !Number.isFinite(this.height) || this.width <= 0) {
      throw new RangeError("ImageData needs a positive width and height");
    }
  }
}

if (typeof globalThis.ImageData === "undefined") {
  (globalThis as unknown as { ImageData: unknown }).ImageData = NodeImageData;
}
