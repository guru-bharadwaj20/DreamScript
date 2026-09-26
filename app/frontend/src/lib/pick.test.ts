/**
 * Phase 16.2.4 - what a chosen file has to survive.
 *
 * `accept="image/*"` is a **hint**, not a gate: a picker can hand back a `.heic`, a PDF named
 * `.jpg`, or on some Android file managers anything at all. So these are real checks, and the
 * ordering is part of what is tested - the free ones must run before the expensive one, because a
 * 40 MB burst shot should be refused without being decoded.
 *
 * HEIC gets more tests than anything else here for a reason. Every photograph on a recent iPhone is
 * HEIC unless the camera was set otherwise, no browser decodes it - Safari included, on the
 * operating system that produced it - and the failure is a `createImageBitmap` rejection with
 * nothing useful in it. Detected by name *and* by its `ftyp` brand, because a share sheet can hand
 * over `image.jpg` containing HEIC.
 */

import { describe, expect, it, vi } from "vitest";

import { ACCEPTED, MAX_BYTES, accept, explainPick, isHeic, isPicked } from "./pick";

/** A `File` with the given bytes, since node has no picker. */
function file(name: string, bytes: number[] | Uint8Array, size?: number): File {
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const blob = new Blob([data]);
  const made = new File([blob], name);
  if (size !== undefined) Object.defineProperty(made, "size", { value: size });
  return made;
}

/** ISO base media header with a given brand at offset 8. */
function ftyp(brand: string, name = "photo.jpg"): File {
  const head = new Uint8Array(16);
  head.set([0, 0, 0, 0x18], 0);
  head.set([..."ftyp"].map((c) => c.charCodeAt(0)), 4);
  head.set([...brand].map((c) => c.charCodeAt(0)), 8);
  return file(name, head);
}

const PNG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3, 4];

function withDecoder(result: { width: number; height: number } | Error) {
  const close = vi.fn();
  globalThis.createImageBitmap = vi.fn(async () => {
    if (result instanceof Error) throw result;
    return { ...result, close } as unknown as ImageBitmap;
  }) as typeof createImageBitmap;
  return { close };
}

describe("accept", () => {
  it("takes a real image and reports its size", async () => {
    withDecoder({ width: 1920, height: 1440 });
    const out = await accept(file("page.png", PNG));
    expect(isPicked(out)).toBe(true);
    if (isPicked(out)) {
      expect(out.width).toBe(1920);
      expect(out.height).toBe(1440);
    }
  });

  it("closes the bitmap once the dimensions are read", async () => {
    // A 12-megapixel photograph is 48 MB of decoded pixels. On a phone that is worth releasing.
    const { close } = withDecoder({ width: 4032, height: 3024 });
    await accept(file("page.jpg", PNG));
    expect(close).toHaveBeenCalledOnce();
  });

  it("refuses an empty file", async () => {
    withDecoder({ width: 10, height: 10 });
    const out = await accept(file("page.png", [], 0));
    expect(out).toEqual({ kind: "empty" });
  });

  it("refuses an oversize file without decoding it", async () => {
    // The order matters: a 40 MB burst shot must not be decoded to find out it is too big.
    const decoder = withDecoder({ width: 10, height: 10 });
    void decoder;
    const out = await accept(file("page.jpg", PNG, MAX_BYTES + 1));
    expect(out).toEqual({ kind: "size", bytes: MAX_BYTES + 1 });
    expect(globalThis.createImageBitmap).not.toHaveBeenCalled();
  });

  it("refuses a suffix the backend would refuse anyway, before any upload", async () => {
    withDecoder({ width: 10, height: 10 });
    const out = await accept(file("notes.pdf", PNG));
    expect(out).toEqual({ kind: "type", suffix: ".pdf" });
  });

  it("refuses a file with no suffix at all", async () => {
    withDecoder({ width: 10, height: 10 });
    const out = await accept(file("IMG0001", PNG));
    expect(out).toEqual({ kind: "type", suffix: "" });
  });

  it("reports a decode failure with what the browser said", async () => {
    const error = new Error("The source image could not be decoded.");
    error.name = "InvalidStateError";
    withDecoder(error);
    const out = await accept(file("page.png", PNG));
    expect(out).toMatchObject({ kind: "decode" });
    if (!isPicked(out) && out.kind === "decode") {
      expect(out.detail).toContain("InvalidStateError");
    }
  });

  it("treats a zero-dimension decode as a failure rather than a pass", async () => {
    withDecoder({ width: 0, height: 0 });
    const out = await accept(file("page.png", PNG));
    expect(out).toMatchObject({ kind: "decode" });
  });

  it("accepts every suffix the backend does", async () => {
    withDecoder({ width: 100, height: 100 });
    for (const suffix of ACCEPTED) {
      expect(isPicked(await accept(file(`page${suffix}`, PNG))), suffix).toBe(true);
    }
  });
});

describe("HEIC", () => {
  it("is caught by its suffix", async () => {
    expect(await isHeic(file("IMG_0001.HEIC", PNG))).toBe(true);
    expect(await isHeic(file("IMG_0001.heif", PNG))).toBe(true);
  });

  it("is caught by its brand when the name lies", async () => {
    // A share sheet can hand over `image.jpg` containing HEIC, which is the case that would
    // otherwise reach `createImageBitmap` and fail with nothing useful in the error.
    for (const brand of ["heic", "heix", "mif1", "msf1"]) {
      expect(await isHeic(ftyp(brand)), brand).toBe(true);
    }
  });

  it("does not mistake an MP4 or a plain JPEG for HEIC", async () => {
    expect(await isHeic(ftyp("isom", "clip.mp4"))).toBe(false);
    expect(await isHeic(file("page.jpg", [0xff, 0xd8, 0xff, 0xe0, 0, 16, 0x4a, 0x46]))).toBe(false);
  });

  it("is refused before the decoder is asked", async () => {
    withDecoder({ width: 10, height: 10 });
    const out = await accept(ftyp("heic"));
    expect(out).toEqual({ kind: "heic" });
    expect(globalThis.createImageBitmap).not.toHaveBeenCalled();
  });

  it("gets a sentence that names the setting to change", async () => {
    const { title, detail } = explainPick({ kind: "heic" });
    expect(title).toContain("HEIC");
    expect(detail).toContain("Most Compatible");
    expect(detail).toContain("Safari");
  });
});

describe("explainPick", () => {
  it("gives every refusal its own words", () => {
    const said = (
      [
        { kind: "heic" },
        { kind: "type", suffix: ".pdf" },
        { kind: "size", bytes: 40 * 1024 * 1024 },
        { kind: "empty" },
        { kind: "decode", detail: "x" },
      ] as const
    ).map((e) => explainPick(e).title);
    expect(new Set(said).size).toBe(5);
  });

  it("puts a real number in the size message", () => {
    const { detail } = explainPick({ kind: "size", bytes: 41_943_040 });
    expect(detail).toContain("40.0 MB");
    expect(detail).toContain("25 MB");
  });

  it("matches the backend's own cap, so neither refuses what the other allows", () => {
    // `main.MAX_UPLOAD_BYTES` and `limits.MAX_BODY_BYTES` are both 25 MiB.
    expect(MAX_BYTES).toBe(25 * 1024 * 1024);
  });
});
