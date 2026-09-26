/**
 * Phase 16.2.4 - choosing a photograph instead of shooting one.
 *
 * Deliberately **not** a `capture` attribute on the input.
 *
 * `<input type="file" accept="image/*" capture="environment">` opens the camera directly, which
 * sounds like a shortcut and is the wrong control: it replaces the *library* picker with a system
 * camera, so the one affordance labelled "choose a photo" would not offer any photo the person
 * already has. 16.2.2 is the camera. This is the other thing.
 *
 * ## What `accept` does and does not do
 *
 * `accept="image/*"` filters the picker on most platforms and is a **hint, not a gate** - a person
 * can still hand back a `.heic`, a PDF renamed to `.jpg`, or on some Android file managers anything
 * at all. So the checks below are real checks, in the order that costs least:
 *
 *   suffix    free, and it is what the backend rejects on before reading a byte
 *   size      free, and a 40 MB burst shot is not going up a mobile uplink to be refused at the far
 *             end after the whole of it has arrived
 *   decode    the only one that is certain. `createImageBitmap` either reads the bytes or does not,
 *             which is the same question the pipeline will ask
 *
 * ## HEIC is named rather than left to fail
 *
 * Every photograph on a recent iPhone is HEIC unless the camera is set otherwise, and **no browser
 * decodes it** - Safari included, which is the operating system that produces it. Sharing one into
 * a web app gives a file that fails `createImageBitmap` with nothing useful in the error. It is
 * detected by suffix and by its `ftyp` brand and given its own sentence, because "this image could
 * not be read" sends someone to look for a corrupt file that is not corrupt.
 */

/** What the backend's `ALLOWED_SUFFIXES` accepts. Checked here so the refusal costs no upload. */
export const ACCEPTED = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"] as const;

/** `main.MAX_UPLOAD_BYTES`. Refused here rather than after 25 MB has crossed a mobile link. */
export const MAX_BYTES = 25 * 1024 * 1024;

export type PickError =
  | { kind: "type"; suffix: string }
  | { kind: "heic" }
  | { kind: "size"; bytes: number }
  | { kind: "empty" }
  | { kind: "decode"; detail: string };

export interface Picked {
  file: File;
  width: number;
  height: number;
}

function suffixOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot < 0 ? "" : name.slice(dot).toLowerCase();
}

/**
 * Is this the HEIC/HEIF an iPhone produces?
 *
 * By suffix, and by the ISO base-media `ftyp` brand at offset 4 for a file whose name lies - which
 * is common, because a share sheet can hand over `image.jpg` containing HEIC.
 */
export async function isHeic(file: File): Promise<boolean> {
  const suffix = suffixOf(file.name);
  if (suffix === ".heic" || suffix === ".heif") return true;
  try {
    const head = new Uint8Array(await file.slice(0, 16).arrayBuffer());
    if (String.fromCharCode(...head.slice(4, 8)) !== "ftyp") return false;
    const brand = String.fromCharCode(...head.slice(8, 12));
    return ["heic", "heix", "hevc", "heim", "heis", "hevm", "mif1", "msf1"].includes(brand);
  } catch {
    return false;
  }
}

/** Check a chosen file, cheapest test first, and decode it to prove it is an image. */
export async function accept(file: File): Promise<Picked | PickError> {
  if (file.size === 0) return { kind: "empty" };
  if (file.size > MAX_BYTES) return { kind: "size", bytes: file.size };

  if (await isHeic(file)) return { kind: "heic" };

  const suffix = suffixOf(file.name);
  if (!(ACCEPTED as readonly string[]).includes(suffix)) return { kind: "type", suffix };

  try {
    // The decisive check, and the same question the pipeline will ask. `close()` because a bitmap
    // holds decoded pixels - a 12-megapixel photograph is 48 MB of them, and on a phone that is
    // worth releasing the moment the dimensions have been read.
    const bitmap = await createImageBitmap(file);
    const { width, height } = bitmap;
    bitmap.close();
    if (!width || !height) return { kind: "decode", detail: "the image decoded to nothing" };
    return { file, width, height };
  } catch (error) {
    return {
      kind: "decode",
      detail: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    };
  }
}

export function isPicked(value: Picked | PickError): value is Picked {
  return "file" in value;
}

/** One sentence per refusal, and the thing the person can actually do about it. */
export function explainPick(error: PickError): { title: string; detail: string } {
  switch (error.kind) {
    case "heic":
      return {
        title: "That is an Apple HEIC photo",
        detail:
          "No browser can read HEIC, including Safari on the phone that made it. In Settings → " +
          "Camera → Formats choose Most Compatible to shoot JPEG, or take the picture with the " +
          "camera button here instead.",
      };
    case "type":
      return {
        title: `${error.suffix || "That file"} is not an image this reads`,
        detail: `Choose a ${ACCEPTED.join(", ")} file — or use the camera.`,
      };
    case "size":
      return {
        title: "That photograph is too large",
        detail:
          `It is ${(error.bytes / 1024 / 1024).toFixed(1)} MB and the limit is ` +
          `${MAX_BYTES / 1024 / 1024} MB. A picture of one page does not need to be that big; ` +
          "the camera here produces about a tenth of it.",
      };
    case "empty":
      return { title: "That file is empty", detail: "There is nothing in it to read." };
    default:
      return {
        title: "That image could not be opened",
        detail: `The browser could not decode it. ${error.detail}`,
      };
  }
}
