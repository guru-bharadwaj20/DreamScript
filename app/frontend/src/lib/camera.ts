/**
 * Phase 16.2.2 - the camera, and the five ways it does not open.
 *
 * `getUserMedia` is one call and a long tail of refusals, and on a phone every one of them has a
 * different fix that only the person holding it can apply. A single "camera unavailable" would send
 * someone to the wrong setting, so each is a named state with the sentence that matches it.
 *
 *   insecure      the page is not on HTTPS or localhost. `navigator.mediaDevices` is **undefined**,
 *                 not an error - so the check has to come before the call. This is the one that
 *                 catches every developer testing on a phone over the LAN
 *   denied        NotAllowedError. The permission was refused, and a retry shows no prompt on iOS:
 *                 Safari remembers, and only Settings can change it back
 *   none          NotFoundError / OverconstrainedError - no camera, or no *rear* camera. A laptop
 *                 has the second problem, and the fix is to fall back rather than to fail
 *   busy          NotReadableError. Another app or tab holds the device. Common on Android
 *   failed        anything else, with the name kept, because an unnamed failure is unreportable
 *
 * ## Why `facingMode` is `ideal` and not `exact`
 *
 * `exact: "environment"` fails outright on a device with no rear camera, which is every laptop and
 * every desktop webcam. `ideal` prefers the rear camera and falls back to whatever exists, so the
 * app works on a phone the way it should and still works where it is developed.
 */

export type CameraState =
  | { kind: "idle" }
  | { kind: "opening" }
  | { kind: "live"; stream: MediaStream }
  | { kind: "insecure" }
  | { kind: "denied" }
  | { kind: "none" }
  | { kind: "busy" }
  | { kind: "failed"; detail: string };

/**
 * The longest edge the pipeline benefits from.
 *
 * 9.1's detector runs at 1280px, so a capture larger than that is megabytes of upload the models
 * discard on their first resize. `ideal` rather than `exact` again: a camera that cannot do 1920
 * gives what it has instead of refusing.
 */
export const IDEAL_WIDTH = 1920;
export const IDEAL_HEIGHT = 1440;

/** JPEG, and 0.92 - see `capture`. */
export const CAPTURE_TYPE = "image/jpeg";
export const CAPTURE_QUALITY = 0.92;

export function secureEnough(): boolean {
  // `isSecureContext` is the browser's own answer to the question, and it is true for localhost as
  // well as HTTPS. Checking the protocol by hand gets the localhost exemption wrong.
  return typeof navigator !== "undefined" && !!navigator.mediaDevices && window.isSecureContext;
}

export async function open(): Promise<CameraState> {
  if (!secureEnough()) return { kind: "insecure" };
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: IDEAL_WIDTH },
        height: { ideal: IDEAL_HEIGHT },
      },
      audio: false,
    });
    return { kind: "live", stream };
  } catch (error) {
    return classify(error);
  }
}

function classify(error: unknown): CameraState {
  const name = error instanceof Error ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") return { kind: "denied" };
  if (name === "NotFoundError" || name === "OverconstrainedError") return { kind: "none" };
  if (name === "NotReadableError" || name === "AbortError") return { kind: "busy" };
  const detail = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  return { kind: "failed", detail };
}

/**
 * Stop every track.
 *
 * Every track, not the stream: `MediaStream` has no stop, and stopping only the first track leaves
 * the camera light on. This is the cleanup that `StrictMode`'s double-invoked effects exist to
 * catch, and it is why the effect that opens a camera must return one.
 */
export function close(stream: MediaStream | null | undefined): void {
  stream?.getTracks().forEach((track) => track.stop());
}

/**
 * The smallest frame worth sending.
 *
 * A stream does not arrive at its full size immediately: `videoWidth` is 0 before the first frame
 * and can report a placeholder for a moment after. Chrome's fake device makes this obvious - it
 * reported **2x2** for over a second - and a real phone does the same thing for a shorter one. The
 * shutter is disabled until the track is past this, because a 2x2 upload is a wasted round trip and
 * a confusing "the detector found nothing" for a person who saw a perfectly good preview.
 *
 * 320 is far below any real camera and far above any placeholder.
 */
export const MIN_USABLE_WIDTH = 320;

/** Is this element showing a frame big enough to be worth capturing? */
export function ready(video: HTMLVideoElement | null): boolean {
  return !!video && video.videoWidth >= MIN_USABLE_WIDTH && video.videoHeight > 0;
}

/**
 * One frame, as a JPEG blob.
 *
 * **JPEG, not PNG.** The subject is a photograph of paper under room light - continuous tone, sensor
 * noise, no flat areas - and PNG compresses that at roughly 4 to 6 times the size for no visible
 * gain. On a mobile uplink that is the difference between a two-second upload and a ten-second one,
 * and 15.11 accepts both.
 *
 * **0.92, not 1.0.** Above about 0.92 JPEG spends bytes on noise; the detector and the recogniser see
 * nothing a human would not. Below about 0.8, ringing appears along the pen strokes, which is
 * precisely the signal being read.
 *
 * `videoWidth`/`videoHeight`, not the element's layout size: the preview is cropped to fit the
 * screen and the element is whatever CSS made it. The frame is the sensor's.
 */
export async function capture(video: HTMLVideoElement): Promise<Blob> {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) throw new Error("the camera has not produced a frame yet");
  if (width < MIN_USABLE_WIDTH) {
    throw new Error(`the camera is still warming up (${width}x${height})`);
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("this browser gave no 2d canvas context");
  context.drawImage(video, 0, 0, width, height);

  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("the frame could not be encoded"))),
      CAPTURE_TYPE,
      CAPTURE_QUALITY,
    );
  });
}

/** A filename the backend will accept, with the time in it so a share sheet offers something sane. */
export function captureName(now = new Date()): string {
  const stamp = now.toISOString().replace(/[:.]/g, "-").slice(0, 19);
  return `dreamscript-${stamp}.jpg`;
}

/** What to tell a person, per state. One sentence, and the fix they can actually apply. */
export function explain(state: CameraState): { title: string; detail: string; retry: boolean } {
  switch (state.kind) {
    case "insecure":
      return {
        title: "The camera needs a secure connection",
        detail:
          "Browsers only hand over a camera on HTTPS or on localhost. If you are testing over a " +
          "local network address, open the app on the machine running it, or serve it over HTTPS.",
        retry: false,
      };
    case "denied":
      return {
        title: "Camera access was refused",
        detail:
          "Your browser is remembering a previous 'no'. On iOS, Settings → Safari → Camera; on " +
          "Android, tap the padlock in the address bar. Or choose a photo from your library instead.",
        retry: true,
      };
    case "none":
      return {
        title: "No camera found",
        detail: "This device has no camera the browser can reach. Choosing a photo works the same way.",
        retry: false,
      };
    case "busy":
      return {
        title: "The camera is in use",
        detail: "Another app or another tab has it open. Close that one and try again.",
        retry: true,
      };
    case "failed":
      return { title: "The camera did not open", detail: state.detail, retry: true };
    default:
      return { title: "", detail: "", retry: false };
  }
}
