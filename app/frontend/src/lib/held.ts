/**
 * Phase 16.2.5 - the photograph this session uploaded, for the overlay to draw on.
 *
 * **One slot, on purpose.** 16.1.1 stores the result on the server and not the picture, and the
 * About screen promises exactly that - so the only copy of a page is the one this browser is
 * holding, and it lives until the next capture replaces it or the tab closes.
 *
 * A module-level slot rather than React state because it is not state: nothing re-renders when it
 * changes, the capture screen writes it and the result screen reads it, and threading it through
 * the router would mean giving a `Blob` to an address bar that cannot hold one.
 *
 * The object URL of the previous page is revoked when a new one arrives. Without that, a session
 * that photographs twenty whiteboards leaks twenty decoded images - which on a phone is the
 * difference between an app that runs and one the system kills.
 */

let heldId: string | null = null;
let heldUrl: string | null = null;

/** Remember the image that produced `id`. Revokes whatever was held before. */
export function hold(id: string, blob: Blob): string {
  release();
  heldId = id;
  heldUrl = URL.createObjectURL(blob);
  return heldUrl;
}

/** The image for `id`, or `null` - a reload, a shared link, or a later capture. */
export function heldFor(id: string): string | null {
  return heldId === id ? heldUrl : null;
}

export function release(): void {
  if (heldUrl) URL.revokeObjectURL(heldUrl);
  heldId = null;
  heldUrl = null;
}
