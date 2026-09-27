# What DreamScript does with your photograph

Phase 16.3.4. Written from the code rather than from intent — every claim below names the file that
makes it true, so it can be checked and so it fails visibly when the code changes.

The short version, which is also the first thing the app says on its landing screen:

> **Your photograph is uploaded to a server.** The models do not run in your browser, and this is
> not a detail of the deployment — it is what the app is.

## Who "the server" is

Nobody's, by default. This is not a hosted service and there is no `dreamscript.app` to sign up to.
The backend and the model server are two processes that somebody runs — on a laptop, on a machine
in a cupboard, on a rented box — and whoever runs them can read everything below. If you did not
start them yourself, the person who did is the person this page is really about.

## What leaves your device

**The page image, every time you press the shutter.** There is no local reading and no offline
recognition: the recogniser is a 334M-parameter handwriting model and the detector runs at 1280px,
and neither belongs on a phone. With no network the app opens and shows five *stored example*
readings and nothing else, which is 16.2.10 and is labelled as stored wherever it appears.

**Nothing else.** No analytics, no error reporting, no third-party requests of any kind. The app
loads no web font, no CDN script and no tracking pixel — `app/frontend/index.html` and the built
bundle contain no external origin, and a test asserts the client fetches only same-origin paths.
The document sets `referrer: no-referrer`.

### What the image is by the time it leaves

A **camera capture** is drawn to a `<canvas>`, straightened (`src/lib/dewarp.ts`) and re-encoded as
JPEG. Re-encoding through a canvas **drops every EXIF field**, so the time, the device, the lens and
any **GPS coordinates** the camera recorded do not leave with it. That is a side effect of how the
capture works rather than a feature that was asked for, and it is worth knowing it is there.

A **photograph you choose from your library** is uploaded **as it is on disk, EXIF included**. It is
checked (suffix, size, that it decodes) and then sent unchanged — `src/lib/pick.ts` hands the
original `File` to the uploader. **If that photograph has location data in it, the location data is
uploaded.** Nothing in this project reads it, and nothing in this project strips it either. If that
matters to you, take the photograph with the in-app camera instead, or strip the metadata before
choosing it.

The filename sent with a camera capture is `dreamscript-<ISO timestamp>.jpg`, so the second at which
you pressed the shutter travels with it and is stored with the result.

## What is kept, where, and for how long

### On the app backend — the result, not the picture

`app/backend/store.py` writes one JSON file per reading:

| field | what it is |
| :--- | :--- |
| `id` | 16 hex characters, minted per upload |
| `created` | when it was read |
| `result` | the shapes, the arrows, the recognised text, the IR and the generated code |
| `filename` | the name your browser sent |
| `bytes_in` | how large the upload was |
| `corrections` | labels you have retyped |

**The image itself is not stored.** That is why the app can show you a result again without a second
upload, and why reopening a shared link shows you the reading with no picture to draw it on — the
only copy of the photograph is the one the browser that took it is still holding, in memory, until
the tab closes or the next capture replaces it (`src/lib/held.ts`).

**Retention is the most recent 500 readings.** `MAX_RECORDS = 500`, oldest evicted first. A reading
that has aged out answers 404, and the app says so in those words.

**Corrections are kept for ever.** When you retype a label, the correction is appended to
`feedback.jsonl` and that file is **never evicted**, including after the reading it refers to has
gone. The same label corrected twice keeps both events. This is deliberate and it is the reason the
feature exists: corrections are the training data that makes the reader better. If you do not want a
correction kept, do not make it.

### On the model server — a temporary file, and a cache that is not temporary

The upload is written into a `tempfile.TemporaryDirectory` for the duration of the run and removed
when it finishes (`src/serve/api.py`).

**The stage cache is the part people do not expect, so it is stated here first.**
`src/pipeline/cache.py` keeps each stage's *output* as JSON under `data/interim/pipeline_cache`,
keyed by a SHA-256 of the image's bytes — so the detected boxes, the recognised handwriting and the
assembled IR of every page ever read persist on that machine. It has **no expiry and no size
limit**; `StageCache.clear()` removes it and nothing calls that automatically. It exists so a page
read twice is not paid for twice.

If you operate a deployment, that directory is the one to think about, and `DREAMSCRIPT_CACHE_DIR`
moves it somewhere you can wipe.

### In your browser

One entry: `dreamscript.theme`, which is `system`, `light` or `dark` (`src/lib/theme.ts`). The
service worker's cache holds the app's own files — its HTML, JavaScript, icons and the five example
sketches — and nothing derived from anything you photographed; a test asserts that the worker never
caches an API response, because a cached reading is a stale reading served without saying so.

## The id in the address bar is a key

`#/r/<id>` is how a result is reopened and shared, and it is **not secret**: anyone who has that id
and can reach that backend can read the reading, the recognised text and the generated code. There
are no accounts and no authentication anywhere in this project — the ids are unguessable, which is
not the same thing as protected. Share a link only with people you would show the whiteboard to.

## Running the generated code

The **Run it** button sends the id to the backend, which executes the generated program in a sandbox
on the *server* (`app/backend/runner.py`, built on 11.2.9's): a Win32 job object with a 128 MB cap, a
timeout, a 64 KiB output cap and 43 refused imports. Nothing runs in your browser and nothing runs on
your phone. That sandbox is a **resource** boundary, not a security one — it stops a runaway loop and
an import of `subprocess`, and it does not stop a program reading a file the service user can read.

## What there is none of

- No account, no login, no email address, no password.
- No cookies. Nothing sets one.
- No analytics, telemetry, crash reporting or usage statistics.
- No advertising, and nothing shared with anyone.
- No location request. The app never calls the geolocation API — but see the note above about EXIF
  in a photograph you choose from your library.
- The camera permission is asked for by the browser, used for the viewfinder, and the stream is
  **released the moment a frame is taken** rather than held through the pipeline run.

## Checking any of this

Everything above is in this repository:

| claim | where |
| :--- | :--- |
| the reading is stored and the picture is not | `app/backend/store.py` |
| 500 readings, oldest evicted | `MAX_RECORDS` in the same file |
| corrections are never evicted | `Store._evict` and `FEEDBACK_FILE` |
| the upload is a temporary file | `src/serve/api.py` |
| stage outputs are cached indefinitely | `src/pipeline/cache.py` |
| the client calls only same-origin paths | `tests/test_app_frontend.py` |
| the worker never caches an API response | `app/frontend/public/sw.js`, and a test |
| one `localStorage` key | `src/lib/theme.ts` |

If a claim here and the code disagree, the code is what happens — and that is a bug in this
document, which is worth reporting.
