# `app/` — Phase 16's client and its backend

Two processes and one rule. **No model lives here.** The recogniser is `trocr-base-handwritten` at
333.9M parameters and the detector is a YOLO pose model at 1280px; both stay in `src/`, behind
15.11's model server, and this tree talks to them over HTTP. A test asserts it: no file under
`app/` may import torch, ultralytics or transformers.

## What is here

| Path | Row | What |
| :--- | :--- | :--- |
| `app/backend/` | 16.1 | the app backend — one upload, cheap reads against its id, the correction log |
| `app/frontend/` | 16.2 | the client — Vite + React + TypeScript, mobile-first, installable |

### `app/backend/` — 16.1

```
uvicorn app.backend.main:app --host 0.0.0.0 --port 3000   # or: make backend
python -m app.backend.main --check                        # self-test, no port, no model
python -m app.backend.main --openapi                      # the published contract
```

| Route | What |
| :--- | :--- |
| `GET /health` | this process. `?upstream=1` also probes the model server |
| `POST /predict` | one page → the model server's answer, stored under a new id |
| `POST /predict/stream` | the same, streamed stage by stage as server-sent events (16.1.2) |
| `GET /predict/{id}` | the whole stored record again |
| `GET /ir/{id}` | just the IR and the traversal |
| `GET /code/{id}` | just the code. `?format=text` for a clipboard or a share sheet |
| `POST /run/{id}` | execute that code in 11.2.9's sandbox and return what it printed |
| `POST /feedback` | one correction against an id. Appended, never overwritten |
| `GET /feedback/{id}` | every correction against one id |

Three things make this a service rather than a client library:

**One upload, then ids.** A page photograph is several megabytes over a mobile uplink, and the app
needs the boxes to draw, the IR to lay out, the code to show and a regeneration after a label is
fixed. Charging the user a second 3 MB upload for each is a phone bill, not an API.

**`/health` does not probe the model server by default.** 15.11's `/health` deliberately does not
load the weights, because a liveness probe that pulls 1.3 GB gets the container killed inside its
own start period. The same argument holds one hop out and is easier to get wrong: a health check
that failed because the *model* container was not ready would make Docker restart this process —
which was never the unhealthy one — forever.

**Degradation is passed through, not flattened.** Every response carries the model server's own
`ok`, `degraded`, `stopped_at`, `needs_confirmation` and per-stage table, verbatim. A phone showing
someone their own whiteboard is the caller with the least context in the system and the most need
for that distinction.

**The progress stream is relayed, never invented.** A cold GPU run of a fixture page is 12.08 s, of
which `assemble` alone is 9.15 s — a spinner held for nine seconds is indistinguishable from a
stalled request, which is why `POST /predict/stream` exists. But this process holds no pipeline, so
the only progress it could report on its own authority would be the expected stage names advanced on
a timer: a bar that moves because seconds passed rather than because work finished. So the observer
lives in `src/pipeline/core.py`, the SSE route on the model server, and this relays it — adding one
`upload` frame (its own fact) and the `id` on the `result` frame (its own, from the store), and
passing everything else through unchanged, keep-alive comments included.

```
curl -N -F "image=@page.jpg" http://localhost:3000/predict/stream

event: upload
data: {"event":"upload","filename":"page.jpg","bytes_in":2901}
event: stage_started
data: {"event":"stage_started","stage":"assemble","index":3}
...
event: result
data: {"id":"9e3462fe95214bf2","ok":true,"code":"...","stages":[...]}
```

**`POST /run/{id}` takes an id, and there is no parameter a program could arrive in.** That is the
difference between a sandbox runner and arbitrary-code-execution as a service. Python goes to
11.2.9's job-object sandbox, tightened for a socket rather than an evaluator: memory capped at
128 MB, two processes (the interpreter's own floor — see `runner.py`, the constant records why 1 does
not work), 43 modules refused by a pre-scan, the timeout clamped to 1–15 s, output capped at 64 KiB,
and at most four programs running at once. SQL, React and SPICE go to 12.3.2's own checkers; a host
with no node or no ngspice answers `*.unavailable`, which is carried through rather than folded into
a failure.

`runner.py` records the probe behind all of it: every emitter was fed labels engineered to break out
of the generated code, and none of them did — labels become identifiers in Python and SQL and
JSON-escaped string literals in JSX. The tightenings exist for the *other* rung: when
`DREAMSCRIPT_MODEL_URL` is set, the program is a fine-tuned model's output and a label is prompt
content.

> 11.2.9 says it about itself and it is repeated here: this is a **resource** sandbox, not a security
> boundary. It caps memory, kills a process tree and refuses names; it does not stop a program
> reading a file the service user can read.

**Four rate budgets, not one.** A page is seconds of GPU; an id lookup is a file read, and a phone
makes three of those per capture. One bucket would be set by the expensive route and would throttle
the cheap one.

| Class | Per minute | Why |
| :--- | :---: | :--- |
| `predict` | 6 | seconds of GPU and megabytes of upload |
| `run` | 10 | a program is a process; the runner already caps four at once |
| `write` | 30 | corrections — cheap, and the thing we most want people to do |
| `read` | 120 | an id lookup; a client polls these |

Every response carries `X-RateLimit-Limit`, `-Remaining` and `-Reset`, so a client can pace itself
instead of discovering the limit by hitting it. A `Content-Length` over 25 MB is refused before a byte
of body is read. `/health` is exempt — a 429 to a liveness probe gets the container restarted.

Two things the limiter is not, both stated on the wire rather than only in a comment.
`X-RateLimit-Scope: process:60s`, because the counters are a dict in this process: two workers give a
client twice the budget, which is why the numbers are conservative and why there is no Redis here.
And `X-Forwarded-For` is ignored unless `DREAMSCRIPT_TRUSTED_PROXY_HOPS` says how many proxies are
real — a limiter keyed on a header the client sets is a limiter with a bypass.

State lives under `runs/app/` (gitignored, derived): one JSON file per prediction, oldest-first
eviction at 500, plus `feedback.jsonl`, which is never evicted because it is training data.
`DREAMSCRIPT_APP_STATE` moves it.

### `app/frontend/` — 16.2

```
npm ci && npm run dev      # or: make frontend — http://localhost:5173
npm run build              # tsc --noEmit, then a hashed production bundle into dist/
```

**A web app, not an APK.** Phase 16's preamble records the amendment and its reasons: it has to run
on iOS as well as Android, nothing on this machine can compile an APK, and writing Kotlin that is
never built would make 16.2.1 the first row in the plan flipped on no evidence. The gesture — a
phone held over paper — is unchanged, and so is the decision not to use an app store.

Same-origin by construction. The client calls `/predict`, `/ir/:id` and so on with no base URL: Vite
proxies those paths to port 3000 in development, and the backend serves the bundle in production. So
there is no CORS middleware anywhere and no `VITE_API_URL` to be wrong in one of three environments.

**The design system is `src/styles/tokens.css`,** and it carries one argument. Gold is the brand and
nothing else; four contrast colours carry the only thing this app knows that a photograph does not —
whether to trust the answer. Emerald read cleanly, orange degraded, rose stopped, violet asking. Each
wears a dot as well as a colour, and *only* those four do, so the dot means "this is about trust".
Orange rather than amber on purpose: amber sits at gold's hue and a degraded pill beside a gold
button would read as brand.

Light mode is not dark inverted. The ground is warm off-white because the subject is paper, and gold
has a second, darker value for it — `#e8c36a` on white is about 1.9:1 and unreadable.

Three theme states, not two: system is the default, because a person choosing light or dark on their
phone already made that choice once.

## The model server, which is not here

`src/serve/api.py`, finished in 15.11, a separate process by design — a phone client, a notebook
and a batch job all reach the same service:

```
uvicorn src.serve.api:app --host 0.0.0.0 --port 8000     # or: make serve
docker compose up model
```

The backend finds it at `MODEL_URL`, which `docker-compose.yml` has set to `http://model:8000`
since 15.10. Outside compose it defaults to `http://localhost:8000`.

## Compose

`docker-compose.yml` declares the `app` service behind the `app` profile, so a default
`docker compose up` does not build it. CI asserts that boundary directly:

```
docker compose config --services                 # must be exactly: model
docker compose --profile app config --services   # must contain: app
```
