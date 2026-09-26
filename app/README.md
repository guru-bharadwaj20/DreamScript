# `app/` — Phase 16's client and its backend

Two processes and one rule. **No model lives here.** The recogniser is `trocr-base-handwritten` at
333.9M parameters and the detector is a YOLO pose model at 1280px; both stay in `src/`, behind
15.11's model server, and this tree talks to them over HTTP. A test asserts it: no file under
`app/` may import torch, ultralytics or transformers.

## What is here

| Path | Row | What |
| :--- | :--- | :--- |
| `app/backend/` | 16.1 | the app backend — one upload, cheap reads against its id, the correction log |
| `app/frontend/` | 16.2 | the client. **Not built yet** |

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
| `GET /predict/{id}` | the whole stored record again |
| `GET /ir/{id}` | just the IR and the traversal |
| `GET /code/{id}` | just the code. `?format=text` for a clipboard or a share sheet |
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

State lives under `runs/app/` (gitignored, derived): one JSON file per prediction, oldest-first
eviction at 500, plus `feedback.jsonl`, which is never evicted because it is training data.
`DREAMSCRIPT_APP_STATE` moves it.

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
