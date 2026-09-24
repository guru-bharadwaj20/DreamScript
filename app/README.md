# `app/` — Phase 16's client

**There is no code here yet, and this file exists so that fact is visible rather than implied by
two `.gitkeep` files.**

`docker-compose.yml` declares an `app` service with `context: ./app`, behind the `app` profile so
a default `docker compose up` does not try to build it. CI asserts that boundary directly:

```
docker compose config --services                 # must be exactly: model
docker compose --profile app config --services   # must contain: app
```

## What is here

Nothing but this file.

## What Phase 16 puts here

| Path | Row | What |
| :--- | :--- | :--- |
| `app/Dockerfile` | 16.1 | the image `docker-compose.yml` already names |
| `app/backend/` | 16.1.2–16.1.4 | streaming stage-by-stage progress, the hardened sandbox runner, rate and size limits — the three things `src/serve/api.py`'s docstring says are **not** in the model server |
| `app/frontend/` | 16.2 | the browser client |

## What is already real, and is not here

The **model server** is `src/serve/api.py` and it is finished (15.11). It is a separate process
by design — a phone client, a notebook and a batch job all reach the same service — and it runs
today:

```
uvicorn src.serve.api:app --host 0.0.0.0 --port 8000     # or: make serve
docker compose up model                                   # or in the container
```

The `app` service talks to it by service name (`MODEL_URL=http://model:8000`), which is why the
two are separate services rather than one image.
