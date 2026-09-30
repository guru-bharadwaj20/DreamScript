# API reference

Phase 17.5. There are two HTTP services and one data contract:

| | Process | Default port | Who calls it |
| :--- | :--- | :--- | :--- |
| [Model server](#model-server-srcserveapi) | `uvicorn src.serve.api:app` | 8000 | the app backend, a notebook, a batch job |
| [App backend](#app-backend-appbackendmain) | `uvicorn app.backend.main:app` | 3000 | the installable client (Phase 16) |
| [IR](#the-ir) | `schemas/*.schema.json` | — | every stage after detection, and every response |

Each service also publishes its own OpenAPI document at `/openapi.json` and interactive docs at
`/docs`, generated from the code. `python -m app.backend.main --openapi` prints the backend's
without starting a server. This page explains what those schemas leave out: why the routes work
the way they do, and what the fields mean.

## Model server (`src.serve.api`)

Loads the pipeline once, lazily, on the first request. It has no state between requests.

| Route | What |
| :--- | :--- |
| `GET /health` | `{"status": "ok", "model_loaded": bool}`. **Never loads the model**, so a liveness probe cannot time out on 1.3 GB of weights |
| `GET /version` | the service version and the registry version of each component |
| `POST /predict` | multipart `image` → one [result](#the-result-object) |
| `POST /predict/stream` | the same, as server-sent events: `run_started`, then `stage_started` / `stage_finished` for each stage, then one `result` frame |

**Uploads.** Accepted suffixes: `.png .jpg .jpeg .webp .bmp .tif .tiff`, with at most 25 MiB per
upload. The contents are checked against the file's magic bytes, so a renamed file is refused.

| Status | When |
| :--- | :--- |
| 200 | the pipeline answered, **including when it could not read the page**. See `ok` and `stopped_at` |
| 400 | empty upload |
| 413 | over 25 MiB |
| 415 | wrong suffix, or contents that are not an image of that type |
| 502 | the pipeline raised. The request was fine, so retry the page, not the request |

### The result object

```json
{
  "ok": true,
  "degraded": false,
  "stopped_at": null,
  "needs_confirmation": false,
  "diagram_type": "flowchart",
  "language": "python",
  "code": "def main():\n    ...",
  "ir": { "...": "see The IR below" },
  "traversal": ["n0", "n1", "n3", "n2"],
  "stages": [{"stage": "detect", "seconds": 0.036, "degraded": false, "...": "..."}],
  "seconds": 2.71
}
```

| Field | Meaning |
| :--- | :--- |
| `ok` | code was produced and passed `verify` |
| `degraded` | some stage answered with a fallback, for example the emitter instead of the fine-tuned model. The `stages` row says which stage |
| `stopped_at` | the first stage that returned no value, or `null`. Earlier stages keep their timings |
| `needs_confirmation` | the type probability was below the 0.60 gate (13.5). There is no code, because the code generators are type-specific |
| `diagram_type` | `flowchart`, `state_machine`, `er_diagram`, `wireframe`, `circuit` or `unknown` |
| `language` | `python`, `react`, `sql` or `spice`, or `null` |
| `ir` | the assembled [IR](#the-ir) |
| `traversal` | node ids in the order the generator reads them |
| `stages` | one row per stage, the same rows as `Result.timing_table()` |

## App backend (`app.backend.main`)

Stores a prediction under an id, so the phone uploads a photograph once and then makes cheap
reads against that id. It serves the client bundle from the same origin (`DREAMSCRIPT_BUNDLE`),
so there is no CORS anywhere. It finds the model server through `MODEL_URL` (default
`http://localhost:8000`).

| Route | What |
| :--- | :--- |
| `GET /health` | this process, plus the bundle directory it serves. `?upstream=1` also probes the model server |
| `POST /predict` | one page → the model server's result, stored and returned with a new `id` |
| `POST /predict/stream` | the same as SSE. Adds an `upload` frame, and an `id` on the `result` frame |
| `GET /predict/{id}` | the whole stored record |
| `GET /ir/{id}` | the IR and traversal only, for the overlay and graph view |
| `GET /code/{id}` | the code only. `?format=text` returns plain text for a clipboard or share sheet |
| `POST /run/{id}` | runs the stored code in the 11.2.9 sandbox and returns its output. `?timeout_s=` is clamped to 1–15 |
| `POST /correct/{id}` | fix a label, log the fix, and regenerate the code from the corrected IR |
| `POST /feedback` | one correction, appended and never overwritten |
| `GET /feedback/{id}` | every correction against one id |

**`/run` takes an id, and no request can carry a program.** It only ever runs code the pipeline
produced. Python runs under a 128 MB memory cap, with 43 modules refused, output capped at 64 KiB
and at most four runs at once. SQL, React and SPICE go to 12.3.2's checkers. A host without node or
ngspice answers `*.unavailable`, and that is reported as unavailable, not as a failure. When every
slot is busy it returns 503 with `Retry-After`.

**Correction body** (`/feedback`, `/correct/{id}`):

```json
{"id": "9e3462fe95214bf2", "kind": "sub_text", "node": "n3", "was": "strat", "now": "start"}
```

`kind` is one of `sub_text`, `sub_type`, `add_node`, `del_node`, `add_edge`, `del_edge`, `other`,
and anything else returns 422. `/correct` reports `logged`, `applied` and `regenerated`
separately. Every correction is logged. Only some kinds are applied to the IR, and the response
says which. What is kept, and for how long, is in [privacy.md](privacy.md).

Unknown ids return 404. The store keeps the 500 most recent predictions. The correction log is
never evicted.

## The IR

One diagram, of any type, as a labelled directed multigraph plus provenance. The JSON Schemas in
`schemas/` are authoritative, and `src/ir` validates against them.

### `Diagram` (`ir.schema.json`)

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `ir_version` | `"M.m"` | the major version changes when a consumer of the old schema would break |
| `id` | string | 1–128 characters |
| `diagram_type` | enum | as above. `unknown` is legal: a converter that cannot tell must not guess |
| `nodes` | `Node[]` | |
| `edges` | `Edge[]` | |
| `unresolved_edges` | `{edge, reason, candidates?, note?}[]` | edges whose endpoint could not be decided. They also stay in `edges` with a `null` end. `reason` is one of `no-source`, `no-target`, `both-ends-open`, `ambiguous-endpoint`, `crosses-other-edge`, `outside-frame` |
| `crossed_out` | array | regions the writer struck through: drawn, then retracted, which is not the same as absent |
| `low_conf_text` | array | transcriptions the producer does not stand behind, kept out of `text` |
| `meta` | object | requires `source` (dataset or process) and `geometry`: `annotated`, `derived` or `absent`, meaning how far `bbox` and `polyline` can be trusted. `image` names the picture the coordinates refer to |

### `Node` (`node.schema.json`)

| Field | Required | Meaning |
| :--- | :---: | :--- |
| `id` | yes | unique within the diagram, and stable across re-runs |
| `shape` | yes | `rectangle`, `rounded-rect`, `diamond`, `ellipse`, `circle`, `double-circle`, `parallelogram`, `octagon`, `arrow`, `line`, `text-block`, `freeform` |
| `bbox` | yes | `[x, y, w, h]` in image pixels, origin top-left, or `null` |
| `text` | yes | the transcription. Empty string when there is none, never `null` |
| `semantic_role` | yes | see below |
| `confidence` | yes | 0–1. It is 1.0 only for ground truth copied from an annotation |
| `source_id` | | the id in the originating dataset, for auditing |
| `attrs` | | type-specific extras, for example a circuit component's value |

`semantic_role`: `start`, `end`, `process`, `decision`, `io`, `fork`, `join`, `event` (flowchart);
`state`, `initial-state`, `final-state`, `transition` (state machine); `entity`, `attribute`,
`relationship` (ER); `container`, `ui-input`, `ui-button`, `ui-label`, `ui-image` (wireframe);
`component`, `wire` (circuit); `unknown`.

### `Edge` (`edge.schema.json`)

| Field | Required | Meaning |
| :--- | :---: | :--- |
| `id` | yes | |
| `src`, `dst` | yes | node ids, or `null` when that end could not be attributed (it is then listed in `unresolved_edges`) |
| `directed` | yes | `true` when an arrowhead was drawn |
| `label` | yes | the `yes` on a decision branch or the `a,b` on a transition, else `""` |
| `polyline` | yes | `[[x, y], …]` in image pixels, at least two points, or `null` |
| `confidence` | yes | 0–1 |
| `source_id`, `attrs` | | as for nodes |

### Example

```json
{
  "ir_version": "1.0",
  "id": "page-0001",
  "diagram_type": "flowchart",
  "nodes": [
    {"id": "n0", "shape": "rounded-rect", "bbox": [40, 20, 120, 50], "text": "start",
     "semantic_role": "start", "confidence": 0.94},
    {"id": "n1", "shape": "diamond", "bbox": [40, 120, 140, 90], "text": "x > 0",
     "semantic_role": "decision", "confidence": 0.81}
  ],
  "edges": [
    {"id": "e0", "src": "n0", "dst": "n1", "directed": true, "label": "",
     "polyline": [[100, 70], [110, 120]], "confidence": 0.88}
  ],
  "unresolved_edges": [],
  "crossed_out": [],
  "low_conf_text": [],
  "meta": {"source": "pipeline", "geometry": "derived", "image": "page.jpg"}
}
```

`src/ir` converts to and from each dataset's native format. `src/codegen/serialise.py` turns the IR
into the compact text that the language model reads (12.1.2).
