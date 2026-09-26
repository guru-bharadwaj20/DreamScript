/**
 * Phase 16.2.1 - the backend's contract, in types.
 *
 * Written from `app/backend/main.py`'s OpenAPI document rather than from memory, and the fields are
 * the ones 13.4 and 16.1.1 argue about: `ok`, `degraded`, `stopped_at`, `needs_confirmation` and the
 * per-stage table travel all the way to the screen, because a phone showing a person their own
 * whiteboard is the caller with the least context and the most need for them.
 *
 * Every path is **same-origin**. Vite proxies them to the backend in development and the backend
 * serves this bundle in production, so there is no base URL, no CORS preflight, and no
 * `VITE_API_URL` to be wrong in one of three environments.
 */

/** One row of 13.7's timing table. */
export interface Stage {
  stage: string;
  ok: boolean;
  seconds: number;
  cached: boolean;
  degraded: boolean;
  confidence: number;
  reason: string;
}

/** What `/predict` answers with: the model server's payload plus the backend's own fields. */
export interface Prediction {
  id: string;
  created: number;
  filename: string | null;
  bytes_in: number | null;
  upstream_seconds: number | null;

  ok: boolean;
  degraded: boolean;
  stopped_at: string | null;
  needs_confirmation: boolean;
  diagram_type: string;
  language: string | null;
  code: string | null;
  ir: Diagram | null;
  traversal: string[];
  stages: Stage[];
  seconds: number;
  corrections: Correction[];
}

/** 10.1's IR, as far as the client reads it. Extra keys are carried without being understood. */
export interface Diagram {
  nodes: Node[];
  edges: Edge[];
  [key: string]: unknown;
}

export interface Node {
  id: string;
  text?: string;
  type?: string;
  bbox?: [number, number, number, number];
  confidence?: number;
  [key: string]: unknown;
}

export interface Edge {
  source: string;
  target: string;
  text?: string;
  confidence?: number;
  [key: string]: unknown;
}

export interface Correction {
  id: string;
  at: number;
  kind: CorrectionKind;
  node?: string | null;
  edge?: string | null;
  was?: string | null;
  now?: string | null;
  note?: string | null;
  diagram_type?: string;
  degraded?: boolean;
}

/** Closed, matching `FEEDBACK_KINDS`. A log that also contains "typo" cannot be counted. */
export type CorrectionKind =
  | "sub_text"
  | "sub_type"
  | "add_node"
  | "del_node"
  | "add_edge"
  | "del_edge"
  | "other";

export interface RunVerdict {
  id: string;
  language: string | null;
  timeout_s: number;
  ok: boolean;
  kind: string;
  detail: string;
  stdout: string;
  stderr: string;
  seconds: number;
  truncated?: boolean;
  peak_memory_bytes?: number | null;
  available?: boolean;
  sandbox?: { memory_limit_mb: number; active_process_limit: number; denied_imports: number };
}

export interface Health {
  status: string;
  service: string;
  version: string;
  model_url: string;
  stored: number;
  limits: {
    budgets: Record<string, number>;
    window_s: number;
    max_body_bytes: number;
    tracked_clients: number;
    scope: string;
    trusted_proxy_hops: number;
  };
  model_server?: { reachable: boolean; url: string; status?: number };
}

/**
 * A failure with the status on it, because the client shows four of them differently.
 *
 * 503 means the model server is down and retrying later can work. 429 means slow down and
 * `retryAfter` says how long. 413 and 415 mean this photograph will never work, so offering "try
 * again" would be a lie. A single `Error` would collapse all four into one apology.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly retryAfter: number | null;

  constructor(status: number, detail: string, retryAfter: number | null = null) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.retryAfter = retryAfter;
  }

  /** Is trying the same thing again a reasonable suggestion to put in front of a person? */
  get retryable(): boolean {
    return this.status === 429 || this.status === 503 || this.status === 504 || this.status >= 500;
  }
}

async function unwrap<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;
  let detail = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json();
    // FastAPI puts a string in `detail` for an HTTPException and a list of objects for a validation
    // error. Both reach a person, so both are turned into a sentence rather than shown as JSON.
    if (typeof body?.detail === "string") detail = body.detail;
    else if (Array.isArray(body?.detail)) {
      detail = body.detail.map((d: { msg?: string }) => d?.msg ?? "invalid").join("; ");
    }
  } catch {
    // A non-JSON error body is not worth a second failure; the status line is enough.
  }
  const after = response.headers.get("retry-after");
  throw new ApiError(response.status, detail, after ? Number(after) : null);
}

export const api = {
  health(probeUpstream = false): Promise<Health> {
    // The query is built before the template rather than as a ternary inside it. Same result, and
    // the route stays readable as a route - which matters because
    // `tests/test_app_frontend.py` reads these literals to check every path the client calls is one
    // the backend registers, and a quoted ternary mid-literal defeats that join.
    const query = probeUpstream ? "?upstream=1" : "";
    return fetch(`/health${query}`).then(unwrap<Health>);
  },

  /** One page, not streamed. 16.2.2 uses the streaming form; this is the fallback and the test path. */
  predict(image: Blob, filename: string): Promise<Prediction> {
    const body = new FormData();
    body.append("image", image, filename);
    return fetch("/predict", { method: "POST", body }).then(unwrap<Prediction>);
  },

  prediction(id: string): Promise<Prediction> {
    return fetch(`/predict/${id}`).then(unwrap<Prediction>);
  },

  ir(id: string): Promise<{ id: string; ir: Diagram | null; traversal: string[] }> {
    return fetch(`/ir/${id}`).then(unwrap<{ id: string; ir: Diagram | null; traversal: string[] }>);
  },

  /** The program as text, which is what a clipboard and a share sheet want. */
  async codeText(id: string): Promise<string> {
    const response = await fetch(`/code/${id}?format=text`);
    if (!response.ok) await unwrap(response);
    return response.text();
  },

  run(id: string, timeoutS?: number): Promise<RunVerdict> {
    const query = timeoutS ? `?timeout_s=${timeoutS}` : "";
    return fetch(`/run/${id}${query}`, { method: "POST" }).then(unwrap<RunVerdict>);
  },

  feedback(correction: {
    id: string;
    kind: CorrectionKind;
    node?: string;
    edge?: string;
    was?: string;
    now?: string;
    note?: string;
  }): Promise<{ stored: Correction; corrections: number }> {
    return fetch("/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(correction),
    }).then(unwrap<{ stored: Correction; corrections: number }>);
  },

  corrections(id: string): Promise<{ id: string; count: number; corrections: Correction[] }> {
    return fetch(`/feedback/${id}`).then(unwrap<{ id: string; count: number; corrections: Correction[] }>);
  },
};
