/**
 * Phase 16.2.1 - the result screen, as far as the skeleton goes.
 *
 * `#/r/<id>` is a real address, and this is what makes it one: the screen loads a prediction from the
 * backend by id alone. That is 16.1.1's store paying off on the client side - a result survives a
 * reload, a shared link and a cold start, because the id in the address bar is the app's state.
 *
 * The overlay (16.2.5), the graph (16.2.6), the code panel (16.2.7) and the correction sheet (16.2.8)
 * fill the tabs in. What is here is the part every one of them sits under: the **trust header**,
 * which is the first thing on the screen because 13.4's whole argument is that a caller must be able
 * to tell an emitter answer from a model answer without asking a second question.
 */

import { useEffect, useState } from "react";

import { type Prediction, ApiError, api } from "../lib/api";
import { heldFor } from "../lib/held";
import { Card, Pill, Segmented, TrustPill } from "../ui";
import { Code } from "../ui/Code";
import { Graph } from "../ui/Graph";
import { LOW_CONFIDENCE, Overlay } from "../ui/Overlay";

type Tab = "overlay" | "graph" | "code";

type Load =
  | { state: "loading" }
  | { state: "ready"; prediction: Prediction }
  | { state: "missing"; detail: string };

export function Result({ id }: { id: string }) {
  const [load, setLoad] = useState<Load>({ state: "loading" });
  const [tab, setTab] = useState<Tab>("code");
  const [selected, setSelected] = useState<string | null>(null);
  const photograph = heldFor(id);

  useEffect(() => {
    let live = true;
    setLoad({ state: "loading" });
    api
      .prediction(id)
      .then((prediction) => {
        if (!live) return;
        setLoad({ state: "ready", prediction });
        // The photograph is the most legible answer to "did it read my page", so it leads when
        // this session still has it. On a reload there is nothing to draw on and the code does.
        if (heldFor(id)) setTab("overlay");
      })
      .catch((error: unknown) => {
        if (!live) return;
        const detail =
          error instanceof ApiError && error.status === 404
            ? "This page is no longer held. The server keeps the most recent 500 and this one has aged out — photograph it again."
            : error instanceof ApiError
              ? error.message
              : "The server did not answer.";
        setLoad({ state: "missing", detail });
      });
    return () => {
      live = false;
    };
  }, [id]);

  if (load.state === "loading") {
    return (
      <div className="page stack">
        <div className="status">
          <span className="status-dot" data-state="checking" />
          <span className="grow muted">Fetching {id}…</span>
        </div>
      </div>
    );
  }

  if (load.state === "missing") {
    return (
      <div className="page stack">
        <Card style={{ padding: "var(--sp-5)" }} className="stack-sm">
          <span className="eyebrow">Not here</span>
          <p className="muted">{load.detail}</p>
          <a href="#/" className="btn btn-primary btn-block">
            Take another photograph
          </a>
        </Card>
      </div>
    );
  }

  const p = load.prediction;

  return (
    <div className="scroll">
      <div className="page stack">
        <header className="stack-sm">
          <div className="row wrap" style={{ gap: "var(--sp-2)" }}>
            <TrustPill
              ok={p.ok}
              degraded={p.degraded}
              stoppedAt={p.stopped_at}
              needsConfirmation={p.needs_confirmation}
            />
            <Pill tone="gold">{p.diagram_type.replace(/_/g, " ")}</Pill>
            {p.language ? <Pill tone="plain">{p.language}</Pill> : null}
            <span className="grow" />
            <span className="dim num" style={{ fontSize: 12.5 }}>
              {p.seconds.toFixed(2)}s
            </span>
          </div>

          {p.needs_confirmation ? (
            <Card
              style={{
                padding: "var(--sp-4)",
                borderColor: "color-mix(in srgb, var(--confirm) 32%, transparent)",
                background: "var(--confirm-soft)",
              }}
            >
              <strong style={{ fontSize: 14 }}>It would rather ask than guess.</strong>
              <p className="muted" style={{ fontSize: 13.5, marginTop: 4 }}>
                Every stage after the diagram type is specific to that type, so a low-confidence guess
                would produce confident, runnable, wrong code. Tell it which kind of diagram this is.
              </p>
            </Card>
          ) : null}

          {p.stopped_at ? (
            <Card
              style={{
                padding: "var(--sp-4)",
                borderColor: "color-mix(in srgb, var(--stopped) 30%, transparent)",
                background: "var(--stopped-soft)",
              }}
            >
              <strong style={{ fontSize: 14 }}>It stopped at {p.stopped_at}.</strong>
              <p className="muted" style={{ fontSize: 13.5, marginTop: 4 }}>
                {reasonFor(p) ?? "The stage reported no value and the run ended there."}
              </p>
            </Card>
          ) : null}
        </header>

        <Segmented<Tab>
          label="Result view"
          value={tab}
          onChange={setTab}
          options={[
            { value: "overlay", label: "Photo" },
            { value: "graph", label: "Graph" },
            { value: "code", label: "Code" },
          ]}
        />

        {tab === "overlay" ? (
          <div className="stack-sm">
            <Overlay
              src={photograph}
              ir={p.ir}
              traversal={p.traversal}
              selected={selected}
              onSelect={setSelected}
              flagLowConfidence
            />
            {photograph ? (
              <>
                <div className="overlay-key">
                  <span className="k-node">
                    <i /> shape found
                  </span>
                  <span className="k-edge">
                    <i /> arrow
                  </span>
                  <span className="k-low">
                    <i /> below {LOW_CONFIDENCE} confidence
                  </span>
                  <span className="k-order">
                    <i /> reading order
                  </span>
                </div>
                <Selected prediction={p} id={selected} />
              </>
            ) : null}
          </div>
        ) : tab === "graph" ? (
          <div className="stack-sm">
            <Graph ir={p.ir} traversal={p.traversal} selected={selected} onSelect={setSelected} />
            <Selected prediction={p} id={selected} />
          </div>
        ) : (
          <Code prediction={p} />
        )}

        <StageTable prediction={p} />
      </div>
    </div>
  );
}

/** The last stage that reported a reason - which for a stopped run is the one that stopped it. */
function reasonFor(prediction: Prediction): string | null {
  const withReason = [...prediction.stages].reverse().find((stage) => stage.reason);
  return withReason?.reason ?? null;
}

/**
 * What was tapped, under the picture.
 *
 * The overlay can show a label but not its confidence, its shape class or its edges - a canvas has
 * nowhere to put four facts about one box without covering the handwriting they are about. So the
 * detail sits below, and 16.2.8's correction will attach to exactly this panel.
 */
function Selected({ prediction, id }: { prediction: Prediction; id: string | null }) {
  if (!id) {
    return (
      <p className="dim" style={{ fontSize: 12.5, textAlign: "center" }}>
        Tap a shape to see what was read there.
      </p>
    );
  }
  const node = (prediction.ir?.nodes ?? []).find((n) => n.id === id);
  if (!node) return null;
  const edges = (prediction.ir?.edges ?? []).filter((e) => e.src === id || e.dst === id);
  const confidence = node.confidence ?? 1;
  const step = prediction.traversal.indexOf(id);

  return (
    <Card style={{ padding: "var(--sp-4)" }} className="stack-sm">
      <div className="row wrap" style={{ gap: "var(--sp-2)" }}>
        <Pill tone="plain">{String(node.shape ?? "shape")}</Pill>
        {step >= 0 ? <Pill tone="gold">step {step + 1}</Pill> : null}
        {confidence < LOW_CONFIDENCE ? (
          <Pill tone="degraded">confidence {confidence.toFixed(2)}</Pill>
        ) : (
          <Pill tone="plain">confidence {confidence.toFixed(2)}</Pill>
        )}
        <span className="grow" />
        <span className="dim mono" style={{ fontSize: 11 }}>
          {id}
        </span>
      </div>
      <p style={{ fontSize: 15 }}>
        {node.text ? (
          <span className="selectable">{String(node.text)}</span>
        ) : (
          <span className="dim">no text was read here</span>
        )}
      </p>
      <p className="dim" style={{ fontSize: 12.5 }}>
        {edges.length === 0
          ? "No arrows touch this shape."
          : `${edges.length} arrow${edges.length === 1 ? " touches" : "s touch"} this shape.`}
      </p>
    </Card>
  );
}

/**
 * 13.7's timing table, on the phone.
 *
 * Kept on the result screen and not only in the progress stream, because "which stage was slow" and
 * "which stage fell back" are the two questions someone asks *after* seeing a wrong answer, and by
 * then the stream is gone. A cached stage says so: a second run of the same page looking instant is
 * the cache working, not a fault.
 */
function StageTable({ prediction }: { prediction: Prediction }) {
  // Over the stages that actually ran. A cached stage reports 0.0 s, so including them would not
  // change the maximum - but it makes the intent explicit, and a future cached-with-a-duration
  // would otherwise flatten every real bar.
  const slowest = Math.max(
    ...prediction.stages.filter((stage) => !stage.cached).map((stage) => stage.seconds),
    0.001,
  );
  return (
    <Card style={{ padding: "var(--sp-4)" }} className="stack-sm">
      <span className="eyebrow">Stages</span>
      <div className="stack-sm" style={{ marginTop: "var(--sp-1)" }}>
        {prediction.stages.map((stage) => (
          <div key={stage.stage} className="row" style={{ gap: "var(--sp-3)" }}>
            <span
              style={{
                width: 7,
                height: 7,
                borderRadius: "50%",
                flex: "none",
                background: stage.ok
                  ? stage.degraded
                    ? "var(--degraded)"
                    : "var(--ok)"
                  : "var(--stopped)",
              }}
            />
            <span style={{ width: 74, fontSize: 13 }}>{stage.stage}</span>
            {/* A cached stage gets an empty track, not a 2%-wide stub. Every stage of a re-run is
                cached, and seven near-zero bars in a row read as a broken chart rather than as the
                cache working - which is exactly the wrong impression, because it is the cache
                working. The word "cached" in the right-hand column is the honest signal. */}
            <span className="grow" style={{ height: 4, background: "var(--line-2)", borderRadius: 2 }}>
              {stage.cached ? null : (
                <span
                  style={{
                    display: "block",
                    height: "100%",
                    borderRadius: 2,
                    width: `${Math.max(2, (stage.seconds / slowest) * 100)}%`,
                    background: stage.degraded ? "var(--degraded)" : "var(--gold)",
                  }}
                />
              )}
            </span>
            <span className="dim num" style={{ fontSize: 12, width: 52, textAlign: "right" }}>
              {stage.cached ? "cached" : `${stage.seconds.toFixed(2)}s`}
            </span>
          </div>
        ))}
      </div>
    </Card>
  );
}
