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
import { type Cached, CAPTURED, CAPTURED_BY, cachedFor, looksOffline } from "../lib/offline";
import { Button, Card, Pill, Segmented, TrustPill } from "../ui";
import { Code } from "../ui/Code";
import { Correct } from "../ui/Correct";
import { Doubts } from "../ui/Doubts";
import { Graph } from "../ui/Graph";
import { Overlay } from "../ui/Overlay";
import { LOW_CONFIDENCE } from "../lib/doubts";

type Tab = "overlay" | "graph" | "code";

type Load =
  | { state: "loading" }
  /**
   * `stored` is set only when this reading came out of 16.2.10's bundle instead of the server.
   *
   * It is on the *ready* state rather than a fourth state on purpose: a stored answer is a real
   * reading of a real page and every panel below should render it exactly as it renders a live one.
   * What changes is one banner and the two controls that need a server - and a separate state would
   * have meant a second copy of this screen, which is how the copies drift apart.
   */
  | { state: "ready"; prediction: Prediction; stored?: Cached }
  | { state: "missing"; detail: string; offline: boolean };

export function Result({ id }: { id: string }) {
  const [load, setLoad] = useState<Load>({ state: "loading" });
  const [tab, setTab] = useState<Tab>("code");
  const [selected, setSelected] = useState<string | null>(null);
  const [correcting, setCorrecting] = useState(false);
  const held = heldFor(id);

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
        //
        // Deliberately not extended to 16.2.10's stored answers, whose bundled image is a request
        // that a lack of network can refuse: landing on an explained-but-empty canvas is a worse
        // first screen than landing on the program, and the Photo tab is one tap away.
        if (heldFor(id)) setTab("overlay");
      })
      .catch((error: unknown) => {
        if (!live) return;
        // 16.2.10, and the order is the rule: the server was asked first and failed. A stored
        // answer that pre-empted the request would show a reading from before a correction someone
        // had just made, silently, on the screen whose job is saying how much to trust it.
        const stored = cachedFor(id);
        if (stored) {
          setLoad({ state: "ready", prediction: stored.prediction, stored });
          return;
        }
        const detail =
          error instanceof ApiError && error.status === 404
            ? "This page is no longer held. The server keeps the most recent 500 and this one has aged out — photograph it again."
            : error instanceof ApiError
              ? error.message
              : looksOffline(error)
                ? "There is no network, and this reading is not one of the stored examples — it only ever existed on the server."
                : "The server did not answer.";
        setLoad({ state: "missing", detail, offline: looksOffline(error) });
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
          {load.offline ? (
            // Offering "take another photograph" with no network would send someone to a camera
            // whose shutter cannot reach anything. The examples are what is left.
            <a href="#/gallery" className="btn btn-primary btn-block">
              See the stored examples
            </a>
          ) : (
            <a href="#/" className="btn btn-primary btn-block">
              Take another photograph
            </a>
          )}
        </Card>
      </div>
    );
  }

  const p = load.prediction;
  const stored = load.stored;
  /**
   * What the overlay draws on.
   *
   * `heldFor` first - that is this session's own photograph and the only one that is certainly
   * right. For a stored example there is a second candidate: the bundled PNG the answer was
   * computed from, which is a real request and can fail with no network. It is *offered* rather than
   * promised, and `Overlay` now reports a src it could not load instead of leaving a blank canvas.
   * After 16.3.1 precaches `examples/`, this stops being a gamble.
   */
  const photograph = held ?? (stored ? `examples/${stored.file}` : null);

  return (
    <div className="scroll">
      <div className="page stack">
        <header className="stack-sm">
          {stored ? <Stored cached={stored} /> : null}
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
                <Selected
                  prediction={p}
                  id={selected}
                  stored={!!stored}
                  onCorrect={() => setCorrecting(true)}
                />
              </>
            ) : null}
          </div>
        ) : tab === "graph" ? (
          <div className="stack-sm">
            <Graph ir={p.ir} traversal={p.traversal} selected={selected} onSelect={setSelected} />
            <Selected
              prediction={p}
              id={selected}
              stored={!!stored}
              onCorrect={() => setCorrecting(true)}
            />
          </div>
        ) : (
          <Code prediction={p} stored={!!stored} />
        )}

        <Doubts
          prediction={p}
          onSelect={(node) => {
            // Jumping to the picture is the point: a list that named a problem and left a person
            // to find it would be a worse version of the flag it replaced.
            setSelected(node);
            setTab(photograph ? "overlay" : "graph");
          }}
        />

        <StageTable prediction={p} />
      </div>

      <Correct
        open={correcting}
        node={(p.ir?.nodes ?? []).find((n) => n.id === selected) ?? null}
        prediction={p}
        onClose={() => setCorrecting(false)}
        onCorrected={(next) => setLoad({ state: "ready", prediction: next })}
      />
    </div>
  );
}

/**
 * Phase 16.2.10 - "this came out of the bundle", above everything else on the screen.
 *
 * Above the trust pill, which is the only place it can go. Every other panel here answers *how much
 * to trust this reading*, and all of them are meaningless if the reader does not first know that the
 * reading is not of anything they did. A stored answer mistaken for a live one is the app claiming
 * to have read a page it never saw.
 *
 * It names the date and the backend version because a cache with no provenance is worse than no
 * cache: "an example" invites the question "of what, and when", and the answer is in the bundle
 * already - `scripts/make_offline_examples.py` records both.
 */
function Stored({ cached }: { cached: Cached }) {
  return (
    <Card
      style={{
        padding: "var(--sp-4)",
        borderColor: "color-mix(in srgb, var(--degraded) 34%, transparent)",
        background: "var(--degraded-soft)",
      }}
      className="stack-sm"
    >
      <div className="row" style={{ gap: "var(--sp-2)" }}>
        <Pill tone="degraded">stored answer</Pill>
        <span className="grow" />
        <span className="dim num" style={{ fontSize: 11.5 }}>
          {CAPTURED}
        </span>
      </div>
      <strong style={{ fontSize: 14 }}>
        Nothing was read just now — this is the bundled {cached.name.toLowerCase()} example.
      </strong>
      <p className="muted" style={{ fontSize: 13.5 }}>
        The server could not be reached, so the app is showing the answer it shipped with. It is a
        real run of the real pipeline over <span className="mono">{cached.file}</span> on {CAPTURED},
        against backend {CAPTURED_BY} — including the parts it got wrong, which are not tidied up
        here.
      </p>
      <p className="dim" style={{ fontSize: 12.5 }}>
        Running the program and correcting a label both need the server and are switched off below.
      </p>
    </Card>
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
function Selected({
  prediction,
  id,
  stored,
  onCorrect,
}: {
  prediction: Prediction;
  id: string | null;
  /** 16.2.10: this reading came from the bundle, so there is no server to send a correction to. */
  stored?: boolean;
  onCorrect: () => void;
}) {
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
      {stored ? (
        // Disabled *and* explained. A live-looking button that answers with a network error is the
        // failure this row exists to remove, and a button that silently does nothing is worse.
        <>
          <Button block disabled>
            {node.text ? "Fix this label" : "Say what is here"}
          </Button>
          <p className="dim" style={{ fontSize: 12 }}>
            A correction is written to the server&apos;s log (16.1.5) and the code is re-emitted
            there, so there is nothing to send this to while it is offline.
          </p>
        </>
      ) : (
        <Button block onClick={onCorrect}>
          {node.text ? "Fix this label" : "Say what is here"}
        </Button>
      )}
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
