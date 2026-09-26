/**
 * Phase 16.2.8 - retype a label, and watch the code change.
 *
 * The plan calls this the highest-value screen in the app, and the arithmetic is 14's error
 * taxonomy: **`sub_text` is 11 of hdbpmn's median 21 edits**. More than half the distance between a
 * read page and a correct one is a word the recogniser got wrong. One tap closes more of that gap
 * than a recogniser retrain, and it closes it now, for the person holding the phone.
 *
 * ## Three things this screen refuses to do
 *
 * **It does not hide what was there.** The recogniser's reading stays on screen above the field,
 * struck through, because a person who cannot see what the machine thought cannot tell whether it
 * was close or nowhere near - and that is the judgement that decides whether to keep going or
 * photograph the page again.
 *
 * **It does not pretend an unapplied correction was applied.** Only `sub_text` rewrites the IR
 * today; everything else is logged as training data and says so. Telling someone their edit took
 * effect when the code did not move is the one thing that would make them stop making edits.
 *
 * **It does not submit on every keystroke.** A correction is an event in an append-only log that
 * is never evicted, and streaming one per character would fill it with the thirteen prefixes of
 * "Receive order" and make the store useless for the thing it exists for.
 *
 * ## The keyboard is the hard part on a phone
 *
 * A bottom sheet with a text field inside it is where iOS puts the software keyboard over the
 * submit button. `autoFocus` with `scrollIntoView` on focus, and the sheet's own `max-height` of
 * 88dvh - `dvh`, so it shrinks when the keyboard opens rather than sliding under it.
 */

import { useEffect, useRef, useState } from "react";

import { type CorrectionKind, type Node, type Prediction, ApiError, api } from "../lib/api";
import { Button, Card, Pill, Sheet } from "../ui";
import "./Correct.css";

/** The kinds a person can report from this screen, with words rather than taxonomy labels. */
const KINDS: { value: CorrectionKind; label: string; hint: string }[] = [
  { value: "sub_text", label: "The words are wrong", hint: "Retype them and the code is rewritten" },
  { value: "sub_type", label: "The shape is wrong", hint: "A decision read as a box, say" },
  { value: "del_node", label: "This is not a shape", hint: "It found something that is not there" },
  { value: "add_edge", label: "An arrow is missing", hint: "Something connects that it did not join" },
  { value: "del_edge", label: "An arrow is wrong", hint: "It joined two things that do not connect" },
  { value: "other", label: "Something else", hint: "Say what, in your own words" },
];

export function Correct({
  open,
  node,
  prediction,
  onClose,
  onCorrected,
}: {
  open: boolean;
  node: Node | null;
  prediction: Prediction;
  onClose: () => void;
  onCorrected: (next: Prediction) => void;
}) {
  const [kind, setKind] = useState<CorrectionKind>("sub_text");
  const [text, setText] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<string | null>(null);
  const field = useRef<HTMLInputElement>(null);

  const was = String(node?.text ?? "");

  useEffect(() => {
    if (!open) return;
    setKind("sub_text");
    // Pre-filled with what the recogniser read, so a one-character fix is a one-character edit
    // rather than retyping a word that was almost right.
    setText(was);
    setNote("");
    setError(null);
    setOutcome(null);
    const timer = window.setTimeout(() => {
      field.current?.focus();
      field.current?.select();
      field.current?.scrollIntoView({ block: "center", behavior: "smooth" });
    }, 120);
    return () => window.clearTimeout(timer);
  }, [open, was, node?.id]);

  if (!node) return null;

  const changed = kind !== "sub_text" || (text.trim() !== was.trim() && text.trim().length > 0);

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const next = await api.correct(prediction.id, {
        kind,
        node: node.id,
        now: kind === "sub_text" ? text.trim() : undefined,
        note: note.trim() || undefined,
      });
      onCorrected(next);
      if (next.correction.applied && next.correction.regenerated) {
        setOutcome("Fixed, and the code has been rewritten.");
        window.setTimeout(onClose, 900);
      } else if (next.correction.applied) {
        setOutcome("Fixed. The code could not be re-emitted from the corrected reading.");
      } else {
        // Logged and not applied. Said plainly: this is the case where pretending would cost the
        // most, because the person is watching for the code to move and it will not.
        setOutcome(
          "Recorded. This kind of correction is kept as training data but does not rewrite the " +
            "reading yet, so the code below has not changed.",
        );
      }
    } catch (failure) {
      setError(
        failure instanceof ApiError
          ? failure.status === 429 && failure.retryAfter
            ? `${failure.message} Try again in ${failure.retryAfter} seconds.`
            : failure.message
          : "The correction could not be saved.",
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <Sheet open={open} onClose={onClose} title="Fix this">
      <div className="stack">
        <Card style={{ padding: "var(--sp-4)" }} className="stack-sm">
          <span className="eyebrow">What it read</span>
          <p className="correct-was">
            {was ? <s>{was}</s> : <span className="dim">nothing was read here</span>}
          </p>
          <div className="row wrap" style={{ gap: "var(--sp-2)" }}>
            <Pill tone="plain">{String(node.shape ?? "shape")}</Pill>
            <Pill tone="plain">{node.id}</Pill>
            {typeof node.confidence === "number" ? (
              <Pill tone={node.confidence < 0.5 ? "degraded" : "plain"}>
                confidence {node.confidence.toFixed(2)}
              </Pill>
            ) : null}
          </div>
        </Card>

        <div className="correct-kinds" role="radiogroup" aria-label="What is wrong">
          {KINDS.map((option) => (
            <button
              key={option.value}
              role="radio"
              aria-checked={kind === option.value}
              className="correct-kind"
              onClick={() => setKind(option.value)}
            >
              <strong>{option.label}</strong>
              <small>{option.hint}</small>
            </button>
          ))}
        </div>

        {kind === "sub_text" ? (
          <label className="correct-field">
            <span className="eyebrow">What it should say</span>
            <input
              ref={field}
              type="text"
              value={text}
              onChange={(event) => setText(event.target.value)}
              onKeyDown={(event) => {
                // Enter submits, because a software keyboard's return key is the obvious way to
                // finish a one-field form and reaching past it to a button is not.
                if (event.key === "Enter" && changed && !busy) void submit();
              }}
              enterKeyHint="done"
              autoComplete="off"
              autoCorrect="off"
              autoCapitalize="none"
              spellCheck={false}
              placeholder="Type the words on the page"
            />
          </label>
        ) : (
          <label className="correct-field">
            <span className="eyebrow">Anything to add</span>
            <input
              type="text"
              value={note}
              onChange={(event) => setNote(event.target.value)}
              placeholder="Optional"
              autoComplete="off"
            />
          </label>
        )}

        {error ? (
          <Card
            style={{
              padding: "var(--sp-3) var(--sp-4)",
              borderColor: "color-mix(in srgb, var(--stopped) 30%, transparent)",
            }}
          >
            <p style={{ fontSize: 13 }}>{error}</p>
          </Card>
        ) : null}

        {outcome ? (
          <Card
            style={{
              padding: "var(--sp-3) var(--sp-4)",
              borderColor: "var(--gold-line)",
              background: "var(--gold-soft)",
            }}
          >
            <p style={{ fontSize: 13 }}>{outcome}</p>
          </Card>
        ) : null}

        <Button variant="primary" block onClick={submit} disabled={!changed || busy}>
          {busy ? "Saving…" : kind === "sub_text" ? "Fix it and rewrite the code" : "Record this"}
        </Button>

        <p className="dim" style={{ fontSize: 12, textAlign: "center" }}>
          Corrections are kept — they are how the reader gets better at handwriting like yours.
        </p>
      </div>
    </Sheet>
  );
}
