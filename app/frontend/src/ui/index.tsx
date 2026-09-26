/**
 * Phase 16.2.1 - the primitives, as components.
 *
 * Six, and no more until a screen needs a seventh. A component library written before the screens is
 * one that gets half used and half worked around.
 *
 * `TrustPill` is the only one here that is not generic, and it is the most important: it is the
 * single place that maps 13.4's four flags onto four colours and four words, so every screen says
 * the same thing about the same state. The alternative - each screen deciding - is how one screen
 * ends up calling a degraded answer "done".
 */

import type { CSSProperties, ReactNode } from "react";
import { useEffect } from "react";

import "./ui.css";

export function Mark({ size = 19 }: { size?: number }) {
  return (
    <span className="mark" style={{ fontSize: size }} aria-label="DreamScript">
      Dream<b>Script</b>
    </span>
  );
}

export function Button({
  children,
  variant = "default",
  block,
  icon,
  className = "",
  ...rest
}: {
  children?: ReactNode;
  variant?: "default" | "primary" | "quiet" | "danger";
  block?: boolean;
  icon?: boolean;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const classes = [
    "btn",
    variant === "primary" ? "btn-primary" : "",
    variant === "quiet" ? "btn-quiet" : "",
    variant === "danger" ? "btn-danger" : "",
    block ? "btn-block" : "",
    icon ? "btn-icon" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <button type="button" className={classes} {...rest}>
      {children}
    </button>
  );
}

export type Tone = "ok" | "degraded" | "stopped" | "confirm" | "neutral" | "gold" | "plain";

export function Pill({
  tone = "neutral",
  children,
  title,
}: {
  tone?: Tone;
  children: ReactNode;
  title?: string;
}) {
  return (
    <span className={`pill pill-${tone}`} title={title}>
      {children}
    </span>
  );
}

/**
 * 13.4's four flags, in one place, in priority order.
 *
 * The order is the point and it is not alphabetical. A page that stopped has no code, so `stopped`
 * outranks everything; a page the router refused to guess at is a *question*, so `confirm` comes
 * next; `degraded` means there is an answer but a lower rung produced it; `ok` is last because it is
 * the only one that claims nothing needs looking at.
 *
 * Showing all four at once would be worse than showing none: four pills is a wall, and the reader
 * has to work out which one dominates. This decides, once.
 */
export function TrustPill({
  ok,
  degraded,
  stoppedAt,
  needsConfirmation,
}: {
  ok: boolean;
  degraded: boolean;
  stoppedAt: string | null;
  needsConfirmation: boolean;
}) {
  if (needsConfirmation) {
    return (
      <Pill tone="confirm" title="The router was below its confidence floor, so it is asking rather than guessing">
        needs a type
      </Pill>
    );
  }
  if (stoppedAt) {
    return (
      <Pill tone="stopped" title={`The pipeline stopped at the ${stoppedAt} stage`}>
        stopped at {stoppedAt}
      </Pill>
    );
  }
  if (degraded) {
    return (
      <Pill tone="degraded" title="A stage answered from a rung below its first choice">
        degraded
      </Pill>
    );
  }
  if (ok) return <Pill tone="ok" title="Every stage answered with its first choice">read cleanly</Pill>;
  return <Pill tone="neutral">no answer</Pill>;
}

export function Card({
  children,
  gold,
  className = "",
  style,
}: {
  children: ReactNode;
  gold?: boolean;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <div className={`card ${gold ? "card-gold" : ""} ${className}`} style={style}>
      {children}
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (next: T) => void;
  label: string;
}) {
  return (
    <div className="seg" role="tablist" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="tab"
          aria-selected={option.value === value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

/**
 * A bottom sheet: the correction UI (16.2.8) and every confirmation live in one.
 *
 * Three things that are easy to leave out and are why this is a component rather than a div.
 * **Escape closes it**, because on a desktop window there is no swipe-down. **The page behind it
 * does not scroll** while it is open - on iOS a sheet over a scrollable page scrolls the page, and
 * the sheet appears to drift. And the scrim is a `button`, so a tap outside dismisses it and a
 * screen reader is told it is dismissable rather than being handed a div with a click listener.
 */
export function Sheet({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = previous;
      window.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <>
      <button className="sheet-scrim" aria-label="Close" onClick={onClose} />
      <div className="sheet" role="dialog" aria-modal="true" aria-label={title}>
        <div className="sheet-grip" aria-hidden="true" />
        <div className="row" style={{ padding: `0 var(--sp-5) var(--sp-3)` }}>
          <strong className="grow" style={{ fontSize: 16, letterSpacing: "-0.01em" }}>
            {title}
          </strong>
          <Button variant="quiet" onClick={onClose} aria-label="Close">
            Done
          </Button>
        </div>
        <div className="sheet-body">{children}</div>
      </div>
    </>
  );
}
