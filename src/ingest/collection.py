"""Phase 1.2 — the chaos-corpus collection protocol, as data rather than prose.

The 260 sketches in Phase 1.2 have to be drawn by human hands; this module is everything
around that act which *can* be automated: what to draw, who drew it, under what conditions,
and whether the corpus collected so far actually satisfies the plan's targets.

    python -m src.ingest.collection --plan       # print the full drawing assignment
    python -m src.ingest.collection --progress   # measure the corpus against its targets
    python -m src.ingest.collection --sheet      # write the printable capture sheet

Every sketch is filed as:

    data/raw/chaos/<diagram_type>/<scribe_id>/<scenario>__<medium>__<condition>.jpg

The filename *is* the metadata. No sidecar file can drift out of sync with it, and
`parse_filename` turns it straight back into a manifest row (Phase 1.3.1).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from src.utils.config import ROOT

CHAOS = ROOT / "data" / "raw" / "chaos"

DIAGRAM_TYPES = ("flowchart", "wireframe", "state_machine", "er_diagram", "circuit")

# Phase 1.2.1 - 1.2.5 targets, straight from plan.md.
TARGETS: dict[str, int] = {
    "flowchart": 60,
    "wireframe": 60,
    "state_machine": 50,
    "er_diagram": 50,
    "circuit": 40,
}

MIN_SCRIBES = 8  # Phase 1.2.6
MIN_ADVERSE_FRACTION = 0.25  # Phase 1.2.7

# Phase 1.2.8 — drawing media. `medium` is one axis of style variation; `condition` is the
# capture axis. They are recorded separately because they fail differently: a marker on a
# whiteboard changes stroke width, while glare changes binarization.
MEDIA = ("pencil", "ballpoint", "marker", "whiteboard", "stylus")

# Phase 1.2.7 — capture conditions. `clean` is the control group.
CONDITIONS = (
    "clean",  # flat, even light, straight on
    "shadow",  # a hand or body shadow across the page
    "angle",  # photographed at 20-40 degrees off perpendicular
    "ruled",  # drawn on ruled or squared notebook paper
    "glare",  # whiteboard or glossy paper reflecting a light source
    "stain",  # coffee ring, smudge, or dirt over part of the drawing
    "crossedout",  # contains elements struck through or scribbled out
    "lowlight",  # dim room, high ISO noise
    "crop",  # part of the diagram runs off the edge of the frame
)
ADVERSE_CONDITIONS = tuple(c for c in CONDITIONS if c != "clean")

FILENAME_RE = re.compile(
    r"^(?P<scenario>[a-z0-9_]+)__(?P<medium>[a-z]+)__(?P<condition>[a-z]+)"
    r"(?:__(?P<index>\d+))?\.(?P<ext>jpg|jpeg|png)$"
)


@dataclass(frozen=True)
class Scenario:
    """One thing to draw, described well enough that two people draw the same diagram."""

    slug: str
    title: str
    brief: str
    must_contain: tuple[str, ...]


# --- Phase 1.2.1 flowcharts -------------------------------------------------------------
FLOWCHART_SCENARIOS = (
    Scenario(
        "login_check",
        "Login validation",
        "Read username and password, check them, branch to success or retry.",
        ("start", "decision", "loop back", "end"),
    ),
    Scenario(
        "age_gate",
        "Age gate",
        "Ask for age; if 18 or over allow entry, otherwise refuse.",
        ("decision on a comparison", "two branches"),
    ),
    Scenario(
        "grade_calc",
        "Grade calculator",
        "Take a mark and print A/B/C/F using a chain of comparisons.",
        ("three or more decisions", "one end"),
    ),
    Scenario(
        "atm_withdraw",
        "ATM withdrawal",
        "Card, PIN, balance check, dispense or decline, with a PIN retry loop.",
        ("loop", "nested decision", "io node"),
    ),
    Scenario(
        "bubble_sort",
        "Bubble sort",
        "Two nested loops with a swap in the middle.",
        ("nested loop", "back edge"),
    ),
    Scenario(
        "order_pipeline",
        "Order processing",
        "Receive order, check stock, take payment, ship or refund.",
        ("four or more processes", "two terminals"),
    ),
    Scenario(
        "fizzbuzz",
        "FizzBuzz",
        "Loop 1..n with divisibility decisions and three print branches.",
        ("loop", "three-way branch"),
    ),
    Scenario(
        "file_retry",
        "Download with retry",
        "Attempt a download, retry up to three times, then fail.",
        ("counter", "loop", "failure terminal"),
    ),
    Scenario(
        "triangle_type",
        "Triangle classifier",
        "Given three sides, decide equilateral, isosceles or scalene.",
        ("chained decisions",),
    ),
    Scenario(
        "password_reset",
        "Password reset",
        "Request reset, send email, verify token, set new password.",
        ("io node", "decision", "linear chain"),
    ),
)

# --- Phase 1.2.2 wireframes -------------------------------------------------------------
WIREFRAME_SCENARIOS = (
    Scenario(
        "login_form",
        "Login screen",
        "Title, email field, password field, submit button, forgot-password link.",
        ("two inputs", "one button", "one link"),
    ),
    Scenario(
        "signup_form",
        "Sign-up screen",
        "Four fields, a checkbox for terms, and a primary button.",
        ("four inputs", "checkbox"),
    ),
    Scenario(
        "dashboard",
        "Analytics dashboard",
        "Header, sidebar nav, four stat tiles, one chart area.",
        ("sidebar", "grid of tiles", "chart placeholder"),
    ),
    Scenario(
        "product_list",
        "Product listing",
        "Search bar, filter column, grid of product cards with images.",
        ("grid", "image placeholders"),
    ),
    Scenario(
        "profile_page",
        "User profile",
        "Avatar, name, bio block, tabbed section, edit button.",
        ("avatar circle", "tabs"),
    ),
    Scenario(
        "settings_page",
        "Settings",
        "List of labelled rows with toggles, grouped into sections.",
        ("repeated rows", "toggles"),
    ),
    Scenario(
        "checkout",
        "Checkout",
        "Order summary on the right, address form on the left, pay button.",
        ("two columns", "form"),
    ),
    Scenario(
        "chat_ui",
        "Chat window",
        "Conversation list, message bubbles, input box with a send button.",
        ("list", "bubbles", "input row"),
    ),
    Scenario(
        "mobile_feed",
        "Mobile feed",
        "Narrow frame, top bar, scrolling cards, bottom tab bar.",
        ("bottom nav", "cards"),
    ),
    Scenario(
        "data_table",
        "Admin table",
        "Toolbar, table with five columns and a pagination row.",
        ("table grid", "pagination"),
    ),
)

# --- Phase 1.2.3 state machines ---------------------------------------------------------
STATE_SCENARIOS = (
    Scenario(
        "traffic_light",
        "Traffic light",
        "Red, amber, green cycling on a timer.",
        ("cycle", "three states"),
    ),
    Scenario(
        "vending_machine",
        "Vending machine",
        "Idle, coin inserted, selection made, dispensing, refund.",
        ("self-loop on coin insert", "accepting state"),
    ),
    Scenario(
        "tcp_handshake",
        "TCP connection",
        "CLOSED, LISTEN, SYN_RCVD, ESTABLISHED, FIN_WAIT, CLOSED.",
        ("labelled transitions", "return to start"),
    ),
    Scenario(
        "elevator",
        "Elevator",
        "Idle, moving up, moving down, doors open, with request transitions.",
        ("bidirectional transitions",),
    ),
    Scenario(
        "turnstile",
        "Turnstile",
        "Locked and unlocked, with coin and push transitions.",
        ("two states", "two self-loops"),
    ),
    Scenario(
        "order_state",
        "Order lifecycle",
        "Placed, paid, shipped, delivered, cancelled.",
        ("terminal state", "branch to cancelled"),
    ),
    Scenario(
        "media_player",
        "Media player",
        "Stopped, playing, paused, with play/pause/stop transitions.",
        ("three states", "cycle"),
    ),
    Scenario(
        "regex_dfa",
        "DFA for (ab)+",
        "Start, q1, accepting state, with a and b transitions and a dead state.",
        ("double-circle accepting state", "dead state"),
    ),
    Scenario(
        "door_lock",
        "Smart lock",
        "Locked, unlocking, unlocked, auto-relock after timeout.",
        ("timeout transition",),
    ),
    Scenario(
        "game_states",
        "Game loop",
        "Menu, playing, paused, game over, back to menu.",
        ("return edge to start",),
    ),
)

# --- Phase 1.2.4 ER diagrams ------------------------------------------------------------
ER_SCENARIOS = (
    Scenario(
        "blog",
        "Blog",
        "User, Post, Comment with one-to-many relationships and key attributes.",
        ("three entities", "two relationships", "cardinality marks"),
    ),
    Scenario(
        "school",
        "School",
        "Student, Course, Enrolment as a many-to-many with a join entity.",
        ("many-to-many", "join entity"),
    ),
    Scenario(
        "ecommerce",
        "E-commerce",
        "Customer, Order, OrderLine, Product.",
        ("four entities", "composite relationship"),
    ),
    Scenario(
        "library",
        "Library",
        "Member, Book, Loan, with due-date attributes.",
        ("attribute ovals", "one-to-many"),
    ),
    Scenario(
        "hospital",
        "Hospital",
        "Patient, Doctor, Appointment, Prescription.",
        ("four entities", "chained relationships"),
    ),
    Scenario(
        "bank",
        "Bank",
        "Customer, Account, Transaction, with a self-referencing transfer.",
        ("self-relationship",),
    ),
    Scenario(
        "cinema",
        "Cinema booking",
        "Film, Screening, Seat, Booking.",
        ("many-to-many", "weak entity"),
    ),
    Scenario(
        "inventory",
        "Warehouse inventory",
        "Product, Warehouse, Stock with quantity on the relationship.",
        ("relationship attribute",),
    ),
    Scenario(
        "social",
        "Social network",
        "User with a self-referencing follows relationship, plus Post.",
        ("self-relationship", "cardinality both ways"),
    ),
    Scenario(
        "hr",
        "HR",
        "Employee, Department, Role, with a manager self-link.",
        ("self-relationship", "three entities"),
    ),
)

# --- Phase 1.2.5 circuits ---------------------------------------------------------------
CIRCUIT_SCENARIOS = (
    Scenario(
        "series_resistors",
        "Series resistors",
        "Battery and three resistors in a single loop.",
        ("closed loop", "battery symbol", "three resistors"),
    ),
    Scenario(
        "parallel_resistors",
        "Parallel resistors",
        "Battery with two resistor branches in parallel.",
        ("junction nodes", "parallel branches"),
    ),
    Scenario(
        "rc_filter",
        "RC low-pass filter",
        "Source, resistor, capacitor to ground, output node.",
        ("capacitor symbol", "ground symbol"),
    ),
    Scenario(
        "led_circuit",
        "LED with series resistor",
        "Battery, resistor, LED, closed loop.",
        ("diode symbol",),
    ),
    Scenario(
        "voltage_divider",
        "Voltage divider",
        "Two resistors between supply and ground with a tapped midpoint.",
        ("labelled midpoint",),
    ),
    Scenario(
        "and_gate",
        "AND gate",
        "Two inputs into an AND gate, one output.",
        ("gate symbol", "two inputs"),
    ),
    Scenario(
        "half_adder",
        "Half adder",
        "XOR and AND gates producing sum and carry.",
        ("two gates", "two outputs"),
    ),
    Scenario(
        "sr_latch", "SR latch", "Two cross-coupled NOR gates.", ("feedback loop", "two gates")
    ),
    Scenario(
        "switch_lamp",
        "Switch and lamp",
        "Battery, switch, lamp in series.",
        ("switch symbol", "lamp symbol"),
    ),
    Scenario(
        "transistor_switch",
        "Transistor switch",
        "Base resistor, NPN transistor, load, supply rail.",
        ("transistor symbol", "supply rail"),
    ),
)

SCENARIOS: dict[str, tuple[Scenario, ...]] = {
    "flowchart": FLOWCHART_SCENARIOS,
    "wireframe": WIREFRAME_SCENARIOS,
    "state_machine": STATE_SCENARIOS,
    "er_diagram": ER_SCENARIOS,
    "circuit": CIRCUIT_SCENARIOS,
}


def assignments() -> list[dict]:
    """Spread each type's target across its scenarios and the ≥8 scribes.

    Every scenario is drawn by several people (that is what makes style variation
    measurable), and every scribe draws across all five types (so no type is confounded
    with one person's handwriting).
    """
    out: list[dict] = []
    for dtype, target in TARGETS.items():
        scen = SCENARIOS[dtype]
        for i in range(target):
            s = scen[i % len(scen)]
            scribe = f"scribe{(i % MIN_SCRIBES) + 1:02d}"
            # A quarter of every type is captured adversely (Phase 1.2.7), cycling through
            # the adverse conditions so no single condition dominates.
            adverse = (i % 4) == 3
            condition = (
                ADVERSE_CONDITIONS[(i // 4) % len(ADVERSE_CONDITIONS)] if adverse else "clean"
            )
            out.append(
                {
                    "diagram_type": dtype,
                    "index": i + 1,
                    "scenario": s.slug,
                    "title": s.title,
                    "scribe_id": scribe,
                    "medium": MEDIA[i % len(MEDIA)],
                    "condition": condition,
                    "adverse": adverse,
                    "filename": f"{s.slug}__{MEDIA[i % len(MEDIA)]}__{condition}__{i + 1:03d}.jpg",
                    "path": f"data/raw/chaos/{dtype}/{scribe}/"
                    f"{s.slug}__{MEDIA[i % len(MEDIA)]}__{condition}__{i + 1:03d}.jpg",
                }
            )
    return out


def parse_filename(path: Path) -> dict | None:
    """Recover the metadata a collected file encodes in its own name and location."""
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    try:
        scribe = path.parent.name
        dtype = path.parent.parent.name
    except Exception:  # noqa: BLE001
        return None
    if dtype not in DIAGRAM_TYPES:
        return None
    return {
        "diagram_type": dtype,
        "scribe_id": scribe,
        "scenario": m["scenario"],
        "medium": m["medium"],
        "condition": m["condition"],
        "adverse": m["condition"] in ADVERSE_CONDITIONS,
        "path": str(path.relative_to(ROOT)),
    }


def collected() -> list[dict]:
    """Every valid file currently sitting in data/raw/chaos/."""
    if not CHAOS.is_dir():
        return []
    rows = []
    for p in sorted(CHAOS.rglob("*")):
        if p.is_file() and (row := parse_filename(p)):
            rows.append(row)
    return rows


def progress() -> dict:
    """Measure the corpus against every Phase 1.2 target. This is the acceptance check."""
    rows = collected()
    by_type = Counter(r["diagram_type"] for r in rows)
    scribes = {r["scribe_id"] for r in rows}
    media = {r["medium"] for r in rows}
    conditions = Counter(r["condition"] for r in rows)
    adverse = sum(1 for r in rows if r["adverse"])

    per_type = {
        t: {"collected": by_type.get(t, 0), "target": n, "complete": by_type.get(t, 0) >= n}
        for t, n in TARGETS.items()
    }
    total, target_total = len(rows), sum(TARGETS.values())

    return {
        "total_collected": total,
        "total_target": target_total,
        "per_type": per_type,
        "scribes": sorted(scribes),
        "n_scribes": len(scribes),
        "media_used": sorted(media),
        "conditions": dict(conditions),
        "adverse_count": adverse,
        "adverse_fraction": round(adverse / total, 3) if total else 0.0,
        "checks": {
            "1.2.1_flowcharts": per_type["flowchart"]["complete"],
            "1.2.2_wireframes": per_type["wireframe"]["complete"],
            "1.2.3_state_machines": per_type["state_machine"]["complete"],
            "1.2.4_er_diagrams": per_type["er_diagram"]["complete"],
            "1.2.5_circuits": per_type["circuit"]["complete"],
            "1.2.6_min_scribes": len(scribes) >= MIN_SCRIBES,
            "1.2.7_adverse_fraction": (adverse / total if total else 0) >= MIN_ADVERSE_FRACTION,
            "1.2.8_all_media": set(MEDIA) <= media,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", action="store_true", help="print the full drawing assignment")
    ap.add_argument("--progress", action="store_true", help="measure the corpus against targets")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    if args.plan:
        rows = assignments()
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for dtype in DIAGRAM_TYPES:
                subset = [r for r in rows if r["diagram_type"] == dtype]
                print(f"\n{dtype}  ({len(subset)} sketches)")
                for r in subset[:6]:
                    print(
                        f"  {r['index']:>3}  {r['scribe_id']}  {r['scenario']:<20} "
                        f"{r['medium']:<11} {r['condition']}"
                    )
                if len(subset) > 6:
                    print(f"  ... {len(subset) - 6} more")
        return 0

    report = progress()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"collected {report['total_collected']} of {report['total_target']} sketches")
    for t, s in report["per_type"].items():
        print(f"  {'OK ' if s['complete'] else '   '} {t:<15} {s['collected']:>3} / {s['target']}")
    print(f"  scribes: {report['n_scribes']} (need {MIN_SCRIBES})")
    print(f"  adverse: {report['adverse_fraction']:.0%} (need {MIN_ADVERSE_FRACTION:.0%})")
    print(f"  media:   {report['media_used']}")
    print("-" * 60)
    for name, ok in report["checks"].items():
        print(f"  {'PASS' if ok else 'PENDING'}  {name}")
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
