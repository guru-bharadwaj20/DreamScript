# Annotation Guide

Phase 2.2.3. How to label a hand-drawn diagram for DreamScript, and - more importantly - what
to do when the drawing does not cooperate.

The rules below are written so that two people following them independently produce the same
labels. Where that is impossible, the rule says *record the doubt* instead of *pick one*: the
IR has fields for uncertainty (`unresolved_edges`, `crossed_out`, `low_conf_text`) precisely so
an annotator never has to guess silently. Phase 2.2.4 measures whether the rules achieve that.

## The one-sentence version

**Label what is on the paper, not what the writer meant.** A squashed circle that is obviously
a decision is still a squashed circle: `shape: ellipse`, `semantic_role: decision`. Those two
fields exist separately for exactly this reason, and collapsing them is the single most common
way to poison this dataset.

---

## 1. What gets a box

| Draw a box around | Do not box |
| :--- | :--- |
| Every closed outline that holds meaning: boxes, diamonds, circles, ovals | The page, the margin, the ruled lines |
| Text that stands alone - a title, an edge label, a caption | Text *inside* a shape (it belongs to that shape's `text`) |
| A component symbol in a circuit, even a squiggly one | Doodles, smudges, the writer's initials |
| A pool or lane that visually contains other shapes | Shading or hatching inside a shape |

The box is the **tightest axis-aligned rectangle containing all ink of that element**,
excluding the arrows that touch it. If the writer's pen overshot a corner, the overshoot is
part of the shape and goes inside the box.

## 2. Shape before role

Pick `shape` by looking at the outline only, with the text covered:

| If the outline has | Label it |
| :--- | :--- |
| Four corners, sides roughly parallel to the page | `rectangle` |
| Four corners with visibly rounded ones | `rounded-rect` |
| Four corners pointing up/down/left/right | `diamond` |
| Four corners, one pair of sides slanted | `parallelogram` |
| No corners, roughly as wide as tall | `circle` |
| No corners, clearly wider than tall (or taller than wide) | `ellipse` |
| Two closed curves, one inside the other | `double-circle` |
| More than four corners, roughly as wide as tall | `octagon` |
| A stroke with an arrowhead | `arrow` |
| A stroke without one | `line` |
| Words with no outline around them | `text-block` |
| None of the above, and you are not guessing | `freeform` |

Then pick `semantic_role` using the text, the diagram type and the neighbours. The role
vocabulary allowed in each diagram type is in `src/ir/vocab.py::ROLES_BY_TYPE`; the interface
does not restrict it, so it is on the annotator not to put a `wire` in a flowchart.

**When shape and role disagree, both are still right.** `shape: rectangle` with
`semantic_role: decision` is a legitimate, common labelling of a writer who could not be
bothered to draw a diamond.

---

## 3. The hard cases

These are the rules that stop two annotators diverging. Each one names the failure it prevents.

### 3.1 A broken arrow - the arrow does not reach its box

Very common: the pen stops 5 mm short, or the arrowhead lands beside the box rather than on it.

> **Rule.** If exactly one shape is plainly the nearest thing in the arrow's direction of
> travel, connect it and mark the relation `flow`. If two shapes are comparably close, or the
> arrow points into empty space, connect what you can and use the relation type `uncertain`.
> Never connect an arrow to a shape it is *pointing away from*.

*Why:* Phase 10.1.4 is a model that learns to repair broken arrows. It can only learn from
examples where a human said "this one was repairable and this one was not". Guessing removes
the second class from the data entirely.

### 3.2 Overlapping or nested boxes

> **Rule.** Box each shape at its own full extent, even where they overlap. If one shape
> visually contains another *and the containment is meaningful* (a lane holding tasks, a card
> holding buttons), label the outer one `container` and draw a `contains` relation. If the
> overlap is accidental - the writer ran out of room - draw no relation.

*Why:* Overlap is geometry, containment is semantics. An IoU-based detector metric in Phase 9
needs the first; a code generator in Phase 12 needs the second.

### 3.3 Crossed-out content

> **Rule.** Label the crossed-out element normally - shape, role, text and all - and then tick
> `crossed-out` on it. Do not skip it and do not label the scribble as its own shape.

*Why:* A retracted box is evidence about how people edit on paper, and it is the only signal
that teaches a model to ignore scribble rather than read it. Dropping it teaches nothing;
labelling the scribble as a shape teaches the opposite of what is wanted.

### 3.4 Illegible text

> **Rule.** Transcribe your best reading and tick `text-uncertain`. Type nothing at all only
> when there is genuinely no text. Never type `???` or `illegible` - those become the node's
> text and a model will learn to output them.

*Why:* `low_conf_text` lets Phase 13 ask the user instead of inventing a variable name. A
placeholder string destroys that option, because nothing downstream can tell it from a reading.

### 3.5 One arrow, several branches

A single line that splits, or one arrowhead serving two boxes.

> **Rule.** One relation per source-target pair. A line from A that splits to B and C is two
> relations, A→B and A→C, not one.

### 3.6 An arrow with no source: start markers

Standard notation in state machines - a short arrow from nowhere into the initial state.

> **Rule.** Do not draw a relation. Label the target state `initial-state`. If the marker
> stroke is prominent, box it as `arrow` with role `transition` and leave it unconnected; the
> converter records it in `unresolved_edges` with the reason `no-source`.

### 3.7 Self-loops

> **Rule.** A relation from a region to itself. Label it exactly as any other transition,
> including its label text.

### 3.8 Text that could belong to two things

An `a,b` sitting between two transitions.

> **Rule.** Attach it to the element whose ink it is *closest* to, measured from the text's
> centre to the nearest ink of each candidate - not to the nearest bounding box, which is
> biased towards large shapes. If it is genuinely equidistant, attach it to neither and box it
> as a standalone `text-block` with role `unknown`.

### 3.9 A shape you cannot name

> **Rule.** `freeform`, with `shape-uncertain` ticked. Do not stretch the vocabulary: a
> hexagon is not an octagon, and a cylinder is not an ellipse.

*Why:* The vocabulary is frozen (Phase 2.1.4). `freeform` counts are the evidence for whether
it was frozen too narrowly, and forcing bad fits destroys that evidence.

### 3.10 Two diagrams on one page

> **Rule.** Label both, in one annotation. Do not draw relations between them. The page is the
> unit of work; a diagram is not.

---

## 4. Order of work

Doing it in this order roughly halves the time per image and, more importantly, makes two
annotators produce the same boxes:

1. **Boxes first**, all of them, shape labels only. Do not think about roles yet.
2. **Roles**, in a second pass over the same boxes. By now the whole diagram is visible, which
   is what a role depends on.
3. **Text**, third. Zoom in; do not transcribe from the overview.
4. **Relations**, last. Every arrow, then every containment.
5. **Flags**, on the way out: crossed-out, text-uncertain, shape-uncertain.

Pre-annotations from `src.ir.labelstudio` are the converter's output, not truth. **Correct them
by the rules above; do not accept them because they are there.** For hdBPMN in particular the
pre-annotated *shape* is a BPMN drawing convention rather than an observation, and it is wrong
often enough to matter - see `reports/annotator_agreement.md`.

## 5. What to do when a rule does not cover it

Do not invent a convention. Flag the region as uncertain, finish the image, and record the case
in the issue log. A rule added after the fact is cheap; a dataset labelled two different ways is
not recoverable.

## 6. Definitions of the shared terms

| Term | Means |
| :--- | :--- |
| **Element** | Anything that gets a box: a shape, a standalone text block, an arrow. |
| **Ink** | The pen strokes themselves, not the bounding box. |
| **Adjacent** | Nearest by ink-to-ink distance, not by box-to-box distance. |
| **Meaningful containment** | The inner element is *part of* the outer one in the diagram's own terms, not merely drawn inside it. |
| **Ground truth** | Copied from an annotation a human made. Anything computed is not ground truth, however confident. |
