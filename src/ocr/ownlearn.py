"""Learn which text region an element owns, instead of ranking them by a hand-written cost.

    python -m src.ocr.ownlearn --build     # recognise every candidate, write the training table
    python -m src.ocr.ownlearn --train     # fit the regressor the assignment then uses

`src.ocr.ownership` decides ownership with a cost function written by hand: distance, an "inside"
bonus, a "below" bonus, a centring penalty. Every term in it was justified by a measurement, and
it is still a guess about how those terms trade off against each other. This module replaces the
trade-off with a fitted one, and 9.1.2's missing text-region annotation is the reason it can be
fitted at all without ever labelling a text region by hand.

## Where the training signal comes from

The corpus does not say which pixels are a label. It says what each element's label *reads*. So
the target is manufactured by reading: for every element, each candidate crop is recognised and
scored against that element's transcript, and **the model is trained to predict a candidate's
CER from geometry alone**. At assembly time the transcript is gone and only the prediction is
left, which is exactly the quantity the assignment wants to minimise.

That makes the target honest but noisy in one specific way, and it is worth naming: a candidate
is punished both for being the wrong region *and* for being a region the recogniser reads badly.
The second is not ownership. It is tolerable here because the recogniser's error on a correct
crop is small next to the gap between a right and a wrong region - 0.065 against roughly 1.0 -
so the target is dominated by ownership, not by legibility.

## Why an anchor and not a set

`ownership.match` gives each element up to three slots and lets the assignment pick a set of
lines. That makes the cost of a *set* the sum of the costs of its members, which is wrong - a
label is a run of stacked lines, and the second line is only worth having if it stacks under the
first. Here each element picks one **anchor** line and the block is grown from it deterministically
by `anchored`. One choice per element, one slot per element, and the Hungarian is over anchors.

The measured ceiling of this design is what motivated it: choosing the anchor perfectly, with the
label in hand, gives Flow 0.2450 and Event 0.1707 against the hand-written cost's 0.4191 and
0.2935. That gap is what this module is trying to close, and it is also the most it can win.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.ocr import ownership as ow
from src.utils.config import ROOT

TABLE = ROOT / "experiments" / "ocr" / "ownlearn" / "candidates.parquet"
MODEL = ROOT / "experiments" / "ocr" / "ownlearn" / "ranker.joblib"

#: How many candidates an element is allowed, ranked by the hand-written cost. The learned model
#: re-ranks inside this shortlist rather than replacing the reach rules, so a candidate the
#: geometry says is out of reach never becomes reachable by being scored well. Raised from 8
#: because the shortlist, not the ranker, had become the ceiling: the hand-written cost decides
#: what the model is even allowed to consider, and it is the component measured worst
#: (pick 0.4329 against the ranker's 0.2736 and an oracle of 0.1178).
TOP_K = 14

ROLES = ("internal", "external", "container", "edge")


def variants(seed: list[int], others: list[list[int]]) -> list[list[int]]:
    """The blocks an anchor can grow into: itself, and the first one or two lines stacked under it.

    `anchored` takes the whole stack, which is right for a three-line activity label and wrong
    for a one-line branch condition written directly above another one. Offering the truncations
    as separate candidates lets the choice be made per element instead of by a single rule.
    """
    below = sorted([b for b in others if b[1] >= seed[1] and b != seed], key=lambda b: b[1])
    chain = ow.prune([seed] + below)
    return [ow.with_rotation(ow.union(chain[:k])) for k in range(1, len(chain) + 1)]


def anchored(seed: list[int], others: list[list[int]]) -> list[int]:
    """The label block grown down from one anchor line.

    Only lines at or below the anchor may join it, and only through `ownership.prune`, so the
    block is a function of the anchor alone. Two elements choosing different anchors therefore
    cannot be handed the same block by accident.
    """
    below = sorted([b for b in others if b[1] >= seed[1] and b != seed], key=lambda b: b[1])
    return ow.with_rotation(ow.union(ow.prune([seed] + below)))


def shortlist(element: dict, lines: list[list[int]], page_w: float, k: int = TOP_K):
    """`(index, hand-written cost)` for the candidates this element may claim, best first."""
    row = ow.cost_row(element, lines, page_w, reach=True)
    order = [j for j in np.argsort(row) if row[j] < ow.BIG]
    return [(int(j), float(row[j])) for j in order[:k]]


def features(
    element: dict,
    line: list[int],
    lines: list[list[int]],
    page_shape,
    rank: int,
    cost: float,
) -> list[float]:
    """Geometry only. Nothing here may depend on what the element's label says."""
    height, width = page_shape[:2]
    role = ow.kind_of(element)
    x, y, w, h = (float(v) for v in element["bbox"])
    bx, by, bw, bh = (float(v) for v in line[:4])
    ecx, ecy = x + w / 2.0, y + h / 2.0
    lcx, lcy = bx + bw / 2.0, by + bh / 2.0
    size = max(w, h, 1.0)
    poly = element.get("polyline")
    if poly is not None:
        distance, at = ow.polyline_foot(poly, line)
    else:
        distance, at = ow.gap(element["bbox"], line), 0.0
    strip = ow.header_strip([x, y, w, h])
    near = sorted(ow.gap(element["bbox"], b) for b in lines)
    return [
        distance / width,
        distance / size,
        (lcx - ecx) / width,
        (lcy - ecy) / width,
        (lcx - ecx) / size,
        (lcy - ecy) / size,
        abs(lcx - ecx) / width,
        float(ow.inside(element["bbox"], line)),
        float(by >= y + h * 0.6),
        ow.overlap_fraction(line, [x, y, w, h]),
        ow.overlap_fraction(line, strip),
        bw / width,
        bh / width,
        bw / max(1.0, bh),
        w / width,
        h / width,
        w / max(1.0, h),
        (w * h) / max(1.0, width * height),
        at,
        float(rank),
        cost / width,
        (near[1] if len(near) > 1 else 0.0) / width,
        float(len(lines)),
        *[1.0 if role == r else 0.0 for r in ROLES],
    ]


FEATURES = [
    "dist_page",
    "dist_size",
    "dx_page",
    "dy_page",
    "dx_size",
    "dy_size",
    "adx_page",
    "inside",
    "below",
    "overlap_box",
    "overlap_strip",
    "line_w",
    "line_h",
    "line_aspect",
    "el_w",
    "el_h",
    "el_aspect",
    "el_area",
    "arc_at",
    "rank",
    "hand_cost",
    "next_line",
    "n_lines",
    *[f"role_{r}" for r in ROLES],
]

#: Added by reading the candidate rather than by measuring it. Geometry alone picks the right
#: crop at 0.4676 against a 0.1422 oracle, so the decoder's own mean token log-probability is
#: brought in as evidence: a crop holding a written phrase is easier to read than a crop holding
#: half an arrow, and knowing that needs no label, so it is available at inference too.
READING = ["conf", "ntok"]
MODEL_FEATURES = FEATURES + READING


def build(limit: int | None = None, out: Path = TABLE) -> dict:
    """Recognise every shortlisted candidate on the training pages and store its CER."""
    import cv2
    import pandas as pd
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    from src.detect.dataset import image_for
    from src.ocr.labelcrops import cached_lines, crop, elements_of
    from src.ocr.metrics import edit_distance, normalise
    from src.ocr.s3 import CHECKPOINT, predict
    from src.ocr.textcrops import load_index
    from src.parse.sequences import load_ir
    from src.preprocess.exif import load

    splits = load_index()[["page", "split", "scribe"]].drop_duplicates("page")
    assignment = {r["page"]: (r["split"], r["scribe"]) for _, r in splits.iterrows()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = TrOCRProcessor.from_pretrained(CHECKPOINT)
    model = VisionEncoderDecoderModel.from_pretrained(CHECKPOINT).to(device)

    scratch = out.parent / "crops"
    scratch.mkdir(parents=True, exist_ok=True)
    rows, paths, pages = [], [], 0
    for diagram in load_ir(["hdbpmn"], limit=limit):
        meta = assignment.get(diagram["id"])
        if meta is None or meta[0] != "train":
            continue
        path = image_for(diagram)
        if not path or not Path(path).is_file():
            continue
        image = load(Path(path), grayscale=True)
        if image is None:
            continue
        lines = [list(map(int, b[:4])) for b in (cached_lines(diagram["id"]) or [])]
        if not lines:
            continue
        for element in elements_of(diagram):
            text = normalise(element["text"])
            if not text or ow.kind_of(element) == "internal":
                continue
            for rank, (j, cost) in enumerate(shortlist(element, lines, image.shape[1])):
                for depth, block in enumerate(variants(lines[j], lines)):
                    patch = crop(image, block)
                    if patch is None:
                        continue
                    name = scratch / f"{diagram['id']}__{element['id']}__{rank}_{depth}.png"
                    cv2.imwrite(str(name), patch)
                    paths.append(str(name))
                    rows.append(
                        {
                            "page": diagram["id"],
                            "scribe": meta[1],
                            "element_id": str(element["id"]),
                            "crop": str(name),
                            "depth": depth,
                            "element": str(element["id"]).split("_")[0],
                            "role": ow.kind_of(element),
                            "truth": element["text"],
                            **dict(
                                zip(
                                    FEATURES,
                                    features(element, lines[j], lines, image.shape, rank, cost),
                                    strict=True,
                                )
                            ),
                        }
                    )
        pages += 1
        if pages % 50 == 0:
            print(f"[ownlearn] {pages} pages, {len(rows)} candidates", flush=True)

    print(f"[ownlearn] recognising {len(paths)} candidate crops", flush=True)
    predictions = predict(model, proc, paths, device, beams=1)
    frame = pd.DataFrame(rows)
    frame["cer"] = [
        min(1.5, edit_distance(normalise(t), normalise(p)) / max(1, len(normalise(t))))
        for t, p in zip(frame["truth"], predictions, strict=True)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    frame["prediction"] = predictions
    frame.drop(columns=["truth"]).to_parquet(out)
    summary = {
        "pages": pages,
        "candidates": len(frame),
        "elements": int(frame.groupby(["page", "element"]).ngroups),
        "mean_cer": round(float(frame["cer"].mean()), 4),
        "oracle_cer": round(float(frame.groupby(["page", "element_id"])["cer"].min().mean()), 4),
    }
    (out.parent / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train(table: Path = TABLE, model_path: Path = MODEL, dev_writers: int = 15) -> dict:
    """Fit the candidate-CER regressor, holding the S3 dev writers out of it.

    The dev writers are the ones `src.ocr.s3` selects its epoch on. If the ranker learned from
    them, the epoch choice and the ownership rule would both have seen the same handwriting and
    dev would stop being an honest check on either.
    """
    import joblib
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor

    frame = pd.read_parquet(table)
    names = sorted(frame["scribe"].dropna().unique())
    rng = np.random.default_rng(42)
    held = set(rng.permutation(names)[:dev_writers].tolist())
    fit = frame[~frame["scribe"].isin(held)]
    check = frame[frame["scribe"].isin(held)]

    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=6, l2_regularization=1.0, random_state=42
    )
    model.fit(fit[MODEL_FEATURES].to_numpy(), fit["cer"].to_numpy())
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": MODEL_FEATURES}, model_path)

    def per_element(part, column):
        """Mean CER of the candidate each rule would pick, one pick per element."""
        picked = part.sort_values(column).groupby(["page", "element_id"], sort=False).head(1)
        return round(float(picked["cer"].mean()), 4)

    check = check.copy()
    check["hat"] = model.predict(check[MODEL_FEATURES].to_numpy())
    check["hand"] = check["rank"] * 10.0 + check["depth"]
    return {
        "rows": len(frame),
        "fit_rows": len(fit),
        "check_rows": len(check),
        "held_writers": len(held),
        "mae": round(float(np.mean(np.abs(check["hat"] - check["cer"]))), 4),
        # The three numbers that matter, all one pick per element on writers the fit never saw.
        "hand_cost_pick": per_element(check, "hand"),
        "learned_pick": per_element(check, "hat"),
        "oracle_pick": per_element(check, "cer"),
    }


def reader(checkpoint=None, replicas: int = 1):
    """`replicas` independent copies of the S3 checkpoint, each on its own CUDA stream."""
    global _READER
    if _READER is None or len(_READER[0]) < replicas:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        from src.ocr.s3 import CHECKPOINT

        path = checkpoint or CHECKPOINT
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        proc = TrOCRProcessor.from_pretrained(path)
        models = [
            VisionEncoderDecoderModel.from_pretrained(path).to(device).eval()
            for _ in range(replicas)
        ]
        streams = [torch.cuda.Stream() for _ in models] if device.type == "cuda" else [None]
        _READER = (models, proc, device, streams)
    return _READER


def plan_workers(count: int, batch: int = 256) -> tuple[int, int]:
    """How many replicas and how large a batch this card can actually use.

    **The answer is one worker with a big batch, and that is a measured reversal of the obvious
    guess.** Concurrent replicas on separate streams do help when the batch is small - at a fixed
    batch of 128, one worker reaches 65.7 crops/s, two 79.4 and three 89.0 - because a decode
    step is a tiny kernel and the SMs sit idle between them. But that is the wrong baseline. A
    *single* worker at batch 256 reaches **107.0 crops/s**, beating three workers at batch 128,
    because a bigger batch fills the same idle SMs without paying for a second copy of the
    weights or a second activation arena.

    Stacking the two is worse than either. A worker needs ~9.86 GiB at batch 256, so the three
    this function first returned wanted ~30 GiB on a 24 GiB card; measured end to end, that took
    throughput from **29.9 crops/s to 7.5** - a 4x regression from over-subscribing the card,
    the same cliff batch 1024 falls off at 1.2 crops/s. Filling VRAM is not the objective and
    twice here it has been actively harmful.

    So the default is one worker, and `workers` stays a parameter only because the shared queue
    in `read_confidence` is correct for any number and a future card with real headroom could
    use it.
    """
    import torch

    if not torch.cuda.is_available():
        return 1, min(batch, 32)
    free, _ = torch.cuda.mem_get_info()
    # Measured cost of one worker at batch 256, plus margin. Crossing the card is a 4-90x
    # penalty while a spare worker is worth at most tens of percent, so the asymmetry says
    # round down hard.
    per_worker_gib = 9.9 * (batch / 256.0)
    room = max(1, min(3, int((free / 2**30 - 2.0) // per_worker_gib)))
    # A batch this size already fills the SMs on its own, and the measurement says one worker at
    # 256 beats three at 128. Extra replicas are only allowed to help where the batch is too
    # small to saturate by itself.
    workers = 1 if batch >= 192 else room
    if count < batch * 2:
        workers = 1
    return workers, batch


def read_confidence(patches, batch: int = 256, workers: int | None = None):
    """Score each crop by how readable the decoder finds it. No label is involved.

    **`use_cache=True` is passed explicitly because the checkpoint ships with it off.** The
    decoder config carries `use_cache: false`, so every generation step was recomputing the whole
    prefix - O(n^2) attention instead of O(n) - and turning it on measured **49.6 -> 107.0
    crops/s at batch 256**, an exact 2.16x for no change of method. It is not bit-identical: the
    cached and uncached paths take different bf16 kernels and **2 of 512 transcripts differed**,
    which is numerical noise rather than a different computation, and is why `src.ocr.s3.evaluate`
    is deliberately left on the shipped path for the headline number.

    **Work is pulled from one shared queue, not dealt out in advance.** Static shards
    (`patches[i::n]`) finish at different times because crops differ in width and in how many
    tokens they decode, so the last worker runs alone while the others idle. A worker here takes
    the next unclaimed slice whenever it becomes free, which is the same thing as handing a
    finished worker someone else's remaining work, and no slice can be taken twice because the
    index is handed out under a lock.
    """
    import threading

    import torch
    from PIL import Image

    from src.ocr.s3 import MAX_LENGTH

    if not patches:
        return []
    if workers is None:
        workers, batch = plan_workers(len(patches), batch)
    models, proc, device, streams = reader(replicas=workers)
    out: list[tuple[float, float] | None] = [None] * len(patches)
    cursor = 0
    lock = threading.Lock()

    def claim() -> tuple[int, int] | None:
        """The whole scheduler: one lock, one cursor, every slice handed out exactly once."""
        nonlocal cursor
        with lock:
            if cursor >= len(patches):
                return None
            start = cursor
            cursor = min(cursor + batch, len(patches))
            return start, cursor

    def work(index: int) -> None:
        model = models[index % len(models)]
        stream = streams[index % len(streams)] if streams[0] is not None else None
        context = torch.cuda.stream(stream) if stream is not None else _null()
        with context:
            while True:
                claimed = claim()
                if claimed is None:
                    return
                start, stop = claimed
                images = [Image.fromarray(p).convert("RGB") for p in patches[start:stop]]
                px = proc(images=images, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                    ):
                        gen = model.generate(
                            px,
                            max_length=MAX_LENGTH,
                            num_beams=1,
                            use_cache=True,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )
                    scores = model.compute_transition_scores(
                        gen.sequences, gen.scores, normalize_logits=True
                    )
                seq = gen.sequences[:, 1:]
                mask = (seq != proc.tokenizer.pad_token_id) & (seq != proc.tokenizer.eos_token_id)
                summed = scores.masked_fill(~mask, 0.0).float()
                n = mask.sum(1).clamp(min=1)
                means = (summed.sum(1) / n).cpu().numpy().tolist()
                counts = n.cpu().numpy().tolist()
                for offset, pair in enumerate(zip(means, counts, strict=True)):
                    out[start + offset] = pair

    threads = [threading.Thread(target=work, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if device.type == "cuda":
        torch.cuda.synchronize()
    missing = [i for i, value in enumerate(out) if value is None]
    if missing:
        raise RuntimeError(f"{len(missing)} crops were never scored, first at {missing[0]}")
    return out


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def assign(image, elements: list[dict], lines: list[list[int]]) -> dict[str, list[int]]:
    """Ownership by predicted CER, using both the geometry and how readable each candidate is.

    The hand-written cost still draws the shortlist - reach rules are geometry and stay - but the
    ordering inside it is the model's. Each element scores every anchor it can reach at every
    depth `variants` offers; `linear_sum_assignment` then hands out anchors so no two elements
    are given the same one. Candidate blocks are deduplicated across elements before they are
    read, so a page costs one pass over its distinct blocks rather than one per element.
    """
    from scipy.optimize import linear_sum_assignment

    from src.ocr.labelcrops import crop

    bundle = ranker()
    if bundle is None:
        return ow.blocks(image.shape, elements, lines)
    model = bundle["model"]

    page_shape = image.shape
    height, width = page_shape[:2]
    page_area = float(height * width)
    groups: dict[str, list[dict]] = {"container": [], "external": [], "edge": [], "internal": []}
    for element in elements:
        groups[ow.kind_of(element)].append(element)

    out: dict[str, list[int]] = {}
    free = [list(map(int, b[:4])) for b in lines]
    taken: set = set()
    # Activities keep 9.3.1's inset box - it reads them at 0.0656 and nothing here beats that -
    # and reserve the lines they cover, exactly as in `ownership.blocks`.
    for element in groups["internal"]:
        out[element["id"]] = ow.with_rotation(ow.inset_box(element))
        if element["bbox"][2] * element["bbox"][3] < 0.20 * page_area:
            box = [float(v) for v in element["bbox"]]
            taken.update(tuple(b) for b in free if ow.overlap_fraction(b, box) >= ow.RESERVE)
    free = [b for b in free if tuple(b) not in taken]

    outside = groups["container"] + groups["external"] + groups["edge"]
    if outside and free:
        wanted: dict[tuple[int, int], list[int]] = {}
        rows: list[tuple[int, int, int, list[float]]] = []
        for i, element in enumerate(outside):
            for rank, (j, cost) in enumerate(shortlist(element, free, width)):
                options = variants(free[j], free)
                geometry = features(element, free[j], free, page_shape, rank, cost)
                for depth, block in enumerate(options):
                    wanted.setdefault((j, depth), block)
                    rows.append((i, j, depth, geometry))

        keys = list(wanted)
        patches, usable = [], []
        for key in keys:
            patch = crop(image, wanted[key])
            if patch is not None:
                patches.append(patch)
                usable.append(key)
        scored = dict(zip(usable, read_confidence(patches), strict=True)) if patches else {}

        cost = np.full((len(outside), len(free)), ow.BIG)
        best_depth: dict[tuple[int, int], int] = {}
        batch = [r for r in rows if (r[1], r[2]) in scored]
        if batch:
            matrix = np.asarray([r[3] + list(scored[(r[1], r[2])]) for r in batch], dtype=float)
            for (i, j, depth, _), value in zip(batch, model.predict(matrix), strict=True):
                if value < cost[i, j]:
                    cost[i, j] = float(value)
                    best_depth[(i, j)] = depth
        for i, j in zip(*linear_sum_assignment(cost), strict=True):
            if cost[i, j] < ow.BIG:
                out[outside[i]["id"]] = wanted[(j, best_depth[(i, j)])]

    for element in outside:
        if element["id"] in out:
            continue
        role = ow.kind_of(element)
        if role == "edge":
            out[element["id"]] = ow.with_rotation(ow.edge_fallback(element))
        elif role == "container":
            out[element["id"]] = ow.with_rotation(
                ow.header_strip([float(v) for v in element["bbox"]])
            )
        else:
            out[element["id"]] = ow.with_rotation(ow.below_box(element, (height, width)))
    return out


def best_crops(table: Path = TABLE) -> dict[tuple[str, str], str]:
    """`(page, element id) -> the candidate crop that actually reads as that element's label`.

    Training-time only. This is the one place the transcript is allowed to choose a crop, and it
    is what makes the fine-tuning set clean: a third of Flow crops and nearly half of Participant
    crops previously carried *another element's* text, and pairing those with this element's
    transcript trains the decoder to invent in-domain phrases rather than read the ink. Ties go
    to the candidate the geometry liked best, so the label breaks ties it did not create.
    """
    import pandas as pd

    frame = pd.read_parquet(table)
    frame = frame.sort_values(["cer", "rank", "depth"])
    first = frame.groupby(["page", "element_id"], sort=False).head(1)
    return {(r.page, r.element_id): r.crop for r in first.itertuples()}


_RANKER = None
_READER = None


def ranker():
    """The fitted model, or None when it has not been built - callers fall back to the cost."""
    global _RANKER
    if _RANKER is None:
        if not MODEL.is_file():
            return None
        import joblib

        _RANKER = joblib.load(MODEL)
    return _RANKER


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    if args.build:
        print(json.dumps(build(args.limit), indent=2))
        return 0
    if args.train:
        print(json.dumps(train(), indent=2))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
