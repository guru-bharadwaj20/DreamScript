"""Where a downloaded base checkpoint lives, decided once.

    from src.detect.pretrained import weights
    YOLO(weights("yolov8n-pose.pt"))

ultralytics downloads a bare weight name into the **current working directory**. Every training
command in this repo is run from the repo root, so four base checkpoints - `yolov8n.pt`,
`yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt`, ~90 MB - accumulated there. They are
gitignored by `*.pt`, so `git status` never mentioned them, and they were in the Docker build
context until a `.dockerignore` existed.

`weights()` returns a path under `models/pretrained/`, creating the directory. A name that is
already a path (a trained checkpoint under `experiments/`) is returned unchanged, so a caller can
pass either and does not have to know which it has.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.config import ROOT

#: Downloaded, not produced. `experiments/` is where this project's own checkpoints go, and the
#: difference matters: those are DVC outputs and these are a cache.
DIR = ROOT / "models" / "pretrained"


def weights(name: str | Path) -> str:
    """The path ultralytics should load `name` from.

    A bare `yolov8n.pt` becomes `models/pretrained/yolov8n.pt`; anything with a directory in it
    is already a real path and comes back as it went in.
    """
    path = Path(name)
    if path.parent != Path("."):
        return str(path)
    DIR.mkdir(parents=True, exist_ok=True)
    return str(DIR / path.name)
