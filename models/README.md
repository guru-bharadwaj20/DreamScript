# `models/`

Weights that are **downloaded, not produced here**.

`models/pretrained/` holds the ultralytics base checkpoints the detector and the arrow pose model
fine-tune from: `yolov8n.pt`, `yolov8n-pose.pt`, `yolov8s-pose.pt`, `yolov8m-pose.pt`.

They used to sit in the repository root. Four files, ~90 MB, gitignored by `*.pt` and therefore
invisible to `git status` — which is why nobody noticed them, and why they were still in the
Docker build context until 15.x added a `.dockerignore`. ultralytics downloads a bare weight name
into the current working directory, so running a training command from the repo root put them
there.

Nothing in git: the directory is gitignored apart from this file. They re-download on demand, and
`src.detect.pretrained.weights()` is where a module asks for one, so the location is decided in a
single place rather than by whichever directory a command was run from.

What is *produced* here lives elsewhere, and deliberately:

    experiments/detect/final/           9.1's trained detector
    experiments/detect/arrows/pose_m/   10.1.6's arrow pose model
    experiments/ocr/trocr_large/        9.3's recogniser

Those are DVC outputs. These are not.
