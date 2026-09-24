<#
.SYNOPSIS
    DreamScript task runner for Windows (Phase 0.2.5).

.DESCRIPTION
    GNU make is not installed on the target machine, so this script mirrors every Makefile
    target one-for-one. The Makefile stays the canonical reference for Linux/CI; this file
    is what actually runs here.

    "One-for-one" was a claim and not a fact. Four targets were missing entirely (env, clean,
    clean-experiments, verify-classical); `lint` ran ruff where the Makefile and CI run ruff,
    black and isort, so the Windows lint path was weaker than the gate it stands in for;
    `format` ran black where the Makefile runs isort, black and ruff --fix; and `repro` called
    dvc with no PATH prepend, walking straight into the Windows-Store python stub trap that
    dvc.yaml documents and `make repro` guards against. tests/test_task_runner.py compares the
    two files now, so the claim is checked rather than asserted.

.EXAMPLE
    ./tasks.ps1 help
    ./tasks.ps1 verify
    ./tasks.ps1 train-clf -Overrides "model=knn","cv.n_splits=10"
    ./tasks.ps1 eval -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Task = "help",

    [string[]]$Overrides = @(),

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $Root ".venv\Scripts\python.exe"

# task name -> @{ Cmd = <argument list>; Help = <one-line description> }
$Tasks = [ordered]@{
    # environment
    "verify"            = @{ Cmd = @("scripts/verify_env.py");                 Help = "Phase 0.1 - check every installed stack" }
    "verify-gpu"        = @{ Cmd = @("scripts/check_cuda.py");                 Help = "Phase 0.1.2 - CUDA, bf16 and VRAM" }
    "verify-cv"         = @{ Cmd = @("scripts/smoke_cv.py");                   Help = "Phase 0.1.3 - CV stack binarization smoke test" }
    "verify-classical"  = @{ Cmd = @("scripts/verify_env.py", "--only", "classical"); Help = "Phase 0.1.4 - sklearn, xgboost, lightgbm, hmmlearn, imblearn" }
    "verify-genai"      = @{ Cmd = @("scripts/check_4bit_load.py");            Help = "Phase 0.1.5 - load the 7B base model in 4-bit" }
    "determinism"       = @{ Cmd = @("scripts/determinism_check.py", "--compare"); Help = "Phase 0.1.6 - two runs must be identical" }
    "dummy-run"         = @{ Cmd = @("scripts/dummy_run.py");                  Help = "Phase 0.2.4 - write a complete run directory" }

    # quality
    "test"              = @{ Cmd = @("-m", "pytest");                          Help = "run the full pytest suite" }
    "test-fast"         = @{ Cmd = @("-m", "pytest", "-m", "not slow");        Help = "skip anything marked slow" }
    # All three, because CI runs all three: a lint task that passes here and fails on the
    # runner is worse than no lint task.
    "lint"              = @{ Steps = @(
                                @("-m", "ruff", "check", "src", "tests", "scripts"),
                                @("-m", "black", "--check", "src", "tests", "scripts"),
                                @("-m", "isort", "--check-only", "src", "tests", "scripts"),
                                @("scripts/typecheck.py"));
                             Help = "ruff + black --check + isort --check + mypy baseline" }
    "typecheck"         = @{ Cmd = @("scripts/typecheck.py"); Help = "mypy, held to its recorded baseline" }
    "format"            = @{ Steps = @(
                                @("-m", "isort", "src", "tests", "scripts"),
                                @("-m", "black", "src", "tests", "scripts"),
                                @("-m", "ruff", "check", "--fix", "src", "tests", "scripts"));
                             Help = "apply isort + black + ruff --fix" }
    # PrependPath is the whole point: without the venv's Scripts directory in front, `dvc repro`
    # spawns `python`, Windows resolves it to the Store stub and every stage exits 9009. The
    # Makefile's repro target prepends for the same reason, and dvc.yaml documents the trap.
    "repro"             = @{ Cmd = @("-m", "dvc", "repro"); PrependPath = $true;
                             Help = "Phase 15.4 - run the DVC DAG with the venv on PATH" }
    "dag"               = @{ Cmd = @("-m", "dvc", "dag");                     Help = "Phase 15.4 - print the pipeline graph" }

    # pipeline stages. These ran `-m src.<pkg> --config configs/<pkg>.yaml`, which every stage
    # entry point answered with StageNotImplemented and exit 2 - the same fourteen do-nothing
    # targets the Makefile had. Same commands as the Makefile now, one for one.
    "data"              = @{ Steps = @(@("-m", "src.ingest", "manifest"), @("-m", "src.ingest", "splits"));
                             Help = "Phase 1 - build the corpus manifest and splits" }
    "preprocess"        = @{ Cmd = @("-m", "src.preprocess", "layers"); Help = "Phase 3 - photos to clean strokes, layers and primitives" }
    "features"          = @{ Cmd = @("-m", "src.features", "build");    Help = "Phase 4 - build the handcrafted feature table" }
    "train-clf"         = @{ Cmd = @("-m", "src.classify", "s1");       Help = "Phase 5 - Decision Tree / KNN / Logistic Regression" }
    "train-nn"          = @{ Steps = @(@("-m", "src.classify", "mlp"), @("-m", "src.classify", "svm"));
                             Help = "Phase 6 - MLP and SVM on embeddings" }
    "train-ens"         = @{ Steps = @(@("-m", "src.classify", "forest"), @("-m", "src.classify", "boosting"));
                             Help = "Phase 7 - Random Forest / boosting" }
    "detect"            = @{ Cmd = @("-m", "src.detect", "train");      Help = "Phase 9.1 - component detector" }
    "ocr"               = @{ Cmd = @("-m", "src.ocr", "s3");            Help = "Phase 9.3 - handwriting recognition" }
    "parse"             = @{ Cmd = @("-m", "src.parse", "s4");          Help = "Phases 7.3/10 - HMM roles and graph assembly" }
    "rl"                = @{ Cmd = @("-m", "src.rl", "dqn");            Help = "Phase 11 - traversal policy" }
    "finetune"          = @{ Cmd = @("-m", "src.llm.run", "--config", "configs/llm.yaml");
                             Help = "Phase 12.2 - one QLoRA training run" }
    "eval"              = @{ Steps = @(@("-m", "src.eval", "stagewise"), @("-m", "src.eval", "ablate"),
                                       @("-m", "src.eval", "compute"), @("-m", "src.eval", "humanbaseline"),
                                       @("-m", "src.eval", "master"));
                             Help = "Phase 14 - the master table and everything it reads" }
    "serve"             = @{ Cmd = @("-m", "uvicorn", "src.serve.api:app", "--host", "127.0.0.1", "--port", "8000");
                             Help = "Phase 15.11 - run the inference service" }
    "app"               = @{ Cmd = @("-m", "uvicorn", "src.serve.api:app", "--host", "127.0.0.1", "--port", "8000");
                             Help = "alias for serve" }
    "stages"            = @{ Stages = $true;            Help = "list every command each pipeline package can run" }

    # housekeeping
    "env"               = @{ Env = $true;               Help = "create the venv and install every requirements layer" }
    "clean"             = @{ Clean = @(".pytest_cache", ".ruff_cache", "build", "dist");
                             Help = "remove caches and build artifacts" }
    "clean-experiments" = @{ CleanRuns = $true;         Help = "delete every run directory (irreversible)" }
}

function Show-Help {
    Write-Host "DreamScript tasks:`n"
    foreach ($name in $Tasks.Keys) {
        Write-Host ("  {0,-16} {1}" -f $name, $Tasks[$name].Help)
    }
    Write-Host "`nUsage: ./tasks.ps1 <task> [-Overrides key=value,...] [-DryRun]"
}

if ($Task -in @("help", "-h", "--help")) { Show-Help; exit 0 }

if (-not $Tasks.Contains($Task)) {
    Write-Error "unknown task '$Task'. Run './tasks.ps1 help' for the list."
    exit 1
}

$spec = $Tasks[$Task]

if ($spec.Env) {
    $layers = @("base.txt", "torch.txt", "classical.txt", "cv.txt", "serve.txt", "genai.txt",
                "test.txt", "dev.txt")
    Write-Host "> uv venv --python 3.11 .venv" -ForegroundColor DarkGray
    if (-not $DryRun) { & uv venv --python 3.11 .venv }
    foreach ($layer in $layers) {
        $extra = @()
        if ($layer -eq "torch.txt") {
            $extra = @("--index-url", "https://download.pytorch.org/whl/cu124")
        }
        Write-Host "> uv pip install -r requirements/$layer $($extra -join ' ')" -ForegroundColor DarkGray
        if (-not $DryRun) {
            $env:VIRTUAL_ENV = Join-Path $Root ".venv"
            & uv pip install -r "requirements/$layer" @extra
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }
    exit 0
}

if ($spec.Clean) {
    foreach ($dir in $spec.Clean) {
        Write-Host "> remove $dir" -ForegroundColor DarkGray
        if (-not $DryRun) { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $Root $dir) }
    }
    if (-not $DryRun) {
        Get-ChildItem -Path $Root -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
    exit 0
}

if ($spec.CleanRuns) {
    # Irreversible, and `make clean-experiments` says so too. Timestamped run directories only:
    # the per-phase artefact directories under experiments/ are what later stages load by name.
    $pattern = "^[0-9]{8}-[0-9]{6}"
    $runs = Get-ChildItem -Path (Join-Path $Root "experiments") -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match $pattern }
    Write-Host "> $(@($runs).Count) timestamped run director$(if (@($runs).Count -eq 1) {'y'} else {'ies'}) under experiments/"
    foreach ($run in $runs) {
        Write-Host "> remove $($run.FullName)" -ForegroundColor DarkGray
        if (-not $DryRun) { Remove-Item -Recurse -Force $run.FullName }
    }
    exit 0
}

if ($spec.Stages) {
    $packages = @("ingest", "preprocess", "features", "classify", "detect", "ocr", "parse",
                  "rl", "synth", "eval", "serve")
    foreach ($pkg in $packages) {
        if ($DryRun) { Write-Host "$Py -m src.$pkg" } else { & $Py -m "src.$pkg"; Write-Host "" }
    }
    exit 0
}

if ($spec.Steps) {
    foreach ($step in $spec.Steps) {
        $line = "$Py " + ($step -join " ")
        if ($DryRun) { Write-Host $line; continue }
        Write-Host "> $line" -ForegroundColor DarkGray
        & $Py @step
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    exit 0
}

# Every task is a command line now. The `Module` form this replaced built
# `-m src.<pkg> --config configs/<pkg>.yaml`, which each stage entry point answered with
# StageNotImplemented and exit 2 - the same fourteen do-nothing targets the Makefile had.
$argv = $spec.Cmd
if ($Overrides) { $argv += $Overrides }

$display = "$Py " + ($argv -join " ")
if ($DryRun) { Write-Host $display; exit 0 }

if (-not (Test-Path $Py)) {
    Write-Error "venv not found at $Py - run 'uv venv --python 3.11 .venv' first (see Makefile 'env')."
    exit 1
}

if ($spec.PrependPath) {
    $env:PATH = (Join-Path $Root ".venv\Scripts") + [System.IO.Path]::PathSeparator + $env:PATH
}

Write-Host "> $display" -ForegroundColor DarkGray
& $Py @argv
exit $LASTEXITCODE
