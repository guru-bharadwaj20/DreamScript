<#
.SYNOPSIS
    DreamScript task runner for Windows (Phase 0.2.5).

.DESCRIPTION
    GNU make is not installed on the target machine, so this script mirrors every Makefile
    target one-for-one. The Makefile stays the canonical reference for Linux/CI; this file
    is what actually runs here.

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
    "verify-genai"      = @{ Cmd = @("scripts/check_4bit_load.py");            Help = "Phase 0.1.5 - load the 7B base model in 4-bit" }
    "determinism"       = @{ Cmd = @("scripts/determinism_check.py", "--compare"); Help = "Phase 0.1.6 - two runs must be identical" }
    "dummy-run"         = @{ Cmd = @("scripts/dummy_run.py");                  Help = "Phase 0.2.4 - write a complete run directory" }

    # quality
    "test"              = @{ Cmd = @("-m", "pytest");                          Help = "run the full pytest suite" }
    "test-fast"         = @{ Cmd = @("-m", "pytest", "-m", "not slow");        Help = "skip anything marked slow" }
    "lint"              = @{ Cmd = @("-m", "ruff", "check", "src", "tests", "scripts"); Help = "ruff check" }
    "format"            = @{ Cmd = @("-m", "black", "src", "tests", "scripts");Help = "apply black formatting" }

    # pipeline stages
    "data"              = @{ Module = "src.ingest";     Help = "Phase 1 - build the corpus manifest and splits" }
    "preprocess"        = @{ Module = "src.preprocess"; Help = "Phase 3 - photos to clean strokes and primitives" }
    "features"          = @{ Module = "src.features";   Help = "Phase 4 - build the handcrafted feature table" }
    "train-clf"         = @{ Module = "src.classify";   Help = "Phase 5 - Decision Tree / KNN / Logistic Regression" }
    "train-nn"          = @{ Module = "src.classify";   Extra = @("model=mlp"); Help = "Phase 6 - MLP and SVM on embeddings" }
    "train-ens"         = @{ Module = "src.classify";   Extra = @("model=rf");  Help = "Phase 7 - Random Forest / boosting" }
    "detect"            = @{ Module = "src.detect";     Help = "Phase 9.1 - component detector" }
    "ocr"               = @{ Module = "src.ocr";        Help = "Phase 9.3 - handwriting recognition" }
    "parse"             = @{ Module = "src.parse";      Help = "Phases 7.3/10 - HMM roles and graph assembly" }
    "rl"                = @{ Module = "src.rl";         Help = "Phase 11 - traversal policy" }
    "finetune"          = @{ Module = "src.synth";      Help = "Phase 12 - QLoRA code synthesis" }
    "eval"              = @{ Module = "src.eval";       Help = "Phase 14 - metrics, ablations, error analysis" }
    "serve"             = @{ Module = "src.serve";      Help = "Phase 16 - run the web demo" }
    "app"               = @{ Module = "src.serve";      Help = "alias for serve" }
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
if ($spec.Module) {
    $configName = ($spec.Module -split "\.")[-1]
    $argv = @("-m", $spec.Module, "--config", "configs/$configName.yaml")
    if ($spec.Extra) { $argv += $spec.Extra }
    if ($Overrides) { $argv += $Overrides }
}
else {
    $argv = $spec.Cmd
    if ($Overrides) { $argv += $Overrides }
}

$display = "$Py " + ($argv -join " ")
if ($DryRun) { Write-Host $display; exit 0 }

if (-not (Test-Path $Py)) {
    Write-Error "venv not found at $Py - run 'uv venv --python 3.11 .venv' first (see Makefile 'env')."
    exit 1
}

Write-Host "> $display" -ForegroundColor DarkGray
& $Py @argv
exit $LASTEXITCODE
