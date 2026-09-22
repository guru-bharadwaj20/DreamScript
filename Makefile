# DreamScript task runner (Phase 0.2.5).
#
#   make help          list every target
#   make verify        prove the environment is intact
#   make data          build the corpus manifest
#   make train-clf     train the Phase 5 classifiers
#   make eval          run the evaluation suite
#   make app           serve the web demo
#
# Windows note: this repo is developed on Windows with Git Bash, where `make` may be absent.
# Every target is a one-line command you can also run directly, and `tasks.ps1` mirrors the
# whole file for PowerShell users (`./tasks.ps1 verify`).

SHELL := /bin/bash
PY := .venv/Scripts/python.exe
ifeq ($(OS),)
PY := .venv/bin/python
endif

CONFIG ?= configs/base.yaml
OVERRIDES ?=

.DEFAULT_GOAL := help
.PHONY: help env verify verify-gpu verify-cv verify-classical verify-genai determinism \
        test test-fast lint format data preprocess features train-clf train-nn train-ens \
        detect ocr parse rl finetune eval app serve clean clean-experiments repro dag

## -- environment -------------------------------------------------------------

help:  ## list available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

env:  ## create the venv and install every requirements layer
	uv venv --python 3.11 .venv
	VIRTUAL_ENV=$(PWD)/.venv uv pip install -r requirements/base.txt
	VIRTUAL_ENV=$(PWD)/.venv uv pip install -r requirements/torch.txt \
		--index-url https://download.pytorch.org/whl/cu124
	VIRTUAL_ENV=$(PWD)/.venv uv pip install -r requirements/cv.txt \
		-r requirements/classical.txt -r requirements/genai.txt

verify:  ## Phase 0.1 — check every installed stack
	$(PY) scripts/verify_env.py

verify-gpu:  ## Phase 0.1.2 — CUDA, bf16 and VRAM
	$(PY) scripts/check_cuda.py

verify-cv:  ## Phase 0.1.3 — CV stack binarization smoke test
	$(PY) scripts/smoke_cv.py

verify-genai:  ## Phase 0.1.5 — load the 7B base model in 4-bit
	$(PY) scripts/check_4bit_load.py

determinism:  ## Phase 0.1.6 — two runs must produce identical metrics
	$(PY) scripts/determinism_check.py --compare

dummy-run:  ## Phase 0.2.4 — write a complete experiment run directory
	$(PY) scripts/dummy_run.py

## -- quality -----------------------------------------------------------------

test:  ## run the full pytest suite
	$(PY) -m pytest

test-fast:  ## skip anything marked slow
	$(PY) -m pytest -m "not slow"

lint:  ## ruff + black --check + isort --check
	$(PY) -m ruff check src tests scripts
	$(PY) -m black --check src tests scripts
	$(PY) -m isort --check-only src tests scripts

format:  ## apply black + isort + ruff --fix
	$(PY) -m isort src tests scripts
	$(PY) -m black src tests scripts
	$(PY) -m ruff check --fix src tests scripts

## -- pipeline ----------------------------------------------------------------

data:  ## Phase 1 — build the corpus manifest and splits
	$(PY) -m src.ingest --config configs/ingest.yaml $(OVERRIDES)

preprocess:  ## Phase 3 — photos to clean strokes and primitives
	$(PY) -m src.preprocess --config configs/preprocess.yaml $(OVERRIDES)

features:  ## Phase 4 — build the handcrafted feature table
	$(PY) -m src.features --config configs/features.yaml $(OVERRIDES)

train-clf:  ## Phase 5 — Decision Tree / KNN / Logistic Regression
	$(PY) -m src.classify --config configs/classify.yaml $(OVERRIDES)

train-nn:  ## Phase 6 — MLP and SVM on image embeddings
	$(PY) -m src.classify --config configs/classify.yaml model=mlp $(OVERRIDES)

train-ens:  ## Phase 7 — Random Forest / boosting ensembles
	$(PY) -m src.classify --config configs/classify.yaml model=rf $(OVERRIDES)

detect:  ## Phase 9.1 — component detector
	$(PY) -m src.detect --config configs/detect.yaml $(OVERRIDES)

ocr:  ## Phase 9.3 — handwriting recognition
	$(PY) -m src.ocr --config configs/ocr.yaml $(OVERRIDES)

parse:  ## Phases 7.3/10 — HMM roles and graph assembly into the IR
	$(PY) -m src.parse --config configs/parse.yaml $(OVERRIDES)

rl:  ## Phase 11 — traversal policy
	$(PY) -m src.rl --config configs/rl.yaml $(OVERRIDES)

finetune:  ## Phase 12 — QLoRA code synthesis
	$(PY) -m src.synth --config configs/synth.yaml $(OVERRIDES)

eval:  ## Phase 14 — metrics, ablations and error analysis
	$(PY) -m src.eval --config configs/eval.yaml $(OVERRIDES)

app: serve  ## alias for `make serve`

serve:  ## Phase 16 — run the web demo
	$(PY) -m src.serve --config configs/serve.yaml $(OVERRIDES)

## -- housekeeping ------------------------------------------------------------

clean:  ## remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ build dist *.egg-info

clean-experiments:  ## delete every run directory (irreversible)
	rm -rf experiments/*/

## -- pipeline DAG (Phase 15.4) ------------------------------------------------

repro:  ## Phase 15.4 — run the DVC DAG with the project venv on PATH
	PATH="$(CURDIR)/$(dir $(PY)):$$PATH" $(PY) -m dvc repro $(STAGES)

dag:  ## Phase 15.4 — print the pipeline graph
	$(PY) -m dvc dag
