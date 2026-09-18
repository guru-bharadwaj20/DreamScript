"""Phase 12.2.8 - merge the LoRA adapter into bf16 weights and export GGUF for llama.cpp.

    python -m src.llm.export --adapter experiments/llm/runs/<run>/adapter --out experiments/llm/export

Steps, each with its own check, because "the file was written" is not "the export works":

1. **merge** - the base is loaded in **bf16 on the CPU** (not NF4: merging a LoRA delta into
   4-bit weights would re-quantise the sum and bake quantisation error into the merged model),
   the adapter applied, `merge_and_unload`, `save_pretrained` as safetensors. Check: on held-out
   prompts, the merged model's next-token logits equal the unmerged bf16 base + adapter's
   (max absolute difference reported) and greedy continuations are identical.
2. **gguf** - llama.cpp's own `convert_hf_to_gguf.py` (pinned release tag, `gguf-py` from the same
   checkout) writes an f16 GGUF; `llama-quantize` writes Q8_0 and Q4_K_M.
3. **serve** - `llama-server` loads each GGUF on the GPU and answers the frozen template's chat
   messages; its greedy replies are scored by 12.3 exactly like the HF generations, so the export
   is judged on functional pass rate, not on a perplexity number.

vLLM is not attempted: it does not support native Windows. The merged safetensors directory is
the vLLM-loadable artefact (standard HF layout, `config.json` + `model-*.safetensors`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "experiments" / "llm" / "tools"
LLAMA_CPP = TOOLS / "llama.cpp"
LLAMA_BIN = TOOLS / "bin"
LLAMA_TAG = "b10909"


#: Windows build flavour to fetch. cu124 matches requirements/torch.txt; a newer driver runs it.
LLAMA_BUILD = "win-cuda-12.4-x64"
LLAMA_RELEASE = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"


def toolchain_ready() -> bool:
    return (LLAMA_CPP / "convert_hf_to_gguf.py").is_file() and all(
        (LLAMA_BIN / exe).is_file() for exe in ("llama-quantize.exe", "llama-server.exe")
    )


def ensure_toolchain() -> bool:
    """Fetch llama.cpp at `LLAMA_TAG` - the converter script and the prebuilt binaries.

    `experiments/` is gitignored, so this tree never survives a fresh clone and used to be built
    by hand; a missing converter then failed the export stage with a bare `exit status 2`. Pinned
    to a tag rather than a branch, because `convert_hf_to_gguf.py` and the GGUF it writes have to
    agree with the `llama-quantize` that reads it.
    """
    import shutil
    import urllib.request
    import zipfile

    if toolchain_ready():
        return True
    TOOLS.mkdir(parents=True, exist_ok=True)

    if not (LLAMA_CPP / "convert_hf_to_gguf.py").is_file():
        if LLAMA_CPP.exists():
            shutil.rmtree(LLAMA_CPP, ignore_errors=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                LLAMA_TAG,
                LLAMA_REPO,
                str(LLAMA_CPP),
            ],
            check=True,
        )

    if not toolchain_ready():
        LLAMA_BIN.mkdir(parents=True, exist_ok=True)
        # The binaries and the CUDA runtime ship as two archives; both unpack flat into bin/.
        for name in (
            f"llama-{LLAMA_TAG}-bin-{LLAMA_BUILD}.zip",
            f"cudart-llama-bin-{LLAMA_BUILD}.zip",
        ):
            archive = TOOLS / name
            if not archive.is_file():
                url = LLAMA_RELEASE.format(tag=LLAMA_TAG, name=name)
                with urllib.request.urlopen(url, timeout=900) as response, archive.open("wb") as fh:
                    shutil.copyfileobj(response, fh)
            with zipfile.ZipFile(archive) as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    target = LLAMA_BIN / Path(member.filename).name
                    with zf.open(member) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    return toolchain_ready()


def merge(base_id: str, adapter: Path, out: Path) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    began = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(adapter))
    merged = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(out)
    size = sum(f.stat().st_size for f in out.glob("*.safetensors"))
    return {"merge_s": round(time.perf_counter() - began, 1), "merged_gb": round(size / 1e9, 2)}


def check_merge(base_id: str, adapter: Path, merged_dir: Path, prompts: list[str]) -> dict:
    """Merged vs unmerged (bf16 base + adapter): logits max |diff| and greedy agreement, on GPU."""
    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_id)

    def run(model) -> tuple[list, list]:
        logits, conts = [], []
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                logits.append(model(**ids).logits[0, -1].float().cpu())
                out = model.generate(**ids, do_sample=False, max_new_tokens=64)
            conts.append(out[0, ids["input_ids"].shape[1] :].tolist())
        return logits, conts

    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    unmerged = PeftModel.from_pretrained(base, str(adapter)).eval()
    la, ca = run(unmerged)
    del unmerged, base
    gc.collect()
    torch.cuda.empty_cache()
    merged = AutoModelForCausalLM.from_pretrained(
        merged_dir, torch_dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    lb, cb = run(merged)
    del merged
    gc.collect()
    torch.cuda.empty_cache()
    diffs = [float((a - b).abs().max()) for a, b in zip(la, lb, strict=True)]
    return {
        "prompts": len(prompts),
        "max_abs_logit_diff": round(max(diffs), 5),
        "argmax_agree": sum(int(a.argmax() == b.argmax()) for a, b in zip(la, lb, strict=True)),
        "greedy_64_identical": sum(int(a == b) for a, b in zip(ca, cb, strict=True)),
    }


def to_gguf(merged_dir: Path, out: Path, quants: tuple[str, ...] = ("Q8_0", "Q4_K_M")) -> dict:
    if not ensure_toolchain():
        raise RuntimeError(f"llama.cpp {LLAMA_TAG} toolchain unavailable under {TOOLS}")
    env = dict(os.environ, PYTHONPATH=str(LLAMA_CPP / "gguf-py"))
    f16 = out / "model-f16.gguf"
    began = time.perf_counter()
    convert = subprocess.run(
        [
            sys.executable,
            str(LLAMA_CPP / "convert_hf_to_gguf.py"),
            str(merged_dir),
            "--outtype",
            "f16",
            "--outfile",
            str(f16),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if convert.returncode != 0:
        # `check=True` here raised CalledProcessError with the converter's own diagnosis captured
        # and discarded, which is how a missing script read as a bare "exit status 2".
        raise RuntimeError(
            f"convert_hf_to_gguf.py failed rc={convert.returncode}:\n"
            f"{(convert.stderr or convert.stdout or '')[-2000:]}"
        )
    report: dict[str, Any] = {
        "llama_cpp_tag": LLAMA_TAG,
        "f16": {"path": str(f16), "gb": round(f16.stat().st_size / 1e9, 2)},
        "convert_s": round(time.perf_counter() - began, 1),
    }
    for q in quants:
        target = out / f"model-{q}.gguf"
        began = time.perf_counter()
        subprocess.run(
            [str(LLAMA_BIN / "llama-quantize.exe"), str(f16), str(target), q],
            check=True,
            capture_output=True,
        )
        report[q] = {
            "path": str(target),
            "gb": round(target.stat().st_size / 1e9, 2),
            "quantize_s": round(time.perf_counter() - began, 1),
        }
    return report


class LlamaServer:
    """`llama-server` on a GGUF, as a context manager; OpenAI-compatible chat endpoint."""

    def __init__(self, gguf: Path, port: int = 8765, parallel: int = 1, ctx: int = 8192) -> None:
        self.gguf, self.port, self.parallel, self.ctx = gguf, port, parallel, ctx
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> LlamaServer:
        import urllib.request

        log = open(self.gguf.with_suffix(f".server{self.parallel}.log"), "w")  # noqa: SIM115
        self.proc = subprocess.Popen(
            [
                str(LLAMA_BIN / "llama-server.exe"),
                "-m",
                str(self.gguf),
                "--port",
                str(self.port),
                "-ngl",
                "999",
                "-c",
                str(self.ctx * self.parallel),
                "-np",
                str(self.parallel),
                "--temp",
                "0",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return self
            except OSError:
                time.sleep(1)
        raise RuntimeError("llama-server did not become healthy")

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()

    def chat(self, messages: list[dict], max_tokens: int = 1600, stream: bool = False) -> dict:
        """Greedy reply; with `stream`, also time-to-first-token."""
        import urllib.request

        body = {
            "messages": messages,
            "temperature": 0,
            "top_k": 1,
            "max_tokens": max_tokens,
            "stream": stream,
            "cache_prompt": True,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        began = time.perf_counter()
        first = None
        with urllib.request.urlopen(req, timeout=600) as resp:
            if not stream:
                data = json.loads(resp.read())
                return {
                    "text": data["choices"][0]["message"]["content"],
                    "new_tokens": data["usage"]["completion_tokens"],
                    "seconds": time.perf_counter() - began,
                }
            parts, n = [], 0
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[5:])["choices"][0]["delta"].get("content")
                if delta:
                    if first is None:
                        first = time.perf_counter() - began
                    parts.append(delta)
                    n += 1
            return {
                "text": "".join(parts),
                "new_tokens": n,
                "seconds": time.perf_counter() - began,
                "ttft_s": first,
            }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - heavy
    parser = argparse.ArgumentParser(description="Phase 12.2.8 - merge and export")
    parser.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/llm/export")
    parser.add_argument("--skip-merge", action="store_true")
    args = parser.parse_args(argv)
    merged_dir = args.out / "merged"
    report: dict[str, Any] = {"adapter": str(args.adapter)}
    if not args.skip_merge:
        report["merge"] = merge(args.base, args.adapter, merged_dir)
    report["gguf"] = to_gguf(merged_dir, args.out)
    (args.out / "export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
