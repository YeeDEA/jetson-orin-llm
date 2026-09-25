#!/usr/bin/env python3
"""Latency and memory benchmark for the quantized checkpoints in this repo.

Reports decode throughput, time-to-first-token and peak memory for one model
under one quantization setting. Prefill and decode are timed separately because
they bottleneck on different things: prefill is compute-bound on the prompt,
decode is bandwidth-bound on the weights.

The numbers this produces are only comparable against numbers produced the same
way, so every run records its own environment (GPU name, driver, torch version,
prompt and generation length, batch size) into the output JSON alongside the
measurements.

Usage
-----
    python scripts/bench.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --quant fp16 --out results/bench_t4_fp16.json

    python scripts/bench.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --quant int4-bnb --out results/bench_t4_int4.json

Run the same model at every quantization you want to compare, then feed the
JSON files to scripts/bench_table.py to get the README table.

NOTE: --quant int4-bnb measures bitsandbytes NF4, which is a *memory* format:
weights are stored in 4 bits and dequantized to fp16 at compute time. It is the
training-side quantization used in notebooks/02-qlora-llama3-8b.ipynb, not the
TensorRT-LLM INT4 AWQ engine used for deployment. Those are different things and
the README must not present one as a proxy for the other.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_FILLER = (
    "The following is a technical description of an embedded inference system. "
)


def build_prompt_ids(tokenizer, target_tokens: int) -> list[int]:
    """Return exactly target_tokens token ids (BOS included) of filler text.

    Truncating ids directly instead of decode -> re-encode keeps the length
    exact (re-encoding added BOS again and drifted by a couple of tokens).
    """
    text = PROMPT_FILLER
    while len(tokenizer(text).input_ids) < target_tokens:
        text += PROMPT_FILLER
    return tokenizer(text).input_ids[:target_tokens]


def load_model(model_id: str, quant: str):
    kwargs = {"device_map": "cuda:0"}

    if quant == "fp16":
        kwargs["dtype"] = torch.float16
    elif quant == "int8-bnb":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "int4-bnb":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ValueError(f"unknown quant: {quant}")

    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


@torch.inference_mode()
def time_prefill(model, input_ids) -> float:
    """Seconds until the first token is produced."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    model.generate(input_ids, attention_mask=torch.ones_like(input_ids),
                   max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    return time.perf_counter() - start


@torch.inference_mode()
def time_decode(model, input_ids, gen_tokens: int) -> tuple[float, int]:
    """Seconds for a full generation, and how many tokens were actually produced."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),  # batch 1, no padding
        max_new_tokens=gen_tokens,
        min_new_tokens=gen_tokens,
        do_sample=False,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    produced = out.shape[-1] - input_ids.shape[-1]
    return elapsed, produced


def gpu_info() -> dict:
    info = {
        "name": torch.cuda.get_device_name(0),
        "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(0)),
        "total_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "transformers": transformers.__version__,
    }
    try:
        import bitsandbytes

        info["bitsandbytes"] = bitsandbytes.__version__
    except Exception:
        info["bitsandbytes"] = None
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        info["driver"] = driver
    except Exception:
        info["driver"] = None
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument(
        "--quant",
        required=True,
        choices=["fp16", "int8-bnb", "int4-bnb"],
        help="weight format to load",
    )
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible; this benchmark needs a GPU")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = load_model(args.model, args.quant)
    model.eval()
    load_seconds = time.perf_counter() - load_start
    weights_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 3)

    ids = build_prompt_ids(tokenizer, args.prompt_tokens)
    input_ids = torch.tensor([ids] * args.batch, device="cuda:0")
    actual_prompt_tokens = input_ids.shape[-1]

    for _ in range(args.warmup):
        time_decode(model, input_ids, min(16, args.gen_tokens))

    torch.cuda.reset_peak_memory_stats()

    ttfts, decode_rates = [], []
    for _ in range(args.runs):
        ttfts.append(time_prefill(model, input_ids))
        elapsed, produced = time_decode(model, input_ids, args.gen_tokens)
        # Subtract the prefill cost so the rate reflects decode alone.
        decode_only = elapsed - ttfts[-1]
        decode_rates.append((produced * args.batch) / decode_only)

    peak_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 3)

    result = {
        "model": args.model,
        "quant": args.quant,
        "config": {
            "prompt_tokens": actual_prompt_tokens,
            "gen_tokens": args.gen_tokens,
            "batch": args.batch,
            "warmup": args.warmup,
            "runs": args.runs,
            "sampling": "greedy",
            "command": " ".join(sys.argv),
        },
        "measurements": {
            "decode_tokens_per_s_median": round(statistics.median(decode_rates), 2),
            "decode_tokens_per_s_all": [round(r, 2) for r in decode_rates],
            "ttft_s_median": round(statistics.median(ttfts), 4),
            "ttft_s_all": [round(t, 4) for t in ttfts],
            "peak_memory_gb": peak_gb,
            "weights_only_memory_gb": weights_gb,
            "model_load_s": round(load_seconds, 2),
        },
        "environment": gpu_info(),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    m = result["measurements"]
    print(f"{args.model}  [{args.quant}]  on {result['environment']['name']}")
    print(f"  decode      {m['decode_tokens_per_s_median']:>8.2f} tok/s (median of {args.runs})")
    print(f"  TTFT        {m['ttft_s_median']:>8.4f} s  @ {actual_prompt_tokens} prompt tokens")
    print(f"  peak memory {m['peak_memory_gb']:>8.3f} GB  (weights {m['weights_only_memory_gb']:.3f} GB)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
