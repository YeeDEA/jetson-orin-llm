#!/usr/bin/env python3
"""Turn bench.py output files into the markdown table the README wants.

    python scripts/bench_table.py results/*.json

Prints a table plus the environment footnote. Paste both into the README —
the footnote is what makes the numbers mean anything to someone else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

QUANT_LABEL = {
    "fp16": "FP16",
    "int8-bnb": "INT8 (bitsandbytes)",
    "int4-bnb": "NF4 (bitsandbytes)",
}


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit("usage: bench_table.py results/*.json")

    rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    rows.sort(key=lambda r: -r["measurements"]["peak_memory_gb"])

    baseline = next(
        (r["measurements"]["decode_tokens_per_s_median"] for r in rows if r["quant"] == "fp16"),
        None,
    )

    print("| Precision | Decode (tok/s) | TTFT (s) | Peak memory (GB) | vs FP16 |")
    print("|---|---|---|---|---|")
    for r in rows:
        m = r["measurements"]
        speedup = (
            f"{m['decode_tokens_per_s_median'] / baseline:.2f}x"
            if baseline
            else "—"
        )
        print(
            f"| {QUANT_LABEL.get(r['quant'], r['quant'])} "
            f"| {m['decode_tokens_per_s_median']:.1f} "
            f"| {m['ttft_s_median']:.3f} "
            f"| {m['peak_memory_gb']:.2f} "
            f"| {speedup} |"
        )

    env = rows[0]["environment"]
    cfg = rows[0]["config"]
    print()
    print(
        f"Measured on **{env['name']}** (CUDA {env['cuda']}, torch {env['torch']}), "
        f"batch {cfg['batch']}, {cfg['prompt_tokens']}-token prompt, "
        f"{cfg['gen_tokens']} generated tokens, greedy, "
        f"median of {cfg['runs']} runs after {cfg['warmup']} warmups. "
        f"Prefill is excluded from the decode rate and reported separately as TTFT."
    )


if __name__ == "__main__":
    main()
