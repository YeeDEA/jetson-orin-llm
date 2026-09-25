"""Peak GPU memory of QLoRA training with vs without prepare_model_for_kbit_training.

Tests the README's hypothesis for the Llama-3 8B OOM ("the manual k-bit prep
fp32 upcast is what blew the memory") on a model that fits the 8 GB card.
Uses the notebook's batch shape (batch 4, every example padded to 512 tokens,
LoRA r=64 on all projections) and fp16 compute, like the T4 run.

Variants (run each in its own process so peak stats are clean):
  prep       cleaned path, prepare_model_for_kbit_training (fp32 upcast + grad ckpt)
  noprep     cleaned path, no upcast, gradient checkpointing still on
  notebook   the notebook cell-4 path (fp16 + manual prep + LoRA cast to fp16)

    python scripts/kbit_prep_memory.py --variant prep --out results/kbit_prep_prep.json
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from jetson_llm.data import make_splits
from jetson_llm.train import (build_trainer, load_model, load_model_notebook, load_tokenizer,
                              tokenize_example)


def gb(x):
    return round(x / 1024**3, 3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--variant", choices=["prep", "noprep", "notebook"], required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    tok = load_tokenizer(args.model)
    torch.cuda.reset_peak_memory_stats()
    if args.variant == "notebook":
        model = load_model_notebook(args.model)
    else:
        model = load_model(args.model, prepare_kbit=args.variant == "prep",
                           compute_dtype=torch.float16)
    after_load = {"allocated_gb": gb(torch.cuda.memory_allocated()),
                  "peak_gb": gb(torch.cuda.max_memory_allocated())}
    dtypes = {}
    for _, prm in model.named_parameters():
        k = f"{prm.dtype}".replace("torch.", "") + ("_trainable" if prm.requires_grad else "")
        dtypes[k] = dtypes.get(k, 0) + prm.numel()

    train, _ = make_splits(args.steps * 4, 10, seed=0)
    feats = [tokenize_example(tok, s, max_length=512) for s in train]
    trainer = build_trainer(model, tok, feats, "runs/kbit_mem", torch.float16,
                            batch_size=4, max_steps=args.steps, pad_to=512, logging_steps=1)
    torch.cuda.reset_peak_memory_stats()
    error = None
    try:
        trainer.train()
    except Exception as e:  # record, don't hide: the notebook path may not train at all
        error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    result = {
        "model": args.model, "variant": args.variant, "command": " ".join(sys.argv),
        "config": {"batch": 4, "seq_len_padded": 512, "lora_r": 64, "compute": "fp16",
                   "steps": args.steps},
        "after_load_and_prep": after_load,
        "param_count_by_dtype": dtypes,
        "train_peak_allocated_gb": None if error else gb(torch.cuda.max_memory_allocated()),
        "train_peak_reserved_gb": None if error else gb(torch.cuda.max_memory_reserved()),
        "error": error,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("variant", "after_load_and_prep",
                                             "param_count_by_dtype", "train_peak_allocated_gb",
                                             "error")}))


if __name__ == "__main__":
    main()
