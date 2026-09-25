"""QLoRA fine-tune on a fixed-seed split of the synthetic robot-command dataset.

Builds the train/held-out split with ``jetson_llm.data.make_splits`` (held-out
is a separate seeded draw and is never passed to the trainer), trains a LoRA
adapter on a 4-bit NF4 base, and writes run stats to ``--stats``.

Usage (the run reported in the README):
    python scripts/train_qlora.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --train-count 8000 --test-count 500 --seed 0 \
        --adapter-dir runs/tinyllama_lora --stats results/train_tinyllama.json

For Llama-3 8B pass --model-id meta-llama/Meta-Llama-3-8B-Instruct (gated; needs
HF access and more GPU memory than the 8 GB card used here).
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jetson_llm.data import make_splits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--train-count", type=int, default=8000)
    p.add_argument("--test-count", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset-version", type=int, choices=[1, 2], default=2,
                   help="1 = notebook labels (random sign on direction-less rotate), 2 = fixed")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--no-kbit-prep", action="store_true",
                   help="skip prepare_model_for_kbit_training (no fp32 upcast)")
    p.add_argument("--adapter-dir", default="runs/tinyllama_lora")
    p.add_argument("--stats", type=Path, default=Path("results/train_tinyllama.json"))
    args = p.parse_args()

    train_samples, _ = make_splits(args.train_count, args.test_count, args.seed,
                                      legacy_rotate=args.dataset_version == 1)

    from jetson_llm.train import train  # heavy imports after arg parsing
    _, stats = train(model_id=args.model_id, train_samples=train_samples,
                     output_dir="runs/trainer", adapter_dir=args.adapter_dir,
                     prepare_kbit=not args.no_kbit_prep, epochs=args.epochs,
                     batch_size=args.batch_size, lr=args.lr, lora_r=args.lora_r)
    stats.update({"model": args.model_id, "command": " ".join(sys.argv),
                  "train_count": args.train_count, "seed": args.seed, "dataset_version": args.dataset_version,
                  "kbit_prep": not args.no_kbit_prep, "lora_r": args.lora_r,
                  "batch_size": args.batch_size, "lr": args.lr, "epochs": args.epochs})
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print({k: v for k, v in stats.items() if k != "log_history"})


if __name__ == "__main__":
    main()
