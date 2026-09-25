"""Exact-match / valid-code accuracy on the held-out split.

The held-out set is rebuilt with the same ``make_splits`` call used for
training (seeded, so identical), and is never passed to the trainer.

Usage (the runs reported in the README):
    # base model, zero-shot and 3-shot
    python scripts/eval_accuracy.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant fp16 \
        --tag base_fp16_0shot --out results/eval_base_fp16_0shot.json
    python scripts/eval_accuracy.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant fp16 --shots 3 \
        --tag base_fp16_3shot --out results/eval_base_fp16_3shot.json
    # merge the LoRA adapter into an fp16 checkpoint and evaluate it, then evaluate it at NF4
    python scripts/eval_accuracy.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --merge-adapter runs/tinyllama_lora --merged-dir runs/tinyllama_merged \
        --quant fp16 --tag ft_fp16 --out results/eval_ft_fp16.json
    python scripts/eval_accuracy.py --model runs/tinyllama_merged --quant int4-bnb \
        --tag ft_nf4 --out results/eval_ft_nf4.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from jetson_llm.data import make_splits, unseen_mask
from jetson_llm.evaluate import score

FEWSHOT = [  # fixed, hand-written, not drawn from the held-out set
    ("Go left for 30cm", "robot.move('left', 30)"),
    ("Pick up the cup", "robot.grab('cup')"),
    ("Go to the kitchen and bring me the book", "robot.navigate('kitchen')\nrobot.grab('book')"),
]


def load(model_id, quant):
    if quant == "fp16":
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="cuda:0")
    cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    return AutoModelForCausalLM.from_pretrained(model_id, quantization_config=cfg, device_map="cuda:0")


def messages(instruction, shots):
    msgs = []
    for q, a in FEWSHOT[:shots]:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return msgs + [{"role": "user", "content": instruction}]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--quant", choices=["fp16", "int4-bnb"], default="fp16")
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--train-count", type=int, default=8000)
    p.add_argument("--test-count", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset-version", type=int, choices=[1, 2], default=2,
                   help="1 = notebook labels (random sign on direction-less rotate), 2 = fixed")
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--merge-adapter", help="LoRA adapter dir to merge into --model first (fp16)")
    p.add_argument("--merged-dir", help="where to save the merged fp16 checkpoint")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    train, test = make_splits(args.train_count, args.test_count, args.seed,
                                      legacy_rotate=args.dataset_version == 1)
    unseen = unseen_mask(train, test)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if args.merge_adapter:
        if args.quant != "fp16":
            raise SystemExit("merge in fp16, then evaluate --model <merged-dir> --quant int4-bnb")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="cuda:0")
        model = PeftModel.from_pretrained(base, args.merge_adapter).merge_and_unload()
        if args.merged_dir:
            model.save_pretrained(args.merged_dir)
            tok.save_pretrained(args.merged_dir)
    else:
        model = load(args.model, args.quant)
    model.eval()

    prompts = [tok.apply_chat_template(messages(s["instruction"], args.shots), tokenize=False,
                                       add_generation_prompt=True) for s in test]
    preds = []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(0, len(prompts), args.batch):
            enc = tok(prompts[i:i + args.batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to("cuda:0")
            out = model.generate(**enc, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            preds += tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    elapsed = time.perf_counter() - t0

    result = {
        "tag": args.tag, "model": args.model, "quant": args.quant, "shots": args.shots,
        "adapter": args.merge_adapter, "command": " ".join(sys.argv),
        "split": {"train_count": args.train_count, "test_count": args.test_count,
                  "seed": args.seed, "dataset_version": args.dataset_version, "unseen_instruction_count": sum(unseen)},
        "scores": score(preds, [s["output"] for s in test], unseen),
        "eval_seconds": round(elapsed, 1),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "gpu": torch.cuda.get_device_name(0),
        "examples": [{"instruction": s["instruction"], "reference": s["output"], "prediction": p}
                     for s, p in list(zip(test, preds))[:20]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.tag, json.dumps(result["scores"]["overall"]), "unseen:",
          json.dumps(result["scores"].get("unseen_instruction")))


if __name__ == "__main__":
    main()
