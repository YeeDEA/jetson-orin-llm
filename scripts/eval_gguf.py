"""Held-out accuracy of a GGUF model served by llama.cpp's ``llama-server``.

Same protocol as ``eval_accuracy.py``: the held-out split is rebuilt with
``make_splits``, prompts are built with the HF tokenizer's chat template and
tokenized by the HF tokenizer (no extra BOS, matching ``add_special_tokens=False``
in ``eval_accuracy.py``), the token ids are sent to the server as-is, and
decoding is greedy (temperature 0, top_k 1) for at most 48 new tokens.

Start the server first, e.g.:
    llama-server -m runs/gguf/ft_v2_q4_k_m.gguf -ngl 99 -c 2048 --port 8089
then:
    python scripts/eval_gguf.py --tokenizer runs/tinyllama_merged_v2 \
        --gguf runs/gguf/ft_v2_q4_k_m.gguf --tag gguf_q4_k_m_v2 \
        --out results/eval_gguf_q4_k_m_v2.json
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoTokenizer

from jetson_llm.data import make_splits, unseen_mask
from jetson_llm.evaluate import score


def complete(url, ids):
    body = json.dumps({"prompt": ids, "n_predict": 48, "temperature": 0, "top_k": 1,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(url + "/completion", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["content"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--gguf", required=True, help="recorded in the result; the server must be serving it")
    p.add_argument("--url", default="http://127.0.0.1:8089")
    p.add_argument("--train-count", type=int, default=8000)
    p.add_argument("--test-count", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset-version", type=int, choices=[1, 2], default=2)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    train, test = make_splits(args.train_count, args.test_count, args.seed,
                              legacy_rotate=args.dataset_version == 1)
    unseen = unseen_mask(train, test)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    preds = []
    t0 = time.perf_counter()
    for s in test:
        text = tok.apply_chat_template([{"role": "user", "content": s["instruction"]}],
                                       tokenize=False, add_generation_prompt=True)
        preds.append(complete(args.url, tok(text, add_special_tokens=False).input_ids))
    elapsed = time.perf_counter() - t0

    result = {
        "tag": args.tag, "gguf": args.gguf, "command": " ".join(sys.argv),
        "split": {"train_count": args.train_count, "test_count": args.test_count, "seed": args.seed,
                  "dataset_version": args.dataset_version, "unseen_instruction_count": sum(unseen)},
        "scores": score(preds, [s["output"] for s in test], unseen),
        "eval_seconds": round(elapsed, 1),
        "examples": [{"instruction": s["instruction"], "reference": s["output"], "prediction": p}
                     for s, p in list(zip(test, preds))[:20]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.tag, json.dumps(result["scores"]["overall"]))


if __name__ == "__main__":
    main()
