"""CLI wrapper for QLoRA fine-tuning of Llama-3 8B (notebooks/02-qlora-llama3-8b.ipynb, cell 4 variant).

Requires: CUDA GPU, gated Llama-3 HF access (run `huggingface-cli login` or set
HF_TOKEN first — the notebook used huggingface_hub.login() in Colab).

NOTE: this run never completed on a free Colab T4 (FP32-upcast OOM); see the
module docstring of src/jetson_llm/train.py and the README.

Usage:
    python scripts/train_qlora.py --data robot_dataset.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Llama-3 8B on the robot-command dataset")
    parser.add_argument("--model-id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data", default="robot_dataset.json", help="dataset JSON (from generate_dataset.py)")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--adapter-dir", default="./trained_adapter")
    args = parser.parse_args()

    from jetson_llm.train import train  # heavy imports deferred until after arg parsing
    train(model_id=args.model_id, data_files=args.data,
          output_dir=args.output_dir, adapter_dir=args.adapter_dir)


if __name__ == "__main__":
    main()
