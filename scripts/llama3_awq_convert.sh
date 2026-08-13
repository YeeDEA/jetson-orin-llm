#!/usr/bin/env bash
# Llama-3 8B INT4 AWQ conversion (x86 Colab side).
# Command sequence extracted from notebooks/04-llama3-awq-orin-deploy.ipynb, cells 0-3.
# Originally run on Colab (/content/... paths, Google Drive upload); WORKDIR replaces those.
# NOTE: Meta-Llama-3-8B-Instruct is gated — authenticate with HF first.
set -euo pipefail

WORKDIR="${WORKDIR:-$PWD/work}"
mkdir -p "$WORKDIR"

# 1. Install TensorRT-LLM (x86)
#    Note (from the notebook): prefer a version compatible with what will be installed on the Jetson.
pip install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com
pip install --upgrade transformers

# 2. Clone TensorRT-LLM source for the example scripts
git clone https://github.com/NVIDIA/TensorRT-LLM.git "$WORKDIR/TensorRT-LLM"
(cd "$WORKDIR/TensorRT-LLM" && git submodule update --init --recursive)
pip install -r "$WORKDIR/TensorRT-LLM/examples/llama/requirements.txt"

# 3. Download the model (git-lfs; large — takes a while)
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct "$WORKDIR/model_input"

# 4. Convert + quantize
#    --use_weight_only: quantize weights only (memory saving)
#    --weight_only_precision int4_awq: INT4 precision
cd "$WORKDIR/TensorRT-LLM/examples/llama"
python convert_checkpoint.py --model_dir "$WORKDIR/model_input" \
                             --output_dir "$WORKDIR/tllm_checkpoint_int4" \
                             --dtype float16 \
                             --use_weight_only \
                             --weight_only_precision int4_awq

# 5. Pack the checkpoint for transfer to the Jetson
#    (the notebook then copied it to Google Drive; move it to the Orin however you like)
tar -czvf "$WORKDIR/tllm_checkpoint_int4.tar.gz" "$WORKDIR/tllm_checkpoint_int4"
