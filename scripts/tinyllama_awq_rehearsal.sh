#!/usr/bin/env bash
# TinyLlama-1.1B INT4 AWQ rehearsal: convert -> build -> run.
# Command sequence extracted from notebooks/03-tinyllama-awq-rehearsal.ipynb.
# Originally run on Colab (paths were /content/...); WORKDIR replaces those.
# NOTE: TensorRT-LLM was installed as a Dec-2025 pre-release; flags of
# convert_checkpoint.py / trtllm-build may have changed since.
set -euo pipefail

WORKDIR="${WORKDIR:-$PWD/work}"
mkdir -p "$WORKDIR"

# 1. Install TensorRT-LLM and deps (the notebook also ran `huggingface_hub.login()` first)
pip install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com
pip install --upgrade transformers  # required for recent model support

# 2. Clone TensorRT-LLM example scripts
git clone https://github.com/NVIDIA/TensorRT-LLM.git "$WORKDIR/TensorRT-LLM"
(cd "$WORKDIR/TensorRT-LLM" && git submodule update --init --recursive)
pip install -r "$WORKDIR/TensorRT-LLM/examples/llama/requirements.txt"

# 3. Download the model with git-lfs
git lfs install
git clone https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0 "$WORKDIR/tmp/model"

# 4. Convert checkpoint with INT4 AWQ weight-only quantization
cd "$WORKDIR/TensorRT-LLM/examples/llama"
python convert_checkpoint.py --model_dir "$WORKDIR/tmp/model" \
                             --output_dir "$WORKDIR/tmp/tllm_checkpoint" \
                             --dtype float16 \
                             --use_weight_only \
                             --weight_only_precision int4_awq

# 5. Build the engine
trtllm-build --checkpoint_dir "$WORKDIR/tmp/tllm_checkpoint" \
             --output_dir "$WORKDIR/tmp/tllm_engine" \
             --gemm_plugin float16

# 6. Run inference
python ../run.py --engine_dir "$WORKDIR/tmp/tllm_engine" \
                 --max_output_len 100 \
                 --tokenizer_dir "$WORKDIR/tmp/model" \
                 --input_text "What is the capital of South Korea?"
