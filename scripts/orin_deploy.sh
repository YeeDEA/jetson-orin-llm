#!/usr/bin/env bash
# Jetson Orin Nano deployment: unpack checkpoint, build engine, run inference.
# Command sequence extracted from notebooks/04-llama3-awq-orin-deploy.ipynb, cells 5-9.
# Run ON THE JETSON. Steps 2-3 happen INSIDE the dustynv/tensorrt_llm container —
# this script documents the sequence; run the container step interactively.
set -euo pipefail

# 1. On the Jetson terminal: unpack the checkpoint produced by llama3_awq_convert.sh
mkdir -p ~/llm_project
mv tllm_checkpoint_int4.tar.gz ~/llm_project
cd ~/llm_project
tar -xzvf tllm_checkpoint_int4.tar.gz

# 2. Start the container (dustynv's pre-built image is the easiest to use;
#    downloaded automatically if missing — takes a while). Drops you into a shell.
sudo docker run -it --rm --runtime nvidia --network host \
    -v ~/llm_project:/data \
    dustynv/tensorrt_llm:r36.2.0 \
    bash

# --- Inside the container ---

# 3. Build the engine
# trtllm-build --checkpoint_dir /data/tllm_checkpoint_int4 \
#              --output_dir /data/llama3_int4_engine \
#              --gemm_plugin float16 \
#              --max_batch_size 1 \
#              --max_input_len 2048 \
#              --max_output_len 512

# 4. Test inference (run.py is usually at /opt/TensorRT-LLM/examples/run.py in the container)
# python3 /opt/TensorRT-LLM/examples/run.py \
#     --engine_dir /data/llama3_int4_engine \
#     --tokenizer_dir /data/tllm_checkpoint_int4 \
#     --max_output_len 100 \
#     --input_text "Hello, tell me about Jetson Orin Nano."
