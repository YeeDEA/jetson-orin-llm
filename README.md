# jetson-orin-llm

A QLoRA fine-tune of Llama-3 8B, quantized to INT4 AWQ and built into a TensorRT-LLM engine for a Jetson Orin Nano 8GB — trained on a free Colab T4, run on the board inside the `dustynv/tensorrt_llm` container.

*Work period: 2025-12-15 – 2025-12-16, in Google Colab. Published here Aug 2026, when those notebooks were reorganized into this repository — so the git history starts at that import, not at the work. File-by-file mapping in [Provenance](#provenance).*

The problem is a squeeze from both ends. An 8B-parameter model does not fit on an 8GB edge board in half precision, and it does not train on a free Colab T4 without 4-bit tricks either. So the pipeline works both ends of it:

```
synthesize 50k NL-command -> robot-code pairs    (CPU, pure Python templating)
QLoRA fine-tune Llama-3 8B: NF4 + LoRA r=64      (Colab T4)
INT4 AWQ convert -> trtllm-build -> run          (Colab, then Orin Nano 8GB)
```

The full flow was rehearsed end-to-end on TinyLlama-1.1B first — it iterates in minutes instead of hours — before the same commands were pointed at the 8B model.

## Where it broke

Two days, two walls, both in the QLoRA step. This is where the actual engineering happened, so it goes first rather than in a footnote. Everything below is kept verbatim from the outputs of `notebooks/02-qlora-llama3-8b.ipynb`.

### Wall 1 — BF16 gradients inside a run I had forced to FP16

The first `SFTTrainer` run (Llama-3.1-8B-Instruct, LoRA handed to the trainer via `peft_config`) died mid-training:

```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
```

despite `fp16=True, bf16=False` in the training args. The traceback ends in `torch/amp/grad_scaler.py` → `unscale_` → `_unscale_grads_` → `torch._amp_foreach_non_finite_check_and_unscale_`: some parameters were still materializing as bfloat16, and the FP16 grad scaler has no bfloat16 kernel to unscale them with. Setting `bf16=False` controls what the trainer does; it does not control what dtype the parameters arrive in.

So the next cell stops asking politely and does the dtype work by hand: load in FP16, run `prepare_model_for_kbit_training` manually, attach LoRA with `get_peft_model` (and pass `peft_config=None` to the trainer, since LoRA is already on), then sweep every parameter and cast any surviving bfloat16 tensor to float16 — plus a second sweep forcing every `lora`/`adapter` module to float16. That variant is preserved in `src/jetson_llm/train.py`, comments and all.

### Wall 2 — that fix bought me an out-of-memory

The manual `prepare_model_for_kbit_training` call is not free. Inside peft it walks the parameters and upcasts every non-`Params4bit` fp16/bf16 tensor to fp32, and on the T4 that is exactly where it died:

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 1.96 GiB.
GPU 0 has a total capacity of 14.74 GiB of which 1.25 GiB is free.
Process 101462 has 13.49 GiB memory in use. Of the allocated memory
12.27 GiB is allocated by PyTorch, and 1.09 GiB is reserved by PyTorch
but unallocated.
```

Crash line: `param.data = param.data.to(torch.float32)` in `peft/utils/other.py`.

### What I take from the memory math

- **The 4-bit weights were never the problem.** 13.49 GiB was already in use when the FP32 upcast asked for another 1.96 GiB and found 1.25 GiB free. NF4 buys you the room to load the model; it does not buy you the room to then promote everything that isn't `Params4bit` to fp32 while the fp16 copy is still resident.
- **Fixing a dtype bug by hand moves the memory peak.** Pulling `prepare_model_for_kbit_training` out of `SFTTrainer` was the right move for controlling dtypes and the wrong move for controlling the peak — I now owned the ordering of load → prep → LoRA → sweep, and put the most expensive step at the worst moment.
- **The recovery cell matters as much as the training cell.** `del model / del trainer; gc.collect(); torch.cuda.empty_cache()` is kept in the notebook because on a single-GPU Colab session you otherwise pay for a fresh runtime after every failed attempt.
- **Untested hypothesis for the next attempt** (not something this repo demonstrates): drop the manual upcast and let the trainer do k-bit prep, or move to a card with more headroom (A100/L4). Which of the two is actually needed is a memory measurement nobody has taken — see [What nobody measured](#what-nobody-measured).

The Llama-3 8B QLoRA run therefore never completed. The FP32-upcast OOM is the last recorded state.

## What ran clean

- Dataset synthesis — pure templating, no GPU, 50,000 instruction/output pairs into `robot_dataset.json`.
- The TinyLlama-1.1B INT4 AWQ rehearsal: `convert_checkpoint.py --use_weight_only --weight_only_precision int4_awq` → `trtllm-build` → `run.py`.
- The Orin container and deployment procedure: `docker run --runtime nvidia` with `dustynv/tensorrt_llm:r36.2.0` and the engine files mounted from `/data`.

## The pipeline in full

```
 SFT data synthesis          QLoRA fine-tune           INT4 AWQ quantize
 (templated NL command  -->  (Llama-3 8B, NF4 +   -->  (TensorRT-LLM
  -> robot-control code,      LoRA r=64, Colab T4)      convert_checkpoint,
  50k pairs)                                            weight-only int4_awq)
                                                              |
                                                              v
       Jetson Orin deploy            <--            TensorRT-LLM engine build
       (dustynv/tensorrt_llm:r36.2.0                (trtllm-build)
        Docker on Orin Nano 8GB)
```

## Running it

Everything is Colab-oriented; run the notebooks in numeric order. `scripts/` holds the same steps as thin CLIs and shell command sequences if you would rather not open a notebook.

**First: HF access.** Llama-3 is gated. Store your token as a Colab Secret named `HF_TOKEN` (the notebooks call `huggingface_hub.login()`; never paste a token into a cell).

1. `notebooks/01-dataset-synthesis.ipynb` — produces `robot_dataset.json` (50k pairs). CPU-only.
   `python scripts/generate_dataset.py --count 50000 --output robot_dataset.json`
2. `notebooks/02-qlora-llama3-8b.ipynb` — needs a T4 (or better). Expect to fight the dtype/OOM issues above on a 16GB card; the last cell variant is the furthest-debugged one.
   `python scripts/train_qlora.py --data robot_dataset.json`
3. `notebooks/03-tinyllama-awq-rehearsal.ipynb` — validates your TensorRT-LLM install end-to-end in minutes.
   `bash scripts/tinyllama_awq_rehearsal.sh`
4. `notebooks/04-llama3-awq-orin-deploy.ipynb` — convert/build for the 8B model, copy the engine to the Orin, and run it inside `dustynv/tensorrt_llm:r36.2.0` (`docker run -it --rm --runtime nvidia --network host ...` with the engine directory mounted).
   `bash scripts/llama3_awq_convert.sh` on the Colab side, then `scripts/orin_deploy.sh` as the on-board procedure.

## The notebooks

| # | Notebook | What it does |
|---|---|---|
| 1 | `notebooks/01-dataset-synthesis.ipynb` | Synthesizes 50,000 instruction/output pairs (natural-language command → robot control code) into `robot_dataset.json` using pure-Python templating |
| 2 | `notebooks/02-qlora-llama3-8b.ipynb` | QLoRA fine-tuning of Llama-3 8B Instruct on the T4: NF4 double quantization, LoRA r=64 on all attention + MLP projections, TRL `SFTTrainer`. Contains the BF16/OOM debugging log quoted above |
| 3 | `notebooks/03-tinyllama-awq-rehearsal.ipynb` | Full TensorRT-LLM rehearsal on **TinyLlama-1.1B-Chat**: `convert_checkpoint.py --use_weight_only --weight_only_precision int4_awq` → `trtllm-build` → `run.py` inference. Small model, same commands |
| 4 | `notebooks/04-llama3-awq-orin-deploy.ipynb` | The same INT4 AWQ conversion applied to **Llama-3 8B**, plus the Jetson Orin deployment procedure: `docker run --runtime nvidia` with `dustynv/tensorrt_llm:r36.2.0` and engine files mounted from `/data` |

Notebooks 3 and 4 are deliberately separate: 3 is the cheap rehearsal that validates the toolchain, 4 is the real 8B run targeting the board.

## Hardware and versions

| Component | Value |
|---|---|
| Training GPU | Colab T4 (14.74 GiB usable, per the OOM traces) |
| Target device | Jetson Orin Nano 8GB |
| Orin runtime | `dustynv/tensorrt_llm:r36.2.0` (JetPack 6 / L4T r36.2) |
| TensorRT-LLM (Colab side) | pre-release from `https://pypi.nvidia.com` (`pip install tensorrt_llm -U --pre`), Dec 2025 |
| Models | `meta-llama/Meta-Llama-3-8B-Instruct` (also a 3.1-8B attempt), `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Training stack | `torch` / `transformers` / `peft` / `trl` / `bitsandbytes` / `datasets`, latest-at-the-time (no pins in the notebooks; see `requirements.txt` for loose bounds) |

The notebooks install unpinned latest versions, which is the single biggest reproducibility hazard here — the TensorRT-LLM pre-release channel and `transformers` both moved fast in this period.

## What nobody measured

No inference throughput or latency numbers were captured in the notebook outputs, so none are quoted here. That is a gap in the work, not a stylistic choice, and it is the first thing I would close. Concretely, the claims in this README become verifiable when these exist — and only when each is tagged with the precision it was taken at:

- **Decode throughput (tokens/s)** for the INT4 AWQ weight-only engine on the Orin Nano 8GB, at batch size 1 and the engine's build settings (`--max_batch_size 1`, `--max_input_len 2048`, `--max_output_len 512`).
- **Time to first token**, at a stated prompt length, on the same engine — separate from decode throughput, because prefill and decode are limited by different things and a single "speed" number hides which one is the bottleneck.
- **Peak GPU memory during inference on the board**, which is the number that actually decides whether "it fits in 8GB" is true, plus peak memory during `trtllm-build` (the build, not the run, is often what fails on an edge device).
- **The same three for the TinyLlama-1.1B engine**, as a control: it went through an identical command path, so it isolates whether a problem is model-size-related or toolchain-related.
- **A baseline to compare against** — the same three numbers for the FP16 checkpoint the engine was converted from. INT4 AWQ figures with no FP16 reference say nothing about what quantization bought.
- **Peak training memory for the QLoRA run with and without the manual `prepare_model_for_kbit_training` upcast.** That single pair of numbers would settle whether the 8B fine-tune needs a bigger card or just a different ordering, which is currently guesswork.

None of these are recorded anywhere in the repo. Notebooks 03 and 04 were saved with their output cells empty, so even the qualitative run logs are gone; the only quantitative outputs preserved are the two failure tracebacks in notebook 02. Until the list above is filled in, treat this repository as a recorded command path with a documented failure analysis, not as a performance result.

## Repository layout

```
notebooks/                     original Colab experiments (run in numeric order)
  01-dataset-synthesis.ipynb
  02-qlora-llama3-8b.ipynb
  03-tinyllama-awq-rehearsal.ipynb
  04-llama3-awq-orin-deploy.ipynb
src/jetson_llm/                the same code extracted into importable modules
  data.py                      dataset synthesis (from notebook 01)
  train.py                     QLoRA setup + training (from notebook 02, cell-4 variant)
scripts/                       thin entrypoints over src/ and the shell command sequences
  generate_dataset.py          CLI for dataset synthesis
  train_qlora.py               CLI for the QLoRA run
  tinyllama_awq_rehearsal.sh   convert -> build -> run (from notebook 03)
  llama3_awq_convert.sh        Llama-3 8B INT4 AWQ conversion, Colab side (notebook 04)
  orin_deploy.sh               Jetson container/build/run procedure (notebook 04)
docs/README.ko.md              original Korean README
```

The notebooks are the original Colab experiments; `src/` is the same code extracted into importable modules (no new functionality, bugs preserved).

## Provenance

These were Colab notebooks organized later; original names and dates:

| Current file | Original name | Date |
|---|---|---|
| `notebooks/01-dataset-synthesis.ipynb` | `Llama 데이터셋 만들기.ipynb` | 2025-12-16 |
| `notebooks/02-qlora-llama3-8b.ipynb` | `Quantization_0.ipynb` | 2025-12-16 |
| `notebooks/03-tinyllama-awq-rehearsal.ipynb` | `양자화테스트.ipynb` | 2025-12-16 |
| `notebooks/04-llama3-awq-orin-deploy.ipynb` | `양자화테스트_orin_초기버전.ipynb` | 2025-12-15 |

Original Korean README preserved at [`docs/README.ko.md`](docs/README.ko.md).

## Status, limits, open ends

**Status: archived experiment notebooks** — a 2-day sprint (December 2025), kept as a record of the pipeline and its failure modes. Not a maintained tool.

- The Llama-3 8B QLoRA run never completed on the free T4 — the last recorded state is the FP32-upcast OOM. A larger card (A100/L4) or removing the manual upcast would be the next step.
- No on-device throughput/memory measurements were logged; benchmarking the built engine on the Orin is the obvious missing piece, spelled out in [What nobody measured](#what-nobody-measured).
- The synthetic dataset is templated, not model-generated — coverage is narrow by construction.
- Versions are unpinned pre-releases; expect command-line flags of `convert_checkpoint.py` / `trtllm-build` to have changed since.

## License

MIT — see [LICENSE](LICENSE).
