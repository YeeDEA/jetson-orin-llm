# jetson-orin-llm

[![tests](https://github.com/YeeDEA/jetson-orin-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/YeeDEA/jetson-orin-llm/actions/workflows/tests.yml)

A pipeline aimed at running a QLoRA fine-tune of Llama-3 8B as an INT4 AWQ TensorRT-LLM engine on a Jetson Orin Nano 8GB: synthetic data, QLoRA on a free Colab T4, AWQ conversion, and an on-board deployment procedure for the `dustynv/tensorrt_llm` container.

**The Jetson Orin board is no longer available, so no on-device Orin or TensorRT-LLM number was ever measured and none will be.** The Llama-3 8B QLoRA run also never completed (see [Where it broke](#where-it-broke)). Every measured result in this repository comes from an RTX 5050 laptop GPU: a completed TinyLlama-1.1B QLoRA fine-tune, its held-out accuracy at FP16 and NF4, bitsandbytes latency/memory, and a memory measurement of the k-bit prep step that caused the 8B OOM. The Orin notebooks and scripts are kept as the record of the intended deployment path, not as validated results.

*Work period: 2025-12-15 – 2025-12-16, in Google Colab. Published here Aug 2026, when those notebooks were reorganized into this repository — so the git history starts at that import, not at the work. File-by-file mapping in [Provenance](#provenance).*

The problem is a squeeze from both ends. An 8B-parameter model does not fit on an 8GB edge board in half precision, and it does not train on a free Colab T4 without 4-bit tricks either. So the pipeline works both ends of it:

```
synthesize 50k NL-command -> robot-code pairs    (CPU, pure Python templating)
QLoRA fine-tune Llama-3 8B: NF4 + LoRA r=64      (Colab T4)
INT4 AWQ convert -> trtllm-build -> run          (Colab, then Orin Nano 8GB)
```

The full flow was rehearsed end-to-end on TinyLlama-1.1B first — it iterates in minutes instead of hours — before the same commands were pointed at the 8B model.

## Measured results (RTX 5050 laptop GPU, not the Orin)

Added Sep 2026. **Everything in this section was measured on an NVIDIA GeForce RTX 5050 Laptop GPU (8 GB, driver 592.00), Intel Core 5 210H, Windows 11, Python 3.12.10, torch 2.13.0+cu130, transformers 4.57.6, bitsandbytes 0.50.2, peft 0.21.0** (pinned in `requirements.txt`). None of it is a Jetson Orin number and none of it uses TensorRT-LLM: the quantization here is bitsandbytes NF4/INT8 in plain PyTorch, which is a different thing from the INT4 AWQ TensorRT-LLM engine the pipeline targets. Every number below is copied from a JSON file in `results/`, and each JSON records the exact command that produced it.

### Accuracy: what fine-tuning bought, what 4-bit cost

TinyLlama-1.1B-Chat, QLoRA (NF4 base, LoRA r=64/alpha=16 on all attention + MLP projections, bf16 compute, lr 2e-4, batch 16, 1 epoch over 8,000 synthetic training examples = 500 steps). Training took 373 s, peak training memory 1.82 GB, final logged loss 0.0136 (`results/train_tinyllama.json`).

Held-out: 500 examples drawn by `make_splits(8000, 500, seed=0)` from a separate seeded RNG stream and never passed to the trainer. Greedy decoding, max 48 new tokens. *Exact match* = output string equals the reference; *valid code* = output parses and every statement is `robot.<known method>(<literals>)`.

| Model | Weights | Exact match (500) | Valid code (500) | Exact match, unseen-string subset (92) | Eval peak mem (GB) |
|---|---|---|---|---|---|
| TinyLlama base, zero-shot | FP16 | 0.0% | 0.0% | 0.0% | 2.202 |
| TinyLlama base, 3-shot | FP16 | 47.8% | 58.8% | 83.7% | 2.57 |
| TinyLlama + LoRA, merged | FP16 | **91.0%** | 100% | 100% | 2.18 |
| TinyLlama + LoRA, merged, then quantized | NF4 | **91.0%** | 100% | 100% | 0.85 |

So on this task **NF4 cost 0.0 points of exact match** (455/500 in both) for **2.6x less eval memory**, and, from the benchmark below, about 21% lower decode throughput on this GPU. The base model zero-shot answers in prose ("To navigate to the bathroom in your home, follow these steps..."), so the zero-shot 0% is a formatting failure; the 3-shot row is the fairer baseline.

Two things about the dataset that these numbers exposed, and that change how to read them:

- **All 45 fine-tuned misses are `rotate` commands, and most are unanswerable.** Patterns like "Turn 90 degrees" or "Spin 180 degrees" contain no direction, but the generator still picks a random direction and signs the angle with it (`robot.rotate(90)` vs `robot.rotate(-90)`). 99 of the 135 held-out rotate examples have no direction word, so their label is a coin flip; rotate exact match is 66.7%, every other command type is 100%. About 91% is roughly the ceiling this dataset allows, not a model limit.
- **The template space is tiny.** There are only 24 distinct "grab" instructions and 90 distinct compound ones, so a 50k-example dataset is mostly exact repeats. Only 92 of the 500 held-out examples have an instruction string that never appears in the 8,000 training examples (almost all `move` with an unseen distance). That subset is reported separately above; it is 100% for the fine-tuned model, but it is a weak generalization test.

Commands:
```
python scripts/train_qlora.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --train-count 8000 --test-count 500 --seed 0 --adapter-dir runs/tinyllama_lora --stats results/train_tinyllama.json
python scripts/eval_accuracy.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant fp16 --tag base_fp16_0shot --out results/eval_base_fp16_0shot.json
python scripts/eval_accuracy.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant fp16 --shots 3 --tag base_fp16_3shot --out results/eval_base_fp16_3shot.json
python scripts/eval_accuracy.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --merge-adapter runs/tinyllama_lora --merged-dir runs/tinyllama_merged --quant fp16 --tag ft_fp16 --out results/eval_ft_fp16.json
python scripts/eval_accuracy.py --model runs/tinyllama_merged --quant int4-bnb --tag ft_nf4 --out results/eval_ft_nf4.json
```

### Latency and memory

Batch 1, 512-token prompt, 128 generated tokens, greedy, median of 10 timed runs after 3 warmups; the decode rate excludes prefill, and TTFT is reported separately. Peak memory is `torch.cuda.max_memory_allocated` during the timed runs (the load-time peak is in the JSON as `weights_only_memory_gb`).

| Model | Precision | Decode tok/s, median (min-max) | TTFT (s) | Peak memory (GB) | Decode vs FP16 |
|---|---|---|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 | FP16 | 22.3 (15.6-24.7) | 0.118 | 2.19 | 1.00x |
| TinyLlama-1.1B-Chat-v1.0 | INT8 (bitsandbytes) | 7.0 (5.3-8.1) | 0.206 | 1.31 | 0.32x |
| TinyLlama-1.1B-Chat-v1.0 | NF4 (bitsandbytes) | 17.5 (11.8-18.8) | 0.127 | 0.86 | 0.79x |
| Qwen2.5-7B-Instruct | NF4 (bitsandbytes) | 12.5 (9.3-15.4) | 0.673 | 5.43 | n/a |

How to read it:

- **On this setup bitsandbytes 4-bit is a memory format, not a speed format.** NF4 cut TinyLlama's peak memory 2.5x and made decode slower, not faster; INT8 (LLM.int8) was about 3x slower. Weight-only INT4 *kernels* (AWQ in TensorRT-LLM) are what can turn fewer bytes into more tokens/s; these rows do not measure that.
- **TinyLlama decode here looks host-bound, not bandwidth-bound**: the 7B model runs at more than half the 1.1B model's rate despite about 6x the parameters, and run-to-run spread is wide (15.6-24.7 tok/s for one config). Eager-mode HF `generate` on a laptop CPU spends much of each step launching kernels. Treat the TinyLlama rows as "this stack on this laptop", not as a GPU capability figure.
- **An 8B-class model fits in 8 GB at NF4** with a 512-token prompt (5.43 GB peak). Llama-3-8B itself is gated, so the ungated Qwen2.5-7B-Instruct stands in; its FP16 weights (about 15 GB on disk) do not fit this card and were not attempted.
- AWQ/GPTQ via `autoawq` / `gptqmodel` were tried and skipped: neither installs as a wheel on this Windows + torch 2.13 setup, and both source builds failed (`No module named 'torch'` in the isolated build; `unable to find vswhere.exe`).

Commands: `python scripts/bench.py --model <id> --quant {fp16,int8-bnb,int4-bnb} --out results/bench_rtx5050_<name>_<quant>.json`; table from `python scripts/bench_table.py results/bench_*.json`.

### The k-bit prep upcast, measured

Wall 2 below leaves one question open: does the manual `prepare_model_for_kbit_training` fp32 upcast explain the 8B OOM? Measured on TinyLlama with the notebook's training shape (batch 4, padded to 512 tokens, LoRA r=64, fp16 compute, 10 steps), each variant in a fresh process (`python scripts/kbit_prep_memory.py --variant {prep,noprep,notebook} --out results/kbit_prep_<variant>.json`):

| Variant | Non-4-bit params | After load + prep (GB) | Training peak (GB) |
|---|---|---|---|
| `prepare_model_for_kbit_training` (upcast + grad checkpointing) | 131.2M in fp32 | 1.153 | 2.436 |
| no prep, grad checkpointing on | 131.2M in fp16 | 0.909 | 1.898 |
| notebook cell-4 path (prep + LoRA cast to fp16) | 131.2M in fp32 | 1.058 | did not train: `ValueError: Attempting to unscale FP16 gradients.` |

- Removing the upcast lowered peak training memory by **0.54 GB (22%)** on TinyLlama, with gradient checkpointing held constant. The 131.2M non-4-bit parameters are the embedding, `lm_head` and norms.
- Scaling that to the 8B run is arithmetic, not a measurement: Llama-3-8B's embedding matrix alone is 128,256 x 4,096 = 525M parameters, which at fp32 is 1.96 GiB, the size of the allocation that failed in Wall 2. That is consistent with the OOM being the upcast of the embedding (and then `lm_head`), not the 4-bit weights. It is still not an 8B measurement.
- The notebook's dtype workaround would not have trained even with enough memory, at least on today's library versions: with the LoRA weights cast to fp16, the fp16 GradScaler refuses to unscale them. Keeping LoRA in fp32 (the fixed `load_model` path in `src/jetson_llm/train.py`) is what fixes Wall 1 cleanly.

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
- **Hypothesis for the next attempt**: drop the manual upcast, or move to a card with more headroom (A100/L4). The with/without-upcast pair has since been measured on TinyLlama (see [The k-bit prep upcast, measured](#the-k-bit-prep-upcast-measured)); it has not been measured on the 8B model.

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
   `python scripts/train_qlora.py --model-id meta-llama/Meta-Llama-3-8B-Instruct` (the script builds its own seeded split; the TinyLlama run in the results section is the one that was actually completed)
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
| Training GPU | Colab T4 (14.74 GiB usable, per the OOM traces); Sep 2026 re-runs: RTX 5050 Laptop 8 GB |
| Target device | Jetson Orin Nano 8GB (no longer available; never measured) |
| Orin runtime | `dustynv/tensorrt_llm:r36.2.0` (JetPack 6 / L4T r36.2) |
| TensorRT-LLM (Colab side) | pre-release from `https://pypi.nvidia.com` (`pip install tensorrt_llm -U --pre`), Dec 2025 |
| Models | `meta-llama/Meta-Llama-3-8B-Instruct` (also a 3.1-8B attempt), `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Training stack | `torch` / `transformers` / `peft` / `trl` / `bitsandbytes` / `datasets`, latest-at-the-time (no pins in the notebooks); `requirements.txt` pins the versions used for the Sep 2026 measurements |

The notebooks install unpinned latest versions, which is the single biggest reproducibility hazard here — the TensorRT-LLM pre-release channel and `transformers` both moved fast in this period.

## Not measured (board no longer available)

This list is closed: the Jetson Orin Nano is no longer available, so none of the following was measured and none will be.

- Decode throughput, TTFT and peak memory of the INT4 AWQ TensorRT-LLM engine on the Orin (and of the TinyLlama engine or its FP16 source as a control), and peak memory during `trtllm-build` on the board.
- Accuracy of the AWQ engine on the held-out split (the accuracy numbers above are bitsandbytes NF4 on the laptop GPU, not AWQ).
- The Llama-3 8B QLoRA run itself, with or without the upcast (only TinyLlama was measured).

Notebooks 03 and 04 were saved with their output cells empty, so the Dec 2025 TensorRT-LLM runs left no logs, and no TensorRT-LLM install was available for the Sep 2026 laptop measurements.

## Repository layout

```
notebooks/                     original Colab experiments (run in numeric order)
  01-dataset-synthesis.ipynb
  02-qlora-llama3-8b.ipynb
  03-tinyllama-awq-rehearsal.ipynb
  04-llama3-awq-orin-deploy.ipynb
src/jetson_llm/                importable modules
  data.py                      dataset synthesis (notebook 01) + seeded train/held-out splits
  train.py                     QLoRA training, dtype-fixed; notebook cell-4 path kept for comparison
  evaluate.py                  exact-match / valid-robot-code scoring
scripts/                       thin entrypoints over src/ and the shell command sequences
  generate_dataset.py          CLI for dataset synthesis
  train_qlora.py               QLoRA fine-tune on the seeded split
  eval_accuracy.py             held-out accuracy (base / merged fp16 / NF4)
  bench.py, bench_table.py     decode tok/s, TTFT, peak memory -> JSON -> README table
  kbit_prep_memory.py          training peak memory with vs without k-bit prep
  tinyllama_awq_rehearsal.sh   convert -> build -> run (from notebook 03)
  llama3_awq_convert.sh        Llama-3 8B INT4 AWQ conversion, Colab side (notebook 04)
  orin_deploy.sh               Jetson container/build/run procedure (notebook 04)
results/                       JSON record of every measured number, with its command
tests/                         CPU-only pytest suite (run in GitHub Actions)
docs/README.ko.md              original Korean README
```

The notebooks are the original Colab experiments and are unchanged. `src/` started as the same code extracted verbatim; in Sep 2026 `train.py` was fixed (dtype handling, labels, optional k-bit prep, all listed in its docstring) and `data.py` gained seeded splits.

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

- The Llama-3 8B QLoRA run never completed on the free T4; the last recorded state is the FP32-upcast OOM. The TinyLlama measurement supports removing the upcast; it is untested at 8B.
- A TinyLlama QLoRA fine-tune was completed and evaluated on a laptop RTX 5050 (91.0% exact match on held-out, unchanged at NF4). No on-device Orin / TensorRT-LLM numbers exist; the board is no longer available, see [Not measured](#not-measured-board-no-longer-available).
- The synthetic dataset is templated and narrow (24 distinct grab instructions), and direction-less rotate commands get a random sign, which makes about 20% of held-out labels unpredictable.
- `requirements.txt` pins the Sep 2026 laptop environment; the Dec 2025 Colab versions were never recorded, and `convert_checkpoint.py` / `trtllm-build` flags have likely changed since.
- CPU tests (`python -m pytest tests`) cover dataset determinism/format, splits and the scorer; the GPU scripts are not covered by CI.

## License

MIT — see [LICENSE](LICENSE).
