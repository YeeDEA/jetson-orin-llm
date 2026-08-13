# jetson-orin-llm

QLoRA fine-tuning and INT4 AWQ quantization of Llama-3 8B for TensorRT-LLM inference on a Jetson Orin Nano 8GB.

## Overview

An 8B-parameter model does not fit on an 8GB edge board in half precision, and it does not train on a free Colab T4 without 4-bit tricks either. This project works both ends of that squeeze: fine-tune Llama-3 8B Instruct with QLoRA (NF4) on a Colab T4 against a synthetic natural-language-to-robot-control dataset, quantize the result to INT4 AWQ weight-only, build a TensorRT-LLM engine, and run it inside the `dustynv/tensorrt_llm` container on a Jetson Orin Nano 8GB. The full flow was first rehearsed end-to-end on TinyLlama-1.1B (fast to iterate) before being applied to the 8B model.

**Status: archived experiment notebooks** — a 2-day sprint (December 2025), kept as a record of the pipeline and its failure modes. Not a maintained tool.

## Pipeline

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

## Notebooks (execution order)

| # | Notebook | What it does |
|---|---|---|
| 1 | `01-dataset-synthesis.ipynb` | Synthesizes 50,000 instruction/output pairs (natural-language command → robot control code) into `robot_dataset.json` using pure-Python templating |
| 2 | `02-qlora-llama3-8b.ipynb` | QLoRA fine-tuning of Llama-3 8B Instruct on the T4: NF4 double quantization, LoRA r=64 on all attention + MLP projections, TRL `SFTTrainer`. Contains the BF16/OOM debugging log (see below) |
| 3 | `03-tinyllama-awq-rehearsal.ipynb` | Full TensorRT-LLM rehearsal on **TinyLlama-1.1B-Chat**: `convert_checkpoint.py --use_weight_only --weight_only_precision int4_awq` → `trtllm-build` → `run.py` inference. Small model, same commands |
| 4 | `04-llama3-awq-orin-deploy.ipynb` | The same INT4 AWQ conversion applied to **Llama-3 8B**, plus the Jetson Orin deployment procedure: `docker run --runtime nvidia` with `dustynv/tensorrt_llm:r36.2.0` and engine files mounted from `/data` |

Notebooks 3 and 4 are deliberately separate: 3 is the cheap rehearsal that validates the toolchain, 4 is the real 8B run targeting the board.

## Hardware & versions

| Component | Value |
|---|---|
| Training GPU | Colab T4 (14.74 GiB usable, per the OOM traces) |
| Target device | Jetson Orin Nano 8GB |
| Orin runtime | `dustynv/tensorrt_llm:r36.2.0` (JetPack 6 / L4T r36.2) |
| TensorRT-LLM (Colab side) | pre-release from `https://pypi.nvidia.com` (`pip install tensorrt_llm -U --pre`), Dec 2025 |
| Models | `meta-llama/Meta-Llama-3-8B-Instruct` (also a 3.1-8B attempt), `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Training stack | `torch` / `transformers` / `peft` / `trl` / `bitsandbytes` / `datasets`, latest-at-the-time (no pins in the notebooks; see `requirements.txt` for loose bounds) |

The notebooks install unpinned latest versions, which is the single biggest reproducibility hazard here — the TensorRT-LLM pre-release channel and `transformers` both moved fast in this period.

## What worked / what broke

Kept verbatim from the notebook outputs, because this is where the actual engineering happened:

- **BF16 crash inside FP16 training.** The first `SFTTrainer` run died mid-training with `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'` — despite `fp16=True, bf16=False`. Some parameters were still materializing as bfloat16, which the FP16 grad scaler cannot unscale. The fix attempt in the next cell forces the issue: load in FP16, run `prepare_model_for_kbit_training` manually, attach LoRA via `get_peft_model` (passing `peft_config=None` to the trainer), then sweep every parameter and cast any surviving bfloat16 tensor to float16.
- **Then the T4 ran out of memory.** That manual `prepare_model_for_kbit_training` call upcasts non-4-bit params to FP32 and blew the 14.74 GiB card: `CUDA out of memory. Tried to allocate 1.96 GiB ... 13.49 GiB memory in use, 12.27 GiB allocated by PyTorch`. The notebook keeps a `del model/trainer; gc.collect(); torch.cuda.empty_cache()` cell for recovering between attempts.
- **What worked cleanly:** dataset synthesis (pure templating, no GPU), the TinyLlama INT4 AWQ rehearsal (convert → build → run), and the Orin container/deploy procedure.

No inference throughput or latency numbers were captured in the notebook outputs, so none are quoted here.

## Reproduce

Everything is Colab-oriented; run the notebooks in numeric order.

1. **HF access**: Llama-3 is gated. Store your token as a Colab Secret named `HF_TOKEN` (the notebooks call `huggingface_hub.login()`; never paste a token into a cell).
2. `01-dataset-synthesis.ipynb` — produces `robot_dataset.json` (50k pairs). CPU-only.
3. `02-qlora-llama3-8b.ipynb` — needs a T4 (or better). Expect to fight the dtype/OOM issues above on a 16GB card; the last cell variant is the furthest-debugged one.
4. `03-tinyllama-awq-rehearsal.ipynb` — validates your TensorRT-LLM install end-to-end in minutes.
5. `04-llama3-awq-orin-deploy.ipynb` — convert/build for the 8B model, copy the engine to the Orin, and run it inside `dustynv/tensorrt_llm:r36.2.0` (`docker run -it --rm --runtime nvidia --network host ...` with the engine directory mounted).

## Provenance

These were Colab notebooks organized later; original names and dates:

| Current file | Original name | Date |
|---|---|---|
| `01-dataset-synthesis.ipynb` | `Llama 데이터셋 만들기.ipynb` | 2025-12-16 |
| `02-qlora-llama3-8b.ipynb` | `Quantization_0.ipynb` | 2025-12-16 |
| `03-tinyllama-awq-rehearsal.ipynb` | `양자화테스트.ipynb` | 2025-12-16 |
| `04-llama3-awq-orin-deploy.ipynb` | `양자화테스트_orin_초기버전.ipynb` | 2025-12-15 |

Work period: 2025-12-15 – 2025-12-16. Original Korean README preserved at [`docs/README.ko.md`](docs/README.ko.md).

## Limitations & future work

- The Llama-3 8B QLoRA run never completed on the free T4 — the last recorded state is the FP32-upcast OOM. A larger card (A100/L4) or removing the manual upcast would be the next step.
- No on-device throughput/memory measurements were logged; benchmarking the built engine on the Orin is the obvious missing piece.
- The synthetic dataset is templated, not model-generated — coverage is narrow by construction.
- Versions are unpinned pre-releases; expect command-line flags of `convert_checkpoint.py` / `trtllm-build` to have changed since.

## License

MIT — see [LICENSE](LICENSE).
