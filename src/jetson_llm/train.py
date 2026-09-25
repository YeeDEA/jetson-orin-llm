"""QLoRA fine-tuning on the robot-command dataset.

Originally extracted from notebooks/02-qlora-llama3-8b.ipynb (cell 4 variant).
That notebook path is kept as ``load_model_notebook`` because the README's
memory experiment compares against it. The default path (``load_model``) is a
cleaned-up version; what changed and why:

1. **Dtype handling.** The notebook forced ``fp16=True`` and then hand-cast
   every surviving bfloat16 tensor (including LoRA weights) to float16. That
   was a workaround for ``_amp_foreach_non_finite_check_and_unscale_cuda not
   implemented for 'BFloat16'``: the fp16 GradScaler cannot unscale bf16
   grads. The clean fix is to (a) pick ONE compute dtype for the GPU
   (bf16 where supported, fp16 otherwise), use it for both the 4-bit compute
   dtype and the non-quantized weights, and set the matching trainer flag,
   and (b) keep the trainable LoRA parameters in float32 (peft's default) so
   the scaler/optimizer always sees fp32 master weights. No manual sweeps.
2. **k-bit prep is optional** (``prepare_kbit``). ``prepare_model_for_kbit_training``
   upcasts every non-4-bit parameter to fp32 and enables gradient
   checkpointing; that upcast is where the notebook OOMed on the T4. With it
   off, gradient checkpointing is still enabled explicitly, so the two
   settings differ only in the upcast.
3. **Prompt format** comes from ``tokenizer.apply_chat_template`` instead of a
   hard-coded Llama-3 header string, so the same code trains TinyLlama and
   Llama-3.
4. **Labels.** The notebook padded every example to 512 tokens and copied
   ``input_ids`` to ``labels``; with ``pad_token = eos_token`` that trains the
   model on hundreds of padding tokens per example. Labels are now -100 on
   the prompt and on padding, and padding is dynamic per batch.
5. Uses ``transformers.Trainer`` with a small collator instead of
   ``SFTTrainer``: the data is pre-tokenized, and TRL's API has changed
   repeatedly since the notebook was written.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def pick_compute_dtype():
    """bf16 if the GPU supports it, else fp16 (e.g. T4)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_bnb_config(compute_dtype=torch.float16):
    """BitsAndBytes config: 4-bit NF4, double quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(r=64, alpha=16):
    """LoRA on all attention + MLP projections (notebook: r=64, alpha=16)."""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_id=DEFAULT_MODEL_ID, prepare_kbit=True, compute_dtype=None, lora_r=64):
    """Cleaned-up QLoRA load: one compute dtype, fp32 LoRA params, optional k-bit prep."""
    compute_dtype = compute_dtype or pick_compute_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=build_bnb_config(compute_dtype),
        device_map={"": 0},
        dtype=compute_dtype,
    )
    model.config.use_cache = False
    if prepare_kbit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.enable_input_require_grads()
    model = get_peft_model(model, build_lora_config(r=lora_r))
    # LoRA weights stay fp32 (peft default) -> no bf16 grads reach an fp16 GradScaler.
    return model


def load_model_notebook(model_id=DEFAULT_MODEL_ID):
    """The notebook cell-4 path, unchanged in substance (fp16 + manual prep + casts).

    Kept only so the memory comparison in scripts/kbit_prep_memory.py measures
    the code that actually OOMed.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        ),
        device_map={"": 0},
        dtype=torch.float16,
    )
    model.config.torch_dtype = torch.float16
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config())
    for _, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    for name, module in model.named_modules():
        if "lora" in name or "adapter" in name:
            module.to(torch.float16)
    return model


def format_prompt(tokenizer, instruction):
    """Chat-formatted prompt string ending with the assistant turn header."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
    )


def tokenize_example(tokenizer, example, max_length=128):
    """input_ids / labels with loss only on the completion (+ EOS)."""
    prompt_ids = tokenizer(format_prompt(tokenizer, example["instruction"]), add_special_tokens=False).input_ids
    answer_ids = tokenizer(example["output"], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return {"input_ids": ids, "labels": labels}


class PadCollator:
    """Right-pads a batch; pad positions get attention 0 and label -100."""

    def __init__(self, pad_id, pad_to=None):
        self.pad_id = pad_id
        self.pad_to = pad_to

    def __call__(self, feats):
        n = self.pad_to or max(len(f["input_ids"]) for f in feats)
        ids, labels, mask = [], [], []
        for f in feats:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * k)
            labels.append(f["labels"] + [-100] * k)
            mask.append([1] * len(f["input_ids"]) + [0] * k)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


def build_trainer(model, tokenizer, train_features, output_dir, compute_dtype,
                  epochs=1, batch_size=16, grad_accum=1, lr=2e-4, max_steps=-1,
                  pad_to=None, logging_steps=10):
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        optim="paged_adamw_32bit",
        weight_decay=0.001,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        logging_steps=logging_steps,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=0,
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_features,
        data_collator=PadCollator(tokenizer.pad_token_id, pad_to=pad_to),
    )


def train(model_id=DEFAULT_MODEL_ID, train_samples=None, output_dir="./runs/qlora",
          adapter_dir="./trained_adapter", prepare_kbit=True, epochs=1, batch_size=16,
          lr=2e-4, lora_r=64, max_length=128):
    """Load -> tokenize -> train -> save adapter. Returns (trainer, stats)."""
    compute_dtype = pick_compute_dtype()
    tokenizer = load_tokenizer(model_id)
    model = load_model(model_id, prepare_kbit=prepare_kbit, compute_dtype=compute_dtype, lora_r=lora_r)
    feats = [tokenize_example(tokenizer, s, max_length) for s in train_samples]
    trainer = build_trainer(model, tokenizer, feats, output_dir, compute_dtype,
                            epochs=epochs, batch_size=batch_size, lr=lr)
    torch.cuda.reset_peak_memory_stats()
    out = trainer.train()
    stats = {
        "train_runtime_s": round(out.metrics["train_runtime"], 1),
        "train_loss": round(out.metrics["train_loss"], 4),
        "peak_train_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
        "log_history": trainer.state.log_history,
    }
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return trainer, stats
