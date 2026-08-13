"""QLoRA fine-tuning of Llama-3 8B Instruct on the robot-command dataset.

Extracted from notebooks/02-qlora-llama3-8b.ipynb. That notebook contains two
training attempts:

- cell 3: the first attempt (Llama-3.1-8B-Instruct, LoRA passed via
  ``peft_config`` to SFTTrainer) — died mid-training with
  ``NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
  not implemented for 'BFloat16'``.
- cell 4: the furthest-debugged variant (Llama-3-8B-Instruct, manual
  ``prepare_model_for_kbit_training`` + ``get_peft_model`` + a sweep casting
  surviving bfloat16 params to float16). This module preserves that variant.

NOTE: On a Colab T4 (14.74 GiB) the manual ``prepare_model_for_kbit_training``
call upcasts non-4-bit params to FP32 and the run OOMed — that is the last
recorded state in the notebook. The logic is preserved as-is, not fixed.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def build_bnb_config():
    """BitsAndBytes config: 4-bit NF4, FP16 compute, double quantization."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # compute dtype FP16
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )


def build_lora_config():
    """LoRA r=64 on all attention + MLP projections."""
    return LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_model_and_tokenizer(model_id=DEFAULT_MODEL_ID):
    """Load the 4-bit quantized model, apply k-bit prep + LoRA, purge bfloat16.

    Faithful to notebook cell 4, including the dtype "surgery" steps.
    """
    bnb_config = build_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,  # load dtype FP16
    )

    # [surgery step 1] force model config
    model.config.torch_dtype = torch.float16
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # [surgery step 2] k-bit training prep (casts LayerNorm etc. to fp32),
    # pulled out of SFTTrainer to control it directly.
    # NOTE: this upcast is what blew the T4's memory in the notebook.
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # [surgery step 3] attach LoRA manually, then hunt down surviving bfloat16
    model = get_peft_model(model, build_lora_config())

    print("BFloat16 잔재 제거 중...")
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            print(f"변환됨: {name}")
            param.data = param.data.to(torch.float16)

    # make sure LoRA layers are float16 too
    for name, module in model.named_modules():
        if 'lora' in name or 'adapter' in name:
            module.to(torch.float16)
    print("타입 정리 완료.")

    return model, tokenizer


def load_and_preprocess_dataset(tokenizer, data_files="robot_dataset.json"):
    """Load the JSON dataset and tokenize it in Llama-3 chat format."""
    dataset = load_dataset("json", data_files=data_files, split="train")

    def preprocess_function(example):
        text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{example['output']}<|eot_id|>"
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    print("데이터셋 전처리 중...")
    dataset = dataset.map(preprocess_function, batched=False, remove_columns=dataset.column_names)
    print("전처리 완료!")
    return dataset


def build_trainer(model, dataset, output_dir="./results"):
    """SFTTrainer with the notebook's FP16-forced training args.

    ``peft_config=None`` is deliberate: LoRA is already attached in
    ``load_model_and_tokenizer``.
    """
    return SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=None,  # [important] already attached above
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            logging_steps=10,
            learning_rate=2e-4,

            # force FP16, block BF16
            fp16=True,
            bf16=False,

            optim="paged_adamw_32bit",
            weight_decay=0.001,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
        ),
    )


def train(model_id=DEFAULT_MODEL_ID, data_files="robot_dataset.json",
          output_dir="./results", adapter_dir="./trained_adapter"):
    """End-to-end flow of notebook cell 4.

    NOTE: requires a CUDA GPU and gated Llama-3 HF access (the notebook ran
    ``huggingface_hub.login()`` first). Never completed on a free T4 — see
    module docstring.
    """
    model, tokenizer = load_model_and_tokenizer(model_id)
    dataset = load_and_preprocess_dataset(tokenizer, data_files)
    trainer = build_trainer(model, dataset, output_dir)

    print("학습을 시작합니다...")
    trainer.train()

    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print("학습 완료! 어댑터 저장됨.")
