"""
SpendAI - QLoRA Fine-Tuning Pipeline (Google Colab T4 GPU or local GPU)
------------------------------------------------------------------------
Fine-tunes an open-weights LLM (Phi-3-mini / Llama-3-8B) on the SpendAI UNSPSC
spend-classification dataset using 4-bit QLoRA (PEFT).

REQUIREMENTS:
  - A CUDA-capable GPU (Colab T4 or better). This will NOT run on CPU-only
    machines in any reasonable time — bitsandbytes 4-bit quantization requires
    CUDA.
  - Install dependencies first:
        pip install -q -U torch transformers peft bitsandbytes datasets trl

EXPECTED RUNTIME: ~10-20 minutes on a Colab T4 for max_steps=100 with the
default batch size below. Scale max_steps up for a full training run over the
whole dataset.

USAGE:
    python llm_pipeline/train_qlora_colab.py
  or paste directly into a Colab cell.
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"  # or "meta-llama/Meta-Llama-3-8B-Instruct"

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "unspsc_fine_tuning_dataset.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "spendai-qlora-adapter")
FINAL_ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "spendai-qlora-final-adapter")


def format_prompt(example):
    return (
        f"<|user|>\n{example['instruction']}\nContext: {example['input']}<|end|>\n"
        f"<|assistant|>\n{example['output']}<|end|>"
    )


def run_qlora_finetuning():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. QLoRA 4-bit training requires a CUDA-capable GPU "
            "(e.g. Colab T4). Aborting."
        )

    print(f"Loading base model: {MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"Loading SpendAI UNSPSC dataset from: {DATASET_PATH}")
    dataset = load_dataset("json", data_files=DATASET_PATH)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        formatting_func=format_prompt,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting QLoRA Fine-Tuning on BSM Spend Dataset...")
    trainer.train()

    trainer.model.save_pretrained(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
    print(f"Saved SpendAI QLoRA adapter weights to {FINAL_ADAPTER_DIR}")


if __name__ == "__main__":
    run_qlora_finetuning()
