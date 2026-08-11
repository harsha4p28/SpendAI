"""
SpendAI - QLoRA Fine-Tuning Pipeline
------------------------------------
This script fine-tunes Phi-3-mini / Llama-3-8B on the UNSPSC Spend Classification dataset using 4-bit QLoRA.

NOTE: This script requires an NVIDIA GPU (Google Colab T4 GPU or better with CUDA support).
It is NOT intended to run on CPU-only local machines.
Estimated Training Runtime: ~20-30 minutes on a free Google Colab T4 GPU.

Usage (in Colab or GPU environment):
    !pip install -q -U torch transformers peft bitsandbytes datasets trl
    python llm_pipeline/train_qlora_colab.py
"""

import os
import sys
import torch

# GPU Availability Check
if not torch.cuda.is_available():
    print("⚠️ WARNING: CUDA GPU is not available. This QLoRA fine-tuning script requires a GPU (Colab T4 or better).")
    print("If you are running on CPU, training will fail or be extremely slow.")
    sys.exit(1)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "unspsc_fine_tuning_dataset.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "spendai-qlora-final-adapter")

def run_qlora_training():
    print("=" * 60)
    print(f"  SpendAI - Starting QLoRA 4-Bit Fine-Tuning on {MODEL_ID}")
    print("=" * 60)

    # 1. 4-Bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 2. Load Model & Tokenizer
    print(f"Loading base model and tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Adapter Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 4. Load UNSPSC Spend Dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run data_engine/generate_data.py first.")

    dataset = load_dataset("json", data_files=DATASET_PATH)

    def format_prompt(example):
        return f"<|user|>\n{example['instruction']}\nContext: {example['input']}<|end|>\n<|assistant|>\n{example['output']}<|end|>"

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        formatting_func=format_prompt,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args
    )

    print("Starting QLoRA Fine-Tuning loop...")
    trainer.train()

    print(f"Saving fine-tuned QLoRA adapter weights to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Training complete! QLoRA adapter saved successfully.")

if __name__ == "__main__":
    run_qlora_training()
