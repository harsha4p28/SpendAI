"""
SpendAI - Real QLoRA Inference
---------------------------------
Loads the fine-tuned adapter produced by llm_pipeline/train_qlora_colab.py and
runs actual inference, replacing mock_qlora_finetuned_inference in eval_model.py.

This module is intentionally separate from eval_model.py: it requires torch/
transformers/peft/bitsandbytes and a GPU (or a slow CPU fallback), so it's only
imported when real evaluation is explicitly requested — the mock-based benchmark
should keep working with zero extra dependencies.

USAGE (after training an adapter with train_qlora_colab.py):
    from llm_pipeline.real_inference import RealQLoRAInference
    engine = RealQLoRAInference(adapter_path="llm_pipeline/spendai-qlora-final-adapter")
    result = engine.predict("Monthly cloud infrastructure subscription for EC2...")
"""

import os
import json
import re
from typing import Dict, Any

BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


class RealQLoRAInference:
    def __init__(self, adapter_path: str, base_model_id: str = BASE_MODEL_ID):
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"Adapter not found at {adapter_path}. Train it first with "
                f"llm_pipeline/train_qlora_colab.py on a GPU, then place the "
                f"saved adapter directory here."
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        if not torch.cuda.is_available():
            print("WARNING: no CUDA GPU detected — inference will run on CPU and be very slow.")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if torch.cuda.is_available() else None

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        self.torch = torch

    def predict(self, description: str) -> Dict[str, Any]:
        """Runs the fine-tuned adapter on a spend description and parses its
        JSON completion. Falls back to an Uncategorized/parse-error result if
        the model doesn't emit valid JSON — this happens occasionally with
        small models and should be tracked as a metric in its own right."""
        prompt = (
            "<|user|>\nYou are a Business Spend Management AI assistant. Categorize the "
            f"procurement transaction into UNSPSC standard code, category, and risk level.\n"
            f"Context: {description}<|end|>\n<|assistant|>\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        json_match = re.search(r"\{.*\}", generated, re.DOTALL)
        if not json_match:
            return {"category": "Uncategorized", "unspsc_code": "99999900", "risk_assessment": "HIGH",
                     "parse_error": True, "raw_output": generated}
        try:
            parsed = json.loads(json_match.group(0))
            parsed.setdefault("category", "Uncategorized")
            parsed.setdefault("unspsc_code", "99999900")
            parsed.setdefault("risk_assessment", "HIGH")
            return parsed
        except json.JSONDecodeError:
            return {"category": "Uncategorized", "unspsc_code": "99999900", "risk_assessment": "HIGH",
                     "parse_error": True, "raw_output": generated}
