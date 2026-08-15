"""
SpendAI - Real QLoRA Inference
"""

import os
import json
import re
from typing import Dict, Any

BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


class RealQLoRAInference:
    def __init__(self, adapter_path: str, base_model_id: str = BASE_MODEL_ID):
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter not found at {adapter_path}.")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if torch.cuda.is_available() else None

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map={"": 0} if torch.cuda.is_available() else None,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        self.torch = torch

    def predict(self, description: str) -> Dict[str, Any]:
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
