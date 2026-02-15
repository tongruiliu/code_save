from __future__ import annotations

from typing import Any, Dict

from .base import ModelBackend


class LocalBackend(ModelBackend):
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_new_tokens: int = 1024,
        trust_remote_code: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("transformers + torch are required for local backend") from exc

        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = temperature > 1e-6
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-6),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
