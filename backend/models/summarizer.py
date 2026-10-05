from pathlib import Path
from threading import RLock
from typing import Any

from backend.threads.processing import SUMMARY_SYSTEM_PROMPT


TRUNCATION_MARKER = "\n\n[討論串內容因 token 上限而截斷]"


class TaideLoraSummarizer:
    """載入 TAIDE 基底模型，再套用本機 LoRA 摘要 adapter。"""

    def __init__(
        self,
        base_model_source: str,
        adapter_path: Path,
        device: str = "auto",
        local_files_only: bool = True,
        max_input_tokens: int = 2048,
    ) -> None:
        self.base_model_source = base_model_source
        self.adapter_path = adapter_path
        self.requested_device = device
        self.local_files_only = local_files_only
        self.max_input_tokens = max_input_tokens
        self._device = "cpu"
        self._tokenizer: Any = None
        self._model: Any = None
        self._load_lock = RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return

            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            self._device = self._resolve_device(torch)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_source,
                local_files_only=self.local_files_only,
            )
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            load_options: dict[str, Any] = {
                "local_files_only": self.local_files_only,
                "low_cpu_mem_usage": True,
            }
            if self._device == "cuda":
                compute_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                load_options.update(
                    {
                        "quantization_config": BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=compute_dtype,
                            bnb_4bit_use_double_quant=True,
                        ),
                        "device_map": "auto",
                    }
                )
            else:
                load_options["dtype"] = "auto"

            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_source,
                **load_options,
            )
            self._model = PeftModel.from_pretrained(
                base_model,
                str(self.adapter_path),
                local_files_only=True,
            )
            if self._device != "cuda":
                self._model.to(self._device)
            self._model.eval()

    def summarize(
        self,
        thread_text: str,
        max_new_tokens: int = 300,
    ) -> dict[str, Any]:
        self.load()

        import torch

        input_ids, input_truncated = self._fit_thread_prompt(thread_text)
        model_device = next(self._model.parameters()).device
        input_ids = input_ids.to(model_device)
        input_tokens = int(input_ids.shape[-1])

        with torch.inference_mode():
            output = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.15,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated = output[0, input_tokens:]
        summary = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        return {
            "summary": summary,
            "input_tokens": input_tokens,
            "generated_tokens": int(generated.shape[-1]),
            "device": self._device,
            "input_truncated": input_truncated,
        }

    def _render_prompt(self, thread_text: str):
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": thread_text},
        ]
        encoded = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        elif isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        if not hasattr(encoded, "ndim"):
            import torch

            encoded = torch.tensor(encoded, dtype=torch.long)
        if encoded.ndim == 1:
            encoded = encoded.unsqueeze(0)
        return encoded

    def _fit_thread_prompt(self, thread_text: str):
        encoded = self._render_prompt(thread_text)
        if encoded.shape[-1] <= self.max_input_tokens:
            return encoded, False

        thread_ids = self._tokenizer(
            thread_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        low, high = 0, len(thread_ids)
        best = self._render_prompt(TRUNCATION_MARKER)
        while low <= high:
            middle = (low + high) // 2
            kept = self._tokenizer.decode(
                thread_ids[:middle],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).rstrip()
            candidate = self._render_prompt(kept + TRUNCATION_MARKER)
            if candidate.shape[-1] <= self.max_input_tokens:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        if best.shape[-1] > self.max_input_tokens:
            raise RuntimeError("摘要 system prompt 已超過 SUMMARY_MAX_INPUT_TOKENS")
        return best, True

    def _resolve_device(self, torch: Any) -> str:
        if self.requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("MODEL_DEVICE=cuda，但 PyTorch 無法使用 CUDA。")
        return self.requested_device
