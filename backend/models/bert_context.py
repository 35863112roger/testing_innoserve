from pathlib import Path
from threading import RLock
from typing import Any


class BertContextClassifier:
    """延遲載入本機 BertForSequenceClassification。"""

    def __init__(self, model_path: Path, device: str = "auto") -> None:
        self.model_path = model_path
        self.requested_device = device
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
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._device = self._resolve_device(torch)
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            self._model.to(self._device)
            self._model.eval()

    def analyze_reply(self, context_text: str, target_text: str) -> dict[str, Any]:
        self.load()

        import torch

        max_length = min(int(getattr(self._tokenizer, "model_max_length", 512)), 512)
        encoded = self._tokenizer(
            context_text,
            target_text,
            add_special_tokens=True,
            return_tensors="pt",
            truncation="only_first",
            max_length=max_length,
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = self._model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()

        id2label = self._model.config.id2label
        scores = {
            str(id2label.get(index, id2label.get(str(index), index))): float(score)
            for index, score in enumerate(probabilities)
        }
        label = max(scores, key=scores.get)
        return {
            "label": label,
            "confidence": scores[label],
            "probabilities": scores,
        }

    def classify_samples(
        self,
        samples: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """依訓練格式批次分類每一則 reply。"""
        self.load()

        import torch

        max_length = min(int(getattr(self._tokenizer, "model_max_length", 512)), 512)
        target_limit = max_length - 8
        eligible: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for sample in samples:
            target_token_count = len(
                self._tokenizer(
                    sample["target_text"],
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
            )
            if target_token_count > target_limit:
                skipped.append(
                    {
                        "target_id": sample["target_id"],
                        "reasons": ["target_too_long"],
                        "target_token_count": target_token_count,
                    }
                )
                continue
            eligible.append(sample)

        id2label = {
            int(label_id): str(label)
            for label_id, label in self._model.config.id2label.items()
        }
        predictions: list[dict[str, Any]] = []
        for start in range(0, len(eligible), batch_size):
            batch = eligible[start : start + batch_size]
            encoded = self._tokenizer(
                [sample["context_text"] for sample in batch],
                [sample["target_text"] for sample in batch],
                add_special_tokens=True,
                truncation="only_first",
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.inference_mode():
                rows = torch.softmax(self._model(**encoded).logits, dim=-1).cpu().tolist()

            for sample, row in zip(batch, rows):
                probabilities = {
                    id2label[index]: float(probability)
                    for index, probability in enumerate(row)
                }
                label = max(probabilities, key=probabilities.get)
                predictions.append(
                    {
                        "target_id": sample["target_id"],
                        "parent_id": sample["parent_id"],
                        "depth": sample["depth"],
                        "username": sample.get("username"),
                        "text": sample["text"],
                        "permalink": sample.get("permalink"),
                        "label": label,
                        "confidence": probabilities[label],
                        "probabilities": probabilities,
                    }
                )
        return predictions, skipped

    def _resolve_device(self, torch: Any) -> str:
        if self.requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("MODEL_DEVICE=cuda，但 PyTorch 無法使用 CUDA。")
        return self.requested_device
