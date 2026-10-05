from functools import lru_cache

from backend.config import get_settings
from backend.models.bert_context import BertContextClassifier
from backend.models.summarizer import TaideLoraSummarizer


class ModelRegistry:
    def __init__(self) -> None:
        settings = get_settings()
        self.bert = BertContextClassifier(
            settings.bert_model_path,
            settings.model_device,
        )
        self.summarizer = TaideLoraSummarizer(
            base_model_source=settings.summary_base_source,
            adapter_path=settings.summary_adapter_path,
            device=settings.model_device,
            local_files_only=settings.hf_local_files_only,
            max_input_tokens=settings.summary_max_input_tokens,
        )


@lru_cache
def get_model_registry() -> ModelRegistry:
    return ModelRegistry()
