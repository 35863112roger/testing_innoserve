from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Testing Innoserve：Threads 摘要與留言分類"
    bert_model_path: Path
    summary_adapter_path: Path
    summary_base_model_path: Path | None = None
    summary_base_model_id: str = "chengxi0618/taidelx7bchattv1"
    model_device: str = "auto"
    hf_local_files_only: bool = True
    summary_max_input_tokens: int = Field(default=2048, ge=512, le=4096)

    mcp_server_url: str = "http://127.0.0.1:8001/mcp"
    mcp_in_process: bool = True
    mcp_timeout_seconds: float = Field(default=60.0, gt=0)
    cors_origins: str = "http://127.0.0.1:8501,http://localhost:8501"
    threads_tree_data_dir: Path
    threads_collector_script: Path
    threads_collector_profile_dir: Path = PROJECT_ROOT / "data" / "playwright_profile"
    threads_collector_base_timeout_seconds: int = Field(default=600, ge=60, le=3600)
    threads_collector_seconds_per_node: int = Field(default=40, ge=1, le=60)
    threads_collector_max_timeout_seconds: int = Field(default=20000, ge=60, le=86400)
    rag_enabled: bool = False
    rag_data_dir: Path = PROJECT_ROOT / "data" / "rag"
    rag_chroma_collection: str = "bullying_law"
    rag_embed_model: str = "nomic-embed-text"
    rag_generation_model: str = "mistral"
    rag_k_each: int = Field(default=30, ge=1, le=100)
    rag_top_k: int = Field(default=6, ge=1, le=20)
    rag_rrf_k: int = Field(default=60, ge=1, le=200)
    
    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def summary_base_source(self) -> str:
        if self.summary_base_model_path is not None:
            return str(self.summary_base_model_path)
        return self.summary_base_model_id

    def validate_model_paths(self) -> list[str]:
        missing: list[str] = []
        for name, path in (
            ("BERT_MODEL_PATH", self.bert_model_path),
            ("SUMMARY_ADAPTER_PATH", self.summary_adapter_path),
        ):
            if not path.exists():
                missing.append(f"{name}: {path}")
        if self.summary_base_model_path is not None and not self.summary_base_model_path.exists():
            missing.append(f"SUMMARY_BASE_MODEL_PATH: {self.summary_base_model_path}")
        for name, path in (
            ("THREADS_TREE_DATA_DIR", self.threads_tree_data_dir),
            ("THREADS_COLLECTOR_SCRIPT", self.threads_collector_script),
            ("THREADS_COLLECTOR_PROFILE_DIR", self.threads_collector_profile_dir),
        ):
            if not path.exists():
                missing.append(f"{name}: {path}")
        return missing


@lru_cache
def get_settings() -> Settings:
    return Settings()
