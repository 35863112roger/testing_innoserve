from typing import Any

from pydantic import BaseModel, Field


class ProcessThreadRequest(BaseModel):
    url: str = Field(min_length=20)
    max_nodes: int = Field(default=30, ge=1, le=500)
    force_refresh: bool = False
    max_new_tokens: int = Field(default=300, ge=64, le=500)
    classifier_batch_size: int = Field(default=32, ge=1, le=128)
    judgment_limit: int = Field(default=5, ge=1, le=10)


class SummaryResult(BaseModel):
    summary: str
    input_tokens: int
    generated_tokens: int
    device: str
    input_truncated: bool


class CommentPrediction(BaseModel):
    target_id: str
    parent_id: str
    depth: int
    username: str | None = None
    text: str
    permalink: str | None = None
    label: str
    confidence: float
    probabilities: dict[str, float]


class ThreadNode(BaseModel):
    id: str
    parent_id: str | None = None
    depth: int
    username: str | None = None
    text: str
    permalink: str | None = None
    node_type: str


class ThreadInfo(BaseModel):
    requested_post_id: str | None = None
    root_post_id: str | None = None
    root_username: str | None = None
    root_text: str | None = None
    root_permalink: str | None = None
    source: str
    source_url: str
    node_count: int
    used_summary_nodes: int
    classified_count: int
    skipped_count: int
    collection_complete: bool | None = None
    node_limit_truncated: bool
    warnings: list[Any] = Field(default_factory=list)

class RagResult(BaseModel):
    status: str
    thread_id: str | None = None
    report: str = ""
    reason: str | None = None
    risk_node_count: int = 0
    included_node_count: int = 0
    omitted_node_count: int = 0
    queries: list[dict[str, Any]] = Field(
        default_factory=list
    )
    sources: list[dict[str, Any]] = Field(
        default_factory=list
    )
    generation_model: str | None = None

class JudgmentResult(BaseModel):
    status: str
    reason: str | None = None
    selected_risk_node_count: int = 0
    hits: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    attempts: list[dict[str, Any]] = Field(default_factory=list)

class ProcessThreadResult(BaseModel):
    thread: ThreadInfo
    summary: SummaryResult
    statistics: dict[str, int]
    comments: list[CommentPrediction]
    thread_nodes: list[ThreadNode]
    rag: RagResult
    judgments: JudgmentResult
    skipped: list[dict[str, Any]]
