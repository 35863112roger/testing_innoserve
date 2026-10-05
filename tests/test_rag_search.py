from backend.rag.hybrid_search import (
    rrf_fuse,
    zh_tokenize,
)
from backend.rag.schemas import RagGenerationOutput
from backend.rag.service import ThreadRiskRagService


def test_zh_tokenize_contains_chinese_unigrams_and_bigrams() -> None:
    tokens = zh_tokenize("網路霸凌 ABC123")

    assert "網" in tokens
    assert "路" in tokens
    assert "網路" in tokens
    assert "霸凌" in tokens
    assert "abc123" in tokens


def test_rrf_rewards_documents_found_by_both_searchers() -> None:
    vector_hits = [
        ("document_a", 0.10),
        ("document_b", 0.20),
    ]
    bm25_hits = [
        ("document_b", 12.0),
        ("document_c", 8.0),
    ]

    fused = rrf_fuse(
        vector_hits,
        bm25_hits,
        rrf_k=60,
    )

    assert fused[0]["doc_id"] == "document_b"
    assert fused[0]["vector_rank"] == 2
    assert fused[0]["bm25_rank"] == 1


def test_rrf_keeps_documents_from_only_one_searcher() -> None:
    vector_hits = [
        ("vector_only", 0.10),
    ]
    bm25_hits = [
        ("bm25_only", 10.0),
    ]

    fused = rrf_fuse(
        vector_hits,
        bm25_hits,
        rrf_k=60,
    )

    result_ids = {
        result["doc_id"]
        for result in fused
    }

    assert result_ids == {
        "vector_only",
        "bm25_only",
    }


def test_harassment_definition_does_not_fail_when_model_conflates_labels() -> None:
    service = ThreadRiskRagService(
        searcher=None,  # type: ignore[arg-type]
        generation_model="test-model",
    )
    generated = RagGenerationOutput.model_validate(
        {
            "definitions": [
                {
                    "risk_label": "harassment",
                    "explanation": "可能的言語攻擊或網路霸凌行為",
                    "source_numbers": [1],
                }
            ],
            "actions": [
                {
                    "action": "保存完整留言與上下文",
                    "source_numbers": [1],
                },
                {
                    "action": "必要時尋求專業協助",
                    "source_numbers": [1],
                },
            ],
        }
    )
    report, citations = service._render_report(
        statistics={
            "general": 0,
            "friendly": 0,
            "harassment": 1,
            "cyberbullying": 0,
        },
        generated=generated,
        sources=[
            {
                "matched_label": "harassment",
                "title": "言語攻擊與騷擾",
            }
        ],
    )

    assert "言語攻擊或騷擾" in report
    assert "網路霸凌行為" not in report
    assert citations == {1}
