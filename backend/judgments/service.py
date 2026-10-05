from typing import Any

from backend.judgments.core import extract
from backend.judgments.live_search import (
    LiveJudicialSearch,
    search_plan,
)

"""
建立 BERT 到判決搜尋的轉接服務
testing_innoserve 已經有 build_rag_risk_nodes()，會把 BERT 結果整理成：
{
  "target_id": "留言 ID",
  "label": "harassment",
  "confidence": 0.91,
  "text": "留言文字",
  "parent_text": "上一層留言"
}
因此不需要重新分類，只要將標籤轉成 judgment_assistant 可理解的格式。


這裡回傳的不是完整裁判全文，只保留：
- 裁判標題
- 法院
- 日期
- 相關段落
- 主文
- 理由節錄
- 司法院原文連結
可以避免 FastAPI 回應過大。


backend/judgments/service.py 是這次整合新增的轉接層，負責：
- 接收 testing_innoserve 的 BERT 結果。
- 將 harassment／cyberbullying 轉為判決搜尋方向。
- 選出信心最高的代表留言。
- 合併並去除重複裁判。
- 限制最後顯示筆數。
- 移除完整裁判全文，只回傳 UI 需要的內容。
另外，下列也是整合時新增或修改的：
- mcp_servers/internal_tools.py：新增 MCP 工具。
- backend/orchestrator/pipeline.py：呼叫 MCP 判決搜尋。
- backend/models/schemas.py：新增結果格式。
- backend/api/routes.py：傳遞筆數設定。
- frontend/app.py：新增判決分頁與顯示內容。
"""

def _to_decisions(label: str) -> list[dict[str, Any]]:
    return [
        {
            "label": "騷擾",
            "positive": label == "harassment",
            "evidence": [],
        },
        {
            "label": "霸凌",
            "positive": label == "cyberbullying",
            "evidence": (
                ["霸凌風險"]
                if label == "cyberbullying"
                else []
            ),
        },
    ]


def search_similar_cases(
    context_summary: str,
    risk_nodes: list[dict[str, Any]],
    limit: int = 5,
    max_query_nodes: int = 2,
) -> dict[str, Any]:
    if not risk_nodes:
        return {
            "status": "skipped",
            "reason": "沒有騷擾或網路霸凌留言",
            "hits": [],
            "warnings": [],
        }

    # 避免每一則留言都查詢司法院，先取信心最高的代表留言。
    selected_nodes = sorted(
        risk_nodes,
        key=lambda item: float(item.get("confidence") or 0),
        reverse=True,
    )[:max_query_nodes]

    merged_hits: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    attempts: list[dict[str, Any]] = []

    for node in selected_nodes:
        decisions = _to_decisions(str(node["label"]))
        plan = search_plan(
            comment=str(node["text"]),
            context=context_summary,
            results=decisions,
        )

        result = LiveJudicialSearch().search(
            plan,
            k=limit,
        )

        warnings.extend(result.get("warnings") or [])
        attempts.extend(result.get("attempts") or [])

        for hit in result.get("hits") or []:
            doc = hit["doc"]
            excerpt = extract(doc)

            normalized = {
                "case_id": doc["id"],
                "title": doc["title"],
                "court": doc["court"],
                "date": doc["date"],
                "source_url": doc["source_url"],
                "score": hit["score"],
                "preview": hit.get("preview") or hit["text"][:420],
                "match_level": hit.get("match_level"),
                "match_terms": hit.get("match_terms") or [],
                "issue_terms": hit.get("issue_terms") or [],
                "outcome": excerpt["判決結果（主文原文）"],
                "reason_excerpt": excerpt["法院理由（原文節錄）"],
                "matched_comment_ids": [str(node["target_id"])],
            }

            existing = merged_hits.get(doc["id"])
            if existing:
                existing["matched_comment_ids"].append(
                    str(node["target_id"])
                )
                if normalized["score"] > existing["score"]:
                    normalized["matched_comment_ids"] = existing[
                        "matched_comment_ids"
                    ]
                    merged_hits[doc["id"]] = normalized
            else:
                merged_hits[doc["id"]] = normalized

    hits = sorted(
        merged_hits.values(),
        key=lambda item: -item["score"],
    )[:limit]

    return {
        "status": "ok" if hits else "no_match",
        "reason": None if hits else "本次未找到足夠相關的裁判",
        "selected_risk_node_count": len(selected_nodes),
        "hits": hits,
        "warnings": list(dict.fromkeys(warnings)),
        "attempts": attempts,
    }