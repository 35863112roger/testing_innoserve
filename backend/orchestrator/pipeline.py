from collections import Counter
from pathlib import Path
from typing import Any

from backend.mcp_client.gateway import MCPToolGateway
from backend.models.registry import ModelRegistry
from backend.threads.processing import (
    build_rag_risk_nodes,
    build_reply_samples,
    build_thread_text,
    clean_node_text,
    load_thread_tree,
    thread_tree_quality_error,
)
from backend.threads.progress import set_analysis_progress


LABELS = ("general", "friendly", "harassment", "cyberbullying")


class ProcessingPipeline:
    def __init__(
        self,
        models: ModelRegistry,
        tools: MCPToolGateway,
        threads_tree_data_dir: Path,
        rag_enabled: bool = False,
    ) -> None:
        self.models = models
        self.tools = tools
        self.threads_tree_data_dir = threads_tree_data_dir
        self.rag_enabled = rag_enabled

    async def run_thread(
        self,
        url: str,
        max_nodes: int,
        force_refresh: bool,
        max_new_tokens: int,
        classifier_batch_size: int,
        judgment_limit: int = 5,
    ) -> dict[str, Any]:
        collection = await self.tools.call_tool(
            "collect_thread",
            {
                "url": url,
                "force_refresh": force_refresh,
                "max_nodes": max_nodes,
            },
        )
        if "tree_path" not in collection:
            raise RuntimeError(f"collect_thread 未回傳 tree_path：{collection}")

        tree = load_thread_tree(
            collection["tree_path"],
            self.threads_tree_data_dir,
        )

        quality_error = thread_tree_quality_error(tree)
        if quality_error:
            raise ValueError(quality_error)

        thread_text, node_limit_truncated, used_summary_nodes = build_thread_text(
            tree,
            max_nodes,
        )
        if not thread_text.strip():
            raise ValueError("這篇 Threads 串文沒有可供摘要的文字")

        samples, structural_skips = build_reply_samples(tree)
        progress_post_id = str(collection.get("requested_post_id") or "")
        if progress_post_id:
            set_analysis_progress(
                progress_post_id,
                state="classifying",
                message=f"討論樹整理完成，正在分類 {len(samples)} 則留言……",
                current=len(tree.get("nodes") or []),
                total=len(tree.get("nodes") or []),
            )
        # 本機單使用者服務刻意在主執行緒推論。這組 PyTorch/OpenMP 環境在
        # asyncio worker thread 內會嚴重停滯；API 在模型完成前本來就必須等待。
        predictions, model_skips = self.models.bert.classify_samples(
            samples,
            classifier_batch_size,
        )
        if progress_post_id:
            set_analysis_progress(
                progress_post_id,
                state="summarizing",
                message="留言分類完成，正在產生串文摘要……",
            )
        summary = self.models.summarizer.summarize(
            thread_text,
            max_new_tokens,
        )

        counts = Counter(item["label"] for item in predictions)
        statistics = {label: counts.get(label, 0) for label in LABELS}
        skipped = structural_skips + model_skips
        root_post_id = collection.get("root_post_id") or tree.get("conversation_root_id")
        root_node = next(
            (
                node
                for node in (tree.get("nodes") or [])
                if node.get("post_id") == root_post_id
            ),
            {},
        )
        thread_nodes = [
            {
                "id": str(node["post_id"]),
                "parent_id": (
                    str(node["parent_id"])
                    if node.get("parent_id") is not None
                    else None
                ),
                "depth": int(node.get("depth") or 0),
                "username": node.get("username"),
                "text": clean_node_text(node),
                "permalink": node.get("permalink"),
                "node_type": str(node.get("node_type") or "reply"),
            }
            for node in (tree.get("nodes") or [])
            if node.get("post_id")
        ]

        root_text = clean_node_text(root_node)

        risk_nodes = build_rag_risk_nodes(
            predictions,
            thread_nodes,
        )

        if not risk_nodes:
            judgments_result = {
                "status": "skipped",
                "reason": "沒有 harassment 或 cyberbullying 留言",
                "selected_risk_node_count": 0,
                "hits": [],
                "warnings": [],
                "attempts": [],
            }
        else:
            if progress_post_id:
                set_analysis_progress(
                    progress_post_id,
                    state="judgment_searching",
                    message=(
                        "留言分類與摘要已完成，正在搜尋"
                        f"最多 {judgment_limit} 筆類似案件判決……"
                    ),
                )
            try:
                judgments_result = await self.tools.call_tool(
                    "search_similar_judgments",
                    {
                        "context_summary": summary["summary"],
                        "risk_nodes": risk_nodes,
                        "limit": judgment_limit,
                    },
                )
            except Exception as exc:
                judgments_result = {
                    "status": "error",
                    "reason": str(exc),
                    "selected_risk_node_count": 0,
                    "hits": [],
                    "warnings": [],
                    "attempts": [],
                }

        thread_id = str(
            root_post_id
            or collection.get("requested_post_id")
            or ""
        )

        if not self.rag_enabled:
            rag_result = {
                "status": "disabled",
                "thread_id": thread_id,
                "report": "",
                "reason": "RAG 功能未啟用",
                "risk_node_count": len(risk_nodes),
                "included_node_count": 0,
                "omitted_node_count": 0,
                "queries": [],
                "sources": [],
            }

        elif not risk_nodes:
            rag_result = {
                "status": "skipped",
                "thread_id": thread_id,
                "report": "",
                "reason": (
                    "本次沒有 harassment 或 "
                    "cyberbullying 分類節點"
                ),
                "risk_node_count": 0,
                "included_node_count": 0,
                "omitted_node_count": 0,
                "queries": [],
                "sources": [],
            }

        else:
            if progress_post_id:
                set_analysis_progress(
                    progress_post_id,
                    state="rag_analyzing",
                    message=(
                        f"正在針對 {len(risk_nodes)} 個"
                        "風險節點產生整體 RAG 報告……"
                    ),
                )

            try:
                rag_result = await self.tools.call_tool(
                    "analyze_thread_risk",
                    {
                        "thread_id": thread_id,
                        "root_text": root_text,
                        "statistics": statistics,
                        "risk_nodes": risk_nodes,
                    },
                )
            except Exception as exc:
                rag_result = {
                    "status": "error",
                    "thread_id": thread_id,
                    "report": "",
                    "reason": str(exc),
                    "risk_node_count": len(risk_nodes),
                    "included_node_count": 0,
                    "omitted_node_count": 0,
                    "queries": [],
                    "sources": [],
                }
        if progress_post_id:
            set_analysis_progress(
                progress_post_id,
                state="complete",
                message=(
                    f"分析完成：{len(tree.get('nodes') or [])} 個節點，"
                    f"分類 {len(predictions)} 則留言"
                ),
            )
        return {
            "thread": {
                "requested_post_id": collection.get("requested_post_id"),
                "root_post_id": root_post_id,
                "root_username": root_node.get("username"),
                "root_text": root_text,
                "root_permalink": root_node.get("permalink") or url,
                "source": collection.get("source", "unknown"),
                "source_url": url,
                "node_count": len(tree.get("nodes") or []),
                "used_summary_nodes": used_summary_nodes,
                "classified_count": len(predictions),
                "skipped_count": len(skipped),
                "collection_complete": collection.get("collection_complete"),
                "node_limit_truncated": node_limit_truncated,
                "warnings": collection.get("warnings") or [],
            },
            "summary": summary,
            "statistics": statistics,
            "comments": predictions,
            "thread_nodes": thread_nodes,
            "rag": rag_result,
            "judgments": judgments_result,
            "skipped": skipped,
        }
