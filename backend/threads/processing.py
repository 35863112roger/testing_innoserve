from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SUMMARY_SYSTEM_PROMPT = """\
你是一個協助整理 Threads 討論串的助手。
你會收到一則 Threads 貼文與其下方回覆的純文字紀錄（依照回覆層級縮排）。

請用繁體中文寫一段摘要，長度約 3~6 句，需涵蓋：
1. 原始貼文在講什麼（主題、原 PO 的觀點或訴求）
2. 留言區主要在討論什麼、有沒有明顯的正反意見或共識
3. 整體氛圍（例如：支持、爭議、玩梗、無關閒聊等）

如果輸入中沒有任何 [reply]，只摘要原始貼文，
並明確說明本次沒有取得可供分析的留言。
不得推測或虛構留言區的意見、共識及氛圍。

只輸出摘要本文，不要加標題、不要用 Markdown 條列、不要覆述這份指示。
"""

_UI_NOISE_RE = re.compile(
    r"Translate|No replies yet|View activity|Top\b|Reply to [^\n]*\.\.\.|Spoiler",
    re.IGNORECASE,
)
_ENGAGEMENT_LINE_RE = re.compile(r"^\s*[\d,.]+(?:千|萬|億|[KkMmBb])?\s*$")
_RELATIVE_TIME_RE = re.compile(r"^\d+\s*(秒|分鐘|小時|天|週|周|個月|月|年)$")
_LEADING_DATE_RE = re.compile(
    r"^(?:"
    r"(?:19|20)\d{2}[/-]\d{1,2}[/-]\d{1,2}"
    r"|\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})"
    r"|(?:19|20)\d{2}年\d{1,2}月\d{1,2}日"
    r")$"
)
_UI_EXACT_TEXTS = {"尚無回覆", "查看動態", "查看翻譯", "隱藏", "更多", "回覆"}


def clean_node_text(node: dict[str, Any] | None) -> str:
    """遵循訓練資料邏輯，優先尊重 Collector 的 text_clean。"""
    if not node:
        return ""
    if "text_clean" in node:
        raw = node.get("text_clean") or ""
    else:
        raw = node.get("text_raw") or ""

    username = str(node.get("username") or "").strip()
    cleaned: list[str] = []
    for value in str(raw).replace("\ufffc", "").splitlines():
        line = value.strip()
        if not line:
            continue
        line = _UI_NOISE_RE.sub("", line).strip()
        if not line or line in _UI_EXACT_TEXTS:
            continue
        if username and line in {username, f"@{username}"}:
            continue
        if _RELATIVE_TIME_RE.fullmatch(line) or _ENGAGEMENT_LINE_RE.fullmatch(line):
            continue
        if not cleaned and _LEADING_DATE_RE.fullmatch(line):
            continue
        if line.startswith("回覆") and username and username in line:
            continue
        cleaned.append(line)
    return " ".join(cleaned).strip()


def load_thread_tree(tree_path: str, allowed_root: Path) -> dict[str, Any]:
    path = Path(tree_path).expanduser().resolve()
    root = allowed_root.expanduser().resolve()
    if not path.is_relative_to(root):
        raise ValueError("MCP 回傳的 tree 路徑不在允許的 Threads 資料目錄內")
    if not path.is_file():
        raise FileNotFoundError(f"找不到 Threads tree：{path}")

    tree = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tree.get("nodes"), list):
        raise ValueError("Threads tree 缺少 nodes 陣列")
    if not (tree.get("conversation_root_id") or tree.get("root_post_id")):
        raise ValueError("Threads tree 缺少 conversation root")
    return tree


def thread_tree_quality_error(tree: dict[str, Any]) -> str | None:
    """Return a user-facing error when a collected tree is not trustworthy."""
    nodes = tree.get("nodes") or []
    stats = tree.get("stats") or {}
    node_count = len(nodes)
    raw_count = int(
        stats.get("raw_discovered_node_count")
        or node_count
    )
    excluded_count = int(
        stats.get("excluded_unresolved_count")
        or 0
    )

    if raw_count > 1 and node_count <= 1:
        return (
            "Collector 發現多個候選節點，但沒有成功建立"
            "可連回原始貼文的留言樹，已停止摘要與分類。"
        )

    excessive_exclusions = (
        raw_count >= 10
        and excluded_count / raw_count >= 0.90
    )
    if not excessive_exclusions:
        return None

    stopped_by_limit = bool(stats.get("stopped_by_max_nodes"))
    processed_count = int(
        stats.get("processed_detail_pages")
        or node_count
    )
    excluded_processed = stats.get("excluded_processed_count")
    if excluded_processed is None:
        retained_processed = node_count
    else:
        retained_processed = max(
            processed_count - int(excluded_processed),
            0,
        )
    processed_retention = retained_processed / max(processed_count, 1)

    # 因 max_nodes 截斷時，未逐頁確認的候選留言會被列為
    # missing_parent；這些不是首頁污染。已處理頁面的保留率正常即可使用。
    if stopped_by_limit and processed_retention >= 0.50:
        return None

    return (
        "Collector 排除的無法連接節點比例過高，"
        "本次資料可能混入首頁推薦內容，已停止分析。"
    )


def build_thread_text(tree: dict[str, Any], max_nodes: int) -> tuple[str, bool, int]:
    """轉成摘要 LoRA 訓練時使用的縮排純文字格式。"""
    nodes = tree.get("nodes") or []
    selected = nodes[:max_nodes]
    lines: list[str] = []
    used_nodes = 0
    for node in selected:
        text = clean_node_text(node)
        if not text:
            continue
        depth = max(int(node.get("depth") or 0), 0)
        node_type = str(node.get("node_type") or "reply")
        username = str(node.get("username") or "unknown")
        lines.append(f"{'  ' * depth}[{node_type}] @{username}: {text}")
        used_nodes += 1

    truncated = len(nodes) > max_nodes
    if truncated:
        lines.append("\n（註：討論串節點數較多，僅取前面部分內容）")
    return "\n".join(lines), truncated, used_nodes


def build_reply_samples(
    tree: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """將一棵 tree 轉成每個 reply 各一筆的 context/target samples。"""
    nodes = tree.get("nodes") or []
    node_map = {
        str(node["post_id"]): node
        for node in nodes
        if node.get("post_id")
    }
    root_id = str(tree.get("conversation_root_id") or tree.get("root_post_id") or "")
    root_node = node_map.get(root_id)
    if not root_node:
        raise ValueError("conversation root 不在 nodes 中")
    root_text = clean_node_text(root_node)

    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for target_node in nodes:
        if target_node.get("node_type") != "reply":
            continue

        target_id = str(target_node.get("post_id") or "")
        target_text = clean_node_text(target_node)
        parent_id = str(target_node.get("parent_id") or "")
        parent_node = node_map.get(parent_id)
        parent_text = clean_node_text(parent_node)
        path_ids = [str(item) for item in (target_node.get("context_path_ids") or [])]

        reasons: list[str] = []
        structure_complete = bool(
            target_id
            and parent_id
            and len(path_ids) >= 2
            and path_ids[0] == root_id
            and path_ids[-1] == target_id
            and path_ids[-2] == parent_id
            and all(path_id in node_map for path_id in path_ids)
        )
        if not structure_complete:
            reasons.append("incomplete_reply_path")
        if not target_text:
            reasons.append("target_has_no_text")
        if not root_text or not parent_text:
            reasons.append("incomplete_text_context")

        ancestor_ids = path_ids[1:-2] if structure_complete else []
        ancestor_texts = [clean_node_text(node_map.get(item)) for item in ancestor_ids]
        if any(not text for text in ancestor_texts):
            reasons.append("incomplete_text_context")

        if reasons:
            skipped.append({"target_id": target_id, "reasons": sorted(set(reasons))})
            continue

        context_sections = [f"[ROOT] {root_text}"]
        context_sections.extend(f"[ANC] {text}" for text in ancestor_texts)
        depth = int(target_node.get("depth") or 0)
        if depth > 1 and parent_text:
            context_sections.append(f"[PARENT] {parent_text}")

        samples.append(
            {
                "target_id": target_id,
                "parent_id": parent_id,
                "depth": depth,
                "username": target_node.get("username"),
                "text": target_text,
                "permalink": target_node.get("permalink"),
                "context_text": "\n".join(context_sections),
                "target_text": target_text,
            }
        )
    return samples, skipped

RAG_RISK_LABELS = frozenset(
    {
        "harassment",
        "cyberbullying",
    }
)


def build_rag_risk_nodes(
    predictions: list[dict[str, Any]],
    thread_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    將 BERT 分類結果轉成 RAG MCP Tool 所需要的格式。

    只保留 harassment 與 cyberbullying，
    並依照 parent_id 找到目標留言的直接父節點文字。
    """
    node_map = {
        str(node["id"]): node
        for node in thread_nodes
        if node.get("id")
    }

    risk_nodes: list[dict[str, Any]] = []

    for prediction in predictions:
        label = str(
            prediction.get("label") or ""
        ).strip().lower()

        if label not in RAG_RISK_LABELS:
            continue

        target_id = str(
            prediction.get("target_id") or ""
        )

        parent_id = str(
            prediction.get("parent_id") or ""
        )

        parent_node = node_map.get(
            parent_id,
            {},
        )

        risk_nodes.append(
            {
                "target_id": target_id,
                "parent_id": parent_id,
                "label": label,
                "confidence": float(
                    prediction.get("confidence")
                    or 0
                ),
                "text": str(
                    prediction.get("text")
                    or ""
                ).strip(),
                "parent_text": str(
                    parent_node.get("text")
                    or ""
                ).strip(),
            }
        )

    return risk_nodes
