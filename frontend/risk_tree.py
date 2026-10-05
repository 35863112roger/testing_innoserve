from __future__ import annotations

import html as html_lib
from collections import defaultdict
from typing import Any
from urllib.parse import urlsplit

import streamlit as st


RISK_LABELS = {"harassment", "cyberbullying"}
LABEL_NAMES = {
    "general": "一般",
    "friendly": "友善",
    "harassment": "騷擾／攻擊",
    "cyberbullying": "網路霸凌風險",
    "context": "脈絡節點",
    "root": "原始貼文",
}

NODE_WIDTH = 286
NODE_HEIGHT = 112
LEVEL_GAP = 86
LEAF_GAP = 142


def build_risk_subtree(
    comments: list[dict[str, Any]],
    root_id: str | None,
    source_url: str,
    root_username: str | None = None,
    root_text: str | None = None,
    root_permalink: str | None = None,
    thread_nodes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """依完整 parent chain 保留風險留言與其所有祖先節點。"""
    predictions_by_id = {
        str(item["target_id"]): item
        for item in comments
        if item.get("target_id")
    }
    risk_ids = [
        target_id
        for target_id, item in predictions_by_id.items()
        if item.get("label") in RISK_LABELS
    ]
    if not risk_ids:
        return {
            "nodes": [],
            "edges": [],
            "risk_count": 0,
            "context_count": 0,
        }

    structure_source = thread_nodes or [
        {
            "id": item.get("target_id"),
            "parent_id": item.get("parent_id"),
            "depth": item.get("depth"),
            "username": item.get("username"),
            "text": item.get("text"),
            "permalink": item.get("permalink"),
            "node_type": "reply",
        }
        for item in comments
    ]
    structure_by_id = {
        str(item["id"]): item
        for item in structure_source
        if item.get("id")
    }

    visible_ids: set[str] = set()
    for risk_id in risk_ids:
        current_id: str | None = risk_id
        visited: set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            if current_id == root_id:
                break
            item = structure_by_id.get(current_id)
            if item is None:
                break
            visible_ids.add(current_id)
            parent_id = item.get("parent_id")
            if parent_id == root_id:
                break
            current_id = str(parent_id) if parent_id else None

    visual_root_id = root_id or "__thread_root__"
    nodes: list[dict[str, Any]] = [
        {
            "id": visual_root_id,
            "parent_id": None,
            "depth": 0,
            "username": root_username,
            "text": root_text or "原始貼文",
            "label": "root",
            "confidence": None,
            "probabilities": {},
            "permalink": root_permalink or source_url,
            "is_risk": False,
        }
    ]

    for structure_item in structure_source:
        target_id = str(structure_item.get("id") or "")
        if target_id not in visible_ids:
            continue
        prediction = predictions_by_id.get(target_id) or {}
        original_label = str(prediction.get("label") or "unclassified")
        parent_id = structure_item.get("parent_id")
        if parent_id != visual_root_id and str(parent_id) not in visible_ids:
            parent_id = visual_root_id
        nodes.append(
            {
                "id": target_id,
                "parent_id": str(parent_id) if parent_id else visual_root_id,
                "depth": int(structure_item.get("depth") or 1),
                "username": structure_item.get("username"),
                "text": str(structure_item.get("text") or ""),
                "label": original_label if original_label in RISK_LABELS else "context",
                "original_label": original_label,
                "confidence": prediction.get("confidence"),
                "probabilities": prediction.get("probabilities") or {},
                "permalink": structure_item.get("permalink"),
                "is_risk": original_label in RISK_LABELS,
            }
        )

    node_ids = {node["id"] for node in nodes}
    edges = [
        {"from": node["parent_id"], "to": node["id"]}
        for node in nodes
        if node["parent_id"] in node_ids
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "risk_count": len(risk_ids),
        "context_count": len(nodes) - len(risk_ids) - 1,
    }


def _layout_tree(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, str]],
) -> tuple[dict[str, tuple[int, float]], int, int]:
    children: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        children[edge["from"]].append(edge["to"])

    root_id = nodes[0]["id"]
    positions: dict[str, tuple[int, float]] = {}
    next_leaf = 0
    maximum_level = 0

    def place(node_id: str, level: int, active: set[str]) -> float:
        nonlocal next_leaf, maximum_level
        maximum_level = max(maximum_level, level)
        if node_id in active:
            y = 54 + next_leaf * LEAF_GAP
            next_leaf += 1
        else:
            active = {*active, node_id}
            child_y = [place(child, level + 1, active) for child in children[node_id]]
            if child_y:
                y = sum(child_y) / len(child_y)
            else:
                y = 54 + next_leaf * LEAF_GAP
                next_leaf += 1
        x = 28 + level * (NODE_WIDTH + LEVEL_GAP)
        positions[node_id] = (x, y)
        return y

    place(root_id, 0, set())
    canvas_width = max(900, 56 + (maximum_level + 1) * (NODE_WIDTH + LEVEL_GAP))
    canvas_height = max(420, 108 + max(next_leaf, 1) * LEAF_GAP)
    return positions, canvas_width, canvas_height


def _safe_url(value: Any) -> str | None:
    if not value:
        return None
    url = str(value)
    if urlsplit(url).scheme not in {"http", "https"}:
        return None
    return html_lib.escape(url, quote=True)


def _shorten(text: str, limit: int = 72) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _tooltip(node: dict[str, Any]) -> str:
    probabilities = node.get("probabilities") or {}
    lines = [
        f"留言 ID：{node['id']}",
        f"作者：@{node.get('username') or 'unknown'}",
        f"內容：{node.get('text') or ''}",
    ]
    if node.get("confidence") is not None:
        lines.append(f"信心：{float(node['confidence']):.2%}")
    for label in ("general", "friendly", "harassment", "cyberbullying"):
        if label in probabilities:
            lines.append(f"{LABEL_NAMES[label]}：{float(probabilities[label]):.2%}")
    return html_lib.escape("\n".join(lines), quote=True)


def render_risk_tree(tree: dict[str, Any]) -> None:
    """在 Streamlit iframe 內畫出可捲動的左至右風險分支樹。"""
    nodes = tree["nodes"]
    edges = tree["edges"]
    positions, canvas_width, canvas_height = _layout_tree(nodes, edges)

    edge_html: list[str] = []
    nodes_by_id = {node["id"]: node for node in nodes}
    for edge in edges:
        parent_x, parent_y = positions[edge["from"]]
        child_x, child_y = positions[edge["to"]]
        start_x = parent_x + NODE_WIDTH
        end_x = child_x
        middle_x = (start_x + end_x) / 2
        child_label = nodes_by_id[edge["to"]]["label"]
        stroke = {
            "harassment": "#ea580c",
            "cyberbullying": "#dc2626",
        }.get(child_label, "#64748b")
        arrow_left = end_x - 12
        vertical_top = min(parent_y, child_y)
        vertical_height = abs(child_y - parent_y)
        edge_html.append(
            f'<div class="edge-horizontal" aria-hidden="true" '
            f'style="left:{start_x}px;top:{parent_y - 1.5}px;'
            f'width:{middle_x - start_x + 1.5}px;background:{stroke}"></div>'
            f'<div class="edge-vertical" aria-hidden="true" '
            f'style="left:{middle_x - 1.5}px;top:{vertical_top}px;'
            f'height:{vertical_height}px;background:{stroke}"></div>'
            f'<div class="edge-horizontal" aria-hidden="true" '
            f'style="left:{middle_x}px;top:{child_y - 1.5}px;'
            f'width:{max(arrow_left - middle_x, 0)}px;background:{stroke}"></div>'
            f'<div class="edge-arrow" aria-hidden="true" '
            f'style="left:{arrow_left}px;top:{child_y - 7}px;'
            f'border-left-color:{stroke}"></div>'
        )

    node_html: list[str] = []
    for node in nodes:
        x, center_y = positions[node["id"]]
        y = center_y - NODE_HEIGHT / 2
        label = str(node["label"])
        title = LABEL_NAMES.get(label, label)
        if node.get("confidence") is not None and node.get("is_risk"):
            title += f" · {float(node['confidence']):.1%}"
        username = (
            f"@{node['username']}"
            if node.get("username")
            else ("" if label == "root" else "@unknown")
        )
        text = _shorten(str(node.get("text") or ""))
        href = _safe_url(node.get("permalink"))
        tag = "a" if href else "div"
        link_attributes = f' href="{href}" target="_blank" rel="noopener noreferrer"' if href else ""
        node_html.append(
            f'<{tag} class="node {label}" style="left:{x}px;top:{y}px"'
            f'{link_attributes} title="{_tooltip(node)}">'
            f'<span class="kind">{html_lib.escape(title)}</span>'
            f'<strong>{html_lib.escape(username)}</strong>'
            f'<span class="text">{html_lib.escape(text)}</span>'
            f'</{tag}>'
        )

    document = f"""
<meta charset="utf-8">
<style>
  #risk-tree-component, #risk-tree-component * {{ box-sizing: border-box; }}
  #risk-tree-component {{ color: #172033; font-family: Arial, "Microsoft JhengHei", sans-serif; }}
  #risk-tree-component .shell {{ border: 1px solid #d7dee8; border-radius: 12px; overflow: hidden; background: #f8fafc; }}
  #risk-tree-component .legend {{ height: 54px; display: flex; align-items: center; gap: 18px; padding: 0 16px;
             border-bottom: 1px solid #d7dee8; background: white; font-size: 13px; }}
  #risk-tree-component .legend span {{ white-space: nowrap; }}
  #risk-tree-component .dot {{ display: inline-block; width: 12px; height: 12px; margin-right: 5px;
          border: 2px solid; border-radius: 3px; vertical-align: -2px; }}
  #risk-tree-component .dot.root {{ background: #d9eaf7; border-color: #2f6b9a; }}
  #risk-tree-component .dot.context {{ background: #e8edf3; border-color: #64748b; }}
  #risk-tree-component .dot.harassment {{ background: #ffedd5; border-color: #ea580c; }}
  #risk-tree-component .dot.cyberbullying {{ background: #fee2e2; border-color: #dc2626; }}
  #risk-tree-component .hint {{ margin-left: auto; color: #64748b; }}
  #risk-tree-component .viewport {{ width: 100%; height: 650px; overflow: auto; background: #f8fafc; }}
  #risk-tree-component .canvas {{ position: relative; width: {canvas_width}px; height: {canvas_height}px;
             background-image: radial-gradient(#dbe3ed 0.7px, transparent 0.7px);
             background-size: 18px 18px; }}
  #risk-tree-component .edge-horizontal {{ position: absolute; height: 3px; z-index: 1; }}
  #risk-tree-component .edge-vertical {{ position: absolute; width: 3px; z-index: 1; }}
  #risk-tree-component .edge-arrow {{ position: absolute; width: 0; height: 0; z-index: 1;
           border-top: 7px solid transparent; border-bottom: 7px solid transparent;
           border-left: 12px solid #64748b; }}
  #risk-tree-component .node {{ position: absolute; width: {NODE_WIDTH}px; min-height: {NODE_HEIGHT}px;
           padding: 12px 14px; border: 2px solid; border-radius: 10px; color: #111827;
           text-decoration: none; box-shadow: 0 3px 9px rgba(15,23,42,.10); background: white;
           z-index: 2; }}
  #risk-tree-component .node:hover {{ transform: translateY(-2px); box-shadow: 0 7px 18px rgba(15,23,42,.18); }}
  #risk-tree-component .node .kind {{ display: block; margin-bottom: 7px; font-size: 13px; font-weight: 700; }}
  #risk-tree-component .node strong {{ display: block; margin-bottom: 5px; font-size: 13px; }}
  #risk-tree-component .node .text {{ display: block; font-size: 13px; line-height: 1.35; }}
  #risk-tree-component .node.root {{ background: #d9eaf7; border-color: #2f6b9a; }}
  #risk-tree-component .node.context {{ background: #e8edf3; border-color: #64748b; }}
  #risk-tree-component .node.harassment {{ background: #ffedd5; border-color: #ea580c; }}
  #risk-tree-component .node.cyberbullying {{ background: #fee2e2; border-color: #dc2626; }}
</style>
<div id="risk-tree-component"><div class="shell">
  <div class="legend">
    <span><i class="dot root"></i>原始貼文</span>
    <span><i class="dot context"></i>必要上文</span>
    <span><i class="dot harassment"></i>騷擾／攻擊</span>
    <span><i class="dot cyberbullying"></i>網路霸凌</span>
    <span class="hint">可水平／垂直捲動；點擊節點開啟原文</span>
  </div>
  <div class="viewport">
    <div class="canvas" role="img" aria-label="風險留言分支樹">
      {''.join(edge_html)}
      {''.join(node_html)}
    </div>
  </div>
</div></div>
"""
    st.html(document)
