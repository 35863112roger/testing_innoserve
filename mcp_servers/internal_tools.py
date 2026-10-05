import asyncio
import json
import os
import re
import subprocess
import sys
from collections import Counter, deque
from typing import Any
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from backend.config import get_settings
from backend.threads.progress import set_analysis_progress
from backend.threads.processing import thread_tree_quality_error


mcp = MCPServer(
    "internal-text-tools",
    instructions="提供文字分析，以及將 Threads 公開串文蒐集至本機資料目錄的工具。",
)


@mcp.tool()
async def text_stats(text: str) -> dict:
    """計算文字長度、行數及常見字元等基本統計。"""
    visible_characters = [char for char in text if not char.isspace()]
    common_characters = Counter(visible_characters).most_common(10)
    return {
        "characters": len(text),
        "non_whitespace_characters": len(visible_characters),
        "lines": len(text.splitlines()) or 1,
        "ascii_words": len(re.findall(r"[A-Za-z0-9_]+", text)),
        "common_characters": [
            {"character": character, "count": count}
            for character, count in common_characters
        ],
    }


@mcp.tool()
async def find_urls(text: str) -> dict:
    """從文字中找出 HTTP/HTTPS 網址，不進行網路連線。"""
    urls = re.findall(r"https?://[^\s<>\]\[()]+", text)
    return {"urls": urls, "count": len(urls)}


THREADS_POST_RE = re.compile(
    r"https?://(?:www\.)?threads\.(?:com|net)/@(?P<username>[^/?#]+)/post/(?P<post_id>[^/?#]+)",
    re.IGNORECASE,
)


def _normalize_threads_url(url: str) -> tuple[str, str]:
    match = THREADS_POST_RE.search(url.strip())
    if not match:
        raise ValueError(
            "請輸入單篇 Threads 串文網址，例如 "
            "https://www.threads.com/@username/post/POST_ID"
        )
    matched_url = match.group(0)
    parts = urlsplit(matched_url)
    path = re.sub(r"/media/?$", "", parts.path.rstrip("/"), flags=re.IGNORECASE)
    normalized = urlunsplit(("https", "www.threads.com", path, "", ""))
    return normalized, match.group("post_id")


def _tree_path(post_id: str) -> Path:
    settings = get_settings()
    return settings.threads_tree_data_dir / post_id / f"{post_id}_tree.json"


def _tree_metadata(tree_path: Path, source: str) -> dict:
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    return {
        "tree_path": str(tree_path.resolve()),
        "source": source,
        "requested_post_id": tree.get("requested_post_id"),
        "root_post_id": tree.get("conversation_root_id") or tree.get("root_post_id"),
        "node_count": len(tree.get("nodes") or []),
        "collection_complete": tree.get("collection_complete"),
        "warnings": tree.get("warnings") or [],
    }


def _collector_timeout_seconds(max_nodes: int) -> int:
    """Allow large crawls more time without making small crawls wait forever."""
    settings = get_settings()
    calculated = max(
        settings.threads_collector_base_timeout_seconds,
        max_nodes * settings.threads_collector_seconds_per_node,
    )
    return min(calculated, settings.threads_collector_max_timeout_seconds)


def _timeout_output(exc: subprocess.TimeoutExpired) -> str:
    output = exc.stderr or exc.stdout or ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    return str(output)[-1200:].strip()


def _timeout_progress(exc: subprocess.TimeoutExpired) -> str | None:
    """Extract the latest collector page counter from captured output."""
    output = exc.stderr or exc.stdout or ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    matches = re.findall(r"\[(\d+)/(\d+)\]", str(output))
    if not matches:
        return None
    current, total = matches[-1]
    return f"已處理到 {current}/{total} 個詳細頁"


async def _run_collector(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
    post_id: str,
) -> tuple[int, str]:
    """Run the collector while consuming stdout so progress stays observable."""
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    if process.stdout is None:
        process.kill()
        await process.wait()
        raise RuntimeError("Collector 未建立輸出管線")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    output_lines: deque[str] = deque(maxlen=200)
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    command,
                    timeout_seconds,
                    output="".join(output_lines),
                )
            try:
                raw_line = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=min(1.0, remaining),
                )
            except asyncio.TimeoutError:
                continue

            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace")
            output_lines.append(line)
            print(f"[collector:{post_id}] {line}", end="", flush=True)
            match = re.search(r"\[(\d+)/(\d+)\]", line)
            if match:
                current, total = (int(value) for value in match.groups())
                set_analysis_progress(
                    post_id,
                    state="collecting",
                    message=f"正在擷取第 {current}/{total} 個詳細頁節點",
                    current=current,
                    total=total,
                )

        return await process.wait(), "".join(output_lines)
    except BaseException:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise

def _cache_is_usable(
    tree_path: Path,
    requested_max_nodes: int,
) -> bool:
    try:
        tree = json.loads(
            tree_path.read_text(encoding="utf-8")
        )
    except Exception:
        return False

    stats = tree.get("stats") or {}
    cached_max_nodes = int(
        stats.get("requested_max_nodes")
        or stats.get("processed_detail_pages")
        or 0
    )

    if thread_tree_quality_error(tree):
        return False

    # 舊快取因節點上限停止，新要求又比較大時必須重抓。
    if (
        stats.get("stopped_by_max_nodes")
        and requested_max_nodes > cached_max_nodes
    ):
        return False

    return True


@mcp.tool()
async def collect_thread(
    url: str,
    force_refresh: bool = False,
    max_nodes: int = 30,
) -> dict:
    """讀取或抓取一則 Threads 串文，回傳安全的 tree.json 路徑及基本資訊。"""
    if not 1 <= max_nodes <= 500:
        raise ValueError("max_nodes 必須介於 1 與 500")

    normalized_url, post_id = _normalize_threads_url(url)
    tree_path = _tree_path(post_id)
    if (
        tree_path.is_file()
        and not force_refresh
        and _cache_is_usable(tree_path, max_nodes)
    ):
        metadata = _tree_metadata(tree_path, "cache")
        set_analysis_progress(
            post_id,
            state="cache",
            message=f"已讀取快取，共 {metadata['node_count']} 個節點",
            current=metadata["node_count"],
            total=metadata["node_count"],
        )
        return metadata

    settings = get_settings()
    timeout_seconds = _collector_timeout_seconds(max_nodes)
    runner_path = Path(__file__).with_name("threads_collector_runner.py")
    collector_env = os.environ.copy()
    collector_env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "THREADS_COLLECTOR_SOURCE_SCRIPT": str(settings.threads_collector_script),
            "THREADS_COLLECTOR_OUTPUT_DIR": str(settings.threads_tree_data_dir),
            "THREADS_COLLECTOR_PROFILE_DIR": str(settings.threads_collector_profile_dir),
        }
    )
    set_analysis_progress(
        post_id,
        state="collecting",
        message=f"正在啟動 Collector，最多處理 {max_nodes} 個節點",
        current=0,
        total=max_nodes,
    )
    try:
        returncode, output = await _run_collector(
            [
                sys.executable,
                "-u",
                str(runner_path),
                normalized_url,
                "--max-nodes",
                str(max_nodes),
                "--max-expand-rounds",
                "8",
                "--delay",
                "1.0",
                "--headless",
            ],
            cwd=runner_path.parent.parent,
            env=collector_env,
            timeout_seconds=timeout_seconds,
            post_id=post_id,
        )
    except subprocess.TimeoutExpired as exc:
        if tree_path.is_file() and _cache_is_usable(
            tree_path,
            max_nodes,
        ):
            metadata = _tree_metadata(
                tree_path,
                "stale_cache_after_timeout",
            )

            metadata["warnings"] = [
                *metadata["warnings"],
                "Collector 執行逾時，暫時使用通過品質檢查的既有快取。",
            ]
            return metadata

        progress_summary = _timeout_progress(exc)
        progress = _timeout_output(exc)
        message = (
            f"Threads 抓取超過 {timeout_seconds} 秒仍未完成。"
            f"{progress_summary + '；' if progress_summary else ''}"
            "這是逐頁確認父子關係的作業，節點越多耗時越長。"
            "可先降低『最多處理節點』，或提高 Collector 逾時設定後重試。"
        )
        if progress:
            message += f" Collector 最後輸出：{progress}"
        set_analysis_progress(post_id, state="failed", message=message)
        raise ToolError(message) from exc
    if returncode != 0:
        detail = (output or "未知錯誤")[-1200:]
        set_analysis_progress(post_id, state="failed", message=f"Threads 抓取失敗：{detail}")
        raise ToolError(f"Threads 抓取失敗：{detail}")
    if not tree_path.is_file():
        set_analysis_progress(
            post_id,
            state="failed",
            message="Collector 執行完成，但找不到預期的 tree.json",
        )
        raise ToolError("Collector 執行完成，但找不到預期的 tree.json")
    set_analysis_progress(
        post_id,
        state="finalizing",
        message="擷取完成，正在讀取並整理討論樹……",
    )
    return _tree_metadata(tree_path, "live")

# 取得全域共用的 RAG Service。
# 把 MCP 收到的資料轉交給 analyze_thread()。
def _run_rag_analysis(
    thread_id: str,
    root_text: str,
    statistics: dict[str, int],
    risk_nodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    在背景執行緒中建立並執行 RAG Service。

    採用延遲匯入，避免只使用 collect_thread 時，
    就提前載入 ChromaDB、BM25 與 Ollama 相關模組。
    """
    from backend.rag.service import get_rag_service

    service = get_rag_service()

    return service.analyze_thread(
        thread_id=thread_id,
        root_text=root_text,
        statistics=statistics,
        risk_nodes=risk_nodes,
    )


@mcp.tool()
async def analyze_thread_risk(
    thread_id: str,
    root_text: str,
    statistics: dict[str, int],
    risk_nodes: list[dict[str, Any]],
) -> dict:
    """
    根據整篇 Threads 串文的 BERT 風險分類結果，
    檢索知識庫並產生一份整體風險與處置資訊報告。

    BERT 分類只是模型訊號，不代表事實認定或法律判決。
    """
    settings = get_settings()

    if not settings.rag_enabled:
        raise ToolError(
            "RAG 功能尚未啟用，請在 .env 設定 RAG_ENABLED=true"
        )

    normalized_thread_id = str(thread_id or "").strip()
    normalized_root_text = str(root_text or "").strip()

    if not normalized_thread_id:
        raise ToolError("thread_id 不可為空")

    if not normalized_root_text:
        raise ToolError("root_text 不可為空")

    if len(risk_nodes) > 500:
        raise ToolError("risk_nodes 不可超過 500 個節點")

    labels = (
        "general",
        "friendly",
        "harassment",
        "cyberbullying",
    )

    try:
        normalized_statistics = {
            label: max(
                0,
                int(statistics.get(label, 0)),
            )
            for label in labels
        }
    except (TypeError, ValueError) as exc:
        raise ToolError(
            "statistics 的數量必須是整數"
        ) from exc

    valid_risk_labels = {
        "harassment",
        "cyberbullying",
    }

    normalized_nodes: list[dict[str, Any]] = []

    for index, node in enumerate(
        risk_nodes,
        start=1,
    ):
        if not isinstance(node, dict):
            raise ToolError(
                f"risk_nodes 第 {index} 筆不是物件"
            )

        label = str(
            node.get("label") or ""
        ).strip().lower()

        # MCP 邊界再次過濾，general/friendly 不送入 RAG。
        if label not in valid_risk_labels:
            continue

        text = str(
            node.get("text") or ""
        ).strip()

        if not text:
            continue

        try:
            confidence = float(
                node.get("confidence") or 0
            )
        except (TypeError, ValueError) as exc:
            raise ToolError(
                f"risk_nodes 第 {index} 筆的 confidence 格式錯誤"
            ) from exc

        if not 0 <= confidence <= 1:
            raise ToolError(
                f"risk_nodes 第 {index} 筆的 confidence 必須介於 0 與 1"
            )

        normalized_nodes.append(
            {
                "node_id": str(
                    node.get("node_id")
                    or node.get("target_id")
                    or ""
                ),
                "label": label,
                "confidence": confidence,
                "text": text,
                "parent_text": str(
                    node.get("parent_text")
                    or ""
                ).strip(),
            }
        )

    try:
        return await asyncio.to_thread(
            _run_rag_analysis,
            normalized_thread_id,
            normalized_root_text,
            normalized_statistics,
            normalized_nodes,
        )
    except Exception as exc:
        raise ToolError(
            f"整體串文 RAG 分析失敗：{exc}"
        ) from exc


def _run_judgment_search(
    context_summary: str,
    risk_nodes: list[dict],
    limit: int,
) -> dict:
    from backend.judgments.service import search_similar_cases

    return search_similar_cases(
        context_summary=context_summary,
        risk_nodes=risk_nodes,
        limit=limit,
    )


@mcp.tool()
async def search_similar_judgments(
    context_summary: str,
    risk_nodes: list[dict],
    limit: int = 5,
) -> dict:
    """
    使用 BERT 判定的騷擾／網路霸凌留言與串文摘要，
    查詢司法院公開裁判並回傳相關案件。
    """
    if not context_summary.strip():
        raise ToolError("context_summary 不可為空")

    if not 1 <= limit <= 10:
        raise ToolError("limit 必須介於 1 到 10")

    if len(risk_nodes) > 500:
        raise ToolError("risk_nodes 不可超過 500 筆")

    valid_labels = {"harassment", "cyberbullying"}

    normalized_nodes = [
        node
        for node in risk_nodes
        if str(node.get("label") or "").lower() in valid_labels
        and str(node.get("text") or "").strip()
    ]

    if not normalized_nodes:
        return {
            "status": "skipped",
            "reason": "沒有可供搜尋的風險留言",
            "hits": [],
            "warnings": [],
        }

    try:
        return await asyncio.to_thread(
            _run_judgment_search,
            context_summary,
            normalized_nodes,
            limit,
        )
    except Exception as exc:
        raise ToolError(
            f"類似案件判決搜尋失敗：{exc}"
        ) from exc

app = mcp.streamable_http_app()
