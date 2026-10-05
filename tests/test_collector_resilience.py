import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from backend.mcp_client.gateway import MCPToolGateway
from backend.threads.progress import get_analysis_progress, start_analysis_progress
from mcp_servers.internal_tools import (
    _cache_is_usable,
    _collector_timeout_seconds,
    _run_collector,
    _timeout_output,
    _timeout_progress,
)


def _write_tree(tmp_path: Path, tree: dict) -> Path:
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(json.dumps(tree), encoding="utf-8")
    return tree_path


def test_cache_rejects_single_node_when_multiple_nodes_were_discovered(
    tmp_path: Path,
) -> None:
    tree_path = _write_tree(
        tmp_path,
        {
            "nodes": [{"post_id": "root"}],
            "stats": {
                "raw_discovered_node_count": 1203,
                "excluded_unresolved_count": 1202,
            },
        },
    )

    assert not _cache_is_usable(tree_path, requested_max_nodes=30)


def test_cache_rejects_truncated_result_for_larger_request(
    tmp_path: Path,
) -> None:
    tree_path = _write_tree(
        tmp_path,
        {
            "nodes": [{"post_id": str(index)} for index in range(30)],
            "stats": {
                "raw_discovered_node_count": 30,
                "requested_max_nodes": 30,
                "stopped_by_max_nodes": True,
            },
        },
    )

    assert not _cache_is_usable(tree_path, requested_max_nodes=200)
    assert _cache_is_usable(tree_path, requested_max_nodes=30)


def test_cache_accepts_healthy_nodes_when_discovery_was_truncated(
    tmp_path: Path,
) -> None:
    tree_path = _write_tree(
        tmp_path,
        {
            "nodes": [
                {"post_id": str(index)}
                for index in range(10)
            ],
            "stats": {
                "raw_discovered_node_count": 137,
                "excluded_unresolved_count": 127,
                "excluded_processed_count": 0,
                "excluded_unprocessed_count": 127,
                "processed_detail_pages": 10,
                "requested_max_nodes": 10,
                "stopped_by_max_nodes": True,
            },
        },
    )

    assert _cache_is_usable(tree_path, requested_max_nodes=10)


def test_collector_timeout_scales_with_node_count() -> None:
    assert _collector_timeout_seconds(30) == 1200
    assert _collector_timeout_seconds(120) == 4800
    assert _collector_timeout_seconds(150) == 6000
    assert _collector_timeout_seconds(200) == 8000
    assert _collector_timeout_seconds(500) == 20000


def test_timeout_output_handles_bytes() -> None:
    exc = subprocess.TimeoutExpired(["collector"], 10, output=b"progress")
    assert _timeout_output(exc) == "progress"


def test_timeout_progress_uses_latest_counter() -> None:
    exc = subprocess.TimeoutExpired(
        ["collector"],
        10,
        output=b"[82/120] processing\n[83/120] processing",
    )
    assert _timeout_progress(exc) == "已處理到 83/120 個詳細頁"


def test_collector_stdout_updates_live_progress(tmp_path: Path) -> None:
    post_id = "stream_progress_test"
    start_analysis_progress(post_id, 10)
    returncode, output = asyncio.run(
        _run_collector(
            [sys.executable, "-u", "-c", "print('[3/10] processing')"],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_seconds=10,
            post_id=post_id,
        )
    )

    progress = get_analysis_progress(post_id)
    assert returncode == 0
    assert "[3/10]" in output
    assert progress is not None
    assert progress["current"] == 3
    assert progress["total"] == 10


def test_nested_task_group_messages_are_unwrapped() -> None:
    leaf = RuntimeError("Threads 抓取逾時")

    class FakeTaskGroupError(RuntimeError):
        exceptions = [leaf]

    assert MCPToolGateway._exception_messages(FakeTaskGroupError()) == [
        "Threads 抓取逾時"
    ]
