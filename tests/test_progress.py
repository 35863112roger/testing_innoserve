import os
from pathlib import Path

from backend.threads.progress import (
    extract_post_id,
    get_analysis_progress,
    set_analysis_progress,
    start_analysis_progress,
)
from mcp_servers import threads_collector_runner


def test_progress_tracks_node_counter_and_stage() -> None:
    post_id = "progress_test_post"
    start_analysis_progress(post_id, 120)
    set_analysis_progress(
        post_id,
        state="collecting",
        message="正在擷取第 83/120 個詳細頁節點",
        current=83,
        total=120,
    )
    progress = get_analysis_progress(post_id)

    assert progress is not None
    assert progress["state"] == "collecting"
    assert progress["current"] == 83
    assert progress["total"] == 120
    assert progress["elapsed_seconds"] >= 0
    assert progress["idle_seconds"] >= 0

    set_analysis_progress(post_id, state="classifying", message="正在分類")
    progress = get_analysis_progress(post_id)
    assert progress is not None
    assert progress["current"] == 83
    assert progress["total"] == 120


def test_extract_post_id() -> None:
    assert (
        extract_post_id("https://www.threads.com/@user/post/DRUhSlQjyvu?x=1")
        == "DRUhSlQjyvu"
    )
    assert extract_post_id("https://example.com/not-threads") is None


def test_collector_runner_overrides_original_runtime_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source_collector.py"
    output = tmp_path / "isolated_output"
    profile = tmp_path / "isolated_profile"
    probe = tmp_path / "probe.txt"
    source.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "OUTPUT_DIR = Path('original-output')\n"
        "PROFILE_DIR = Path('original-profile')\n"
        "def main():\n"
        "    Path(os.environ['COLLECTOR_TEST_PROBE']).write_text(\n"
        "        f'{OUTPUT_DIR}|{PROFILE_DIR}', encoding='utf-8'\n"
        "    )\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("THREADS_COLLECTOR_SOURCE_SCRIPT", str(source))
    monkeypatch.setenv("THREADS_COLLECTOR_OUTPUT_DIR", str(output))
    monkeypatch.setenv("THREADS_COLLECTOR_PROFILE_DIR", str(profile))
    monkeypatch.setenv("COLLECTOR_TEST_PROBE", str(probe))

    threads_collector_runner.main()

    assert probe.read_text(encoding="utf-8") == f"{output.resolve()}|{profile.resolve()}"
