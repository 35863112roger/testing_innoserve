"""Run the original threads_tree collector with isolated runtime directories."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def main() -> None:
    source_path = Path(os.environ["THREADS_COLLECTOR_SOURCE_SCRIPT"]).resolve()
    output_dir = Path(os.environ["THREADS_COLLECTOR_OUTPUT_DIR"]).resolve()
    profile_dir = Path(os.environ["THREADS_COLLECTOR_PROFILE_DIR"]).resolve()

    spec = importlib.util.spec_from_file_location(
        "testing_innoserve_original_threads_collector",
        source_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"無法載入原始 Threads Collector：{source_path}")

    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    collector.OUTPUT_DIR = output_dir
    collector.PROFILE_DIR = profile_dir
    collector.main()


if __name__ == "__main__":
    main()
