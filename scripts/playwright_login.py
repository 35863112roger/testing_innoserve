"""Log in to Threads using testing_innoserve's isolated browser profile."""

from __future__ import annotations

import importlib.util

from backend.config import get_settings


def main() -> None:
    settings = get_settings()
    source_path = settings.threads_collector_script.with_name("playwright_login.py")
    spec = importlib.util.spec_from_file_location(
        "testing_innoserve_original_playwright_login",
        source_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"無法載入原始 Threads 登入程式：{source_path}")

    login_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(login_module)
    login_module.PROFILE_DIR = settings.threads_collector_profile_dir
    login_module.main()


if __name__ == "__main__":
    main()
