from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from playwright.sync_api import (
    Locator,
    Page, 
    TimeoutError as PlaywrightTimeoutError, 
    sync_playwright
)

"""
新版會分成三份輸出：

DbYsGF5EqJl_tree.json
DbYsGF5EqJl_nodes.jsonl
DbYsGF5EqJl_audit.json
-----------------------------------------------------------------------------------
tree.json

正式資料與視覺化使用，只保留：

貼文內容
作者資訊
父子關係
節點類型
回覆深度
上下文路徑
媒體與外部連結

---------------------------------------------------------------------------------
nodes.jsonl

一行一個節點，之後適合用於：

pandas
人工標註
分類模型
圖結構特徵
資料庫匯入

--------------------------------------------------------------------------------
audit.json

放研究稽核與爬蟲除錯資料：

如何判斷父節點
詳細頁出現哪些祖先
哪些頁面曾發現該節點
DOM 位置資訊
無法解析警告

這樣正式 JSON 不會被偵錯欄位塞滿，但你仍然保有檢查依據。
"""
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILE_DIR = PROJECT_ROOT / "data" / "playwright_profile"
OUTPUT_DIR = PROJECT_ROOT / "data" / "threads_tree"

POST_URL_RE = re.compile(
    r"https?://(?:www\.)?threads\.(?:com|net)/@(?P<username>[^/]+)/post/(?P<post_id>[^/?#]+)",
    re.IGNORECASE,
)

MORE_REPLIES_RE = re.compile(
    r"(?:"
    r"查看更多(?:\s*\d+\s*(?:則|個)?)?\s*回覆|"
    r"查看(?:其他|更多)?(?:\s*\d+\s*(?:則|個)?)?\s*回覆|"
    r"顯示(?:更多)?(?:\s*\d+\s*(?:則|個)?)?\s*回覆|"
    r"更多回覆|"
    r"View(?:\s+\d+)?\s+more\s+repl(?:y|ies)|"
    r"See(?:\s+\d+)?\s+more\s+repl(?:y|ies)|"
    r"Show(?:\s+\d+)?\s+repl(?:y|ies)"
    r")",
    re.IGNORECASE,
)

RELATIVE_TIME_RE = re.compile(
    r"^\d+\s*(?:秒|分鐘|小時|天|週|星期|個月|月|年|s|m|h|d|w|mo|y)$",
    re.IGNORECASE,
)

ABSOLUTE_DATE_RE = re.compile(
    r"^(?:"
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|"
    r"\d{4}年\d{1,2}月\d{1,2}日?"
    r"|"
    r"\d{1,2}月\d{1,2}日"
    r")$"
)

UI_EXACT_LINES = {
    "作者",
    "翻譯",
    "熱門",
    "最新",
    "查看動態",
    "更多",
    "回覆",
    "串文",
    "查看洞察",
    "尚無回覆",
    "顯示回覆",
    "查看更多回覆",
    "查看更多回覆",
    "更多回覆",
}

INTERACTION_COUNT_RE = re.compile(
    r"^\d{1,7}(?:[,.]\d{1,3})?(?:千|萬|億|[KkMmBb])?$"
)

PAGE_VIEW_RE = re.compile(
    r"(?P<count>"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?"
    r"\s*(?:千|萬|億|[KkMmBb])?"
    r")"
    r"\s*次?\s*(?:瀏覽|觀看|views?)",
    re.IGNORECASE,
)

IMAGE_CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/avif": ".avif",
}

class CollectionError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_post_url(url: str) -> str:
    """Normalize a Threads post URL and remove /media, query string and fragment."""
    parts = urlsplit(url)
    path = re.sub(r"/media/?$", "", parts.path.rstrip("/"), flags=re.IGNORECASE)
    return urlunsplit((parts.scheme or "https", parts.netloc, path, "", ""))


def parse_post_url(url: str) -> tuple[str, str]:
    match = POST_URL_RE.search(normalize_post_url(url))
    if not match:
        raise ValueError(
            "請輸入單篇 Threads 串文網址，例如："
            "https://www.threads.com/@username/post/shortcode"
        )
    return match.group("username"), match.group("post_id")


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

def download_image_file(
    context,
    url: str | None,
    base_path: Path,
) -> Path | None:
    """
    下載單一圖片。

    base_path 不需要副檔名，例如：
    media/DbivyP_k55o/image_01
    """
    if not url:
        return None

    base_path.parent.mkdir(parents=True, exist_ok=True)

    # 已經下載過就直接沿用，
    # 避免同一節點在不同頁面重複下載。
    existing_files = list(
        base_path.parent.glob(base_path.name + ".*")
    )

    if existing_files:
        return existing_files[0]

    try:
        response = context.request.get(
            url,
            timeout=30_000,
        )

        if not response.ok:
            return None

        content_type = (
            response.headers
            .get("content-type", "")
            .split(";")[0]
            .strip()
            .lower()
        )

        extension = IMAGE_CONTENT_TYPE_EXTENSIONS.get(
            content_type
        )

        if not extension:
            extension = Path(
                urlsplit(url).path
            ).suffix.lower()

        if extension not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
            ".avif",
        }:
            extension = ".jpg"

        output_path = base_path.with_suffix(
            extension
        )

        output_path.write_bytes(
            response.body()
        )

        return output_path

    except Exception as exc:
        print(f"[media] 圖片下載失敗：{url}")
        print(f"[media] 原因：{exc}")
        return None


def save_node_media(
    context,
    candidate: dict[str, Any],
    run_dir: Path,
) -> None:
    """
    保存單一 node 自己的圖片，
    以及影片的 poster 封面。
    """

    post_id = safe_filename(
        candidate["post_id"]
    )

    node_media_dir = (
        run_dir / "media" / post_id
    )

    # ---------------------------------------------------------
    # 1. 保存貼文 / 留言圖片
    # ---------------------------------------------------------
    for index, image in enumerate(
        candidate.get("images") or [],
        start=1,
    ):
        saved_path = download_image_file(
            context=context,
            url=image.get("src"),
            base_path=(
                node_media_dir /
                f"image_{index:02d}"
            ),
        )

        image["local_path"] = (
            saved_path
            .relative_to(run_dir)
            .as_posix()
            if saved_path
            else None
        )

    # ---------------------------------------------------------
    # 2. 影片目前只保存 poster
    # ---------------------------------------------------------
    for index, video in enumerate(
        candidate.get("videos") or [],
        start=1,
    ):
        poster_path = download_image_file(
            context=context,
            url=video.get("poster"),
            base_path=(
                node_media_dir /
                f"video_{index:02d}_poster"
            ),
        )

        video["poster_local_path"] = (
            poster_path
            .relative_to(run_dir)
            .as_posix()
            if poster_path
            else None
        )


def add_media_features(
    node: dict[str, Any],
) -> None:
    """
    加入方便後續篩選圖片 / 影片事件的欄位。
    """

    images = node.get("images") or []
    videos = node.get("videos") or []

    node["has_image"] = int(
        len(images) > 0
    )
    node["image_count"] = len(images)

    node["has_video"] = int(
        len(videos) > 0
    )
    node["video_count"] = len(videos)

def author_hash(username: str) -> str:
    return hashlib.sha256(username.lower().encode("utf-8")).hexdigest()[:16]


def parse_thread_part(text: str) -> tuple[int | None, int | None]:
    # Threads often renders 1 / 2 as three separate lines.
    match = re.search(r"(?:^|\n)\s*(\d+)\s*\n\s*/\s*\n\s*(\d+)\s*(?:\n|$)", text)
    if not match:
        match = re.search(r"(?:^|\n)\s*(\d+)\s*/\s*(\d+)\s*(?:\n|$)", text)
    if not match:
        return None, None
    part, total = int(match.group(1)), int(match.group(2))
    if 1 <= part <= total <= 100:
        return part, total
    return None, None

COUNT_MULTIPLIERS = {
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
    "千": 1_000,
    "萬": 10_000,
    "億": 100_000_000,
}

def parse_display_count(raw_value: str | None) -> int | None:
    """
    將 Threads 顯示的互動數轉成整數。

    範例：
    1,528 -> 1528
    1.5萬 -> 15000
    13 萬 -> 130000
    2.3K -> 2300
    """

    if raw_value is None:
        return None

    value = str(raw_value).strip()

    # 移除逗號以及各種空白字元
    value = value.replace(",", "")
    value = re.sub(r"\s+", "", value)

    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)(億|萬|千|[KMBkmb])?",
        value,
    )

    if not match:
        return None

    number = float(match.group(1))
    unit = match.group(2)

    if unit:
        multiplier = COUNT_MULTIPLIERS.get(
            unit.upper(),
            1,
        )

        if unit in {"千", "萬", "億"}:
            multiplier = COUNT_MULTIPLIERS[unit]

        number *= multiplier

    return int(round(number))

def normalize_count_token(value: str | None) -> str:
    """統一互動數文字，供 text_clean 清理時比對。"""

    normalized = str(value or "").strip()
    normalized = normalized.replace(",", "")
    normalized = re.sub(r"\s+", "", normalized)

    return normalized.upper()

def clean_card_text(
    raw_text: str,
    username: str,
    engagement_raw: dict[str, str | None] | None = None,
    topic_tags: list[str] | None = None,
) -> str:
    """
    清除 Threads 貼文卡片中的明顯 UI 資訊。

    text_raw 永遠保留原始證據；
    text_clean 僅保留後續 NLP 所需要的正文。
    """

    lines = [
        line.strip()
        for line in raw_text.replace("\u00a0", " ").splitlines()
    ]

    # ---------------------------------------------------------
    # 1. 找正文與 Threads UI 的分界
    # ---------------------------------------------------------
    boundary_index: int | None = None

    for index, line in enumerate(lines):

        if line in {
            "熱門",
            "最新",
            "查看動態",
            "尚無回覆",
            "顯示回覆",
        }:
            boundary_index = index
            break

        # 查看更多回覆 / 更多回覆 / 顯示更多回覆...
        if MORE_REPLIES_RE.fullmatch(line):
            boundary_index = index
            break

        # Threads Meta AI 提示
        if re.search(
            r"在貼文中提及\s*@meta\.ai",
            line,
            re.IGNORECASE,
        ):
            boundary_index = index
            break

        # 回覆 username……
        if re.fullmatch(
            r"回覆\s*@?[^…\.]+[…\.]{2,}",
            line,
            re.IGNORECASE,
        ):
            boundary_index = index
            break

    content_lines = (
        lines[:boundary_index]
        if boundary_index is not None
        else lines
    )

    output: list[str] = []

    # ---------------------------------------------------------
    # 2. 正規化 Threads topic tag
    # ---------------------------------------------------------
    normalized_topic_tags = {
        re.sub(
            r"^[#＃]\s*",
            "",
            str(tag),
        ).strip().casefold()
        for tag in (topic_tags or [])
        if str(tag).strip()
    }

    # ---------------------------------------------------------
    # 3. 清除作者、時間、UI、topic tag
    # ---------------------------------------------------------
    for index, line in enumerate(content_lines):

        if not line:
            continue

        # 作者名稱通常位於最前面
        if (
            index < 4
            and line.lower()
                .lstrip("@")
                .rstrip("：:")
            == username.lower()
        ):
            continue

        # 相對時間，例如：
        # 3小時、2天、1週
        #
        # 絕對日期，例如：
        # 2026-7-24、2026/7/24
        if index < 6 and (
            RELATIVE_TIME_RE.fullmatch(line)
            or ABSOLUTE_DATE_RE.fullmatch(line)
        ):
            continue

        # Threads topic tag
        if index < 6:
            normalized_line = re.sub(
                r"^[#＃]\s*",
                "",
                line,
            ).strip().casefold()

            if normalized_line in normalized_topic_tags:
                continue

        # Threads UI
        if line in UI_EXACT_LINES or line == "·":
            continue

        output.append(line)

    # ---------------------------------------------------------
    # 4. 移除 1 / 2、2 / 2 等續串標記
    # ---------------------------------------------------------
    joined = "\n".join(output)

    joined = re.sub(
        r"(?:^|\n)"
        r"\s*\d+\s*"
        r"\n\s*/\s*\n"
        r"\s*\d+\s*"
        r"(?=\n|$)",
        "\n",
        joined,
    )

    output = [
        line
        for line in joined.splitlines()
        if line.strip()
    ]

    # ---------------------------------------------------------
    # 5. 如果已找到 UI boundary，
    #    移除正文尾端連續的互動數
    #
    # 例如：
    #
    # 148的超可愛
    # 1
    # 2
    #
    # -> 148的超可愛
    # ---------------------------------------------------------
    if boundary_index is not None:
        removed = 0

        while (
            output
            and removed < 5
            and INTERACTION_COUNT_RE.fullmatch(output[-1])
        ):
            output.pop()
            removed += 1

    # ---------------------------------------------------------
    # 6. 再用已解析的 engagement 數字做一次保險
    # ---------------------------------------------------------
    raw_metrics = engagement_raw or {}

    metric_tokens = {
        normalize_count_token(value)
        for value in raw_metrics.values()
        if value is not None
    }

    removed = 0

    while output and removed < 5:

        last_token = normalize_count_token(
            output[-1]
        )

        if (
            last_token
            and last_token in metric_tokens
        ):
            output.pop()
            removed += 1
            continue

        break

    return re.sub(
        r"\n{3,}",
        "\n\n",
        "\n".join(output),
    ).strip()


def get_canonical_url(page: Page) -> str | None:
    locator = page.locator('link[rel="canonical"]')
    if locator.count() == 0:
        return None
    return locator.first.get_attribute("href")


def wait_for_page(page: Page) -> None:
    page.wait_for_timeout(1800)
    try:
        page.wait_for_load_state("networkidle", timeout=8000)
    except PlaywrightTimeoutError:
        # Threads may keep background requests alive; DOM readiness is enough for this collector.
        pass

def extract_page_view_raw(page: Page) -> str | None:
    """
    從 Threads 詳細頁頂部取得串文總瀏覽數。

    例如：
    12 萬次瀏覽 -> "12 萬"
    1.5萬次瀏覽 -> "1.5萬"

    找不到時回傳 None。
    """
    try:
        body_text = page.locator("body").inner_text(timeout=3000)
    except Exception:
        return None

    # 瀏覽數位於頁面頂部，避免誤抓下面留言中的文字
    top_lines = body_text.splitlines()[:60]
    top_text = "\n".join(top_lines)

    match = PAGE_VIEW_RE.search(top_text)

    if not match:
        return None

    return match.group("count").strip()

# 限制展開回覆的按鈕範圍
def expand_visible_replies(
    page: Page,
    scope: Locator,
    max_rounds: int = 18,
) -> dict[str, Any]:
    """
    盡可能展開 Threads 當下公開可見的回覆。

    與舊版不同：
    1. 每一輪都立即抽取目前 DOM 中的 cards。
    2. 用 post_id 判斷是否真的發現新節點，
       不再只依賴 <time> 數量。
    3. innerText / aria-label / title 都拿來判斷
       是否為「顯示更多回覆」控制項。
    4. 所有曾經出現在 DOM 的 card 都累積保存。
    """

    clicked_total = 0
    stable_rounds = 0
    rounds_used = 0

    # ---------------------------------------------------------
    # 所有曾經在此詳細頁被看見的 card
    #
    # key:
    #     post_id
    #
    # value:
    #     candidate
    # ---------------------------------------------------------
    seen_cards: dict[str, dict[str, Any]] = {}

    # 保留第一次發現的順序。
    # 對後續 parent inference 很重要。
    first_seen_order: list[str] = []


    def absorb_current_cards() -> tuple[int, int]:
        """
        把目前 DOM 中能取得的 cards 加入 seen_cards。

        回傳：
            new_count:
                本輪首次發現的 post_id 數

            visible_count:
                本輪 DOM 中抽取到的 card 數
        """

        current_cards = extract_card_candidates(scope)

        new_count = 0

        for card in current_cards:

            post_id = card.get("post_id")

            if not post_id:
                continue

            if post_id not in seen_cards:

                seen_cards[post_id] = card
                first_seen_order.append(post_id)

                new_count += 1

            else:
                # 同一個 node 在不同 scroll round
                # 可能取得比較完整的文字 / media / engagement。
                #
                # 使用既有 merge_node() 合併。
                seen_cards[post_id] = merge_node(
                    seen_cards[post_id],
                    card,
                )

        return new_count, len(current_cards)


    # =========================================================
    # Round 0：
    # 還沒開始捲動前先保存一次。
    #
    # 這非常重要，避免後續 DOM virtualization
    # 把一開始看得到的 nodes 移除。
    # =========================================================
    initial_new, initial_visible = absorb_current_cards()

    print(
        "[expand] initial "
        f"visible={initial_visible} "
        f"unique={len(seen_cards)}"
    )


    # =========================================================
    # 正式展開
    # =========================================================
    for round_index in range(max_rounds):

        rounds_used = round_index + 1

        clicked_this_round = 0

        # 每輪重新取得 controls，
        # 因為 Threads DOM 會動態變更。
        controls = scope.locator(
            'button, [role="button"], a'
        )

        count = min(
            controls.count(),
            500,
        )

        # -----------------------------------------------------
        # A. 尋找「顯示更多回覆」
        # -----------------------------------------------------
        for index in range(count):

            control = controls.nth(index)

            try:
                if not control.is_visible(
                    timeout=200
                ):
                    continue

                description_parts: list[str] = []

                # 1. visible text
                try:
                    inner_text = (
                        control.inner_text(
                            timeout=300
                        )
                        or ""
                    ).strip()

                    if inner_text:
                        description_parts.append(
                            inner_text
                        )

                except Exception:
                    pass


                # 2. aria-label
                try:
                    aria_label = (
                        control.get_attribute(
                            "aria-label"
                        )
                        or ""
                    ).strip()

                    if aria_label:
                        description_parts.append(
                            aria_label
                        )

                except Exception:
                    pass


                # 3. title
                try:
                    title = (
                        control.get_attribute(
                            "title"
                        )
                        or ""
                    ).strip()

                    if title:
                        description_parts.append(
                            title
                        )

                except Exception:
                    pass


                description = " ".join(
                    description_parts
                ).strip()


                if (
                    not description
                    or not MORE_REPLIES_RE.search(
                        description
                    )
                ):
                    continue


                # 避免按鈕剛好在 viewport 邊界外。
                try:
                    control.scroll_into_view_if_needed(
                        timeout=1000
                    )
                except Exception:
                    pass


                control.click(
                    timeout=2500
                )

                clicked_total += 1
                clicked_this_round += 1

                # Threads 回覆是動態載入，
                # 不要點完馬上往下滑。
                page.wait_for_timeout(
                    750
                )

            except Exception:
                continue


        # -----------------------------------------------------
        # B. 往下移動，觸發 lazy loading
        # -----------------------------------------------------
        try:
            scope.hover(timeout=1000)
        except Exception:
            pass

        page.mouse.wheel(
            0,
            1800,
        )

        page.wait_for_timeout(
            1200
        )


        # -----------------------------------------------------
        # C. 立刻保存這一輪目前 DOM 中的 cards
        # -----------------------------------------------------
        new_count, visible_count = (
            absorb_current_cards()
        )


        print(
            f"[expand] round={round_index + 1} "
            f"clicked={clicked_this_round} "
            f"visible={visible_count} "
            f"new={new_count} "
            f"unique={len(seen_cards)} "
            f"stable={stable_rounds}"
        )


        # -----------------------------------------------------
        # D. 判斷是否真的穩定
        #
        # 不再使用：
        #
        #     time.count() 有沒有變
        #
        # 而是看：
        #
        #     有沒有新 post_id
        #     有沒有成功展開回覆
        # -----------------------------------------------------
        if (
            clicked_this_round == 0
            and new_count == 0
        ):
            stable_rounds += 1

        else:
            stable_rounds = 0


        # 原本是 2 輪。
        # Threads lazy loading 偶爾會停頓，
        # 改成 4 輪比較保守。
        if stable_rounds >= 4:
            break


    # =========================================================
    # 根據「第一次被看到的順序」組回 cards。
    #
    # 不用最後 DOM 的位置排序，
    # 避免前面出現過的 node 因 virtualization 消失。
    # =========================================================
    accumulated_cards = [
        seen_cards[post_id]
        for post_id in first_seen_order
        if post_id in seen_cards
    ]


    return {
        "more_reply_clicks": clicked_total,

        # 保留舊欄位，方便 audit / debug。
        "time_element_count": (
            scope.locator("time").count()
        ),

        # 新增診斷欄位。
        "expand_rounds_used": rounds_used,
        "unique_post_count": len(
            accumulated_cards
        ),
        "stable_rounds_at_end": (
            stable_rounds
        ),

        # 最重要：
        # 傳回所有 round 曾經看過的 cards。
        "cards": accumulated_cards,
    }


def extract_page_metadata(page: Page) -> dict[str, Any]:
    def meta(property_name: str | None = None, name: str | None = None) -> str | None:
        if property_name:
            locator = page.locator(f'meta[property="{property_name}"]')
        else:
            locator = page.locator(f'meta[name="{name}"]')
        if locator.count() == 0:
            return None
        return locator.first.get_attribute("content")

    return {
        "final_url": normalize_post_url(page.url),
        "canonical_url": normalize_post_url(get_canonical_url(page) or page.url),
        "og_title": meta(property_name="og:title"),
        "og_description": meta(property_name="og:description"),
        "og_image": meta(property_name="og:image"),
    }


def extract_card_candidates(scope: Locator) -> list[dict[str, Any]]:
    """
    Extract post-card candidates from time anchors.

    Primary boundary: closest data-pressable-container.
    Fallback boundary: the smallest useful ancestor containing the current post URL,
    visible text/media and interactive controls.
    """
    raw_candidates = scope.evaluate(
        r"""
        (root) => {
          function normalizePostUrl(rawUrl) {
            const url = new URL(rawUrl, location.href);
            url.search = "";
            url.hash = "";
            url.pathname = url.pathname
              .replace(/\/media\/?$/i, "")
              .replace(/\/$/, "");
            return url.href;
          }

          function parsePostUrl(rawUrl) {
            const url = normalizePostUrl(rawUrl);
            const match = url.match(/threads\.(?:com|net)\/@([^/]+)\/post\/([^/?#]+)/i);
            if (!match) return null;
            return { url, username: match[1], postId: match[2] };
          }

          function distinctPostIds(element) {
            const ids = new Set();
            for (const anchor of element.querySelectorAll('a[href*="/post/"]')) {
              const parsed = parsePostUrl(anchor.href);
              if (parsed) ids.add(parsed.postId);
            }
            return Array.from(ids);
          }

          function usefulCard(timeElement, postId) {
            let card = timeElement.closest('[data-pressable-container="true"]');
            if (card) {
              const text = (card.innerText || "").trim();
              const hasMedia = !!card.querySelector('img, video');
              if (text.length > 4 || hasMedia) return card;
            }

            let node = timeElement;
            let best = null;
            for (let depth = 0; depth < 15 && node; depth += 1) {
              node = node.parentElement;
              if (!node) break;

              const rect = node.getBoundingClientRect();
              const text = (node.innerText || "").trim();
              const postIds = distinctPostIds(node);
              const hasCurrent = postIds.includes(postId);
              const hasMedia = !!node.querySelector('img, video');
              const controls = node.querySelectorAll('button, [role="button"]').length;

              if (
                hasCurrent && rect.width >= 240 && rect.height >= 40 &&
                (text.length > 4 || hasMedia)
              ) {
                best = node;
                if (controls >= 2 && rect.height >= 70) break;
              }
            }
            return best;
          }

          function isAvatar(img) {
            const alt = (img.alt || "").toLowerCase();
            return alt.includes("大頭貼") || alt.includes("profile picture") || alt.includes("avatar");
          }

          function insideAny(element, blocks) {
            return blocks.some(block => block.contains(element));
          }

          function findEmbeddedBlocks(card, ownPostId) {
            const blocks = [];
            const seen = new Set();

            for (const nestedTime of card.querySelectorAll("time")) {
              const nestedLink = nestedTime.closest('a[href*="/post/"]');
              if (!nestedLink) continue;

              const nestedParsed = parsePostUrl(nestedLink.href);
              if (!nestedParsed || nestedParsed.postId === ownPostId) continue;

              let block = nestedTime.closest('[data-pressable-container="true"]');
              if (!block || block === card || !card.contains(block)) continue;

              if (!seen.has(block)) {
                seen.add(block);
                blocks.push(block);
              }
            }

            // 若容器彼此巢狀，只保留較內層、較接近被引用貼文的卡片。
            return blocks.filter(
              block => !blocks.some(other => other !== block && block.contains(other))
            );
          }

          function topicText(rawText) {
            return (rawText || "")
              .replace(/^[#＃]\s*/, "")
              .replace(/\s+/g, " ")
              .trim();
          }

          function isTopicLink(rawHref) {
            let url;
            try { url = new URL(rawHref, location.href); } catch { return false; }
            if (!/threads\.(?:com|net)$/i.test(url.hostname)) return false;

            const path = decodeURIComponent(url.pathname || "");
            const serpType = (url.searchParams.get("serp_type") || "").toLowerCase();

            return (
              /\/(?:topic|topics|tag|tags|t)\//i.test(path) ||
              (path.toLowerCase().includes("/search") && serpType === "tags") ||
              (path.toLowerCase().includes("/search") && url.searchParams.has("q"))
            );
          }

          function extractTopicTags(card, excludedBlocks = []) {
            const result = [];
            const seen = new Set();

            for (const anchor of card.querySelectorAll("a[href]")) {
              if (insideAny(anchor, excludedBlocks)) continue;
              if (!isTopicLink(anchor.href)) continue;

              const text = topicText(anchor.innerText || anchor.textContent || "");
              if (!text || seen.has(text)) continue;

              seen.add(text);
              result.push(text);
            }

            return result;
          }

          function controlDescription(control) {
            const parts = [];
        
            const ownAria = control.getAttribute("aria-label");
            const ownTitle = control.getAttribute("title");
        
            if (ownAria) parts.push(ownAria);
            if (ownTitle) parts.push(ownTitle);
            if (control.innerText) parts.push(control.innerText);
        
            for (const element of control.querySelectorAll("[aria-label]")) {
              const label = element.getAttribute("aria-label");
              if (label) parts.push(label);
            }
        
            return parts.join(" ").trim();
          }
        
        
          function extractCountToken(text) {
            if (!text) return null;
        
            /*
             * 支援：
             * 1,528
             * 503
             * 1.5萬
             * 2.3K
             */
            const matches = text.match(
              /\d[\d,]*(?:\.\d+)?\s*(?:億|萬|千|K|M|B)?/gi
            );
        
            if (!matches || matches.length === 0) {
              return null;
            }
        
            /*
             * 若 aria-label 中同時存在其他數字，
             * 通常最靠後的數字較可能是互動數。
             */
            return matches[matches.length - 1]
              .replace(/\s+/g, "")
              .trim();
          }
        
        
          function extractEngagement(card, excludedBlocks = []) {
            const result = {
              view: null,
              like: null,
              reply: null,
              repost: null,
              share: null
            };
        
            const controls = Array.from(
              card.querySelectorAll('button, [role="button"]')
            );
        
            for (const control of controls) {
              if (insideAny(control, excludedBlocks)) continue;
              const description = controlDescription(control);
              const lower = description.toLowerCase();
        
              let type = null;
              if (
                /瀏覽次數|觀看次數|瀏覽|觀看|views?\b/.test(lower)
              ){
                type = "view";
              } else if (
                /按讚|讚|喜歡|like/.test(lower)
              ) {
                type = "like";
              } else if (
                /回覆|留言|reply|comment/.test(lower)
              ) {
                type = "reply";
              } else if (
                /轉發|重新發佈|重新發布|repost/.test(lower)
              ) {
                type = "repost";
              } else if (
                /分享|傳送|share|send/.test(lower)
              ) {
                type = "share";
              }
        
              if (!type || result[type] !== null) {
                continue;
              } 
        
              const count =
                extractCountToken(control.innerText || "") ||
                extractCountToken(description);
        
              if (count !== null) {
                result[type] = count;
              }
            }
            
            /*
             * 瀏覽數有時不是 button，而是一般文字或 aria-label。
             * 只針對含「瀏覽／觀看／views」的元素做備援，
             * 避免誤抓正文中的其他數字。
             */
            if (result.view === null) {
              const viewCandidates = Array.from(
                card.querySelectorAll(
                  'span, [aria-label], [title]'
                )
              );
            
              for (const element of viewCandidates) {
                if (insideAny(element, excludedBlocks)) {
                  continue;
                }
            
                const description = [
                  element.getAttribute("aria-label") || "",
                  element.getAttribute("title") || "",
                  element.innerText || ""
                ].join(" ").trim();
            
                if (
                  !/瀏覽次數|觀看次數|瀏覽|觀看|views?\b/i.test(description)
                ) {
                  continue;
                }
            
                const count = extractCountToken(description);
            
                if (count !== null) {
                  result.view = count;
                  break;
                }
              }
            }
            return result;
          }
          
          const results = [];
          const timeElements = Array.from(
            root.querySelectorAll("time")
          );

          for (const timeElement of timeElements) {
            const link = timeElement.closest('a[href*="/post/"]');
            if (!link) continue;

            const parsed = parsePostUrl(link.href);
            if (!parsed) continue;

            const card = usefulCard(timeElement, parsed.postId);
            if (!card) continue;

            const rect = card.getBoundingClientRect();
            const allPostLinks = [];
            const linkSeen = new Set();
            for (const anchor of card.querySelectorAll('a[href*="/post/"]')) {
              const candidate = parsePostUrl(anchor.href);
              if (!candidate || linkSeen.has(candidate.postId)) continue;
              linkSeen.add(candidate.postId);
              allPostLinks.push(candidate);
            }

            let ancestor = card.parentElement?.closest('[data-pressable-container="true"]') || null;
            let nested = false;
            if (ancestor) {
              const ancestorIds = distinctPostIds(ancestor);
              nested = ancestorIds.some(id => id !== parsed.postId);
            }

            const embeddedBlocks = findEmbeddedBlocks(card, parsed.postId);

            const images = Array.from(card.querySelectorAll("img"))
              .filter(img => !isAvatar(img) && !insideAny(img, embeddedBlocks))
              .map(img => ({
                src: img.currentSrc || img.src || null,
                alt: img.alt || null,
                width: img.naturalWidth || null,
                height: img.naturalHeight || null
              }));

            const videos = Array.from(card.querySelectorAll("video"))
              .filter(video => !insideAny(video, embeddedBlocks))
              .map(video => ({
                src: video.currentSrc || video.src || null,
                poster: video.poster || null
              }));

            const topicTags = extractTopicTags(card, embeddedBlocks);

            const embeddedPosts = [];
            const embeddedSeen = new Set();
            for (const block of embeddedBlocks) {
              const nestedTime = block.querySelector("time");
              const nestedLink = nestedTime?.closest('a[href*="/post/"]') ||
                block.querySelector('a[href*="/post/"]');
              if (!nestedLink) continue;

              const nestedParsed = parsePostUrl(nestedLink.href);
              if (!nestedParsed || nestedParsed.postId === parsed.postId) continue;
              if (embeddedSeen.has(nestedParsed.postId)) continue;
              embeddedSeen.add(nestedParsed.postId);

              const nestedImages = Array.from(block.querySelectorAll("img"))
                .filter(img => !isAvatar(img))
                .map(img => ({
                  src: img.currentSrc || img.src || null,
                  alt: img.alt || null,
                  width: img.naturalWidth || null,
                  height: img.naturalHeight || null
                }));

              const nestedVideos = Array.from(block.querySelectorAll("video")).map(video => ({
                src: video.currentSrc || video.src || null,
                poster: video.poster || null
              }));

              embeddedPosts.push({
                post_id: nestedParsed.postId,
                username: nestedParsed.username,
                permalink: nestedParsed.url,
                text_raw: (block.innerText || "").trim(),
                engagement_raw: extractEngagement(block),
                datetime: nestedTime?.getAttribute("datetime") || null,
                topic_tags: extractTopicTags(block),
                images: nestedImages,
                videos: nestedVideos
              });
            }

            function unwrapExternalUrl(rawHref) {
              let url;
              try { url = new URL(rawHref, location.href); } catch { return null; }
              if (!/^https?:$/.test(url.protocol)) return null;

              // Threads wraps external URLs as https://l.threads.com/?u=<encoded-url>.
              if (/^(?:l\.)?threads\.(?:com|net)$/i.test(url.hostname)) {
                const wrapped = url.searchParams.get("u");
                if (!wrapped) return null;
                try { url = new URL(decodeURIComponent(wrapped)); } catch { return null; }
              }

              if (/threads\.(?:com|net)$/i.test(url.hostname)) return null;
              url.hash = "";
              return url.href;
            }

            const externalLinks = [];
            const externalSeen = new Set();
            for (const anchor of card.querySelectorAll('a[href]')) {
              if (insideAny(anchor, embeddedBlocks)) continue;
              const normalized = unwrapExternalUrl(anchor.href);
              if (!normalized || externalSeen.has(normalized)) continue;
              externalSeen.add(normalized);
              externalLinks.push({
                url: normalized,
                text: (anchor.innerText || "").trim() || null
              });
            }
            
            let textRaw = (card.innerText || "").trim();
            for (const block of embeddedBlocks) {
              const nestedText = (block.innerText || "").trim();
              if (nestedText) textRaw = textRaw.replace(nestedText, "");
            }
            textRaw = textRaw.replace(/\n{3,}/g, "\n\n").trim();

            const engagementRaw = extractEngagement(card, embeddedBlocks);
            
            results.push({
              post_id: parsed.postId,
              permalink: parsed.url,
              username: parsed.username,
              text_raw: textRaw,
              engagement_raw: engagementRaw,
              datetime: timeElement.getAttribute("datetime"),
              topic_tags: topicTags,
              images,
              videos,
              embedded_posts: embeddedPosts,
              external_links: externalLinks,
              linked_post_ids: allPostLinks.map(item => item.postId),
              linked_posts: allPostLinks,
              nested_candidate: nested,
              rect: {
                top: rect.top + window.scrollY,
                left: rect.left + window.scrollX,
                width: rect.width,
                height: rect.height
              }
            });
          }

          return results;
        }
        """
    )

    # Dedupe /media, repeated time links and overly-small candidate boundaries.
    best_by_id: dict[str, dict[str, Any]] = {}
    for candidate in raw_candidates:
        score = len(candidate.get("text_raw") or "")
        score += 500 * len(candidate.get("images") or [])
        score += 700 * len(candidate.get("videos") or [])
        if candidate.get("nested_candidate"):
            score -= 5000
        candidate["_score"] = score

        previous = best_by_id.get(candidate["post_id"])
        if previous is None or score > previous["_score"]:
            best_by_id[candidate["post_id"]] = candidate

    candidates = list(best_by_id.values())
    candidates.sort(key=lambda item: (item.get("rect", {}).get("top", 0), item["post_id"]))

    for candidate in candidates:
        candidate.pop("_score", None)

        engagement_raw = candidate.pop("engagement_raw", {}) or {}

        candidate["engagement_snapshot"] = {
            "displayed_view_count": parse_display_count(
                engagement_raw.get("view")
            ),
            "displayed_like_count": parse_display_count(
                engagement_raw.get("like")
            ),
            "displayed_reply_count": parse_display_count(
                engagement_raw.get("reply")
            ),
            "displayed_repost_count": parse_display_count(
                engagement_raw.get("repost")
            ),
            "displayed_share_count": parse_display_count(
                engagement_raw.get("share")
            ),
            "raw": {
                "view": engagement_raw.get("view"),
                "like": engagement_raw.get("like"),
                "reply": engagement_raw.get("reply"),
                "repost": engagement_raw.get("repost"),
                "share": engagement_raw.get("share"),
            },
            "collected_at": utc_now_iso(),
        }

        topic_tags = [
            str(tag).strip()
            for tag in (candidate.get("topic_tags") or [])
            if str(tag).strip()
        ]
        candidate["topic_tags"] = list(dict.fromkeys(topic_tags))

        candidate["text_clean"] = clean_card_text(
            candidate.get("text_raw") or "",
            candidate.get("username") or "",
            engagement_raw=engagement_raw,
            topic_tags=candidate["topic_tags"],
        )

        compact_embedded_posts: list[dict[str, Any]] = []
        for embedded in candidate.get("embedded_posts") or []:
            embedded_raw_metrics = embedded.pop("engagement_raw", {}) or {}
            embedded_tags = [
                str(tag).strip()
                for tag in (embedded.get("topic_tags") or [])
                if str(tag).strip()
            ]
            embedded_tags = list(dict.fromkeys(embedded_tags))

            compact_embedded_posts.append({
                "post_id": embedded.get("post_id"),
                "username": embedded.get("username"),
                "text_clean": clean_card_text(
                    embedded.get("text_raw") or "",
                    embedded.get("username") or "",
                    engagement_raw=embedded_raw_metrics,
                    topic_tags=embedded_tags,
                ),
                "datetime": embedded.get("datetime"),
                "permalink": embedded.get("permalink"),
                "topic_tags": embedded_tags,
                "images": [
                    image for image in (embedded.get("images") or [])
                    if "favicon" not in str(image.get("src") or "").lower()
                ],
                "videos": embedded.get("videos") or [],
            })

        candidate["embedded_posts"] = compact_embedded_posts
        part, total = parse_thread_part(candidate.get("text_raw") or "")
        candidate["thread_part"] = part
        candidate["thread_total"] = total
        candidate["author_hash"] = author_hash(candidate.get("username") or "unknown")

        # A card always contains its own permalink. It is not a quoted/linked post.
        own_id = candidate["post_id"]
        linked_posts = [
            item for item in candidate.get("linked_posts", [])
            if item.get("postId") != own_id
        ]
        candidate["linked_posts"] = linked_posts
        candidate["linked_post_ids"] = [item.get("postId") for item in linked_posts]

        # Keep article preview images, but remove tiny favicons from the media list.
        candidate["images"] = [
            image for image in candidate.get("images", [])
            if "favicon" not in str(image.get("src") or "").lower()
        ]

    return candidates


def merge_node(existing: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return dict(candidate)

    merged = dict(existing)
    for key, value in candidate.items():
        if value in (None, "", [], {}):
            continue
        old = merged.get(key)
        if key in {"text_raw", "text_clean"}:
            if len(str(value)) > len(str(old or "")):
                merged[key] = value
        elif key in {
            "images", "videos", "external_links", "linked_posts", "linked_post_ids",
            "topic_tags", "embedded_posts",
        }:
            if len(value) > len(old or []):
                merged[key] = value
        else:
            merged[key] = value
    return merged


def save_debug_page(page: Page, folder: Path, post_id: str, screenshot: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(post_id)
    (folder / f"{stem}.html").write_text(page.content(), encoding="utf-8")
    (folder / f"{stem}.txt").write_text(page.locator("body").inner_text(), encoding="utf-8")
    if screenshot:
        try:
            page.screenshot(path=str(folder / f"{stem}.png"), full_page=True)
        except Exception:
            pass

def open_actual_detail_page(
    page: Page,
    url: str,
) -> tuple[Any, Locator]:
    """
    Threads 可能先把 permalink 導向首頁，
    必須點擊首頁注入的指定貼文，才會開啟真正詳細頁。
    """
    _, requested_post_id = parse_post_url(url)
    response = page.goto(
        url,
        wait_until="domcontentloaded",
        timeout=60_000,
    )
    wait_for_page(page)

    # 若permalink 被導向首頁 點擊注入首頁指定貼文
    if requested_post_id not in page.url:
        target_link = page.locator(
            f'a[href$="/post/{requested_post_id}"], '
            f'a[href$="/post/{requested_post_id}/"]'
        ).first

        if target_link.count() == 0:
            raise CollectionError(
                "Threads 已導向首頁，但找不到指定貼文連結："
                f"{requested_post_id}"
            )

        target_link.click(timeout=10_000)

        try:
            page.wait_for_url(
                re.compile(
                    rf".*/post/{re.escape(requested_post_id)}"
                    r"(?:[/?#].*)?$"
                ),
                timeout=20_000,
            )
        except PlaywrightTimeoutError as exc:
            raise CollectionError(
                "已找到指定貼文，但無法進入真正詳細頁："
                f"{requested_post_id}"
            ) from exc

        wait_for_page(page)

    if requested_post_id not in page.url:
        raise CollectionError(
            "目前頁面不是指定串文詳細頁："
            f"要求={requested_post_id}，實際={page.url}"
        )

    # Threads 可能同時保留首頁與詳細頁兩個直欄。
    # 由後往前找可見且包含指定貼文的 region。
    regions = page.locator('[role="region"]')
    detail_scope: Locator | None = None

    for index in range(regions.count() - 1, -1, -1):
        region = regions.nth(index)

        try:
            if not region.is_visible(timeout=500):
                continue
        except Exception:
            continue

        matching_links = region.locator(
            f'a[href$="/post/{requested_post_id}"], '
            f'a[href$="/post/{requested_post_id}/"]'
        )

        if matching_links.count() > 0:
            detail_scope = region
            break

    if detail_scope is None:
        raise CollectionError(
            "已進入指定串文網址，但找不到詳細頁內容區域："
            f"{requested_post_id}"
        )

    return response, detail_scope

def collect_detail_page(
    page: Page,
    url: str,
    max_expand_rounds: int,
    debug_folder: Path | None = None,
) -> dict[str, Any]:
    response, detail_scope = open_actual_detail_page(
        page,
        url,
    )

    metadata = extract_page_metadata(page)

    # 先在頁面仍停留頂端時取得瀏覽數
    page_view_raw = extract_page_view_raw(page)

    print(
        f"[view] page={url} "
        f"raw={page_view_raw!r}"
    )

    # ============================================================
    # 展開回覆，同時累積每一輪曾經看到的 cards
    # ============================================================
    expand_result = expand_visible_replies(
        page,
        detail_scope,
        max_rounds=max_expand_rounds,
    )

    wait_for_page(page)

    # ============================================================
    # 第一次沒有取得瀏覽數時，再做一次備援
    # ============================================================
    if page_view_raw is None:
        page_view_raw = extract_page_view_raw(
            page
        )

    # ============================================================
    # expand_visible_replies() 已經累積所有 round 的 cards。
    #
    # pop 出來避免 cards 同時又被 **expand_stats 展開。
    # ============================================================
    cards = expand_result.pop(
        "cards",
        None,
    )

    # 極少數情況如果沒有取得任何 card，
    # 再使用舊方式做一次最後 fallback。
    if not cards:
        cards = extract_card_candidates(detail_scope)

    expand_stats = expand_result

    if debug_folder is not None:
        try:
            _, page_id = parse_post_url(url)
        except ValueError:
            page_id = "unknown"
        save_debug_page(page, debug_folder, page_id)

    _, requested_post_id = parse_post_url(url)

    if not any(
        card.get("post_id") == requested_post_id
        for card in cards
    ):
        raise CollectionError(
            "詳細頁擷取結果沒有包含要求的貼文："
            f"{requested_post_id}"
        )

    return {
        "requested_url": normalize_post_url(url),
        "http_status": response.status if response else None,
        **metadata,
        **expand_stats,
        "page_view_raw": page_view_raw,
        "cards": cards,
    }

def classify_node(
    node: dict[str, Any],
    conversation_root_id: str,
) -> str:
    """
    判斷節點類型。

    root_post：
        真正 conversation root。

    thread_continuation：
        任意作者的編號續串，例如 2/2、3/4。
        不再限制只能是 conversation root 作者。

    reply：
        一般回覆。
    """

    if (
        node["post_id"]
        == conversation_root_id
    ):
        return "root_post"

    if (
        node.get("thread_part")
        and node.get("thread_total")
        and node["thread_part"] > 1
    ):
        return "thread_continuation"

    return "reply"

def resolve_conversation_root(
    requested_post_id: str,
    nodes: dict[str, dict[str, Any]],
    parent_proposals: dict[str, dict[str, Any]],
) -> str:
    """
    從使用者輸入的貼文開始，
    沿著 parent proposal 一直往上尋找，
    找到目前 Collector 可確認的 conversation root。

    範例：

        requested:
        DbLcajRGGAT

        parent:
        DbLcajRGGAT -> DbK-23KlEWM

        DbK-23KlEWM -> None

    回傳：

        DbK-23KlEWM

    若輸入貼文本來就沒有 parent，
    則回傳 requested_post_id 本身。

    seen 用來避免錯誤 parent 關係形成循環。
    """

    current_id = requested_post_id
    seen: set[str] = set()

    while current_id:
        if current_id in seen:
            # 防止意外形成 parent cycle
            break

        seen.add(current_id)

        proposal = parent_proposals.get(
            current_id
        )

        if not proposal:
            break

        parent_id = proposal.get("parent_id")

        if not parent_id:
            break

        # parent 必須真的有被 Collector 收進 nodes
        if parent_id not in nodes:
            break

        current_id = parent_id

    return current_id

def calculate_depths(nodes: dict[str, dict[str, Any]], root_id: str) -> None:
    for node in nodes.values():
        node["depth"] = None

    if root_id in nodes:
        nodes[root_id]["depth"] = 0

    for _ in range(len(nodes) + 1):
        changed = False
        for post_id, node in nodes.items():
            if post_id == root_id or node.get("depth") is not None:
                continue
            parent_id = node.get("parent_id")
            if parent_id and parent_id in nodes and nodes[parent_id].get("depth") is not None:
                node["depth"] = nodes[parent_id]["depth"] + 1
                changed = True
        if not changed:
            break

def build_context_paths(
    nodes: dict[str, dict[str, Any]],
    root_id: str,
) -> None:
    """
    儲存實際 parent 關係形成的 context path。

    不人工補 root，
    避免在 parent 無法確認時產生不存在的路徑。
    """

    for post_id, node in nodes.items():
        path: list[str] = []
        seen: set[str] = set()
        current: str | None = post_id

        while (
            current
            and current in nodes
            and current not in seen
        ):
            seen.add(current)
            path.append(current)
            current = nodes[current].get(
                "parent_id"
            )

        path.reverse()

        node["context_path_ids"] = path

def filter_root_connected_nodes(
    nodes: dict[str, dict[str, Any]],
    root_id: str,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    將 nodes 分成兩類：

    1. graph_nodes:
       parent chain 可以一路追溯到 root_id。
       這些才會進入正式 tree.json / nodes.jsonl / visualizer。

    2. excluded_nodes:
       無法確認父節點、parent 不存在、或出現 cycle。
       這些保留成 excluded_nodes.jsonl，不進正式圖。

    意思是： 每一個nodes都往 parent 一路追蹤 最後追蹤到root 保留 中途斷掉 /parent不存在 /cycle  排除出正式圖graph 令存一個exluded_nodes.jsonl中
    """

    def trace_to_root(
        post_id: str,
    ) -> tuple[bool, str, list[str]]:
        """
        回傳：
        keep:
            是否可一路追到 root

        reason:
            root_reachable / missing_parent / parent_missing / cycle / root_missing

        path:
            從目前 post 往上追的 partial path
        """

        if root_id not in nodes:
            return (
                False,
                "root_missing",
                [],
            )

        current: str | None = post_id
        seen: set[str] = set()
        path: list[str] = []

        while current:

            if current in seen:
                return (
                    False,
                    "cycle",
                    path,
                )

            if current not in nodes:
                return (
                    False,
                    "parent_missing",
                    path + [current],
                )

            seen.add(current)
            path.append(current)

            if current == root_id:
                return (
                    True,
                    "root_reachable",
                    path,
                )

            parent_id = nodes[current].get(
                "parent_id"
            )

            if not parent_id:
                return (
                    False,
                    "missing_parent",
                    path,
                )

            current = parent_id

        return (
            False,
            "missing_parent",
            path,
        )


    keep_ids: set[str] = set()
    excluded_nodes: list[dict[str, Any]] = []

    for post_id, node in nodes.items():

        keep, reason, path = trace_to_root(
            post_id
        )

        if keep:
            keep_ids.add(post_id)
            continue

        excluded = dict(node)
        excluded["exclusion_reason"] = reason
        excluded["partial_context_path_ids"] = path

        excluded_nodes.append(excluded)


    graph_nodes = {
        post_id: dict(nodes[post_id])
        for post_id in keep_ids
    }

    return graph_nodes, excluded_nodes

def compact_node(node: dict[str, Any]) -> dict[str, Any]:
    """Return the stable public schema used by tree.json and nodes.jsonl."""
    ordered_keys = [
        "post_id",
        "root_post_id",
        "parent_id",
        "relation_to_parent",
        "node_type",
        "depth",
        "username",
        "author_hash",
        "text_raw",
        "text_clean",
        "topic_tags",
        "datetime",
        "engagement_snapshot",
        "permalink",
        "thread_part",
        "thread_total",

        "images",
        "videos",

        "has_image",
        "image_count",
        "has_video",
        "video_count",

        "embedded_posts",
        "external_links",
        "context_path_ids",
    ]
    return {key: node.get(key) for key in ordered_keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect a public Threads post, visible replies and an inferred conversation tree."
    )
    parser.add_argument("url", help="單篇 Threads 串文網址")
    parser.add_argument("--max-nodes", type=int, default=500, help="最多處理幾個節點，預設 500")
    parser.add_argument("--max-expand-rounds", type=int, default=18, help="每頁最多展開回覆輪數")
    parser.add_argument("--delay", type=float, default=2.0, help="開啟不同節點間的等待秒數")
    parser.add_argument("--headless", action="store_true", help="隱藏瀏覽器視窗")
    parser.add_argument("--debug", action="store_true", help="保存每個節點頁面的 HTML/TXT/PNG")
    args = parser.parse_args()

    requested_url = normalize_post_url(args.url)

    requested_author, requested_post_id = parse_post_url(
        requested_url
    )

    # 相容目前 Collector 既有流程：
    # crawl 階段仍先從使用者指定貼文開始。
    root_author = requested_author
    root_id = requested_post_id

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT_DIR / requested_post_id
    run_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = run_dir / "pages" if args.debug else None

    nodes: dict[str, dict[str, Any]] = {}
    page_cache: dict[str, dict[str, Any]] = {}
    parent_proposals: dict[str, dict[str, Any]] = {}
    crawl_debug: dict[str, dict[str, Any]] = {}
    discovery_queue: deque[str] = deque([requested_url])
    queued_ids: set[str] = {root_id}
    processed_ids: set[str] = set()
    warnings: list[str] = []
    total_reply_clicks = 0

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=args.headless,
            viewport={"width": 1365, "height": 1000},
        )
        page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(10_000)

        while discovery_queue and len(processed_ids) < args.max_nodes:
            current_url = discovery_queue.popleft()
            try:
                _, current_id = parse_post_url(current_url)
            except ValueError:
                continue
            if current_id in processed_ids:
                continue

            print(f"[{len(processed_ids) + 1}/{args.max_nodes}] 正在處理：{current_id}")

            try:
                detail = collect_detail_page(
                    page,
                    current_url,
                    args.max_expand_rounds,
                    debug_dir,
                )
            except Exception as exc:
                if current_id == root_id:
                    raise CollectionError(
                        f"指定串文詳細頁擷取失敗：{exc}"
                    ) from exc

                warnings.append(
                    f"略過詳細頁 {current_id}：{exc}"
                )
                processed_ids.add(current_id)
                continue

            page_cache[current_id] = detail
            processed_ids.add(current_id)
            total_reply_clicks += int(detail.get("more_reply_clicks") or 0)

            cards = [
                item
                for item in detail["cards"]
                if not item.get("nested_candidate")
            ]
            ordered_ids = [item["post_id"] for item in cards]

            canonical = (
                detail.get("canonical_url")
                or detail.get("final_url")
                or ""
            )

            requested_card_exists = any(
                card.get("post_id") == root_id
                for card in cards
            )

            if (
                current_id == root_id
                and root_id not in str(canonical)
                and not requested_card_exists
            ):
                context.close()
                raise CollectionError(
                    "指定串文未成功開啟，頁面可能重新導向、"
                    "已刪除、非公開或登入狀態失效。\n"
                    f"輸入：{requested_url}\n"
                    f"最終：{detail.get('final_url')}\n"
                    f"Canonical：{canonical}"
                )

            for candidate in cards:
                candidate_id = candidate["post_id"]

                debug_entry = crawl_debug.setdefault(
                    candidate_id,
                    {
                        "seen_on_pages": [],
                        "rect_samples": [],
                        "nested_candidate_seen": False,
                    },
                )
                if current_id not in debug_entry["seen_on_pages"]:
                    debug_entry["seen_on_pages"].append(current_id)
                if candidate.get("rect"):
                    debug_entry["rect_samples"].append(candidate["rect"])
                debug_entry["nested_candidate_seen"] = bool(
                    debug_entry["nested_candidate_seen"] or candidate.get("nested_candidate")
                )

                clean_candidate = dict(candidate)
                clean_candidate.pop("rect", None)
                clean_candidate.pop("nested_candidate", None)
                clean_candidate["root_post_id"] = root_id

                # ============================================================
                # 保存這一個 node 自己的圖片與影片 poster
                # ============================================================
                save_node_media(
                    context=context,
                    candidate=clean_candidate,
                    run_dir=run_dir,
                )

                # Media 下載完成後再 merge，
                # 這樣 local_path 才會一起進入 nodes
                nodes[candidate_id] = merge_node(nodes.get(candidate_id), clean_candidate)

                if (
                    candidate_id not in processed_ids
                    and candidate_id not in queued_ids
                    and len(queued_ids) < args.max_nodes
                ):
                    discovery_queue.append(candidate["permalink"])
                    queued_ids.add(candidate_id)

            # ---------------------------------------------------------
            # 每一個詳細頁都嘗試判斷目前節點的 parent。
            #
            # 注意：
            # 使用者輸入的 requested post 不一定是真正 conversation root，
            # 所以 requested post 本身也必須接受 parent inference。
            # ---------------------------------------------------------
            if current_id in ordered_ids:
                target_index = ordered_ids.index(current_id)
                preceding = [item for item in ordered_ids[:target_index] if item != current_id]
                if preceding:
                    parent_proposals[current_id] = {
                        "parent_id": preceding[-1],
                        "detection_method": "detail_context_predecessor",
                        "relation_status": (
                            "single_context_predecessor"
                            if len(preceding) == 1
                            else "multiple_context_predecessors"
                        ),
                        "visible_ancestor_chain": preceding,
                    }
                else:
                    parent_proposals[current_id] = {
                        "parent_id": None,
                        "detection_method": "not_visible_in_detail_context",
                        "relation_status": "not_visible_in_detail_context",
                        "visible_ancestor_chain": [],
                    }

            time.sleep(max(args.delay, 0.0))

        context.close()

    if requested_post_id not in nodes:
        # Root metadata fallback. This should be rare after successful page verification.
        root_page = page_cache.get(root_id, {})
        description = root_page.get("og_description") or ""
        nodes[requested_post_id] = {
            "post_id": requested_post_id,
            "permalink": requested_url,
            "username": root_author,
            "author_hash": author_hash(root_author),
            "text_raw": description,
            "text_clean": description,
            "datetime": None,
            "images": ([{"src": root_page.get("og_image"), "alt": None}] if root_page.get("og_image") else []),
            "videos": [],
            "external_links": [],
            "thread_part": None,
            "thread_total": None,
            "root_post_id": requested_post_id,
        }
        warnings.append("原始貼文卡片未由 DOM 取得，改用 Open Graph metadata。")

    # ============================================================
    # 決定真正的 conversation root
    # ============================================================

    conversation_root_id = resolve_conversation_root(
        requested_post_id=requested_post_id,
        nodes=nodes,
        parent_proposals=parent_proposals,
    )

    conversation_root_author = (
            nodes.get(conversation_root_id, {}).get("username")
            or requested_author
    )

    # 所有 node 的 root_post_id
    # 統一指向真正 conversation root。
    for node in nodes.values():
        node["root_post_id"] = conversation_root_id

    # root_author = nodes[root_id].get("username") or root_author

    relation_audit_by_child: dict[str, dict[str, Any]] = {}

    for post_id, node in nodes.items():
        node["node_type"] = classify_node(node, conversation_root_id)
        if post_id == conversation_root_id:
            node["parent_id"] = None
            node["relation_to_parent"] = None
        else:
            proposal = parent_proposals.get(post_id) or {
                "parent_id": None,
                "detection_method": "unresolved",
                "relation_status": "unresolved",
                "visible_ancestor_chain": [],
            }
            node["parent_id"] = proposal.get("parent_id")
            node["relation_to_parent"] = "reply" if node.get("parent_id") else None
            relation_audit_by_child[post_id] = {
                "child_id": post_id,
                "parent_id": proposal.get("parent_id"),
                "detection_method": proposal.get("detection_method"),
                "relation_status": proposal.get("relation_status"),
                "visible_ancestor_chain": proposal.get("visible_ancestor_chain", []),
            }

        if (
                node["node_type"]
                == "thread_continuation"
                and node.get("thread_part")
        ):

            expected_previous_part = (
                    node["thread_part"] - 1
            )

            previous_candidates = [
                item
                for item in nodes.values()
                if (
                        item["post_id"] != post_id
                        and item.get("username")
                        == node.get("username")
                        and item.get("thread_total")
                        == node.get("thread_total")
                        and item.get("thread_part")
                        == expected_previous_part
                )
            ]

            if previous_candidates:

                # 優先使用 DOM 原本推定的 parent，
                # 前提是它正好也是上一段續串。
                proposed_parent_id = node.get(
                    "parent_id"
                )

                proposed_previous = next(
                    (
                        item
                        for item
                        in previous_candidates
                        if item["post_id"]
                           == proposed_parent_id
                    ),
                    None,
                )

                if proposed_previous:
                    previous = proposed_previous

                else:
                    # DOM 沒直接指出時，
                    # 再以時間選擇最近的上一段。
                    previous = sorted(
                        previous_candidates,
                        key=lambda item: (
                                item.get("datetime") or ""
                        ),
                    )[-1]

                node["parent_id"] = (
                    previous["post_id"]
                )

                node["relation_to_parent"] = (
                    "thread_continuation"
                )

                relation_audit_by_child[
                    post_id
                ] = {
                    "child_id": post_id,
                    "parent_id": previous["post_id"],
                    "detection_method": (
                        "thread_part_sequence"
                    ),
                    "relation_status": (
                        "numbered_thread_sequence"
                    ),
                    "visible_ancestor_chain": [
                        previous["post_id"]
                    ],
                }
    # ---------------------------------------------------------
    # 原始貼文的頁面層級瀏覽數
    # Threads 的瀏覽數通常顯示在詳細頁標題區，
    # 而不是原始貼文 card 內，因此最後再補進 root node。
    # ---------------------------------------------------------
    # # 原始貼文的頁面層級瀏覽數
    root_page = (
            page_cache.get(conversation_root_id)
            or {}
    )

    root_view_raw = root_page.get(
        "page_view_raw"
    )

    if (
            conversation_root_id in nodes
            and root_view_raw
    ):
        engagement = dict(
            nodes[
                conversation_root_id
            ].get(
                "engagement_snapshot"
            )
            or {}
        )

        raw = dict(
            engagement.get("raw") or {}
        )

        engagement[
            "displayed_view_count"
        ] = parse_display_count(
            root_view_raw
        )

        raw["view"] = root_view_raw
        engagement["raw"] = raw

        nodes[
            conversation_root_id
        ][
            "engagement_snapshot"
        ] = engagement

    # 加入圖片 / 影片統計欄位
    for node in nodes.values():
        add_media_features(node)


    # 保存 raw discovered nodes 數量。
    #
    # 這代表 Collector 曾經發現過的所有 node，
    # 但不代表都能進入正式 graph。
    raw_discovered_node_count = len(nodes)


    # 先保存一份 raw nodes。
    #
    # 目的：
    #   1. 不丟掉已發現但未能接回 root 的資料。
    #   2. 日後若調高 max-nodes 或改 parent inference，
    #      可以重新檢查。
    raw_nodes_path = run_dir / f"{requested_post_id}_raw_nodes.jsonl"

    with raw_nodes_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for node in sorted(
            nodes.values(),
            key=lambda item: (
                item.get("datetime") or "",
                item.get("post_id") or "",
            ),
        ):
            file.write(
                json.dumps(
                    compact_node(node),
                    ensure_ascii=False,
                )
                + "\n"
            )


    # 正式 graph 只保留：
    #
    #   可以從 node → parent → ... → root
    #   一路追溯回 conversation_root_id 的節點。
    #
    # 這一步是為了保證 tree.json / nodes.jsonl / visualizer
    # 不會出現 isolation node。
    graph_nodes, excluded_nodes = filter_root_connected_nodes(
        nodes,
        conversation_root_id,
    )

    excluded_processed_count = sum(
        1
        for node in excluded_nodes
        if node.get("post_id") in processed_ids
    )
    excluded_unprocessed_count = (
        len(excluded_nodes) - excluded_processed_count
    )


    # 從這一行開始，正式輸出的 nodes 改成乾淨的 graph_nodes。
    nodes = graph_nodes


    # 被排除的 node 另外保存。
    #
    # 這些 node 有被 discover，
    # 但不進正式 graph。
    excluded_nodes_path = (
        run_dir /
        f"{requested_post_id}_excluded_nodes.jsonl"
    )

    with excluded_nodes_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for node in sorted(
            excluded_nodes,
            key=lambda item: (
                item.get("exclusion_reason") or "",
                item.get("datetime") or "",
                item.get("post_id") or "",
            ),
        ):
            file.write(
                json.dumps(
                    node,
                    ensure_ascii=False,
                )
                + "\n"
            )


    if excluded_nodes:
        if excluded_unprocessed_count:
            warnings.append(
                f"因詳細頁處理上限，尚有 {excluded_unprocessed_count} 個"
                "已發現候選留言未逐頁確認父子關係；"
                f"詳見 {excluded_nodes_path.name}"
            )
        if excluded_processed_count:
            warnings.append(
                f"已從正式 graph 排除 {excluded_processed_count} 個"
                "處理後仍無法一路連回 root 的節點；"
                f"詳見 {excluded_nodes_path.name}"
            )


    # 對乾淨 graph 重新計算 depth 與 context path。
    #
    # 注意：
    #   這裡一定要在 filter_root_connected_nodes() 之後做。
    calculate_depths(
        nodes,
        conversation_root_id,
    )

    build_context_paths(
        nodes,
        conversation_root_id,
    )


    # ============================================================
    # Edges 只從乾淨 graph nodes 產生。
    #
    # 因為 nodes 已經被 root-connected filter 過，
    # 所以 visualizer 不會再畫出孤立點。
    # ============================================================
    edges = [
        {
            "parent_id": node["parent_id"],
            "child_id": node["post_id"],
            "relation": node.get("relation_to_parent") or "reply",
        }
        for node in nodes.values()
        if node.get("parent_id")
    ]


    # ============================================================
    # 正式 graph 中理論上不應該再有 unresolved。
    # 若還有，代表 filter 或 parent chain 邏輯有問題。
    # ============================================================
    unresolved = [
        node["post_id"]
        for node in nodes.values()
        if (
            node["post_id"] != conversation_root_id
            and not node.get("parent_id")
        )
    ]

    if unresolved:
        warnings.append(
            "正式 graph 仍有無法確認父節點的節點，請檢查 filter 邏輯："
            + ", ".join(unresolved[:20])
        )

    result = {
        "schema_version": "2.0",
        "field_guide": {
            "node_type": "root_post=原始貼文；thread_continuation=原作者續串；reply=一般回覆",
            "parent_id": "此節點直接回覆的上一層節點；原始貼文為 null",
            "relation_to_parent": "reply=回覆；thread_continuation=同作者續串",
            "depth": "原始貼文為 0，直接回覆為 1，依此類推",
            "context_path_ids": "從原始貼文到目前節點的 ID 路徑",
            "topic_tags": "Threads 主題標籤；只保存標籤文字，例如 ['貓咪日常']",
            "engagement_snapshot": (
                "Threads 蒐集當下顯示的互動快照；"
                "view/like/reply/repost/share 可能隨時間改變或未顯示"
            ),
            "embedded_posts": "貼文內嵌／引用的 Threads 貼文；不加入 reply 邊，只作為目前節點的附加內容",
            "collection_complete": "false 代表只能確認已載入內容，不能宣稱取得全部回覆",
            "raw_discovered_node_count": (
                "Collector 曾經發現過的所有節點數；"
                "可能包含無法確認完整父節點鏈的節點"
            ),

            "excluded_unresolved_count": (
                "已發現但無法一路連回 root，"
                "因此被排除於正式 graph 外的節點數"
            ),

            "formal_graph_policy": (
                "正式 tree.json / nodes.jsonl 僅保留可由 parent chain "
                "一路追溯至 conversation_root_id 的節點，"
                "避免 visualizer 與圖模型中出現 isolation node"
            ),
            "images": (
                "此節點包含的圖片；src 為原始來源，"
                "local_path 為成功下載後的本機相對路徑"
            ),

            "videos": (
                "此節點包含的影片資訊；目前僅保存影片 poster，"
                "poster_local_path 為本機封面路徑"
            ),

            "has_image": (
                "1=此節點偵測到至少一張圖片；0=沒有圖片"
            ),

            "image_count": (
                "此節點偵測到的圖片數量"
            ),

            "has_video": (
                "1=此節點偵測到至少一個影片；0=沒有影片"
            ),

            "video_count": (
                "此節點偵測到的影片數量"
            ),
        },
        "platform": "threads",
        "requested_url": requested_url,

        # 使用者真正輸入的貼文
        "requested_post_id": requested_post_id,

        # Collector 判斷的 conversation 最上層節點
        "conversation_root_id": conversation_root_id,

        # 保留既有欄位名稱，
        # 但現在語意正式代表真正 conversation root
        "root_post_id": conversation_root_id,

        "requested_post_role": (
            nodes.get(
                requested_post_id,
                {},
            ).get(
                "node_type",
                "unknown",
            )
        ),
        "collected_at": utc_now_iso(),
        "collection_method": "playwright_dom_detail_context",
        "collection_complete": False,
        "stats": {
            # 正式 graph 的 node 數。
            # visualizer / GNN / context model 會使用這批。
            "node_count": len(nodes),

            "edge_count": len(edges),

            # Collector 曾經 discover 到的所有 node。
            # 可能大於 node_count。
            "raw_discovered_node_count": raw_discovered_node_count,

            # 被排除於正式 graph 外的 node 數。
            "excluded_unresolved_count": len(excluded_nodes),
            "excluded_processed_count": excluded_processed_count,
            "excluded_unprocessed_count": excluded_unprocessed_count,

            # 正式 graph 中仍然 unresolved 的數量。
            # 理想情況應該是 0。
            "unresolved_parent_count": len(unresolved),

            "processed_detail_pages": len(processed_ids),
            "requested_max_nodes": args.max_nodes,
            "total_more_reply_clicks": total_reply_clicks,

            # 修正原本容易誤判的 stopped_by_max_nodes。
            "stopped_by_max_nodes": (
                len(processed_ids) >= args.max_nodes
            ),
        },
        "nodes": [
            compact_node(item)
            for item in sorted(
                nodes.values(),
                key=lambda item: (
                    item.get("depth") is None,
                    item.get("depth") if item.get("depth") is not None else 999,
                    item.get("datetime") or "",
                    item["post_id"],
                ),
            )
        ],
        "edges": edges,
        "warnings": warnings,
        "limitations": [
            "只蒐集目前登入帳號可看見且成功載入的公開內容。",
            "collection_complete=false：無法保證 Threads 顯示了全部回覆。",
            "父節點主要由回覆詳細頁中最接近的可見上文推定；判斷依據另存於 audit JSON。",
            "多段作者串文、引用串文、被刪除內容及網頁改版可能需要人工覆核。",
            "請保留 raw HTML/TXT 作為日後重新解析與研究可重現性的依據。",
        ],
    }

    json_path = run_dir / f"{requested_post_id}_tree.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    jsonl_path = run_dir / f"{requested_post_id}_nodes.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for node in result["nodes"]:
            file.write(json.dumps(node, ensure_ascii=False) + "\n")

    audit_result = {
        "schema_version": "2.0",

        "requested_post_id": (
            requested_post_id
        ),

        "conversation_root_id": (
            conversation_root_id
        ),

        "root_post_id": (
            conversation_root_id
        ),

        "relation_audit": sorted(
            relation_audit_by_child.values(),
            key=lambda item: item["child_id"],
        ),

        "crawl_debug": crawl_debug,
        "warnings": warnings,
    }
    audit_path = run_dir / f"{requested_post_id}_audit.json"
    audit_path.write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n完成")
    print(f"正式 graph 節點：{result['stats']['node_count']}")
    print(f"正式 graph 邊：{result['stats']['edge_count']}")
    print(f"raw discovered 節點：{result['stats']['raw_discovered_node_count']}")
    print(f"排除節點：{result['stats']['excluded_unresolved_count']}")
    print(f"正式 graph 未確認父節點：{result['stats']['unresolved_parent_count']}")
    print(f"JSON：{json_path.resolve()}")
    print(f"JSONL：{jsonl_path.resolve()}")
    print(f"Raw nodes：{raw_nodes_path.resolve()}")
    print(f"Excluded nodes：{excluded_nodes_path.resolve()}")
    print(f"Audit：{audit_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, CollectionError) as exc:
        raise SystemExit(f"錯誤：{exc}")

# python threads_tree_collector_v2_updated.py "https://www.threads.com/@kk__ing19/post/DbecuR-H2R3" --max-nodes 200 --max-expand-rounds 25 --delay 2.5 --debug

"""

"""
