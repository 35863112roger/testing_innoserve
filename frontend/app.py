import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st

from frontend.risk_tree import build_risk_subtree, render_risk_tree
from backend.config import get_settings





API_URL = os.getenv("WORKBENCH_API_URL", "http://127.0.0.1:8000")
APP_SETTINGS = get_settings()

LABEL_NAMES = {
    "general": "一般",
    "friendly": "友善",
    "harassment": "騷擾／攻擊",
    "cyberbullying": "網路霸凌風險",
}
THREADS_POST_RE = re.compile(r"threads\.(?:com|net)/@[^/?#]+/post/([^/?#]+)", re.IGNORECASE)


def _show_live_progress(progress_bar, status_box, progress: dict) -> None:
    state = str(progress.get("state") or "idle")
    current = int(progress.get("current") or 0)
    total = int(progress.get("total") or 0)
    elapsed = int(progress.get("elapsed_seconds") or 0)
    idle = int(progress.get("idle_seconds") or 0)
    message = str(progress.get("message") or "正在準備……")

    if state in {"starting", "collecting"}:
        ratio = min(current / total, 0.94) if total else 0.0
        if idle >= 90 and current:
            message += f"（目前節點已載入 {idle} 秒，仍在等待頁面回應）"
    elif state in {"cache", "finalizing"}:
        ratio = 0.95
    elif state == "classifying":
        ratio = 0.97
    elif state == "summarizing":
        ratio = 0.99
    elif state == "rag_analyzing":
        ratio = 0.995
    elif state == "judgment_searching":
        ratio = 0.985
    elif state == "complete":
        ratio = 1.0
    else:
        ratio = min(current / total, 0.94) if total else 0.0

    progress_bar.progress(ratio)
    status_box.info(f"{message}｜已執行 {elapsed // 60:02d}:{elapsed % 60:02d}")

def _rag_source_icon(
    source: dict,
) -> str:
    """依網址提供輔助辨識圖示。"""
    url = str(
        source.get("url") or ""
    ).lower()

    if ".gov.tw" in url:
        return "🏛️"

    if ".edu.tw" in url:
        return "🎓"

    if "wikipedia.org" in url:
        return "📖"

    return "📚"

def _render_rag_result(
    rag: dict,
) -> None:
    """依照 RAG 狀態顯示報告及引用來源。"""
    status = str(
        rag.get("status") or "missing"
    )

    if status == "disabled":
        st.info(
            "RAG 功能目前未啟用。"
        )
        return

    if status == "skipped":
        st.info(
            rag.get("reason")
            or (
                "沒有騷擾／攻擊或"
                "網路霸凌風險節點，"
                "因此未執行 RAG。"
            )
        )
        return

    if status == "no_match":
        st.warning(
            rag.get("report")
            or rag.get("reason")
            or (
                "有風險節點，但沒有來源"
                "通過目前的檢索門檻。"
            )
        )
        return

    if status == "error":
        st.error(
            "RAG 分析失敗，但摘要與"
            "留言分類結果仍然有效。"
        )
        if rag.get("reason"):
            st.caption(
                str(rag["reason"])
            )
        return

    if status != "ok":
        st.warning(
            "FastAPI 沒有回傳可識別的 "
            f"RAG 狀態：{status}"
        )
        return

    sources = rag.get("sources") or []

    cited_sources = [
        source
        for source in sources
        if source.get("cited", True)
    ]

    metric_columns = st.columns(3)

    metric_columns[0].metric(
        "BERT 風險節點",
        int(
            rag.get("risk_node_count")
            or 0
        ),
    )

    metric_columns[1].metric(
        "送入 RAG 的節點",
        int(
            rag.get(
                "included_node_count"
            )
            or 0
        ),
    )

    metric_columns[2].metric(
        "實際引用來源",
        len(cited_sources),
    )

    omitted_count = int(
        rag.get("omitted_node_count")
        or 0
    )

    if omitted_count:
        st.warning(
            f"共有 {omitted_count} 個"
            "風險節點因 Prompt 長度限制"
            "未逐字送入 LLM；"
            "完整數量仍保留於分類統計。"
        )

    st.subheader(
        "整體風險資訊與處置建議"
    )

    report = str(
        rag.get("report") or ""
    ).strip()

    if report:
        formatted_report = report

        for heading in (
            "一、整體風險概況",
            "二、可能相關的定義與風險",
            "三、目前無法確認的事項",
            "四、建議處置方式",
        ):
            formatted_report = (
                formatted_report.replace(
                    heading,
                    f"#### {heading}",
                )
            )

        st.markdown(
            formatted_report
        )
    else:
        st.warning(
            "RAG 狀態為成功，"
            "但沒有報告內容。"
        )

    st.divider()
    st.subheader("引用來源")

    if not cited_sources:
        st.info(
            "這份報告沒有標記引用來源。"
        )
    else:
        for source in cited_sources:
            number = int(
                source.get("number")
                or 0
            )

            title = str(
                source.get("title")
                or "未命名來源"
            )

            publisher = str(
                source.get("source")
                or "未標示"
            )

            matched_label = str(
                source.get(
                    "matched_label"
                )
                or ""
            )

            label_name = LABEL_NAMES.get(
                matched_label,
                matched_label,
            )

            snippet = str(
                source.get("snippet")
                or ""
            ).strip()

            url = str(
                source.get("url")
                or ""
            ).strip()

            icon = _rag_source_icon(
                source
            )

            with st.expander(
                f"{icon} [來源 {number}] {title}",
                expanded=False,
            ):
                st.caption(
                    f"來源網站：{publisher}"
                    f"｜檢索類別：{label_name}"
                )

                if snippet:
                    st.write(snippet)

                if url.startswith(
                    (
                        "https://",
                        "http://",
                    )
                ):
                    st.link_button(
                        "🔗 開啟原始資料來源",
                        url,
                        key=(
                            "rag_source_"
                            f"{number}_"
                            f"{source.get('doc_id')}"
                        ),
                    )

    generation_model = str(
        rag.get("generation_model")
        or ""
    )

    if generation_model:
        st.caption(
            f"RAG 生成模型："
            f"{generation_model}"
        )

st.set_page_config(
    page_title="輿論放大鏡",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 輿論放大鏡")

st.subheader(
    "Threads 串文摘要、逐留言分類與風險資訊"
)

st.caption(
    "輸入單篇 Threads 網址後，系統會透過 MCP 擷取並還原"
    "討論樹，產生串文摘要，使用 context-aware BERT "
    "逐則分類留言，並透過 RAG 提供風險定義與處置資訊。"
    "若發現騷擾／攻擊或網路霸凌風險留言，系統也會查詢"
    "司法院公開裁判書，提供類似案件、判決主文與法院理由節錄。"
)

with st.expander("ℹ️ 系統功能說明"):
    st.markdown(
        """
        1. **討論串擷取與還原**  
        透過 MCP 與 Playwright 擷取 Threads 貼文及留言，
        並還原留言之間的父子關係。

        2. **串文摘要**  
        使用本機 TAIDE LoRA 模型整理原始貼文、主要討論方向
        與留言區整體氛圍。

        3. **逐留言風險分類**  
        使用 context-aware BERT，結合原始貼文、祖先留言與
        直接父留言，將每則留言分類為一般、友善、騷擾／攻擊
        或網路霸凌風險。

        4. **風險資訊與處置建議**  
        使用 ChromaDB、BM25 與 RRF 混合檢索知識庫，
        再由本機 TAIDE 模型整理相關定義與處置資訊。

        5. **類似案件裁判書**  
        系統依風險留言、BERT 標籤與串文摘要產生搜尋條件，
        透過 MCP 即時查詢司法院公開裁判書，呈現相關案件段落、
        判決主文、法院理由及原文連結。
        """
    )

    st.caption(
        "BERT 分類及裁判書相似度僅供輔助檢視，"
        "不代表事實認定、法律判決或專業法律意見。"
    )

url = st.text_input(
    "Threads 串文網址",
    placeholder="https://www.threads.com/@username/post/POST_ID",
)
with st.expander("進階設定"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        max_nodes = st.slider(
            "最多處理詳細頁",
            10,
            500,
            30,
            10,
            help=(
                "代表 Collector 最多開啟多少個 Threads 詳細頁，"
                "用來確認留言父子關係。"
                "最終輸出節點數不一定等於此數值；"
                "建議先以 30 測試。"
            ),
        )
    with col2:
        summary_length = st.select_slider(
            "摘要輸出長度",
            options=[
                "精簡",
                "標準",
                "較長",
            ],
            value="標準",
            help=(
                "控制摘要模型的最大輸出上限。"
                "實際長度仍會依討論內容而不同；"
                "數值較長會增加生成時間，"
                "不代表會納入更多原始留言。"
            ),
        )

        summary_token_limits = {
            "精簡": 160,
            "標準": 300,
            "較長": 420,
        }

        max_new_tokens = summary_token_limits[
            summary_length
        ]

        st.caption(
            f"目前輸出上限："
            f"{max_new_tokens} tokens"
        )
    with col3:
        classifier_batch_size = st.slider("分類 batch size", 1, 128, 32)
    with col4:
        judgment_limit = st.slider(
            "最多顯示類似判決",
            1,
            10,
            5,
            1,
            help=(
                "控制『類似案件判決書』分頁最多顯示幾筆；"
                "實際筆數可能因司法院查詢結果或相關度門檻而較少。"
            ),
        )
    force_refresh = st.checkbox("忽略快取並重新抓取")
    collector_timeout_seconds = min(
        max(
            APP_SETTINGS.threads_collector_base_timeout_seconds,
            (
                max_nodes
                * APP_SETTINGS.threads_collector_seconds_per_node
            ),
        ),
        APP_SETTINGS.threads_collector_max_timeout_seconds,
    )

    reserved_minutes = (
        collector_timeout_seconds + 59
    ) // 60

    if max_nodes > 50:
        timeout_capped = (
            max_nodes
            * APP_SETTINGS.threads_collector_seconds_per_node
            > APP_SETTINGS.threads_collector_max_timeout_seconds
        )
        calculation_note = (
            f"{max_nodes} 個詳細頁 × "
            f"{APP_SETTINGS.threads_collector_seconds_per_node} 秒"
        )
        if timeout_capped:
            max_timeout_minutes = (
                APP_SETTINGS.threads_collector_max_timeout_seconds + 59
            ) // 60
            calculation_note += (
                f"，已達 {max_timeout_minutes} 分鐘上限"
            )
        st.info(
            "大型串文會逐頁確認父子關係；"
            f"預計最多花費約 {reserved_minutes} 分鐘"
            f"（{calculation_note}）。"
            "實際時間依 Threads 載入速度而定。"
        )

if st.button("取得串文並分析", type="primary", disabled=not url.strip()):
    payload = {
        "url": url.strip(),
        "max_nodes": max_nodes,
        "force_refresh": force_refresh,
        "max_new_tokens": max_new_tokens,
        "classifier_batch_size": classifier_batch_size,
        "judgment_limit": judgment_limit,
    }
    try:
        # Streamlit HTTP 等待時間 = Collector 實際預算 + 15 分鐘。
        # 最低等待 30 分鐘，不另設 135 分鐘硬上限。
        request_timeout = max(
            1800,
            collector_timeout_seconds + 900,
        )

        post_match = THREADS_POST_RE.search(url.strip())
        post_id = post_match.group(1) if post_match else None
        progress_bar = st.progress(0.0)
        status_box = st.empty()
        status_box.info("正在送出分析請求……")

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                requests.post,
                f"{API_URL}/api/process-thread",
                json=payload,
                timeout=request_timeout,
            )
            while not future.done():
                if post_id:
                    try:
                        progress_response = requests.get(
                            f"{API_URL}/api/collection-progress/{post_id}",
                            timeout=2,
                        )
                        if progress_response.ok:
                            _show_live_progress(
                                progress_bar,
                                status_box,
                                progress_response.json(),
                            )
                    except requests.RequestException:
                        pass
                time.sleep(1)
            response = future.result()

            if not response.ok:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                raise RuntimeError(detail)
            result = response.json()
            progress_bar.progress(1.0)
            rag_status = str(
                result.get("rag", {}).get(
                    "status"
                )
                or ""
            )
            judgments = result.get("judgments") or {}
            judgment_status = str(judgments.get("status") or "missing")
            judgment_hit_count = len(judgments.get("hits") or [])

            if rag_status == "ok":
                rag_message = "RAG 風險資訊已完成"
            else:
                rag_message = "請查看 RAG 頁籤了解風險資訊狀態"

            if judgment_status == "ok":
                judgment_message = (
                    "類似案件判決搜尋已完成，"
                    f"共找到 {judgment_hit_count} 筆"
                )
            elif judgment_status == "no_match":
                judgment_message = (
                    "類似案件判決搜尋已完成，"
                    "本次未找到符合相關度門檻的案件"
                )
            elif judgment_status == "skipped":
                judgment_message = (
                    "本次沒有風險留言，已略過類似案件判決搜尋"
                )
            elif judgment_status == "error":
                judgment_message = (
                    "類似案件判決搜尋失敗，"
                    "請查看判決分頁或診斷資訊"
                )
            else:
                judgment_message = "類似案件判決搜尋未回傳可識別狀態"

            completion_message = (
                "擷取、分類與摘要已完成；"
                f"{rag_message}；{judgment_message}。"
            )
            if judgment_status == "error":
                status_box.warning(completion_message)
            else:
                status_box.success(completion_message)

        thread = result["thread"]
        judgment_summary = (
            f"找到 {judgment_hit_count} 筆類似案件判決"
            if judgment_status == "ok"
            else "類似案件判決搜尋完成但沒有符合結果"
            if judgment_status == "no_match"
            else "沒有風險留言，未執行類似案件判決搜尋"
            if judgment_status == "skipped"
            else "類似案件判決搜尋失敗"
            if judgment_status == "error"
            else "類似案件判決搜尋狀態不明"
        )
        st.success(
            f"完成：{thread['node_count']} 個節點，"
            f"分類 {thread['classified_count']} 則留言；"
            f"{judgment_summary}。"
        )

        summary_tab, comments_tab, rag_tab, judgments_tab, diagnostics_tab = st.tabs(
            ["串文摘要", "留言分類", "風險資訊與處置","類似案件判決書","資料與診斷"]
        )
        with summary_tab:
            st.subheader("摘要")
            st.write(result["summary"]["summary"])
            if result["summary"]["input_truncated"] or thread["node_limit_truncated"]:
                st.warning("討論串超過輸入上限，本次摘要只使用部分內容。")
            st.caption(
                f"摘要輸入 {result['summary']['input_tokens']} tokens；"
                f"生成 {result['summary']['generated_tokens']} tokens；"
                f"裝置 {result['summary']['device']}"
            )

        with comments_tab:
            metric_columns = st.columns(4)
            for column, label in zip(metric_columns, LABEL_NAMES):
                column.metric(LABEL_NAMES[label], result["statistics"].get(label, 0))

            risk_tree = build_risk_subtree(
                result["comments"],
                thread.get("root_post_id"),
                thread["source_url"],
                root_username=thread.get("root_username"),
                root_text=thread.get("root_text"),
                root_permalink=thread.get("root_permalink"),
                thread_nodes=result.get("thread_nodes"),
            )
            st.caption(
                "統計數量包含全部已分類留言；下方只顯示騷擾／攻擊與"
                "網路霸凌留言所在的分支，灰色節點是理解分支所需的上文。"
            )
            if risk_tree["risk_count"] == 0:
                st.info("本次分類沒有發現騷擾／攻擊或網路霸凌留言。")
            else:
                st.write(
                    f"風險節點 {risk_tree['risk_count']} 個；"
                    f"保留必要上文 {risk_tree['context_count']} 個。"
                )
                render_risk_tree(risk_tree)

        with rag_tab:
            _render_rag_result(
                result.get("rag") or {}
            )

        with judgments_tab:
            judgments = result.get("judgments") or {}
            status = judgments.get("status")

            if status == "skipped":
                st.info(
                    judgments.get("reason")
                    or "沒有風險留言，因此未搜尋判決。"
                )

            elif status == "error":
                st.error("類似案件判決搜尋失敗。")
                st.caption(str(judgments.get("reason") or ""))

            elif status == "no_match":
                st.warning(
                    judgments.get("reason")
                    or "沒有找到足夠相關的判決。"
                )

            elif status == "ok":
                st.info(
                    "系統根據代表性風險留言、BERT 分類結果與串文摘要，"
                    "產生行為及法律爭點搜尋詞，並即時查詢司法院公開"
                    "裁判書。下列結果僅代表文字與爭點可能相近，"
                    "不表示本串文符合相同犯罪構成要件或法律責任。"
                )

                for index, hit in enumerate(
                    judgments.get("hits") or [],
                    start=1,
                ):
                    with st.container(border=True):
                        st.subheader(
                            f"{index}. {hit.get('title') or '未命名案件'}"
                        )
                        st.write(
                            f"**法院：** {hit.get('court', '未提供')}"
                        )
                        st.write(
                            f"**裁判日期：** {hit.get('date', '未提供')}"
                        )

                        if hit.get("match_level"):
                            st.write(
                                f"**列出原因：** {hit['match_level']}"
                            )

                        st.write("**相關原文段落：**")
                        st.text(hit.get("preview") or "")

                        st.write("**判決結果：**")
                        st.text(hit.get("outcome") or "未擷取到主文")

                        with st.expander("法院理由節錄"):
                            st.text(
                                hit.get("reason_excerpt")
                                or "未擷取到理由"
                            )

                        if hit.get("source_url"):
                            st.link_button(
                                "查核司法院原文",
                                hit["source_url"],
                                key=f"judgment_{index}_{hit['case_id']}",
                            )

        with st.expander("判決搜尋診斷"):
            st.json({
                "status": judgments.get("status"),
                "reason": judgments.get("reason"),
                "selected_risk_node_count": judgments.get(
                    "selected_risk_node_count"
                ),
                "warnings": judgments.get("warnings") or [],
                "attempts": judgments.get("attempts") or [],
            })

        with diagnostics_tab:
            st.json(thread)
            
            rag = result.get("rag") or {}

            with st.expander(
                "RAG 執行診斷"
            ):
                st.json(
                    {
                        "status": rag.get(
                            "status"
                        ),
                        "reason": rag.get(
                            "reason"
                        ),
                        "generation_model": (
                            rag.get(
                                "generation_model"
                            )
                        ),
                        "risk_node_count": (
                            rag.get(
                                "risk_node_count"
                            )
                        ),
                        "included_node_count": (
                            rag.get(
                                "included_node_count"
                            )
                        ),
                        "omitted_node_count": (
                            rag.get(
                                "omitted_node_count"
                            )
                        ),
                        "queries": rag.get(
                            "queries"
                        )
                        or [],
                        "retrieved_source_count": (
                            len(
                                rag.get("sources")
                                or []
                            )
                        ),
                    }
                )

            if result["skipped"]:
                st.warning(f"有 {len(result['skipped'])} 則留言未分類。")
                st.json(result["skipped"])
            if thread["warnings"]:
                st.warning("Collector 回報警告")
                st.json(thread["warnings"])
    except (requests.RequestException, RuntimeError) as exc:
        st.error(f"處理失敗：{exc}")
