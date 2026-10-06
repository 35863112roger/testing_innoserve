# 🔍 輿論透視鏡

本系統整合 Threads 串文擷取、TAIDE LoRA 摘要、context-aware BERT
逐留言分類、RAG 風險資訊與類似案件裁判書查詢。這個 UI 專門處理
單篇 Threads 串文網址，不是一般文字摘要工具。

處理流程：

1. FastAPI 透過 MCP `collect_thread` 工具取得快取或即時抓取的 `tree.json`。
2. 將整棵留言樹轉成摘要模型訓練時使用的 `[root_post]`／`[reply]` 縮排格式。
3. 使用 TAIDE-LX-7B + `taide-lx-7b-short` LoRA 產生 3～6 句短摘要。
4. 將每則 reply 轉成 `[ROOT]`／`[ANC]`／`[PARENT]` context 與 target text pair。
5. BERT 對每則留言分別預測 `general`、`friendly`、`harassment`、`cyberbullying`。
6. 從 BERT 結果擷取 harassment 與 cyberbullying 風險節點。
7. 透過 MCP `search_similar_judgments` 工具，以代表留言及
   串文摘要查詢司法院公開裁判書。
8. 使用中文字元 TF-IDF 重新排序候選案件，擷取相關段落、
   判決主文及法院理由。
9. UI 顯示摘要、留言風險樹、RAG 資訊與類似案件裁判書。


## 模型與來源專案

模型權重沒有複製，`model_artifacts/` 使用符號連結：

- `summary-base`：`chengxi0618/taidelx7bchattv1` 本機快取。
- `summary-adapter`：`summary_training_package/adapters/taide-lx-7b-short`。
- `bert-context`：`threads_tree/.../ckip_baseline_v2_seed42/best_model`。

Threads 快取與 Collector 路徑設定在 `.env`：

- `THREADS_TREE_DATA_DIR`：本系統獨立的 `data/threads_tree` 根目錄。
- `THREADS_COLLECTOR_SCRIPT`：唯讀重用的原始 Playwright Collector 程式。
- `THREADS_COLLECTOR_PROFILE_DIR`：本系統獨立的 Playwright 登入狀態。

原始 Collector 由 `mcp_servers/threads_collector_runner.py` 載入；runner 會將
`OUTPUT_DIR` 與 `PROFILE_DIR` 改到 `testing_innoserve/data/`，因此新抓取的 JSON、
媒體與瀏覽器狀態不會寫入 `threads_tree/data/`。

## 安裝

進入專案並啟用環境：
* 須根據自己的電腦路徑做修正 RAEDME.md 的呈現僅供參考

```bash
cd /media/user/bad06345-7e07-45be-b246-b01172f6655a/11463137/testing_innoserve
source .venv/bin/activate
```

若尚未安裝專案套件：

```bash
python -m pip install -e '.[dev]'
```

摘要模型在 CUDA 上使用 bitsandbytes 4-bit NF4 量化；`bitsandbytes` 已列入
`pyproject.toml`，執行上述指令時會一併安裝。

## 啟動系統

若 Ollama 已經由 systemd 或其他方式在背景執行，完整系統需要三個持續開啟的
Terminal：FastAPI、Streamlit 與 ngrok。Processing Pipeline 和 MCP 工具由
FastAPI 程序內部執行，不需要額外啟動；`.env` 預設的
`MCP_IN_PROCESS=true` 也表示不需要獨立的 MCP Server。

### 0. 確認 Ollama

```bash
curl http://127.0.0.1:11434/api/tags
```

如果無法連線，可使用 systemd 啟動：

```bash
sudo systemctl start ollama
sudo systemctl status ollama --no-pager
```

若未安裝 Ollama systemd 服務，需另開一個 Terminal 執行：

```bash
ollama serve
```

此時總共會需要四個 Terminal。

### Terminal 1：FastAPI 與 Processing Pipeline

```bash
cd /media/user/bad06345-7e07-45be-b246-b01172f6655a/11463137/testing_innoserve
source .venv/bin/activate
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

正式展示時不使用 `--reload`，避免大型模型因程式重新載入而重複初始化。
啟動後可檢查：

```bash
curl http://127.0.0.1:8000/api/health
```

### Terminal 2：Streamlit 前端

```bash
cd /media/user/bad06345-7e07-45be-b246-b01172f6655a/11463137/testing_innoserve
source .venv/bin/activate
WORKBENCH_API_URL=http://127.0.0.1:8000 \
python -m streamlit run frontend/app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true
```

本機可開啟 `http://127.0.0.1:8501`。健康檢查指令：

```bash
curl http://127.0.0.1:8501/_stcore/health
```

### Terminal 3：ngrok 外部連線

本機已安裝 ngrok 且設定檔有效時，將 Streamlit 的 Port 交給 ngrok：

```bash
ngrok http 8501
```

ngrok 會顯示類似下列轉送資訊：

```text
Forwarding  https://example.ngrok-free.app -> http://localhost:8501
```

將 `https://...ngrok-free.app` 提供給外部電腦即可。ngrok 的 Port 必須與
Streamlit 的 `--server.port` 相同；本架構只公開 `8501`，不需要另外執行
`ngrok http 8000`。Streamlit 仍會在伺服器端透過 `127.0.0.1:8000` 呼叫
FastAPI，因此也不需要把 ngrok 網域加入 FastAPI 的 CORS 設定。

ngrok 免費隨機網址在重新啟動後可能改變，且任何取得網址的人都能操作系統。
展示結束後請在 ngrok Terminal 按 `Ctrl+C` 關閉對外連線；正式長期公開前應再
加入身分驗證、請求頻率限制與使用額度保護。

MCP 預設使用 in-process transport，因此不需要另外啟動 MCP Server；工具仍透過 MCP Client 的 `tools/list` 與 `tools/call` 執行。

如需獨立測試 Streamable HTTP MCP Server：

```bash
python -m uvicorn mcp_servers.internal_tools:app --host 127.0.0.1 --port 8001
```

並在 `.env` 設定：

```env
MCP_IN_PROCESS=false
MCP_SERVER_URL=http://127.0.0.1:8001/mcp
```

## RAG 模型

啟用 `RAG_ENABLED=true` 前，需先啟動 Ollama 並安裝嵌入與生成模型：

```bash
ollama pull nomic-embed-text
ollama pull hf.co/tetf/Llama-3.1-TAIDE-LX-8B-Chat-GGUF:Q4_K_M
ollama list
```

`nomic-embed-text` 用於 ChromaDB 向量檢索；TAIDE 8B 用於將檢索來源整理為
含來源編號的風險與處置報告。兩個模型名稱必須與 `.env` 的
`RAG_EMBED_MODEL` 及 `RAG_GENERATION_MODEL` 完全一致。

## API

### 健康檢查

```text
GET /api/health
```

### 處理 Threads 串文

```text
POST /api/process-thread
```

請求範例：

```json
{
  "url": "https://www.threads.com/@username/post/POST_ID",
  "max_nodes": 30,
  "force_refresh": false,
  "max_new_tokens": 300,
  "classifier_batch_size": 32,
  "judgment_limit": 5
}
```

`judgment_limit` 是請求參數，控制最多回傳的類似案件裁判書數量；可設定為
1～10，預設為 5，實際結果可能因相關度門檻而少於設定值。

回傳內容包含：

- `thread`：root、節點數、快取／即時抓取來源、完整度與警告。
- `summary`：摘要及 token／截斷資訊。
- `statistics`：四類留言數量。
- `comments`：每則留言的 ID、parent、深度、文字、標籤、信心與四類機率。
- `thread_nodes`：Collector 的完整精簡節點結構，不受留言是否成功分類影響，供 UI 還原父子鏈。
- `rag`：風險定義、處置建議、引用來源及 RAG 執行狀態。
- `judgments`：類似案件查詢狀態、相關裁判書、警告及查詢診斷。
- `skipped`：因缺文字、脈絡不完整或過長而跳過的留言。

## 風險留言分支樹

UI 的「留言分類」仍以完整 `statistics` 顯示四種分類數量，但樹狀圖只保留：

- 原始貼文（藍色）。
- 分類為 harassment 的留言（橘色）。
- 分類為 cyberbullying 的留言（紅色）。
- 從原始貼文通往上述風險留言所需的祖先留言（灰色）。

與任何風險留言沒有祖先關係的 general／friendly 節點不會進入樹狀圖。
因此視覺節點減少，但模型分類結果與四類統計數量不會被刪除或重新計算。
父鏈以 `thread_nodes.parent_id` 為準；即使直接父留言沒有風險標籤、不是 BERT
分類對象或被模型跳過，仍會作為灰色必要上文保留並畫出連線。
若風險留言的 `parent_id` 就是原始貼文，則直接畫出「原始貼文 → 風險留言」；
不會額外建立灰色占位節點。

## 即時抓取

URL 沒有既有快取時，MCP 工具會呼叫 `threads_tree` 原專案的 Playwright Collector，
但輸出與登入狀態使用本系統的獨立目錄：

```text
testing_innoserve/data/threads_tree
testing_innoserve/data/playwright_profile
```

若登入狀態失效，請在本專案執行下列命令；登入結果只會寫入本系統的獨立
profile，不會修改原始 `threads_tree` 的登入資料：

```bash
python scripts/playwright_login.py
```

即時抓取會逐頁確認留言父子關係。建議第一次先使用 30 個節點；預設至少保留
600 秒，並依每個節點 40 秒自動延長。目前 UI 最多可選 500 個節點，
對應 Collector 預算 20000 秒。Streamlit 的 HTTP 請求會在 Collector 預算之外
再保留 15 分鐘供資料整理與模型推論使用。

Collector 成功完成後，資料會寫入
`THREADS_TREE_DATA_DIR/<requested_post_id>/`，包含 `*_tree.json`、
`*_nodes.jsonl`、`*_raw_nodes.jsonl`、`*_excluded_nodes.jsonl`、`*_audit.json`
及下載的媒體。已有成功快取時，未勾選「忽略快取」會直接重用 `*_tree.json`。
目前原始 Collector 只在整次爬取完成後寫出結構化 JSON；若程序中途逾時，可能
只留下已下載的媒體，不能從中斷的節點自動續跑。

分析期間，Streamlit 會每秒查詢 `/api/collection-progress/<post_id>`，顯示目前
正在處理的詳細頁節點、節點上限、累計時間，以及整理、分類和摘要階段。若同一
節點超過 90 秒沒有進度更新，UI 會提示該節點載入較久；這不一定代表程序失效，
但可用來判斷 Threads 頁面是否可能卡住。

## GPU 狀態

- `MODEL_DEVICE=auto` 會在 `torch.cuda.is_available()` 為 `True` 時使用 CUDA。
- 可在啟動服務的同一個終端環境執行 `nvidia-smi` 及 PyTorch CUDA 測試確認。
- 若 CUDA 不可用，BERT 與摘要模型會回退至 CPU。
- 目前只有短摘要 LoRA，尚未提供長摘要模式。
