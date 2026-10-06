# Model artifacts

此目錄保存本機模型檔案，不將大型權重提交至 GitHub。

- `bert-context`：[`jgigivjry/threads-comment-risk-bert`](https://huggingface.co/jgigivjry/threads-comment-risk-bert)，BERT 四分類完整模型。
- `summary-adapter`：[`jgigivjry/threads-summary-taide`](https://huggingface.co/jgigivjry/threads-summary-taide)，TAIDE 摘要 LoRA adapter。
- `summary-base`：[`chengxi0618/taidelx7bchattv1`](https://huggingface.co/chengxi0618/taidelx7bchattv1)，TAIDE 摘要基礎模型。

在專案根目錄安裝相依套件後，可執行：

```bash
.venv/bin/hf download jgigivjry/threads-comment-risk-bert \
  --local-dir model_artifacts/bert-context

.venv/bin/hf download jgigivjry/threads-summary-taide \
  --local-dir model_artifacts/summary-adapter

.venv/bin/hf download chengxi0618/taidelx7bchattv1 \
  --local-dir model_artifacts/summary-base
```
