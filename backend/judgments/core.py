"""Corpus normalization, traceable retrieval and conservative source extraction."""
from __future__ import annotations

import json
import re
from datetime import datetime
from urllib.parse import quote, urlparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

"""
實際使用：
normalize()
chunks()
retrieve()
extract()
section()
normalized_date()
用途：
- 統一裁判資料格式。
- 將裁判全文切成段落。
- 使用 TF-IDF 計算留言與裁判段落的相關度。
- 擷取判決主文。
- 擷取法院理由。
- 處理民國年與西元日期。
"""


LABELS = ["一般騷擾", "性騷擾", "侮辱貶抑", "威脅恐嚇", "霸凌風險", "網路情境", "校園情境", "職場情境"]


def normalized_date(value):
    value = str(value or "").strip()
    digits = re.sub(r"\D", "", value)
    try:
        if len(digits) == 7:
            digits = str(int(digits[:3]) + 1911) + digits[3:]
        return datetime.strptime(digits, "%Y%m%d").date().isoformat()
    except ValueError:
        return "未提供"


def normalize(raw):
    if not isinstance(raw, dict):
        raise ValueError("每筆裁判書必須是 JSON 物件。")
    jid = str(raw.get("JID") or raw.get("id") or "").strip()
    full = raw.get("JFULLX") or {}
    text = raw.get("text") or raw.get("JFULL") or (full.get("JFULLCONTENT") if isinstance(full, dict) else "")
    if not jid or not isinstance(text, str) or not text.strip():
        raise ValueError("裁判書須有 JID/id 與文字全文；只有 PDF 連結的資料請先轉為文字。")
    if len(text) > 2_000_000:
        raise ValueError("單筆裁判全文超過 200 萬字，請分批處理。")
    court = raw.get("court")
    if not court:
        match = re.search(r"(?:臺灣|台灣|最高|智慧財產|福建|臺北|高雄|臺中)[^\n\r]{0,30}?法院(?:[臺台北中南高雄\s]{0,6}分院)?", text[:500])
        court = re.sub(r"\s+", "", match.group()) if match else "未提供"
    source = str(raw.get("source_url") or "")
    if not source and not raw.get("is_demo"):
        source = "https://judgment.judicial.gov.tw/FJUD/data.aspx?ty=JD&id=" + quote(jid, safe="")
    if source and (urlparse(source).scheme != "https" or urlparse(source).hostname != "judgment.judicial.gov.tw"):
        raise ValueError("來源連結必須使用司法院裁判書系統 HTTPS 網址。")
    return {"id": jid, "court": str(court), "date": normalized_date(raw.get("JDATE") or raw.get("date")),
            "title": str(raw.get("JTITLE") or raw.get("title") or "未提供案由"), "text": text.strip(),
            "source_url": source, "is_demo": bool(raw.get("is_demo", False))}


def parse_corpus(data: bytes):
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("單次匯入上限為 20 MB。")
    content = data.decode("utf-8-sig").strip()
    try:
        raw = json.loads(content)
        rows = raw if isinstance(raw, list) else [raw]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if len(rows) > 3000:
        raise ValueError("此原型單次上限 3000 筆，正式資料庫請改用持久化向量索引。")
    by_id = {}
    for row in rows:
        doc = normalize(row)
        by_id[doc["id"]] = doc
    return list(by_id.values())


def chunks(docs, size=800, overlap=160):
    result = []
    for doc in docs:
        for start in range(0, len(doc["text"]), size - overlap):
            result.append({"doc": doc, "start": start, "end": min(start + size, len(doc["text"])),
                           "text": doc["text"][start:start + size]})
    return result


def retrieve(query, docs, k=5, encoder=None):
    if not query.strip() or not docs:
        return []
    pieces = chunks(docs, size=320, overlap=64) if encoder is not None else chunks(docs)
    # Chinese character n-grams avoid depending on an unvalidated word segmenter.
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=100000, sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform([p["text"] for p in pieces])
    except ValueError:
        return []
    lexical = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
    scores = lexical.copy()
    if encoder is not None:
        vectors = encoder.encode([p["text"] for p in pieces], normalize_embeddings=True)
        q = encoder.encode([query], normalize_embeddings=True)[0]
        semantic = np.maximum(np.asarray(vectors) @ q, 0)
        scores = .45 * lexical + .55 * semantic
    best = {}
    for i in np.argsort(-scores):
        if scores[i] < (0.16 if encoder is not None else 0.015):
            continue
        p = pieces[int(i)]
        if p["doc"]["id"] not in best:
            best[p["doc"]["id"]] = {**p, "score": float(scores[i])}
        if len(best) >= k:
            break
    return list(best.values())


def section(text, heading, stops):
    pattern = r"(?:^|\n)[ \t　]*" + r"[ \t　]*".join(heading) + r"[ \t　]*\r?\n"
    found = re.search(pattern, text)
    if not found:
        return "未擷取到獨立段落，請查看全文。"
    rest = text[found.end():]
    positions = []
    for stop in stops:
        end = re.search(r"(?:^|\n)[ \t　]*" + r"[ \t　]*".join(stop) + r"[ \t　]*\r?\n", rest)
        if end:
            positions.append(end.start())
    rest = rest[:min(positions)] if positions else rest
    return rest.strip()[:1200] + ("…（段落截取）" if len(rest.strip()) > 1200 else "")


def extract(doc):
    return {"判決結果（主文原文）": section(doc["text"], "主文", ["事實", "理由", "事實及理由", "犯罪事實", "犯罪事實及理由"]),
            "法院理由（原文節錄）": section(doc["text"], "理由", ["中華民國", "附錄"])}


def rule_hints(text):
    from backend.judgments.issues import ISSUES
    rules = {issue["label"]: issue["cues"] for issue in ISSUES}
    rules.update({"網路情境": ["留言", "貼文", "私訊", "threads", "網路"],
             "校園情境": ["同學", "老師", "學校", "班上"], "職場情境": ["主管", "同事", "公司"]}
    )
    return [{"label": label, "evidence": [w for w in words if w in text.lower()]} for label, words in rules.items()
            if any(w in text.lower() for w in words)]
