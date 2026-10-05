from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import chromadb
import ollama
from rank_bm25 import BM25Okapi


def zh_tokenize(text: str) -> list[str]:
    """將中文拆成單字、雙字詞，英文與數字保留完整 token。"""
    if not isinstance(text, str):
        return []

    normalized = text.lower()
    english_and_numbers = re.findall(r"[a-z0-9]+", normalized)
    chinese_characters = re.findall(r"[\u4e00-\u9fff]", normalized)
    chinese_bigrams = [
        "".join(chinese_characters[index : index + 2])
        for index in range(len(chinese_characters) - 1)
    ]

    return english_and_numbers + chinese_characters + chinese_bigrams


def rrf_fuse(
    vector_hits: list[tuple[str, float]],
    bm25_hits: list[tuple[str, float]],
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """融合向量名次與 BM25 名次，不直接比較兩種不同尺度的分數。"""
    vector_ranks = {
        doc_id: rank
        for rank, (doc_id, _) in enumerate(vector_hits, start=1)
    }
    bm25_ranks = {
        doc_id: rank
        for rank, (doc_id, _) in enumerate(bm25_hits, start=1)
    }

    vector_distances = dict(vector_hits)
    bm25_scores = dict(bm25_hits)
    all_ids = set(vector_ranks) | set(bm25_ranks)

    fused: list[dict[str, Any]] = []

    for doc_id in all_ids:
        score = 0.0

        if doc_id in vector_ranks:
            score += 1.0 / (rrf_k + vector_ranks[doc_id])

        if doc_id in bm25_ranks:
            score += 1.0 / (rrf_k + bm25_ranks[doc_id])

        fused.append(
            {
                "doc_id": doc_id,
                "rrf_score": score,
                "vector_rank": vector_ranks.get(doc_id),
                "vector_distance": vector_distances.get(doc_id),
                "bm25_rank": bm25_ranks.get(doc_id),
                "bm25_score": bm25_scores.get(doc_id),
            }
        )

    return sorted(
        fused,
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


class ChromaBm25RrfSearcher:
    def __init__(
        self,
        chroma_path: Path,
        collection_name: str,
        embed_model: str,
        k_each: int = 30,
        rrf_k: int = 60,
    ) -> None:
        self.chroma_path = chroma_path.expanduser().resolve()
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.k_each = k_each
        self.rrf_k = rrf_k

        if not self.chroma_path.is_dir():
            raise FileNotFoundError(
                f"找不到 ChromaDB：{self.chroma_path}"
            )

        self.client = chromadb.PersistentClient(
            path=str(self.chroma_path)
        )
        self.collection = self.client.get_collection(
            name=self.collection_name
        )

        self._load_corpus()

    def _load_corpus(self) -> None:
        """從同一次 Chroma 查詢建立 BM25，確保文件與 ID 順序一致。"""
        all_data = self.collection.get(
            include=["documents", "metadatas"]
        )

        ids = all_data.get("ids") or []
        documents = all_data.get("documents") or []
        metadatas = all_data.get("metadatas") or []

        if not ids:
            raise RuntimeError("Chroma collection 沒有文件")

        if not (len(ids) == len(documents) == len(metadatas)):
            raise RuntimeError(
                "Chroma 的 ids、documents、metadatas 數量不一致"
            )

        self.document_ids = [str(item) for item in ids]
        self.documents = [
            str(document or "")
            for document in documents
        ]
        self.metadatas = [
            dict(metadata or {})
            for metadata in metadatas
        ]

        self.documents_by_id = dict(
            zip(self.document_ids, self.documents)
        )
        self.metadatas_by_id = dict(
            zip(self.document_ids, self.metadatas)
        )

        tokenized_corpus = [
            zh_tokenize(document)
            for document in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _embed_query(self, query: str) -> list[float]:
        response = ollama.embeddings(
            model=self.embed_model,
            prompt=query,
        )
        embedding = response["embedding"]

        if not embedding:
            raise RuntimeError("Ollama 沒有回傳 embedding")

        return [float(value) for value in embedding]

    def search(
        self,
        query: str,
        top_k: int = 6,
	require_both: bool = True,
    ) -> list[dict[str, Any]]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("RAG query 不可為空白")

        candidate_count = min(
            self.k_each,
            len(self.document_ids),
        )

        query_embedding = self._embed_query(normalized_query)

        vector_result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_count,
            include=["distances"],
        )

        vector_ids = (vector_result.get("ids") or [[]])[0]
        vector_distances = (
            vector_result.get("distances") or [[]]
        )[0]

        vector_hits = sorted(
            [
                (str(doc_id), float(distance))
                for doc_id, distance in zip(
                    vector_ids,
                    vector_distances,
                )
            ],
            key=lambda item: item[1],
        )

        query_tokens = zh_tokenize(normalized_query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        # BM25 分數為 0 表示完全沒有文字重疊，不加入候選。
        bm25_hits = sorted(
            [
                (doc_id, float(score))
                for doc_id, score in zip(
                    self.document_ids,
                    bm25_scores,
                )
                if float(score) > 0
            ],
            key=lambda item: item[1],
            reverse=True,
        )[:candidate_count]

        fused_hits = rrf_fuse(
            vector_hits,
            bm25_hits,
            rrf_k=self.rrf_k,
        )

        #Chroma 命中＋BM25 命中 → 保留
        #只有 Chroma 命中       → 暫不採用
        #只有 BM25 命中         → 暫不採用
        if require_both:
            fused_hits = [
                hit
                for hit in fused_hits
                if (
                    hit["vector_rank"] is not None
                    and hit["bm25_rank"] is not None
                )
            ]

        results: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        for hit in fused_hits:
            doc_id = hit["doc_id"]
            document = self.documents_by_id.get(doc_id, "")
            metadata = self.metadatas_by_id.get(doc_id, {})

            source_url = str(metadata.get("url") or "")
            deduplication_key = source_url or doc_id

            # 同一篇文章只保留一個 chunk，增加來源多樣性。
            if deduplication_key in seen_urls:
                continue

            seen_urls.add(deduplication_key)

            results.append(
                {
                    **hit,
                    "title": str(
                        metadata.get("title")
                        or "未知標題"
                    ),
                    "url": source_url,
                    "source": str(
                        metadata.get("source")
                        or ""
                    ),
                    "document": document,
                    "snippet": (
                        document[:240]
                        + ("..." if len(document) > 240 else "")
                    ),
                }
            )

            if len(results) >= top_k:
                break

        return results


def main() -> None:
    from backend.config import get_settings

    settings = get_settings()
    query = " ".join(sys.argv[1:]).strip()

    if not query:
        raise SystemExit(
            "用法：python -m backend.rag.hybrid_search <查詢文字>"
        )

    searcher = ChromaBm25RrfSearcher(
        chroma_path=settings.rag_data_dir / "chroma_db",
        collection_name=settings.rag_chroma_collection,
        embed_model=settings.rag_embed_model,
        k_each=settings.rag_k_each,
        rrf_k=settings.rag_rrf_k,
    )

    results = searcher.search(
        query,
        top_k=settings.rag_top_k,
    )

    print(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
