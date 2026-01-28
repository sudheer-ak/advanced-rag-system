import os, json
import numpy as np
import faiss
from typing import List, Tuple
from rank_bm25 import BM25Okapi

from rag.schemas import Chunk, RetrievalResult
from rag.embeddings import Embedder

def tokenize(s: str) -> List[str]:
    return [t for t in s.lower().split() if t.strip()]

class HybridIndex:
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.bm25 = None
        self.embedder = Embedder()
        self.faiss_index = None

    def build(self, chunks: List[Chunk]):
        self.chunks = chunks

        corpus_tokens = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

        vecs = self.embedder.encode([c.text for c in chunks])
        dim = vecs.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(vecs)

    def save(self, folder: str):
        os.makedirs(folder, exist_ok=True)

        with open(os.path.join(folder, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")

        faiss.write_index(self.faiss_index, os.path.join(folder, "dense.faiss"))

        # Store BM25 tokenized corpus (BM25Okapi not easily serializable)
        with open(os.path.join(folder, "bm25_corpus.json"), "w", encoding="utf-8") as f:
            json.dump([tokenize(c.text) for c in self.chunks], f)

    def load(self, folder: str):
        chunks = []
        with open(os.path.join(folder, "chunks.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(Chunk.model_validate_json(line))
        self.chunks = chunks

        self.faiss_index = faiss.read_index(os.path.join(folder, "dense.faiss"))

        with open(os.path.join(folder, "bm25_corpus.json"), "r", encoding="utf-8") as f:
            corpus_tokens = json.load(f)
        self.bm25 = BM25Okapi(corpus_tokens)

    def search_dense(self, query: str, k: int) -> List[Tuple[int, float]]:
        qv = self.embedder.encode([query])
        scores, idx = self.faiss_index.search(qv, k)
        pairs = [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]
        return pairs

    def search_bm25(self, query: str, k: int) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top_idx = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_idx]

    def search_hybrid(self, query: str, k: int = 6, dense_k: int = 20, bm25_k: int = 20) -> List[RetrievalResult]:
        dense = self.search_dense(query, dense_k)
        bm25 = self.search_bm25(query, bm25_k)

        # Reciprocal Rank Fusion (RRF)
        def rrf(rank: int, c: int = 60) -> float:
            return 1.0 / (c + rank)

        scores = {}
        for r, (i, _) in enumerate(dense, start=1):
            scores[i] = scores.get(i, 0.0) + rrf(r)
        for r, (i, _) in enumerate(bm25, start=1):
            scores[i] = scores.get(i, 0.0) + rrf(r)

        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for i, s in merged:
            results.append(
                RetrievalResult(chunk=self.chunks[i], score=float(s), method="hybrid")
            )
        return results

