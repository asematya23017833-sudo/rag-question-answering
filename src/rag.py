"""RAG Question Answering Pipeline - rebuild (Nov 2026) of the lost original.

PDF-based retrieval pipeline:
- chunking: 1,600 characters with 320-character overlap
- embeddings: BAAI/bge-small-en-v1.5
- hybrid retrieval: FAISS (dense) + BM25 (sparse), fused 0.6 / 0.4
- reranking: BAAI/bge-reranker-base
- evidence gate: answers only when the reranker score clears a threshold,
  with citations to the source chunks; otherwise says the evidence is insufficient.
Corpus in this rebuild: three public arXiv papers (Attention Is All You Need,
BERT, XGBoost) - a substitute corpus, the original PDFs were lost.
"""
import json
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdfs"
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 320
DENSE_W, SPARSE_W = 0.6, 0.4
TOP_K_RETRIEVE = 20
RERANK_CANDIDATES = 10
TOP_K_RERANK = 3
EVIDENCE_THRESHOLD = 0.35  # calibrated on this corpus; see results.json

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"


def load_chunks():
    chunks = []
    for pdf in sorted(PDF_DIR.glob("*")):
        reader = PdfReader(str(pdf))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for i, start in enumerate(range(0, max(len(text) - CHUNK_OVERLAP, 1), step)):
            piece = text[start:start + CHUNK_SIZE].strip()
            if piece:
                chunks.append({"source": pdf.name, "chunk_id": f"{pdf.name}#{i}", "text": piece})
    return chunks


class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.chunks = load_chunks()
        embs = self.embedder.encode(
            ["Represent this sentence for searching relevant passages: " + c["text"] for c in self.chunks],
            normalize_embeddings=True, show_progress_bar=False,
        )
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(np.asarray(embs, dtype="float32"))
        self.bm25 = BM25Okapi([c["text"].lower().split() for c in self.chunks])

    def retrieve(self, query):
        q = self.embedder.encode(["Represent this sentence for searching relevant passages: " + query],
                                 normalize_embeddings=True, show_progress_bar=False)
        dense = self.index.search(np.asarray(q, dtype="float32"), TOP_K_RETRIEVE)[1][0]
        dense_scores = self.index.search(np.asarray(q, dtype="float32"), TOP_K_RETRIEVE)[0][0]
        sparse_scores = self.bm25.get_scores(query.lower().split())
        top_sparse = np.argsort(sparse_scores)[::-1][:TOP_K_RETRIEVE]

        def norm(vals):
            vals = {int(k): float(v) for k, v in vals}
            lo, hi = min(vals.values(), default=0), max(vals.values(), default=1)
            return {k: (v - lo) / (hi - lo + 1e-9) for k, v in vals.items()}

        dn = norm(zip(dense, dense_scores))
        sn = norm((i, sparse_scores[i]) for i in top_sparse)
        fused = {}
        for k, v in dn.items():
            fused[k] = fused.get(k, 0) + DENSE_W * v
        for k, v in sn.items():
            fused[k] = fused.get(k, 0) + SPARSE_W * v
        return [self.chunks[i] for i in sorted(fused, key=fused.get, reverse=True)[:TOP_K_RETRIEVE]]

    def answer(self, query):
        candidates = self.retrieve(query)[:RERANK_CANDIDATES]
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs, batch_size=16)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:TOP_K_RERANK]
        best = float(ranked[0][0])
        if best < EVIDENCE_THRESHOLD:
            return {"question": query, "answered": False, "best_rerank_score": round(best, 3),
                    "message": "Insufficient evidence in the corpus to answer with citations.", "citations": []}
        citations = [{"source": c["source"], "chunk_id": c["chunk_id"],
                      "rerank_score": round(float(s), 3), "excerpt": c["text"][:400]} for s, c in ranked]
        return {"question": query, "answered": True, "best_rerank_score": round(best, 3), "citations": citations}


if __name__ == "__main__":
    pipe = RAGPipeline()
    print(f"chunks: {len(pipe.chunks)}")
    questions = [
        "How does the multi-head attention mechanism work in the Transformer?",
        "What two tasks is BERT pre-trained on?",
        "What makes XGBoost scale to billions of examples?",
        "What is the capital of France?",  # out-of-corpus: should hit the evidence gate
    ]
    out = [pipe.answer(q) for q in questions]
    (ROOT / "results.json").write_text(json.dumps(out, indent=2))
    for r in out:
        print(r["question"], "->", "ANSWERED" if r["answered"] else "GATED", r["best_rerank_score"])
