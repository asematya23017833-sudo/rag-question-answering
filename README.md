# RAG Question Answering Pipeline

PDF-based retrieval-augmented QA: 1,600-character chunks with 320-character overlap, `bge-small-en-v1.5` embeddings, FAISS + BM25 hybrid retrieval fused 0.6/0.4, `bge-reranker-base` reranking, and an evidence gate that answers only with citations when the reranker score clears the threshold.

> **Provenance:** Rebuilt in November 2026 after the original project files were lost. All results below were measured on this rebuild against the public dataset named here; they are not carried over from the lost original.

## Corpus
Three public arXiv papers (`data/pdfs/`): Attention Is All You Need (1706.03762), BERT (1810.04805), XGBoost (1603.02754). Substitute corpus; the original PDFs were lost.

## Verified behavior (this rebuild)
- 129 chunks indexed. Real questions answered with source citations, e.g. "What two tasks is BERT pre-trained on?" (rerank 0.998).
- Out-of-corpus question ("What is the capital of France?") correctly refused by the evidence gate (rerank 0.001 < 0.35 threshold).
- Answers are citation-based extracts; no generative LLM is called in this rebuild.

## Run
```bash
pip install -r requirements.txt
python src/rag.py    # indexes the PDFs, runs demo questions, writes results.json
```
First run downloads the two BGE models from Hugging Face (~1.2 GB).
