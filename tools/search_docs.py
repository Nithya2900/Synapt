"""
search_docs.py

Semantic search over unstructured annual report text files.
Uses sentence-transformers for embeddings and FAISS for vector search.

Tool contract (for LLM):
  Use this tool when the question asks about qualitative information, strategy,
  management commentary, reasons, explanations, or anything that would appear
  in a company's annual report narrative. Do NOT use for numbers, margins, or
  revenue figures — use query_data for those.

Input:  query (str) — natural language question
Output: list of dicts with keys: text, source, chunk_id, score
"""

import os
import json
import pickle
import re
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data" / "pdfs"
INDEX_PATH = Path(__file__).parent.parent / "data" / "faiss.index"
META_PATH = Path(__file__).parent.parent / "data" / "faiss_meta.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400          # characters per chunk
CHUNK_OVERLAP = 80        # overlap between chunks
TOP_K = 3


def _chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "source": source,
                "chunk_id": chunk_id,
            })
            chunk_id += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_index() -> None:
    """Read all .txt files in data/pdfs/, embed them, and save a FAISS index."""
    print("[search_docs] Building FAISS index...")
    model = SentenceTransformer(MODEL_NAME)

    all_chunks = []
    for txt_file in sorted(DATA_DIR.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        source = txt_file.name
        chunks = _chunk_text(text, source)
        all_chunks.extend(chunks)
        print(f"  Loaded {len(chunks)} chunks from {source}")

    if not all_chunks:
        raise FileNotFoundError(f"No .txt files found in {DATA_DIR}")

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"[search_docs] Index built: {len(all_chunks)} chunks, dim={dim}")


def _load_index():
    """Load or build the FAISS index and metadata."""
    if not INDEX_PATH.exists() or not META_PATH.exists():
        build_index()
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def search_docs(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Semantic search over annual report documents.

    Args:
        query: Natural language question about company strategy, management
               commentary, reasons, risks, or qualitative information.
        top_k: Number of chunks to return (default 3).

    Returns:
        List of dicts: [{"text": ..., "source": ..., "chunk_id": ..., "score": ...}]
    """
    model = SentenceTransformer(MODEL_NAME)
    index, meta = _load_index()

    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = meta[idx].copy()
        chunk["score"] = float(dist)
        results.append(chunk)

    return results


def format_results(results: list[dict]) -> str:
    """Format results for display in trace log."""
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"  [{i}] source={r['source']} chunk={r['chunk_id']} score={r['score']:.2f}")
        lines.append(f"      {r['text'][:200]}...")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick manual test
    print("=== search_docs standalone test ===\n")
    queries = [
        "What was the main reason for Infosys margin decline?",
        "What strategic priorities did TCS highlight?",
        "How did Wipro handle leadership transition?",
        "What is the headcount reduction strategy?",
        "What AI platforms did companies launch?",
    ]
    for q in queries:
        print(f"Query: {q}")
        results = search_docs(q)
        print(format_results(results))
        print()
