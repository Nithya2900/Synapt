"""
setup.py

One-time setup: builds the FAISS vector index and SQLite database.
Run this ONCE before using agent.py or evaluate.py.

Usage:
    python setup.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from tools.search_docs import build_index
from tools.query_data import _build_db


def main():
    print("=" * 60)
    print("Agentic RAG — Setup")
    print("=" * 60)
    print()

    print("Step 1: Building SQLite database from financials.csv...")
    _build_db()
    print("  Done.\n")

    print("Step 2: Building FAISS vector index from annual reports...")
    print("  (This downloads sentence-transformers model on first run, ~90MB)")
    build_index()
    print("  Done.\n")

    print("=" * 60)
    print("Setup complete! You can now run:")
    print("  python agent.py 'What was Infosys margin in FY2024?'")
    print("  python agent.py   (interactive mode)")
    print("  python evaluate.py   (run full evaluation set)")
    print("=" * 60)


if __name__ == "__main__":
    main()
