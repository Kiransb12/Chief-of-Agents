"""
Usage:
    python scripts/ingest_docs.py --path ./data/sample_docs
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rag.ingest import ingest_directory  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="./data/sample_docs")
    args = parser.parse_args()
    ingest_directory(args.path)
