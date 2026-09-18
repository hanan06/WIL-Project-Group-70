"""
prepare_bm25_corpus.py
-----------------------
Why this file exists:
Pyserini's BM25 indexer (--collection JsonCollection) expects a FOLDER
containing one or more files, where each line is a JSON object with an
"id" field and a "contents" field.

Our collection.jsonl already has one JSON object per line, but the text
field is called "text" (used for the dense/FAISS pipeline), not
"contents". So index-bm25.sh was pointing at a folder that didn't exist,
in a format that wouldn't have worked anyway.

Run this once before bash index-bm25.sh:
    python prepare_bm25_corpus.py
"""

import os
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
SOURCE_JSONL = os.path.join(DATA_DIR, "collection.jsonl")
BM25_CORPUS_DIR = os.path.join(DATA_DIR, "collection_bm25")
OUTPUT_FILE = os.path.join(BM25_CORPUS_DIR, "docs.jsonl")


def main():
    os.makedirs(BM25_CORPUS_DIR, exist_ok=True)

    count = 0
    with open(SOURCE_JSONL, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            fout.write(json.dumps({
                "id": doc["id"],
                "contents": doc["text"],
            }) + "\n")
            count += 1

    print(f"Wrote {count} documents to {OUTPUT_FILE}")
    print(f"You can now run: bash index-bm25.sh")


if __name__ == "__main__":
    main()
