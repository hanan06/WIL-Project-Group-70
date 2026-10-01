# BM25 baseline: writes a TREC run file in the same format as search.py
import pandas as pd
from pyserini.search.lucene import LuceneSearcher

TOPICS = "../../data/topics.csv"
INDEX = "../../target/indexes/bm25"
OUTPUT_PATH = "../../target/runs/rag-bm25.txt"
NUM_HITS = 100

topics = pd.read_csv(TOPICS)
searcher = LuceneSearcher(INDEX)

with open(OUTPUT_PATH, "w") as out:
    for question_id, question in topics[["question_id", "question"]].values:
        for rank, hit in enumerate(searcher.search(str(question), NUM_HITS), 1):
            out.write(f"{question_id} Q0 {hit.docid} {rank} {hit.score:.6f} bm25\n")
print(f"Saved {OUTPUT_PATH}")