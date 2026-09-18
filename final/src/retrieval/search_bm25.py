import os
from pyserini.search.lucene import LuceneSearcher
from pyserini.output_writer import OutputFormat, get_output_writer
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
TOPICS = os.path.join(DATA_DIR, "eval", "topics.csv")

INDEX = os.path.join(os.path.dirname(__file__), "..", "..", "target", "indexes", "bm25")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "target", "runs", "rag-bm25.txt")
RUN = "bm25"


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    topics = pd.read_csv(TOPICS)

    searcher = LuceneSearcher(INDEX)

    tag = "foodsafety.rag." + RUN
    num_hits = 100

    output_writer = get_output_writer(
        OUTPUT_PATH,
        OutputFormat('trec'),
        'w',
        max_hits=num_hits,
        tag=tag,
        topics=topics,
    )

    with output_writer:
        for question_id, question in topics[['question_id', 'question']].values:
            hits = searcher.search(str(question), num_hits)
            output_writer.write(question_id, hits)

    print(f"Wrote BM25 run to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
