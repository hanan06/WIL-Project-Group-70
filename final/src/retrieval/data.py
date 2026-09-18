import os
import pandas as pd
from rag_core import retrieve

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
EVAL_DIR = os.path.join(DATA_DIR, "eval")
TOPICS = os.path.join(EVAL_DIR, "topics.csv")
POOL_OUTPUT = os.path.join(EVAL_DIR, "pool.csv")

POOL_DEPTH = 10  # how many candidate passages to pull per question for judging


def build_pool():
    os.makedirs(EVAL_DIR, exist_ok=True)
    topics = pd.read_csv(TOPICS)

    rows = []
    for question_id, question in topics[["question_id", "question"]].values:
        hits = retrieve(question, top_k=POOL_DEPTH)
        for rank, hit in enumerate(hits, start=1):
            rows.append({
                "question_id": question_id,
                "question": question,
                "rank": rank,
                "passage_id": hit["passage_id"],
                "passage_text": hit["text"][:300],  # trimmed for readability while judging
                "relevance": "",  # <-- fill this in by hand: 0, 1, or 2
            })

    pool_df = pd.DataFrame(rows)
    pool_df.to_csv(POOL_OUTPUT, index=False)
    print(f"Wrote {len(pool_df)} candidate (question, passage) pairs to {POOL_OUTPUT}")
    print("Next step: open this file, fill in the 'relevance' column by hand, "
          "then save the judged rows as data/eval/qrels.csv")


def build_qrels_from_judged_pool(judged_pool_path=POOL_OUTPUT, output_path=None):
    """
    Once you've filled in the 'relevance' column of pool.csv (or a copy of
    it) by hand, run this to produce the clean qrels.csv that eval.py
    expects (only the judged, relevant rows, in the right column format).
    """
    if output_path is None:
        output_path = os.path.join(EVAL_DIR, "qrels.csv")

    df = pd.read_csv(judged_pool_path)
    df = df[pd.to_numeric(df["relevance"], errors="coerce").notna()]
    df["relevance"] = df["relevance"].astype(int)
    df = df[df["relevance"] > 0]  # keep only judged-relevant rows

    qrels = df[["question_id", "passage_id", "relevance"]]
    qrels.to_csv(output_path, index=False)
    print(f"Wrote {len(qrels)} relevance judgments to {output_path}")


if __name__ == "__main__":
    build_pool()

