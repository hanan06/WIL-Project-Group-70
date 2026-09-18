import argparse
import os
import sys

import pandas as pd


def run_retrieval_eval(qrels_path, run_paths):
    from ranx import compare, Qrels, Run

    qrels_df = pd.read_csv(qrels_path)  # columns: question_id, passage_id, relevance
    qrels = Qrels.from_df(
        qrels_df,
        q_id_col="question_id",
        doc_id_col="passage_id",
        score_col="relevance",
    )

    runs = [Run.from_file(p, kind="trec") for p in run_paths]

    report = compare(
        qrels=qrels,
        runs=runs,
        metrics=["ndcg@1", "ndcg@3", "ndcg@5"],
        make_comparable=True,
        rounding_digits=4,
    )

    print(report)
    return report


def run_generation_eval():
    from rag_core import answer_question, is_unanswered

    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    TOPICS = os.path.join(DATA_DIR, "eval", "topics.csv")
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "target", "eval_results")
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, "generation_review.csv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    topics = pd.read_csv(TOPICS)

    rows = []
    for question_id, question in topics[["question_id", "question"]].values:
        answer, passages = answer_question(question, top_k=3)
        rows.append({
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "sources": " | ".join(p["passage_id"] for p in passages),
            "auto_flagged_unanswered": is_unanswered(answer),
            # These two columns are intentionally left blank - fill them
            # in by hand (1 = yes, 0 = no) while reading each answer.
            "faithful_to_sources": "",
            "sources_correct": "",
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_PATH, index=False)

    unanswered_pct = 100 * results_df["auto_flagged_unanswered"].mean()
    print(f"Ran {len(results_df)} questions.")
    print(f"Auto-flagged as unanswered/out-of-scope: {unanswered_pct:.1f}%")
    print(f"Full results + manual-review sheet written to: {OUTPUT_PATH}")
    print("Next step: open that CSV and fill in 'faithful_to_sources' and "
          "'sources_correct' for each row by hand, then report the "
          "percentages in your Findings section.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["retrieval", "generation"])
    parser.add_argument("qrels", nargs="?", help="path to qrels.csv (retrieval mode only)")
    parser.add_argument("runs", nargs="*", help="one or more TREC run files (retrieval mode only)")
    args = parser.parse_args()

    if args.mode == "retrieval":
        if not args.qrels or not args.runs:
            sys.exit("retrieval mode needs: eval.py retrieval <qrels.csv> <run1.txt> [run2.txt ...]")
        run_retrieval_eval(args.qrels, args.runs)
    else:
        run_generation_eval()


if __name__ == "__main__":
    main()
