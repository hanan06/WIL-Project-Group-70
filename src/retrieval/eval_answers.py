# Answer-level evaluation sheet: runs in-scope + out-of-scope questions through the full RAG and saves a CSV
# for manual scoring (two people score independently, then compare).
import os, time
import pandas as pd
import run_rag

OUT_SCOPE = [
    "How do I file my tax return in Australia?",
    "What is the best way to train for a marathon?",
    "How do I change a flat tyre?",
    "What is the capital of Peru?",
    "How do I invest in shares?",
]
OUTPUT = "../../target/answer_eval.csv"

topics = pd.read_csv("../../data/topics.csv")
questions = [(q, "in_scope") for q in topics["question"]] + [(q, "out_of_scope") for q in OUT_SCOPE]

rows = []
for q, kind in questions:
    hits = run_rag.searcher.search(q, 3)
    ids = [h.docid for h in hits]
    passages = []
    for i in ids:
        m = run_rag.collection_df[run_rag.collection_df[run_rag.id_col].astype(str) == str(i)]
        passages.append(m.iloc[0][run_rag.text_col])
    ans = run_rag.generate(q, passages)
    rows.append({
        "question": q, "type": kind, "retrieved_ids": " ".join(ids), "answer": ans,
        "auto_refused": any(p in ans.lower().replace("’", "'") for p in ["isn't covered", "not covered", "not contain", "do not provide", "does not provide", "not cover"]),
        # fill these in by hand (1 = yes, 0 = no):
        "faithful_to_sources": "", "factually_correct": "", "correct_refusal_if_out_of_scope": "", "notes": "",
    })
    print(f"[{kind}] {q}")
    time.sleep(2)  # avoid Groq rate limits

pd.DataFrame(rows).to_csv(OUTPUT, index=False)
print(f"Saved {OUTPUT} - now score the blank columns manually.")