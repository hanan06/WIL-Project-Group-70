# Retrieval, RAG and evaluation scripts

Run these from this folder (`cd src/retrieval`). Setup and the Groq key are explained in the top-level `README.md`.

| Script | Purpose |
|---|---|
| `process_pdfs.py` | Extract text from `data/pdfs/`, split into 200-word chunks (30-word overlap), write `collection.csv` and `collection.jsonl`. Only run if the PDFs change (see warning in the top-level README). |
| `encode.sh`, `index.sh` | Create TCT-ColBERT embeddings and the FAISS HNSW index in `target/`. Pre-built copies are included. |
| `search.py` | Dense retrieval for every test question, written as a TREC run file. |
| `prepare_bm25.py`, `index-bm25.sh`, `search_bm25.py` | BM25 keyword baseline (needs Java). |
| `random_baseline.py` | Random ranking sanity baseline (seed 42). |
| `data.py` | Builds `data/qrels.txt` from `topics.csv` and `groundtruth.csv`. Also builds the illustrative intent run. |
| `eval.py` | Hit@k, Recall, Precision, nDCG, MRR with 95% bootstrap CIs. `python eval.py --selftest` runs the built-in checks. |
| `eval_answers.py` | Runs the 10 in-scope and 5 out-of-scope questions through the full RAG and writes `target/answer_eval.csv` for manual scoring. Needs the Groq key. |
| `run_rag.py` | Interactive RAG assistant: answer plus the three source passages it used. Needs the Groq key (variable near the top of the file). |
| `test_rag.py` | Interactive retrieval-only inspection. No key needed. |
| `main.sh` | Runs the whole evaluation end to end. |
| `search.sh` | DEPRECATED. Kept only to document a failed approach (Pyserini's command line ignores our question file). |

Outputs of `main.sh` go to `../../target/` (runs, BM25 index, `answer_eval.csv`). The scored copy used in the
report is `../../results/answer_eval.csv`.
