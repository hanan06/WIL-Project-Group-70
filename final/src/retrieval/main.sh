#!/bin/bash
# End-to-end pipeline for the Food Safety RAG Assistant.
# Run each numbered step from inside src/retrieval/, in order.
# Steps 1-3 only need to be re-run if you add/change PDFs in data/pdfs/.
set -e

echo "## 1. Extract + chunk PDFs into data/collection.csv and collection.jsonl"
python process_pdfs.py

echo "## 2. Encode passages into dense embeddings (target/embeddings/)"
bash encode.sh

echo "## 3. Build the FAISS dense index (target/indexes/)"
bash index.sh

echo "## 4. (Optional) Build + index the BM25 baseline for comparison"
python prepare_bm25_corpus.py
bash index-bm25.sh

echo "## 5. Run retrieval over the evaluation questions (data/eval/topics.csv)"
python search.py
python search_bm25.py

echo "## 6. Build a judgment pool for manual relevance labelling (first time only)"
python data.py
echo ">>> Now open data/eval/pool.csv, fill in the 'relevance' column by hand,"
echo ">>> then run: python -c \"from data import build_qrels_from_judged_pool; build_qrels_from_judged_pool()\""

echo "## 7. Once data/eval/qrels.csv exists, compute retrieval metrics:"
echo "    python eval.py retrieval ../../data/eval/qrels.csv ../../target/runs/rag-dense-faiss.txt ../../target/runs/rag-bm25.txt"

echo "## 8. Compute generation coverage + build the manual faithfulness review sheet:"
echo "    python eval.py generation"
