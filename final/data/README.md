# Data

This folder holds the knowledge base for the Food Safety & Allergen Risk
Assistant, plus the files used to evaluate it.

## Knowledge base

- **`pdfs/`** — the six source documents (FSANZ, Better Health Channel, SA
  Health, NSW Food Authority, and allergen guidance) that the chatbot's
  answers are grounded in.
- **`collection.csv`** — the knowledge base after processing: every PDF is
  split into ~200-word overlapping chunks ("passages"), one row per chunk,
  with columns `passage_id` and `passage`. Produced by
  `src/retrieval/process_pdfs.py`.
- **`collection.jsonl`** — the same passages in the `{"id": ..., "text": ...}`
  format Pyserini's dense encoder expects, also produced by
  `process_pdfs.py`.
- **`collection_bm25/`** — a copy of the passages reformatted with a
  `"contents"` field instead of `"text"`, which is what Pyserini's BM25
  (Lucene) indexer requires. Produced by `src/retrieval/prepare_bm25_corpus.py`.
  You don't need to edit this by hand; regenerate it any time the main
  collection changes.

## Evaluation

- **`eval/`** — the test question set and relevance judgments used to measure
  retrieval and answer quality. See `eval/README.md` for the full workflow.

## Regenerating this folder

If you add or change PDFs in `pdfs/`, re-run from `src/retrieval/`:

```
python process_pdfs.py
bash encode.sh
bash index.sh
python prepare_bm25_corpus.py   # only if you're using the BM25 baseline
bash index-bm25.sh
```
