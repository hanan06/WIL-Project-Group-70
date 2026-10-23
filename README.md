# Allergen Safety Assistant: a test-driven RAG assistant for food-allergen questions

Group 70, COSC2669/COSC2816 Work Integrated Learning Project, RMIT University, 2026.

A retrieval-augmented generation (RAG) prototype that answers front-of-house staff questions about food allergens
using only six official Australian guidance documents (719 passages). It shows the passages it used and replies
"not covered" instead of guessing. The pipeline adapts [Walert](https://github.com/rmit-ir/walert).

- Retrieval: TCT-ColBERT v2 embeddings, FAISS HNSW index (Pyserini), top 3 passages.
- Generation: `openai/gpt-oss-20b` through the free Groq API, temperature 0.1.
- Evaluation: Hit@k, MRR, nDCG@10 with bootstrap confidence intervals, BM25 and random baselines,
  and answer-level scoring (faithfulness, correctness, correct refusal, unanswered rate).

Full results and discussion are in the written report. A short results summary is in `eval_summary.md`.

## 1. Environment setup

Requires Conda (Miniforge or Miniconda). Java is installed by the environment file (needed for BM25).

```
conda env create -f environment.yml
conda activate walert
```

`requirements.txt` is a pip-only alternative, but the Conda route is the one we tested.

VS Code: if imports such as `pyserini` or `groq` show "could not be resolved", choose
"Python: Select Interpreter" and pick the `walert` environment.

## 2. Groq API key (free, no credit card)

1. Create a key at https://console.groq.com.
2. Open `src/retrieval/run_rag.py` and paste the key between the quotes on the `GROQ_API_KEY = ""` line near the top.
3. Never commit a real key. We submit the file with the key left blank. (A production system should load the key
   from an environment variable instead.)

The key is only needed for `run_rag.py` and `eval_answers.py`. Retrieval-only steps do not need it.

## 3. Run the assistant (pre-built index included)

The FAISS index and embeddings are already in `target/`, so there is nothing to build.

```
cd src/retrieval
python run_rag.py      # full RAG: answer plus the three source passages it used
python test_rag.py     # retrieval only, no API key needed
```

Example questions: "Are food businesses legally obligated to provide allergen information?",
"What is anaphylaxis and why is it dangerous?". Type `exit` to quit.

## 4. Reproduce the evaluation

```
cd src/retrieval
bash main.sh
```

`main.sh` builds the qrels, builds the BM25 index, runs dense, BM25 and random searches, runs the eval
self-test, prints the retrieval table (Hit@k, MRR, nDCG@10 with 95% bootstrap confidence intervals), and finally
runs all 15 questions through the full RAG to write `target/answer_eval.csv`.

- Retrieval numbers are reproducible (seeded random baseline and bootstrap, deterministic FAISS HNSW search).
- LLM wording is not deterministic, so re-running gives slightly different answer text.
- The scored copy used in the report is `results/answer_eval.csv`. `main.sh` writes to `target/`, so it never
  overwrites it.
- `python eval.py --selftest` checks the metrics against hand-computed cases. If `pytrec_eval` is installed it
  also cross-checks reciprocal rank against it.

## 5. Rebuilding the index (only if you change the PDFs)

```
cd src/retrieval
python process_pdfs.py
bash encode.sh
bash index.sh
```

Warning: `process_pdfs.py` reads the PDFs in the order the operating system lists them, and passage IDs
(P0001 to P0719) depend on that order. `data/groundtruth.csv`, `data/qrels.txt` and the document ranges in
`run_rag.py` all use the current IDs, so a rebuild can silently misalign them. Check the IDs and update those
files before trusting any new results. The current corpus does not need a rebuild.

## Repository structure

```
.
├── README.md
├── environment.yml            # Conda environment (tested)
├── requirements.txt           # pip alternative
├── eval_summary.md            # results summary and handover notes
├── data/
│   ├── pdfs/                  # the six source documents
│   ├── collection.csv         # 719 passages (passage_id, passage)
│   ├── collection.jsonl       # same passages in Pyserini format
│   ├── topics.csv             # 10 in-scope test questions (5 topics x 2 paraphrases)
│   ├── groundtruth.csv        # relevance labels per topic
│   ├── qrels.txt              # relevance labels per question (built by data.py)
│   ├── simulated_intent_baseline.csv   # ILLUSTRATIVE only, not a measured system
│   ├── intent_mapping.csv     # Walert leftover used only by the illustrative baseline
│   └── README.md              # describes every data file
├── results/
│   └── answer_eval.csv        # the 15 answers with manual scores
├── src/
│   └── retrieval/
│       ├── process_pdfs.py    # extract and chunk PDFs (200 words, 30 overlap)
│       ├── encode.sh          # TCT-ColBERT embeddings
│       ├── index.sh           # FAISS HNSW index
│       ├── search.py          # dense retrieval run (TREC format)
│       ├── prepare_bm25.py, index-bm25.sh, search_bm25.py   # BM25 baseline
│       ├── random_baseline.py # random sanity baseline
│       ├── data.py            # builds qrels (and the illustrative intent run)
│       ├── eval.py            # retrieval metrics, bootstrap CIs, --selftest
│       ├── eval_answers.py    # runs 15 questions through the RAG, writes the scoring sheet
│       ├── run_rag.py         # interactive RAG assistant (Groq)
│       ├── test_rag.py        # interactive retrieval-only tool
│       ├── main.sh            # runs the whole evaluation
│       └── search.sh          # DEPRECATED (kept to document a failed approach, see report Appendix E.1)
└── target/
    ├── embeddings/            # pre-built TCT-ColBERT embeddings
    └── indexes/               # pre-built FAISS index (BM25 index and runs are generated by main.sh)
```

## Limitations

Prototype only, not a certified food-safety, legal or medical tool. It does not know a specific menu, so it cannot
say whether a particular dish contains an allergen. The test set is small (10 in-scope and 5 out-of-scope
questions, all in-scope questions from one document) and relevance labels are incomplete, so retrieval scores
are lower bounds. Answers were scored by one reviewer. See the report, sections 3 to 4.
