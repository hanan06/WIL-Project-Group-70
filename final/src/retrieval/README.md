# Retrieval & RAG Pipeline

## Setup

1. `conda env create -f environment.yml && conda activate walert`
2. Copy `.env.example` (project root) to `.env` and add your Groq API key.

## Everyday use

| Script | What it does |
|---|---|
| `python run_rag.py` | Interactive command-line chatbot (retrieval + LLM answer). |
| `streamlit run app.py` | The same chatbot as a browser-based web app. |
| `python test_rag.py` | Retrieval only, no LLM/API key needed — quick sanity check of what the FAISS index finds. |

## Rebuilding the knowledge base (only needed if you change the PDFs)

| Script | What it does |
|---|---|
| `process_pdfs.py` | Extracts + chunks text from `data/pdfs/` into `data/collection.csv`/`.jsonl`. |
| `encode.sh` | Generates dense (TCT-ColBERT) embeddings from the chunks. |
| `index.sh` | Builds the FAISS index used for retrieval. |
| `prepare_bm25_corpus.py` | Reformats the chunks for Pyserini's BM25 indexer. |
| `index-bm25.sh` | Builds the BM25 (keyword) index — optional, used as a retrieval baseline. |

## Evaluation

| Script | What it does |
|---|---|
| `search.py` / `search_bm25.py` | Run dense/BM25 retrieval over `data/eval/topics.csv`, producing TREC run files. |
| `data.py` | Builds a candidate-passage pool per question for manual relevance judging, and turns judged results into `qrels.csv`. |
| `eval.py retrieval ...` | Computes nDCG@1/3/5 from `qrels.csv` and the run files. |
| `eval.py generation` | Runs the full chatbot over all test questions, reports the auto-flagged "% unanswered" rate, and writes a CSV for manual faithfulness/source-correctness review. |

Full step-by-step evaluation instructions: see `data/eval/README.md`.

## Shared code

`rag_core.py` holds the actual retrieval + generation logic (loading the
FAISS index, calling the Groq API, etc.). `run_rag.py`, `test_rag.py`, and
`app.py` are thin wrappers around it — this way there's only one place to fix
a bug or change a prompt.
