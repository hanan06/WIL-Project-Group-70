# Food Safety & Allergen Risk Assistant (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) system grounded in official
Australian food safety regulations and guidelines (FSANZ Standards, SA
Health, Better Health Channel, NSW Food Authority).

The pipeline uses dense semantic retrieval (TCT-ColBERT embeddings indexed
with FAISS) and external LLM synthesis via Groq to deliver fast, accurate
answers without local hardware bottlenecks.

## Prerequisites

- **Conda**: Miniforge (recommended for macOS/Apple Silicon) or Miniconda.
- **Groq API Key**: a free developer key from [console.groq.com](https://console.groq.com) (no credit card required).

## 1. Environment Setup

```
conda env create -f environment.yml
conda activate walert
```

**VS Code note** ("Import could not be resolved" under `import pyserini` or
`import groq`): press `Cmd/Ctrl+Shift+P` → "Python: Select Interpreter" →
choose the `walert` environment.

## 2. Set your Groq API key (do this once)

Your API key is a secret and must **never** be pasted directly into a source
file or committed to GitHub. Instead:

```
cp .env.example .env
```

Then open `.env` and replace the placeholder with your real key. `.env` is
already listed in `.gitignore`, so it will never be committed.

## 3. Running the System (Pre-Built Index Included)

The FAISS index and embeddings are already pre-computed in `target/`, so you
can run the assistant immediately.

**Option A — Web app (recommended for demos)**

```
cd src/retrieval
streamlit run app.py
```

Opens a chat interface in your browser at `http://localhost:8501`.

**Option B — Command-line chatbot**

```
cd src/retrieval
python run_rag.py
```

Ask a question (e.g. *"How long can cooked rice sit at room temperature?"*).
Type `exit` or `q` to quit.

**Option C — Retrieval only, no LLM/API key needed**

```
cd src/retrieval
python test_rag.py
```

## 4. Rebuilding the Index (Optional)

Only needed if you add or change files in `data/pdfs/`:

```
cd src/retrieval
python process_pdfs.py
bash encode.sh
bash index.sh
```

## 5. Evaluating the system

The evaluation framework (test questions, relevance judgments, retrieval and
generation metrics) lives in `data/eval/` — see `data/eval/README.md` for the
full step-by-step workflow.

## Repository Structure

```
final/
├── README.md                  # This file
├── .env.example                # Template for your Groq API key (copy to .env)
├── .gitignore
├── environment.yml             # Conda environment
├── requirements.txt            # Pip dependencies backup
├── target/                     # Pre-built embeddings, FAISS/BM25 indexes, eval outputs
│   ├── embeddings/
│   ├── indexes/
│   ├── runs/
│   └── eval_results/
├── data/
│   ├── pdfs/                   # Source regulatory documents
│   ├── collection.csv          # Chunked passages (id + text)
│   ├── collection.jsonl        # Same, for Pyserini's dense encoder
│   ├── collection_bm25/        # Same, reformatted for the BM25 indexer
│   └── eval/                   # Test questions + relevance judgments
└── src/
    └── retrieval/
        ├── rag_core.py         # Shared retrieval + generation logic
        ├── run_rag.py          # CLI chatbot
        ├── app.py              # Streamlit web app
        ├── test_rag.py         # Retrieval-only CLI (no API key needed)
        ├── process_pdfs.py     # PDF -> chunks
        ├── encode.sh / index.sh            # Dense embeddings + FAISS index
        ├── prepare_bm25_corpus.py / index-bm25.sh / search_bm25.py  # BM25 baseline
        ├── search.py           # Dense retrieval run for evaluation
        ├── data.py             # Build/turn judgment pool into qrels.csv
        └── eval.py             # Compute retrieval + generation metrics
```
