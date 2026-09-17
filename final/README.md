Food Safety & Allergen Risk Assistant (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) system grounded in official Australian food safety regulations and guidelines (FSANZ Standards, SA Health, Better Health Channel).

The pipeline uses dense semantic retrieval (ColBERT embeddings indexed with FAISS) and external LLM synthesis via Groq to deliver fast, accurate answers without local hardware bottlenecks.
Prerequisites

    Conda: Miniforge (recommended for macOS/Apple Silicon) or Miniconda installed.

    Groq API Key: A free developer API key from console.groq.com (no credit card required).

1. Environment Setup

Open your terminal, navigate to the project root, and recreate the working Conda environment:

conda env create -f environment.yml
conda activate walert

    VS Code Note (Fix "Import could not be resolved"):
    If VS Code displays red warnings under import pyserini or import groq:

        Press Cmd + Shift + P (macOS) or Ctrl + Shift + P (Windows).

        Search for "Python: Select Interpreter".

        Choose the entry pointing to your walert environment (e.g., ~/miniforge3/envs/walert/bin/python or ~/miniconda3/envs/walert/python.exe).

2. Running the System (Pre-Built Index Included)

The FAISS index and ColBERT embeddings are already pre-computed in target/. You can run the assistant immediately without waiting to re-index.
Option A: End-to-End RAG (Retrieval + Answer Synthesis)

    Open src/retrieval/run_rag.py.

    Insert your Groq API key on line 7:
    GROQ_API_KEY = "your_actual_api_key_here"

    Run the interactive terminal assistant:
    cd src/retrieval
    python run_rag.py

    Ask your question at the prompt (e.g., "How long can cooked rice sit at room temperature?" or "What are the mandatory allergen labelling requirements in Australia?"). Type exit or q to quit.

Option B: Retrieval Inspection Only

To inspect the retrieved document passages directly from the FAISS index without invoking the LLM:

cd src/retrieval
python test_rag.py
3. Rebuilding the Index (Optional)

You only need to run this if you modify the dataset or add new PDFs to data/pdfs/:

cd src/retrieval
python process_pdfs.py
bash encode.sh
bash index.sh
Repository Structure

FINAL_RAG/
├── README.md                  # Setup and execution guide
├── environment.yml            # Exported Conda environment specifications
├── requirements.txt           # Pip dependencies backup
├── target/                    # Pre-built ColBERT embeddings & FAISS index
│   ├── embeddings/
│   └── indexes/
├── data/
│   ├── pdfs/                  # Source regulatory documents & guidelines
│   ├── collection.csv         # Chunked passages mapped by passage_id
│   └── collection.jsonl       # Chunked corpus formatted for Pyserini
└── src/
└── retrieval/
├── process_pdfs.py    # Extracts and chunks text from data/pdfs/
├── encode.sh          # Generates ColBERT embeddings
├── index.sh           # Builds the FAISS HNSW graph index
├── search.py          # TREC-compliant retrieval runner
├── test_rag.py        # Interactive CLI for testing retrieval only
└── run_rag.py         # Full end-to-end RAG assistant via Groq API