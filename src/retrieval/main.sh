set -e
mkdir -p ../../target/runs

## Only if you changed the PDFs / corpus (otherwise skip; the pre-built dense index is reused):
#python process_pdfs.py
#bash encode.sh
#bash index.sh

## 1. Build qrels from topics + groundtruth
python data.py

## 2. Build BM25 index
python prepare_bm25.py
bash index-bm25.sh

## 3. Run searches (dense, BM25, random sanity baseline)
python search.py
python search_bm25.py
python random_baseline.py

## 4. Evaluate retrieval
python eval.py --selftest
python eval.py ../../data/qrels.txt ../../target/runs/rag-dense-faiss.txt ../../target/runs/rag-bm25.txt ../../target/runs/random.txt

## 5. Answer-level evaluation (uses the Groq key on line 7 of run_rag.py)
python eval_answers.py