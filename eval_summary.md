# Food Safety & Allergen Risk Assistant: Evaluation Summary and Handover

This file records the evaluation results already obtained, so nobody has to re-run the pipeline to use them. Numbers come from one full run of `bash main.sh` on 28 Sep 2026.

## 1. What was evaluated

- **System:** RAG over 719 passages chunked from Australian food-allergy guidance PDFs. Dense retrieval (TCT-ColBERT v2 + FAISS HNSW), top 3 passages passed to an LLM (`openai/gpt-oss-20b` via Groq, temperature 0.1), with an instruction to say "not covered" when the passages don't contain the answer.
- **Test set:** 10 in-scope questions (5 topics x 2 paraphrases) and 5 hand-written out-of-scope questions (tax, marathon, tyre, Peru, shares).
- **Important limitation:** all 10 in-scope questions come from **one document** (the Allergy & Anaphylaxis Australia guide). Do not describe them as covering the FSANZ Standards in general.
- **Ground truth (`qrels.txt`):** each question is judged against P0001 or P0002 only (relevance 1). Labels were checked against the passage text and are correct, but **incomplete**: other passages (e.g. P0003, P0024, P0026, P0688) also contain valid answers and count as "non-relevant". Retrieval scores are therefore lower bounds.

## 2. Retrieval results (10 questions, 95% bootstrap CIs)

| Metric | Dense (ColBERT+FAISS) | BM25 | Random |
|---|---|---|---|
| MRR | 0.333 [0.13, 0.58] | 0.453 [0.22, 0.71] | 0.004 |
| Hit@1 | 0.20 [0.0, 0.5] | 0.30 [0.0, 0.6] | 0.00 |
| Hit@3 | 0.30 [0.0, 0.6] | 0.50 [0.2, 0.8] | 0.00 |
| Hit@5 | 0.60 [0.3, 0.9] | 0.60 [0.3, 0.9] | 0.00 |
| Hit@10 | 0.60 [0.3, 0.9] | 0.80 [0.5, 1.0] | 0.00 |
| nDCG@10 | 0.392 [0.17, 0.64] | 0.527 [0.30, 0.76] | 0.00 |

**How to read this**
- Both real methods are far above the random baseline, so the metrics are working and the system retrieves real signal.
- BM25 scores slightly higher than dense retrieval, but the confidence intervals overlap heavily with n=10. **Do not claim either method is better.**
- The generator only sees the top 3, so Hit@3 (dense 0.30) is the number that matters for answer quality.

## 3. How the evaluation code was verified

- `eval.py --selftest` passes on hand-computed cases (MRR, Hit, Recall, nDCG, graded relevance, missing queries, perfect and random runs).
- It was also cross-checked against the independent `pytrec_eval` library (passed in the sandbox where it was tested; skipped on the team's Mac because the library isn't installed).
- Independent cross-check on real data: Hit@3 recomputed by hand from the retrieved IDs saved in `answer_eval.csv` is 0.30, identical to the value `eval.py` reports.
- The original `eval.py` hardcoded 100% and never read a run file. It was replaced.
- Results are reproducible: the random baseline is seeded (42), bootstrap CIs are seeded, and FAISS HNSW search is deterministic. **LLM answers are not** deterministic, so re-running `eval_answers.py` can give different text.

## 4. Answer-level results (`answer_eval.csv`)

Findings from a single reviewer pass over the 15 answers against their retrieved passages (`results/answer_eval.csv`, `notes` column has the reasoning per row). The team decided not to run a second independent scorer and agreement check: it is not required by the assignment brief, which asks only for an effectiveness metric (% unanswered, reported below) and optionally faithfulness/correctness, with no requirement for multiple scorers. These figures are therefore based on one documented pass, not cross-checked by a second reviewer — stated as a limitation in the report and slide 11, not as pending work.

**In-scope (10 questions)**

| Result | Count | Rows (0-indexed) |
|---|---|---|
| Answered, correct and supported by the passages | 6 | 0, 1, 3, 4, 5, 8 |
| Answered correctly, but with small unsupported extras ("rapidly", "if not treated immediately") | 1 | 2 |
| **Refused as "not covered" although the answer exists in the knowledge base** | **3** | 6, 7, 9 |
| Confidently wrong or invented answers | 0 | none seen |

- **Unanswered rate on in-scope questions: 30% (3 of 10).** Cause: retrieval did not return P0001/P0002, which hold the answer. The three affected questions: "only safe way to manage a food allergy", "safe consumption limit" and "deaths mostly outside the home".
- Claims that were spot-checked and found supported: Food Act 2001 and Food Standards Code (P0026), fines or imprisonment up to four years (P0026), "do not guess, seek it out" (P0026), "Some signs and symptoms can lead to death" (P0004).
- Row 9's refusal says the passages only cover reactions in general. That statement was not checked against the passages.

**Out-of-scope (5 questions):** 5 of 5 correctly refused, with no hallucinated answers.

**Interpretation:** when retrieval fails, the system says "not covered" rather than inventing an answer. That is the safer failure mode for allergen advice. The weakness is coverage, not safety.

## 5. Simulated intent baseline (must be labelled as simulated)

`simulated_intent_baseline.csv` mimics an intent-based, Walert-style system in which 5 of 10 paraphrased questions fall back (`AMAZON.FallbackIntent`). It was **not** produced by a deployed system. It is illustrative only.
- Call it "simulated" or "illustrative" in the report, video and slides.
- Do **not** quote "50% vs 30% unanswered" as a measured comparison.
- It is excluded from the quantitative retrieval comparison in section 2.

## 6. Limitations to state in the report

1. Small test set (10 questions, one source document); indicative, not conclusive.
2. Incomplete qrels understate retrieval quality.
3. Manual answer scoring is subjective and was done by one reviewer, not cross-checked by a second independent scorer.
4. Single LLM and single run; answers vary between runs.
5. Chunking is naive (200 words) on noisy infographic text, which likely hurts P0001/P0002 retrieval.
6. Not legal or medical advice.

## 7. Possible improvements (optional, untested)

- Raise top-k from 3 to 5 (dense Hit@5 is 0.60 vs Hit@3 0.30) in `run_rag.py` and in `eval_answers.py` (`search(q, 5)`), then re-run and report both settings.
- Hybrid BM25 + dense retrieval.
- Add extra relevant passages to `groundtruth.csv`, then re-run `data.py` and `main.sh`.
- Clean the noisy text and chunk by section.

## 8. Handover: how to run

**One-time setup:** `conda env create -f environment.yml`, then `conda activate walert`. Java is required for BM25. Put your own Groq key on line 7 of `src/retrieval/run_rag.py` (left blank in the submission; it is only needed for `eval_answers.py` and `run_rag.py`).

**Run everything:**
```
cd src/retrieval
bash main.sh
```
Expected: the retrieval table above (identical numbers), then `../../target/answer_eval.csv` (LLM text will differ slightly).

**Do NOT run** `process_pdfs.py`, `encode.sh` or `index.sh` unless the PDFs change. The pre-built dense index in `target/` matches the current corpus. `process_pdfs.py` writes a `text` field while the current `collection.jsonl` uses `contents`, so a rebuild needs that mismatch resolved first.

**Interactive demo:** `python run_rag.py` (full RAG) or `python test_rag.py` (retrieval only).

## 9. Files

- Keep: `data/` (incl. `pdfs/`, `collection.*`, `topics.csv`, `groundtruth.csv`, `qrels.txt`, `simulated_intent_baseline.csv`), `target/indexes/tct_colbert-v2-hnp-msmarco-faiss` and `target/embeddings/` (dense index, do not delete), and all of `src/retrieval/`.
- Regenerated by `main.sh` (safe to delete for a clean start): `target/runs/*.txt`, `target/indexes/bm25/`, `data/bm25_collection/`, `target/answer_eval.csv`.
- **Scored copy of `answer_eval.csv` goes in `results/`** so a re-run does not overwrite it. `eval_answers.py` overwrites `target/answer_eval.csv`.