"""
Milestone 1 - Preliminary RAG Pipeline
Loads documents from data/knowledge_base, retrieves relevant passages using BM25,
generates answers using a local Ollama model, and logs results to a CSV + Markdown table.

Run this from inside the src/ folder:
    python3 run_pipeline.py
"""

import os
import re
import csv
from pathlib import Path

from PyPDF2 import PdfReader
from rank_bm25 import BM25Okapi
import ollama

# ---------- CONFIG ----------
KB_DIR = "../data/knowledge_base"
TOPICS_FILE = "../data/topics.csv"
OUTPUT_CSV = "../target/results/results_log.csv"
OUTPUT_MD = "../target/results/results_log.md"
OLLAMA_MODEL = "llama3.2"
TOP_K = 3
CHUNK_MAX_WORDS = 120


# ---------- STEP 1: LOAD DOCUMENTS ----------
def load_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, max_words=CHUNK_MAX_WORDS):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for p in paragraphs:
        words = p.split()
        if len(words) <= max_words:
            if len(words) > 5:  # skip tiny fragments
                chunks.append(p)
        else:
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                chunks.append(chunk)
    return chunks


def load_knowledge_base(kb_dir):
    passages = []
    passage_id = 0
    for fname in sorted(os.listdir(kb_dir)):
        path = os.path.join(kb_dir, fname)
        if fname.lower().endswith(".pdf"):
            text = load_pdf(path)
        elif fname.lower().endswith(".txt"):
            text = load_txt(path)
        else:
            continue
        for chunk in chunk_text(text):
            passages.append({
                "passage_id": f"P{passage_id:04d}",
                "source": fname,
                "text": chunk
            })
            passage_id += 1
    return passages


# ---------- STEP 2: BUILD BM25 INDEX ----------
def build_index(passages):
    tokenized = [p["text"].lower().split() for p in passages]
    bm25 = BM25Okapi(tokenized)
    return bm25


# ---------- STEP 3: RETRIEVE ----------
def retrieve(bm25, passages, question, top_k=TOP_K):
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [passages[i] for i in ranked_idx]


# ---------- STEP 4: GENERATE ----------
def generate_answer(question, context_passages):
    context = "\n\n".join(p["text"] for p in context_passages)
    prompt = f"""Answer the question using ONLY the information in the context below.
If the context does not contain the answer, say "I don't have enough information to answer that."
Keep the answer concise (2-4 sentences).

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    return response["response"].strip()


# ---------- MAIN ----------
def read_topics(topics_file):
    topics = []
    with open(topics_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topics.append(row)
    return topics


def main():
    print("Loading knowledge base...")
    passages = load_knowledge_base(KB_DIR)
    print(f"  Loaded {len(passages)} passages from {KB_DIR}")

    print("Building BM25 index...")
    bm25 = build_index(passages)

    print("Reading test questions...")
    topics = read_topics(TOPICS_FILE)
    print(f"  Loaded {len(topics)} questions")

    results = []
    for row in topics:
        qid = row["question_id"]
        question = row["question"]
        print(f"\nProcessing {qid}: {question}")

        retrieved = retrieve(bm25, passages, question)
        sources_used = "; ".join(f"{p['source']} ({p['passage_id']})" for p in retrieved)
        print(f"  Retrieved from: {sources_used}")

        answer = generate_answer(question, retrieved)
        print(f"  Answer: {answer}")

        results.append({
            "question_id": qid,
            "question": question,
            "retrieved_sources": sources_used,
            "generated_answer": answer
        })

    # Save CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "question", "retrieved_sources", "generated_answer"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV log to {OUTPUT_CSV}")

    # Save Markdown table (handy for pasting into your report)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("| ID | Question | Answer | Sources Used |\n")
        f.write("|---|---|---|---|\n")
        for r in results:
            answer_short = r["generated_answer"].replace("\n", " ")
            f.write(f"| {r['question_id']} | {r['question']} | {answer_short} | {r['retrieved_sources']} |\n")
    print(f"Saved Markdown table to {OUTPUT_MD}")

    # Quick summary stat
    unanswered = sum(1 for r in results if "don't have enough information" in r["generated_answer"].lower())
    print(f"\n% of questions answered: {100 * (len(results) - unanswered) / len(results):.1f}%")
    print(f"% of questions unanswered: {100 * unanswered / len(results):.1f}%")


if __name__ == "__main__":
    main()
