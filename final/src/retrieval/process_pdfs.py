import os
import re
import pandas as pd
from pypdf import PdfReader

# UPDATE THIS PATH to point to your PDF folder
PDF_DIR = "../../data/pdfs"

OUTPUT_CSV = "../../data/collection.csv"
OUTPUT_JSONL = "../../data/collection.jsonl"

def chunk_text(text, max_words=200, overlap=30):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.strip()) > 50:  # Skip tiny fragments
            chunks.append(chunk.strip())
        i += max_words - overlap
    return chunks

records = []
passage_count = 1

if not os.path.exists(PDF_DIR):
    print(f"Error: Directory not found: {PDF_DIR}")
    exit(1)

pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith('.pdf')]
print(f"Found {len(pdf_files)} PDF files in {PDF_DIR}")

for filename in pdf_files:
    filepath = os.path.join(PDF_DIR, filename)
    print(f"Processing: {filename}")
    try:
        reader = PdfReader(filepath)
        full_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                # Clean multiple whitespaces and newlines
                t = re.sub(r'\s+', ' ', t)
                full_text += " " + t

        chunks = chunk_text(full_text)
        for c in chunks:
            records.append({
                "passage_id": f"P{passage_count:04d}",
                "passage": c
            })
            passage_count += 1
    except Exception as e:
        print(f"Failed to read {filename}: {e}")

df = pd.DataFrame(records)
print(f"\nGenerated {len(df)} total passage chunks.")

# 1. Save CSV for RAG lookup
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")

# 2. Save JSONL directly for Pyserini encode
jsonl_df = df.rename(columns={"passage_id": "id", "passage": "text"})
jsonl_df.to_json(OUTPUT_JSONL, orient='records', lines=True)
print(f"Saved: {OUTPUT_JSONL}")